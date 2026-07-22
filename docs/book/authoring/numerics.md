# Numerical System Model

A `NumericalSystemModel` (NSM) is a `SystemModel` **plus the scheme choices**:
which Riemann solver, what reconstruction order, how diffusion and sources are
treated, how the depth is regularised. It is the last symbolic object in the
chain — everything downstream (the NumPy runtime, the JAX kernels, the emitted
C++) is generated from it.

```python
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec

nsm = NumericalSystemModel.from_system_model(
    sm, reconstruction=ReconstructionSpec(order=2))
```

The NSM **is a** `SystemModel` (it subclasses it) and `from_system_model`
promotes the object in place — `nsm is sm`. So every operator, the shape
contract, and `describe()` from the [System Model](system-model.md) page carry
over unchanged.

## Building one

`NumericalSystemModel.from_system_model(sm, **kwargs)` — every knob, with its
default:

| Argument | Default | What it selects |
| --- | --- | --- |
| `riemann` | `NonconservativeRusanov` | the numerical flux |
| `reconstruction` | order 1 (constant) | `ReconstructionSpec` |
| `diffusion` | auto | `DiffusionSpec`; enabled iff `diffusion_matrix` is non-zero and `nu > 0` |
| `source_treatment` | `"explicit"` | `"explicit"` / `"local_source"` / `"coupled"` |
| `depth_regularization` | `None` | e.g. Kurganov–Petrova `hinv` |
| `regularization_eps` | `1e-2` | its scale |
| `eigenvalue_treatment` | `"regularize"` | how the spectrum is guarded |
| `eigenvalue_eps` | `1e-8` | regularisation diagonal |
| `eigenvalue_guard` | `None` | extra guard on eigenvalue powers |
| `dt_max` | `5.0` | hard timestep cap |
| `normalize_normal` | `True` | normalise face normals |
| `extra_operations` | `[]` | extra `sm.apply(...)` ops in the pipeline |
| `additional_systems` | `[]` | sub-systems for split solvers |
| `scaled_q_indices` | `None` | rows carrying a scaling |

Two spec objects:

```python
ReconstructionSpec(order=1, limiter="venkatakrishnan",
                   free_surface_aware=False, positivity="")
DiffusionSpec(enabled=True, scheme="crank_nicolson", nu=None)
```

`NumericalSystemModel.from_model(model, **kwargs)` is the shortcut for
`from_system_model(SystemModel.from_model(model), **kwargs)`.

```{note}
**No numerical constant belongs in a backend.** Every threshold, tolerance,
epsilon and safety factor is set here and emitted through the code printer, so
all backends are compatible by construction. A float literal next to a
comparison in backend source is a defect, not a detail.
```

## Riemann solvers

From `zoomy_core.fvm.riemann_solvers`:

| Solver | Notes |
| --- | --- |
| `Rusanov` | local Lax–Friedrichs; bed row zeroed when `b` is in `Q` |
| `HLL` | Davis two-wave; falls back to LLF when `eigenvalues is None` |
| `HLLC` | HLL plus contact/shear wave; requires `h` |
| `NonconservativeRusanov` | **default** — Rusanov + NCP fluctuation viscosity |
| `NonconservativeRoe` | path-conservative Roe |
| `PositiveRusanov` | + Audusse–Bristeau–Klein hydrostatic reconstruction |
| `PositiveHLL`, `PositiveNonconservativeRusanov`, `PositiveNonconservativeHLL` | positivity-preserving compositions |
| `QuasilinearRusanov`, `PositiveQuasilinearRusanov` | driven by `quasilinear_matrix` |
| `WellBalancedNonconservativeRusanov` | well-balanced variant |

A custom variant subclasses `Numerics` (or a built-in) and overrides
`numerical_flux` / `numerical_fluctuations`. The printer picks it up
automatically — no registration:

```python
import sympy as sp, param
from zoomy_core.fvm.riemann_solvers import Rusanov

class MyDampedRusanov(Rusanov):
    name = param.String(default="MyDampedRusanov")
    damping = param.Number(default=0.5)

    def numerical_flux(self):
        return sp.Float(self.damping) * super().numerical_flux()
```

```{note}
Well-balancing, positivity and truncation are **scheme decisions and live
here**, at the symbolic level. Finding one hand-written inside a backend loop
is finding a bug.
```

## Handing the runtime something symbolic can't do

**Numerical eigenvalues.** When the symbolic spectrum is unwieldy, set
`eigenvalue_mode = "numerical"` on the `Model`. You supply nothing else: the
solver builds `A_n = Σ_d n_d · quasilinear_matrix[:,:,d]`, adds the
`eigenvalue_eps` diagonal, and takes `np.real(np.linalg.eigvals(A_n))` per face.
Keep the default `"symbolic"` when the spectrum closes cleanly — it is far
cheaper per step.

**Conditionals.** Write branchy physics as
`sp.Function("conditional")(cond, true, false)` or a `Piecewise`. No
registration needed: every printer maps it to its native form (`np.where`,
`ufl.conditional`, a C/GLSL/JS ternary). `clamp_positive`, `clamp_momentum` and
`safe_denominator` come the same way. Never use a Python `if` inside operator
code — it bakes one branch into the printed kernel.

**Custom callables.** `register_symbolic_function(name, method_ref, sig_struct)`
on the `Model`, `Kernel` or `Numerics` stores the function and installs a proxy
emitting `name(...)`; the printer resolves it against the backend
implementation at lambdify time. The standard operator slots are registered
through exactly this mechanism — a custom function is not a special case.

## Running it

The NumPy solver is the reference implementation; every other backend is ported
from it.

```python
from zoomy_core.mesh import BaseMesh
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.fvm import timestepping

mesh   = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=50)
solver = HyperbolicSolver(time_end=0.5, compute_dt=timestepping.adaptive(CFL=0.9))
Q, Qaux = solver.solve(mesh, nsm, write_output=False)
```

Solvers in `zoomy_core.fvm`: `HyperbolicSolver` (`solver_numpy`), `IMEXSolver`
(`solver_imex_numpy`), `SplittingSolver` (`solver_splitting_numpy`),
`ChorinSplitVAMSolver` (`solver_chorin_vam_numpy`), a DAE solver
(`solver_dae_numpy`), and the σ-3D split solver (`sigma3d_split_solver`).
Timestepping: `timestepping.adaptive(CFL=…)` or `timestepping.constant(dt=…)`.

Two lower-level entry points, if you want the runtime without the solver:

- `nsm.build_numerics()` — instantiate the symbolic Riemann numerics.
- `nsm.build_runtime_numpy()` — lambdify into a runtime with callable `.flux`,
  `.source`, `.eigenvalues`, `.boundary_conditions`.

## Generating code

Every printer accepts a `Model`, a `SystemModel` or an NSM — the input is
normalised at the front door via `to_numerical_system_model`.

| Target | Classes | Entry |
| --- | --- | --- |
| Generic C++ | `GenericCppModel`, `GenericCppNumerics` | `.create_code()` / `.write_code(...)` |
| Plain C++ | `CppModel`, `CppNumerics` | `.create_code()` |
| AMReX | `AmrexModel`, `AmrexNumerics` | `.create_code()` |
| OpenFOAM | `FoamSystemModelPrinter`, `FoamNumericsPrinter` | `.create_code()` |
| GLSL | `GlslModel`, `GlslNumerics` | `.generate()` |
| JavaScript | `JsModel`, `JsNumerics` | `.generate()` |
| NumPy runtime | `NumpyRuntimeModel` | `.from_system_model(sm)` |
| UFL / Firedrake | `UFLRuntimeModel` | `.from_nsm(nsm)` |

```python
from zoomy_core.transformation.to_c import CppModel
print(CppModel(nsm).create_code())
```

Modules live under `zoomy_core.transformation`; the package `__init__` is empty,
so import by module path. Printers are **syntax only** — they translate
expressions, they never make scheme decisions.

## Split solvers

Multi-stage schemes march an ordered list of `Stage(label, kind, sm)`, where
`kind ∈ {hyperbolic, elliptic, pointwise}` selects the executor and `label` is a
stable identifier that code-generating backends key names off (AMReX emits
`Model_<label>.H`). The split is data, not solver code:
`model.chorin_split(dt)` returns the stages, and each stage's `sm` is an
ordinary `SystemModel` that prints like any other.

## Checklist

- `sm.describe()` before running — the operator slots are the cheapest
  bug detector you have.
- Riemann solver matches the physics: NCP terms need a `Nonconservative*`
  variant; wet/dry needs a `Positive*` one.
- Second-order reconstruction needs a limiter appropriate to the mesh.
- Every threshold set here, none in the backend.

Backends: [NumPy](../backends/numpy.md) (reference) ·
[JAX](../backends/jax.md) · [AMReX](../backends/amrex.md) ·
[OpenFOAM](../backends/openfoam.md).
