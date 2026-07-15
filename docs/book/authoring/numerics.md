# Numerics — printers, the NumPy runtime, and Zoomy's SymPy layer

Between a `SystemModel` (frozen operator form, see
[`system-model.md`](system-model.md)) and a running solver sits one thin
layer: code generation. A *printer* walks the operator matrices, resolves
Zoomy's SymPy extensions (opaque basis atoms, runtime limiters, field
handles), and emits either a callable Python runtime (NumPy) or a source
file (C++, GLSL, JS, OpenFOAM, AMReX, UFL). The Riemann-solver registry is
built on the same machinery — every numerical flux is a `SystemModel`-aware
symbolic object the printer compiles like any other operator.

## 0. NumericalSystemModel — the bundle a solver consumes

A solver does not consume a `SystemModel` directly. `NumericalSystemModel`
(`zoomy_core/numerics/numerical_system_model.py`) is its **numerical sibling**:
a read-only wrapper around the frozen `SystemModel` plus everything a solver
needs that is *not* symbolic-PDE-shaped. Build it by **selecting existing
specs** — never by editing a solver:

```python
from zoomy_core.numerics.numerical_system_model import NumericalSystemModel
nsm = NumericalSystemModel.from_system_model(
    sm,
    riemann=NonconservativeRusanov,                              # a Numerics class (§4)
    reconstruction=ReconstructionSpec(order=2, limiter="venkatakrishnan"),
    diffusion=DiffusionSpec(...),
    regularization=RegularizationSpec(eigenvalue_eps=1e-8),
)
```

It carries the Riemann choice, reconstruction/limiter, diffusion and
regularization specs, and any sub-system splits. So the full pipeline is
**`Model → SystemModel → NumericalSystemModel → Solver`**; the printers below are
the code-generation the runtime/solver uses to execute it.

Defaults are filled in for you: `riemann` → `NonconservativeRusanov`, diffusion
auto-enabled iff `diffusion_matrix` is structurally non-zero, and the
least-squares gradient degree is **always derived** from the `aux_registry`
derivative orders (`resolved_lsq_degree()`) — never a knob. The NSM is read-only
over the `SystemModel`; it never mutates the symbolic operators.

### Worked example — the `hinv` aux field

`hinv = 1/h` shows how an aux field threads the whole pipeline. In
`MalpassetSWE` (`model/models/malpasset.py`) it is:

1. **declared** as aux state — `aux_variables=["hinv"]`;
2. **used** symbolically in the operators as the desingularised reciprocal —
   `u = hu·hinv` inside `flux`, `source`, `diffusion_matrix_explicit`,
   `eigenvalues`;
3. **computed** by the registered `update_aux_variables()` method, a
   Kurganov–Petrova desingularisation
   `hinv = √2·h / √(h⁴ + max(h, ε)⁴)` returned as a `ZArray`;
4. **lowered** like `update_variables`: `from_system_model` registers it as a
   runtime callable the solver applies to `Qaux` each step.

So `hinv` is a *runtime-evaluated aux formula* — distinct from the gradient/topography
aux that `expose_aux_atoms` discovers automatically. At wet/dry faces the Riemann
solver additionally overwrites the `hinv` slot through its `FieldHandle` (§3).

## 1. Printers

### 1a. Inventory

Each backend is **one printer subclass**. They differ only in language-specific
syntax — array type, math namespace, kernel decoration — never in the operators
they emit, so a model verified on one backend is the same symbolic system on
every other. The full set lives under
`library/zoomy_core/zoomy_core/transformation/`:
`to_numpy.py` (`NumpyRuntimeModel`, lambdified NumPy runtime — the
reference execution path); `to_c.py` (`CppModel`, `CppNumerics`;
templated C++ header `template <typename T>`); `to_glsl.py`
(`GenericGlslBase`; WebGL2 GLSL ES 3.00 shaders with `out` array
parameters); `to_js.py` (`GenericJsBase`; allocation-free JavaScript
kernels for the browser solver); `to_openfoam.py`
(`FoamSystemModelPrinter` / `FoamNumericsPrinter` / `FoamUpdateAuxPrinter`;
OpenFOAM solver headers with `Foam::scalar` / `Foam::List` — see
[`../backends/openfoam.md`](../backends/openfoam.md));
`to_amrex.py` (`AmrexModel`, `AmrexNumerics`; AMReX C++ using
`amrex::SmallMatrix` and GPU macros); `to_ufl.py` (`UFLRuntimeModel`;
Unified Form Language for Firedrake / FEniCSx).

The package `__init__.py` carries no public re-exports; printers are
imported by full module path. `helpers.py` is the shared substrate
(`regularize_denominator`, `substitute_sympy_attributes_with_symbol_matrix`).

### 1b. Base layer — `generic_c.py`

`GenericCppBase` (CXX11 SymPy printer subclass), `GenericCppModel`
(model-side header) and `GenericCppNumerics` (Riemann-side header) form
the canonical C-family pipeline. `OutParamCodePrinter` is the sibling
base for languages without array return values (GLSL, JS).

Class-level configuration: `_output_subdir` (subdir under
`settings.output.directory`), `_wrapper_name` (outer `struct` name),
`_is_template_class` (wrap in `template <typename T>`), `real_type`
(`"double"` / `"T"` / `"amrex::Real"`), `math_namespace` (`"std::"` /
`"amrex::Math::"`), `gpu_enabled` (emit `PORTABLE_FN` / CUDA
host-device macros).

Per-method generation hooks a subclass may override:
`_generate_signature_from_function`, `wrap_function_signature`,
`get_includes`, `get_simple_array_def`, `get_array_type` /
`get_array_declaration`, `format_accessor`, `format_assignment`,
`format_array_initialization`, `convert_expression_body`, `_print_Pow`,
`_print_Indexed`, `_print_Function`, `_print_Symbol`, `doprint`.

### 1c. Building a backend printer

Both patterns are consumed by `cls.write_code(target, settings)`.

**Pattern A — thin config wrapper.** When the target is templated C++
and the base behaviour is correct, only the configuration attributes
move. `library/zoomy_core/zoomy_core/transformation/to_c.py`:

```python
from zoomy_core.transformation.generic_c import (
    GenericCppModel, GenericCppNumerics,
)

class CppModel(GenericCppModel):
    """Configuration wrapper. Does not override generation logic."""
    _output_subdir = ".c_interface"
    _is_template_class = True

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.real_type = "T"
        self.math_namespace = "std::"
# CppNumerics(GenericCppNumerics) is the same shape on the Numerics side.
```

**Pattern B — mixin + override.** When the backend has structural
differences (different array type, namespacing, kernel decoration), put
them in a mixin and compose. `to_amrex.py:11`:

```python
class AmrexCore:
    """All AMReX-specific syntax rules. Mixed in with
    GenericCppModel or GenericCppNumerics."""
    def __init__(self, *args, **kwargs):
        self.real_type = "amrex::Real"
        self.math_namespace = "amrex::Math::"
        super().__init__(*args, **kwargs)
    # overrides every print hook + signature/wrapper hooks listed in 1b.

class AmrexModel(AmrexCore, GenericCppModel):
    _output_subdir = ".amrex_interface"
    _is_template_class = False
```

The signature override at `to_amrex.py:128` is representative — AMReX
kernels take `amrex::SmallMatrix const&` instead of `const T*` because
the runtime owns `SmallMatrix` value objects passed by const reference
rather than raw pointer buffers:

```python
def _generate_signature_from_function(self, func_obj):
    """Use const references instead of raw pointers."""
    decls = []
    for key, obj in func_obj.args.items():
        cpp_name = self.ARG_MAPPING.get(key, key)
        if cpp_name in ["Q", "Q_minus", "Q_plus"]:
            t_val = self.get_array_type((self.n_dof_q,))
            decls.append(f"{t_val} const& {cpp_name}")
        # Qaux, n, X, p, gradQ the same const-ref pattern;
        # scalars (time, dX, dt, dx, bc_idx) stay by value.
```

`AmrexNumerics(AmrexCore, GenericCppNumerics)` at `to_amrex.py:223`
is the same pattern on the Numerics side.

**Minimal custom printer:**

```python
class MyBackendModel(GenericCppModel):
    _output_subdir = ".mybackend_interface"
    _is_template_class = False

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.real_type = "myreal_t"
        self.math_namespace = "mb::"

    def wrap_function_signature(self, name, args_str, body_str, shape):
        ret_type = self.get_array_type(shape)
        return (f"MB_KERNEL static {ret_type} {name}({args_str}) noexcept\n"
                f"{{\n{body_str}\n}}\n")
```

The contract: (1) consume a `SystemModel` (or a `Model`,
`SystemModel.from_model`-normalised); (2) emit a runtime object or a
source file via `cls.write_code(target, settings, filename=...)`;
(3) register output under a stable `_output_subdir`.

## 2. `NumpyRuntimeModel.from_system_model(sm)`

The lambdify endpoint at
`library/zoomy_core/zoomy_core/transformation/to_numpy.py:289`:

```python
@classmethod
def from_system_model(cls, sm, *, module=None, printer=None):
    """Build a runtime by lambdifying a SystemModel's stored matrices."""
```

The returned object exposes the operator surface as plain callables
taking `(Q, Qaux, p)` and returning NumPy arrays:
`flux` → `(n_eq, n_dim, n_cells)`; `nonconservative_matrix` →
`(n_eq, n_state, n_dim, n_cells)`; `source` / `source_explicit` →
`(n_eq, n_cells)`; `hydrostatic_pressure` → `(n_eq, n_dim, n_cells)`;
`mass_matrix` → `(n_eq, n_state, n_cells)`; `quasilinear_matrix` /
`source_jacobian` per [`system-model.md`](system-model.md);
`eigenvalues(Q, Qaux, p, n)` → `(n_eq, n_cells)`; and
`boundary_conditions(bc_idx, time, position, distance, Q, Qaux, p, n)`
returning the per-face state.

Each operator is compiled through `_lambdify_function`, which routes
through `_vectorize_expression` so constant entries broadcast cleanly
on full-grid arrays (the `zeros_like` / `ones_like` anchor trick).
`nonconservative_matrix` and `quasilinear_matrix` are emitted per-axis
slabs and `np.stack`'d along the trailing axis; `diffusion_matrix` is
rank-4 and stacked along the trailing two. `from_system_model` is the
right entry when the pipeline starts from a `SystemModel` that may not
have a backing `Model` (e.g. a splitter output); the
`NumpyRuntimeModel(model)` constructor remains preferred when the
model's full `functions` registry is required. This is what every NumPy
solver internally invokes — see
[`../backends/numpy.md`](../backends/numpy.md).

## 3. Zoomy's SymPy extensions

SymPy subclasses that look opaque to the printer but carry the
metadata the runtime needs to resolve them.

**`Basisfunction`** —
`library/zoomy_core/zoomy_core/model/models/basisfunctions.py:12`.
Opaque sympy `Function` subclasses with a `_basis` back-reference;
each `phi_k(ζ)` routes back to its basis cache for integration:

```python
self.phi_fn = type(symbol, (sympy.Function,),
                   {"_basis": self, "nargs": 2})
# Sum(amp(k) · basis.phi_fn(k, ζ), (k, 0, L))
```

`EvaluateIntegrals` walks an integrand, reads `func._basis` on every
opaque atom, and dispatches to that basis's `evaluate_integral`.
Distinct bases get distinct sympy classes via the `symbol` argument,
so mixing is automatic and registry-free.

**`limit()`** — `library/zoomy_core/zoomy_core/model/numerics.py:43`.
Inert `sympy.Function` wrapper the runtime resolves to a TVD slope
limiter (`minmod`, `venkatakrishnan`, `barth_jespersen`):

```python
from zoomy_core.model.numerics import limit
S[1, 0] = - 2 * P_1 * limit(sp.Derivative(b, x), sp.Symbol("minmod"))
```

`.doit()` does not strip it; `expose_aux_atoms` lifts each `limit(...)`
to a fresh aux symbol `{target}_{axes}__{scheme}`, and the runtime
aux-refresh applies the named limiter to the LSQ gradient before
evaluating the source.

**`FieldHandle`** —
`library/zoomy_core/zoomy_core/fvm/riemann_solvers.py:22`. Resolves a
named field (`"h"`, `"b"`, `"hinv"`) to either state `Q` or auxiliary
`Qaux` at construction time, exposing `minus` / `plus` / `state`
Symbols on the L / R / cell-centre side of a face. Riemann code reads
`h.access(qL, auxL)` and stays agnostic to whether bathymetry lives in
the state (legacy SWE) or `Qaux` (chain-DAE).

**Regularization** — `(h + ε)^(-n)` rewrite for inverse h-powers near
wet/dry interfaces; declared at construction via the `regularization=`
keyword on the subclass (e.g. `HyperbolicSME(regularization=...)`).
The printer then sees bounded denominators only.

## 4. Riemann solver registry

`library/zoomy_core/zoomy_core/fvm/riemann_solvers.py:80`:

```python
class Numerics(param.Parameterized, SymbolicRegistrar):
    """Symbolic numerics over a SystemModel."""
    name  = param.String(default="NumericsV2")
    model = param.Parameter(default=None)
```

A `Numerics` consumes a `SystemModel` (a `Model` is normalised once via
`SystemModel.from_model`). It builds per-face placeholder vectors
(`Q_minus`, `Q_plus`, `Qaux_minus`, `Qaux_plus`, `flux_minus`,
`flux_plus`, `source_term`), auto-locates standard named fields via
`find_field("h" | "b" | "hinv")`, and registers `numerical_flux` and
`numerical_fluctuations` as symbolic functions the printer lambdifies
exactly like a model operator.

Built-ins: `Rusanov` (local Lax-Friedrichs; identity dissipation,
bed-row zeroed when `b` lives in `Q`); `PositiveRusanov`
(Audusse-Bristeau-Klein hydrostatic reconstruction on top); `HLL`
(Davis-bounds two-wave; falls back to LLF when `eigenvalues is None`);
`HLLC` (HLL plus contact / shear wave for the free-surface family;
requires `h`); `NonconservativeRusanov` (Rusanov with NCP fluctuation
viscosity from `quasilinear_matrix`). Higher compositions
(`PositiveHLL`, `PositiveNonconservativeRusanov`,
`PositiveNonconservativeHLL`, `QuasilinearRusanov`,
`PositiveQuasilinearRusanov`) live alongside.

**User extension point** — `SymbolicRegistrar.register_symbolic_function`
from `library/zoomy_core/zoomy_core/model/basefunction.py:46`:

```python
def register_symbolic_function(self, name, method_ref, sig_struct):
    """Register a symbolic function under `self.functions[name]`,
    install a `self.call[name]` proxy that emits an indexed function
    call (`Model<T>::name(...)` or `name(...)`)."""
```

`SymbolicRegistrar` is a mixin; both `Model` and `Numerics` inherit
it. A custom Riemann variant subclasses `Numerics`, overrides
`numerical_flux` (and optionally `numerical_fluctuations`), and the
parent `_initialize_functions` registers them — the printer picks them
up automatically:

```python
import sympy as sp
import param
from zoomy_core.fvm.riemann_solvers import Rusanov

class MyDampedRusanov(Rusanov):
    name = param.String(default="MyDampedRusanov")
    damping = param.Number(default=0.5)

    def numerical_flux(self):
        return sp.Float(self.damping) * super().numerical_flux()
```

For a truly custom callable used inside an operator expression (not a
Riemann flux), call `register_symbolic_function(name, method_ref, sig)`
directly on the model so the printer resolves the symbolic
`Function(name)(...)` atom against a registered kernel.

## 5. User-supplied functions

Three places let the author hand the runtime something the symbolic layer
cannot, or should not, produce in closed form.

**Numerical eigenvalues.** A symbolic characteristic polynomial is expensive or
impossible for large/strongly-coupled systems. Set `eigenvalue_mode = "numerical"`
on the `Model` and you supply **nothing else**: `eigenvalues()` returns a zero
placeholder, and the solver builds the normal-projected quasilinear matrix
`A_n = Σ_d n_d · quasilinear_matrix[:,:,d]`, adds the `eigenvalue_eps`
regularisation diagonal, and takes `np.real(np.linalg.eigvals(A_n))` per face at
runtime. Use it whenever the symbolic spectrum is unwieldy; keep `"symbolic"`
(the default) when it closes cleanly, because it is far cheaper per step.

**Conditional expressions.** Author branchy physics with
`sp.Function("conditional")(cond, true, false)` (or a `Piecewise`). **No
registration is needed** — every printer maps `conditional` to its native form
(`np.where`, `ufl.conditional`, GLSL/JS/C ternary). Wet/dry clamps
(`clamp_positive`, `clamp_momentum`) and `safe_denominator` are provided the same
way. Prefer `conditional`/`Piecewise` over a Python `if` in operator code — the `if`
would bake one branch into the printed kernel.

**Custom derivatives / symbolic callables.** When an operator expression needs a
helper the framework has no slot for, register it with
`register_symbolic_function(name, method_ref, sig_struct)` on the originating
object (`Model`, `Kernel`, or `Numerics` — all mix in `SymbolicRegistrar`). This
stores `Function(name, definition, args=sig)` and installs a proxy that emits the
symbolic call `name(...)`; the printer then resolves that atom against the
backend implementation in its `module`/`c_functions` map at lambdify time. The
standard operator slots (`flux`, `source`, `update_aux_variables`, …) are
registered through exactly this mechanism — a custom function is not a special
case, just one more entry.

## 6. The solver architecture — a march over a stage list

A solver marches over an **ordered list of stages**. Each stage is
`Stage(label, kind, sm)`:

- **`label`** — a *stable, deterministic* per-stage identifier
  (`"predictor"` / `"pressure"` / `"corrector"` for the Chorin split).
  Code-generating backends key generated names off it — amrex emits
  `Model_<label>.H`. It **must never be derived from the list position**:
  any reordering would silently rename every header and invalidate every
  build cache.
- **`kind`** — the *executor kind*, drawn from a small extensible
  vocabulary: `hyperbolic | elliptic | pointwise`. The kind selects which
  per-kind executor the backend runs for the stage; it is the concept the
  six backends (numpy / jax / dmplex / amrex / OpenFOAM / firedrake) march
  *concept-for-concept* the same way.
- **`sm`** — the `SystemModel` for the stage (frozen, printable exactly
  like any other — §0/§1).

The split is **data, not solver code**: `model.chorin_split(dt)` returns a
`SplitForPressureResult` whose `.stages` property is

```python
[Stage("predictor", "hyperbolic", SM_pred),   # evolution rows
 Stage("pressure",  "elliptic",   SM_press),  # divergence-constraint block
 Stage("corrector", "pointwise",  SM_corr)]   # projection update
```

A **plain `HyperbolicSolver` is the degenerate one-stage list**
`[Stage("evolution", "hyperbolic", sm)]`; a split (Chorin, or any future
pressure-corrector structure) just adds stages. `ChorinSplitVAMSolver`
accepts either the legacy positional triple or `stages=split.stages`, and
binds each sub-model to its executor **by `kind`, never by position**.

### A backend supplies one executor per kind

| kind | what the executor does |
|---|---|
| `hyperbolic` | the Riemann/reconstruction step (§1, §4) — a `HyperbolicSolver` substep; cannot silently under-solve |
| `elliptic`   | the linear/GMRES (or KSP) solve of the stage's constraint block |
| `pointwise`  | the `update_variables` scatter (§2) — a cell-local algebraic update |

Adding a `kind` is additive: a backend that does not implement a kind
refuses that stage list loudly, rather than mis-marching it.

### The `elliptic` executor contract — it MUST surface its residual

An `elliptic` executor **must compute and surface the relative residual**

$$\mathrm{rel\_resid} \;=\; \frac{\lVert b - A x\rVert}{\lVert b\rVert}$$

after its solve (numpy: `solver.last_elliptic_rel_resid`). This is a hard
part of the contract, not a diagnostic nicety:

- An executor that returns only `x` is **indistinguishable between
  "solved" and "gave up"**. jax's `jax_gmres` hard-codes `info = 0` — its
  own docstring calls it a *"placeholder"* — so a converged solve and one
  starved to `maxiter=1` return identical status; the retry branch that
  read `info != 0` was structurally unreachable. dmplex hit the same class
  from the other side: PCNONE **diverges while reporting green
  downstream** — a wrong answer that passed.
- The one extra matvec that measures `‖b − A x‖/‖b‖` costs <1% of a solve
  that already spent ~10² matvecs, and it is the difference between a wall
  you can see and one you find months later in a baseline.

The elliptic executor **must also consume the stage's declared boundary
conditions, and must not substitute a default** (REQ-174). A silently
substituted BC is the same silent-wrong-answer class as an unreported
residual: the numpy Chorin pressure stage hardcoded a homogeneous-Neumann
(`u_boundary_face="extrapolation"`) gradient and never read
`SM_press.boundary_conditions`, so a user-declared Dirichlet P pin at an
outflow was silently replaced by `zeroGradient` — a well-posed solve of the
*wrong* problem (the operator is full-rank without the pin, so this is not a
singularity). The executor must honour `SM_press.boundary_conditions`: the
declared per-field BC drives the derivative-aux stencil, and a declared
Dirichlet is imposed as a real Dirichlet row in the operator/residual (so the
boundary cells carry the value exactly) — no shift, penalty, or rank fix. When
no BC is declared, homogeneous Neumann (`∂ₙP = 0`) remains the default Chorin
pressure BC.

`hyperbolic` and `pointwise` stages need nothing added — neither can
silently under-solve.

> **Cross-backend cost note.** For numpy/jax a stage list is a constructor
> and a loop (the per-stage executors already exist — they were just
> Chorin-named). For amrex/dmplex/OpenFOAM the stage march is *structure*
> (the printer emits N named model headers; the C++ driver allocates
> per-stage state), so "backends follow" is not a uniform-cost rename.

## Running example — SWE end-to-end

From `thesis/notebooks/legacy/modeling/swe/simple_swe_v2.py`
(`SWEModelV2` and `make_model()`; add its directory to `PYTHONPATH`, or
substitute any `Model` subclass):

```python
from zoomy_core.systemmodel import SystemModel
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
from simple_swe_v2 import make_model

m  = make_model()                              # author's Model
sm = SystemModel.from_model(m)                 # freeze
rt = NumpyRuntimeModel.from_system_model(sm)   # lambdify

flux_arr = rt.flux(Q, Qaux, p)                 # (n_eq, n_dim, n_cells)
```

`rt.boundary_conditions`, `rt.aux_boundary_conditions`,
`rt.boundary_gradients`, `rt.eigenvalues` populate on the same object.
For the upstream derivation that produces `SystemModel.flux`, see
[`model.md`](model.md); for the operator tag catalogue and post-freeze
surface, [`system-model.md`](system-model.md). The SME + $\sigma$
thread that pages A and B use is the advanced reference for chain-DAE
printing.

## Authoring checklist

- After `SystemModel.from_model(...)`, no naked `Function` atoms remain
  in the operators — `expose_aux_atoms` has routed them all to
  `aux_state` (cross-check `sm.aux_registry` if a printer emits an
  unresolved symbol).
- Every custom callable used inside an operator expression is registered
  via `SymbolicRegistrar.register_symbolic_function(...)` on the
  originating object so the printer resolves the function atom to a
  real kernel.
- Inverse h-powers are regularised at the `Model` layer (e.g.
  `HyperbolicSME(regularization=...)`); never paper over
  division-by-zero downstream of the printer.
- Targeting a non-NumPy backend: subclass `GenericCppModel` /
  `GenericCppNumerics` (or `OutParamCodePrinter` for languages without
  array return values), set `_output_subdir`, override only the
  language-specific hooks, and validate that the emitted source
  compiles in the target environment.
- Cross-backend invariant: every printer consumes the *same*
  `SystemModel`. If a backend needs different operators, fix the
  derivation, not the printer.
