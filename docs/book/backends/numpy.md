# NumPy backend

The NumPy backend is the **reference implementation** every other backend is
verified against. It ships **inside** `zoomy_core` at `library/zoomy_core/zoomy_core/fvm/`
— there is no separate `zoomy_numpy` package. Same `Model` / `SystemModel` API.

- Pure Python / NumPy / SciPy; no external solver dependency.
- Cell-centred finite-volume; symbolic Riemann solver; NCP path integral.
- Solvers are `param.Parameterized` — every knob is class-level and
  auto-introspected by the GUI / CLI.

## Install

```bash
pip install zoomy_core
```

## Solver inventory

| Class | File | What it adds |
| --- | --- | --- |
| `HyperbolicSolver` | `fvm/solver_numpy.py:274` | Explicit RK1 / SSP-RK2 flux + symbolic Riemann + source. **The reference.** |
| `FreeSurfaceFlowSolver` | `fvm/solver_numpy.py:1072` | Positive (hydrostatic-reconstruction) Rusanov + η=h+b well-balanced reconstruction + wet/dry. Needs `h`, `b`. |
| `RoeFreeSurfaceFlowSolver` | `fvm/solver_numpy.py:1122` | Free-surface with path-conservative Roe `|A|` matrix dissipation. |
| `IMEXSolver` | `fvm/solver_imex_numpy.py:41` | Explicit flux + implicit diffusion + implicit source (Newton / Newton-GMRES). |
| `FSFIMEXSolver` | `fvm/solver_imex_numpy.py:333` | IMEX with the free-surface positive Rusanov flux. |
| `SplittingSolver` | `fvm/solver_splitting_numpy.py:35` | Chorin pressure projection: hyperbolic → viscous diffusion → Poisson correction. |
| `FSFSplittingSolver` | `fvm/solver_splitting_numpy.py:433` | Pressure-splitting for free-surface flow (velocities at `[2..2+dim-1]`). |
| `ColumnIntegratingSolver` | `fvm/solver_column.py:24` | IMEX on extruded `(x,ζ)` meshes; depth integrals evaluated numerically per column. |
| `DAESolver` | `fvm/solver_dae_numpy.py:54` | Monolithic index-1 DAE-PDE via Ascher-Ruuth-Spiteri IMEX-RK (ARS232 / ARS343). Order-1 correctness reference for the VAM chain. |
| `ChorinSplitVAMSolver` | `fvm/solver_chorin_vam_numpy.py:300` | Chorin split of the VAM DAE into predictor + elliptic pressure + corrector sub-systems (from `split_for_pressure`). |

Common base: `Solver` (`fvm/solver_numpy.py:124`, a `param.Parameterized`) holds
the `settings` parameter, state allocation, runtime construction, and the
aux-refresh (it fills the local aux formula leg plus the LSQ-gradient leg from
`sm.aux_registry`); every solver inherits it. Within NumPy the implicit solvers
form a subclass chain `ColumnIntegratingSolver ← SplittingSolver ← IMEXSolver`,
and the free-surface variants override only the numerics/reconstruction build.

## Operator-splitting stages

```
  [ explicit hyperbolic flux + source ]   -- RK1 / SSP-RK2 on F, B·∂_x Q, S_e
              ▼
  [ implicit / explicit diffusion ]        -- A:∇Q (optional, IMEX)
              ▼
  [ pressure Poisson correction ]          -- free-surface / incompressible
              ▼
  [ state update ]                         -- update_q, update_qaux
```

`HyperbolicSolver` runs stage 1; `IMEXSolver` adds stage 2; `SplittingSolver`
adds the Chorin correction in stage 3; `DAESolver` collapses everything into
a monolithic stage Newton (singular `mass_matrix` allowed);
`ChorinSplitVAMSolver` runs predictor → elliptic → corrector on three
sub-system models from `split_for_pressure`.

## Tunable `param` knobs

A solver carries **only time-stepping knobs**. Numerical knobs
(reconstruction order, limiter, eigenvalue regularisation) are *not* solver
params — they live on the `NumericalSystemModel` so the same solver can run any
discretisation. Passing them to a solver constructor raises `TypeError`.

```python
# HyperbolicSolver — time stepping only
time_end   = param.Number(default=0.1, bounds=(0, None))
min_dt     = param.Number(default=1e-6, bounds=(0, None))
compute_dt = param.Parameter(default=None)   # default: timestepping.adaptive(CFL=0.9)

# Numerical knobs — on the NumericalSystemModel specs, NOT the solver:
ReconstructionSpec(order=2, limiter="venkatakrishnan")   # 1=PWC, 2=MUSCL
RegularizationSpec(eigenvalue_eps=1e-8)

# IMEXSolver (plain class attrs, not yet param-typed)
source_mode = "auto"; implicit_tol = 1e-8; implicit_maxiter = 8
gmres_tol = 1e-7; gmres_maxiter = 40; jv_backend = "analytic"

# SplittingSolver
pressure_gmres_tol     = param.Number(default=1e-6)
pressure_gmres_maxiter = param.Integer(default=200)
viscosity              = param.Number(default=0.01)
```

`DAESolver`: `method` (`"ars232" | "ars343"`), `newton_tol`, `newton_maxit`,
`h_index`, `nc_integration_order`, `well_balanced`, `jacobian_mode`
(`"sparse_fd" | "dense_fd"`). `ChorinSplitVAMSolver`: `pressure_tol`,
`pressure_maxit`, `time_order`, `pressure_solver` (`"gmres" | "lu"`),
`riemann_solver` (`"hr" | "ncp"`). Aux-derivative limiting (for a shock
that overshoots into the elliptic/source block) is model-declared via
`zoomy_core.model.numerics.limit` (a `limited_derivative` tag), not a solver knob.

## The numerics menu — what you compose, not subclass

A solver picks its discretisation off a fixed menu carried by the
`NumericalSystemModel`, never by editing the solver. Choose a Riemann flux
(`riemann=`) and a `ReconstructionSpec(order, limiter, free_surface_aware)`:

**Riemann fluxes** (`fvm/riemann_solvers.py`): `Rusanov`, `HLL`, `HLLC`,
`PositiveRusanov` (Audusse hydrostatic reconstruction), `NonconservativeRusanov`
(**the default**, NCP path-integral fluctuations),
`WellBalancedNonconservativeRusanov`, `PositiveHLL`,
`PositiveNonconservativeRusanov` (free-surface default),
`NonconservativeRoe` (Roe `|A|=R|Λ|L` matrix dissipation),
`PositiveNonconservativeHLL`, `QuasilinearRusanov`,
`PositiveQuasilinearRusanov`.

**Reconstruction** (`fvm/reconstruction.py`): `ConstantReconstruction(V2)`
(1st order), `MUSCLReconstruction` / `LSQMUSCLReconstruction` (2nd order MUSCL),
`SurfaceReconstruction` (well-balanced η=h+b), `FreeSurfaceMUSCL` /
`FreeSurfaceLSQMUSCL` (+ wet/dry), `PrimitiveReconstruction`. Limiters:
`venkatakrishnan`, `barth_jespersen`, `minmod`, `van_albada`.

Time integration lives in `fvm/ode.py` (RK1/Heun-RK2/RK3/implicit),
`fvm/imex_ark.py` (IMEX-ARK tableaux: `ars232`, `ars343`), and
`fvm/timestepping.py` (`constant`, `adaptive(CFL=…)`). The full Riemann registry
and how to add a variant are in [`../authoring/numerics.md`](../authoring/numerics.md#riemann-solvers).

## How a solver consumes a Model / SystemModel

Every `setup_simulation(mesh, model, write_output=True)` normalises once:

```python
# inside Solver.setup_simulation (paraphrased)
mesh = ensure_lsq_mesh(mesh, model)
if not isinstance(model, SystemModel):
    model = SystemModel.from_model(model)          # freeze operators
self.sm = model
runtime = NumpyRuntimeModel.from_system_model(model)   # lambdify operators
```

The runtime is the only thing the timestep loop touches: routed solver-tag
callables (`flux`, `nonconservative_flux`, `hydrostatic_pressure`, `source` /
`source_explicit`, `diffusion_matrix`, `boundary_conditions`,
`boundary_gradients`, `update_variables`). See
[`../authoring/system-model.md`](../authoring/system-model.md) and
[`../authoring/numerics.md`](../authoring/numerics.md) for tag routing.

## End-to-end example: SWE

```python
from zoomy_core.systemmodel import SystemModel
from zoomy_core.numerics.numerical_system_model import (
    NumericalSystemModel, ReconstructionSpec)
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.fvm import timestepping
from zoomy_core.mesh import BaseMesh
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
import numpy as np

# Reference SWE model: thesis/notebooks/legacy/modeling/swe/simple_swe_v2.py
# (add its directory to PYTHONPATH, or substitute any Model subclass)
from simple_swe_v2 import SWEModelV2

def ic_func(x):
    Q = np.zeros(2)
    Q[0] = 1.0 + 0.1 * np.exp(-((x[0] - 5.0) ** 2) / 0.5)   # h
    Q[1] = 0.0                                              # hu
    return Q

m = SWEModelV2(
    boundary_conditions=BC.BoundaryConditions(
        [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")]
    ),
    initial_conditions=IC.UserFunction(ic_func),
)

# Numerical knobs go on the NumericalSystemModel, not the solver:
sm  = SystemModel.from_model(m)
nsm = NumericalSystemModel.from_system_model(
    sm, reconstruction=ReconstructionSpec(order=2, limiter="venkatakrishnan"))

mesh   = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=300)
solver = HyperbolicSolver(time_end=0.5, compute_dt=timestepping.adaptive(CFL=0.9))
Q, Qaux = solver.solve(mesh, nsm, write_output=False)
```

`BaseMesh.create_1d` from `zoomy_core.mesh` is the in-package 1D constructor
(`setup_simulation` upgrades it to an `LSQMesh` internally);
richer unstructured 2D / 3D meshes come from `zoomy_mesh` (sibling subrepo at
`library/zoomy_mesh/`).

## Authoring checklist — what the solver expects

- Canonical `M = I`: the model has applied `InvertMassMatrix` before
  `from_model`. Only `DAESolver` tolerates a singular `mass_matrix`.
- Every non-zero term tagged via `solver_tag(...)` and routed to a slot
  (`flux`, `nonconservative_flux`, `hydrostatic_pressure`,
  `implicit_source` / `explicit_source`, `implicit_diffusion` /
  `explicit_diffusion`). Untagged ⇒ runtime returns zero. See the
  [SystemModel tag catalogue](../authoring/system-model.md).
- `parameters` declared with positivity / non-negativity tags.
- `boundary_conditions` and `aux_boundary_conditions` supplied; mesh
  boundary tags match BC tags. Periodic pairs are resolved on the mesh
  before `from_model` freezes the BC into the indexed kernel.
- With `ReconstructionSpec(order=2, ...)` on the `NumericalSystemModel`, choose
  the `limiter` to match smoothness (default `"venkatakrishnan"` for SWE-family).
- For `FreeSurfaceFlowSolver` / `FSFIMEXSolver` / `FSFSplittingSolver`:
  declare `h` and (when bathymetry is present) `b` in `variables` — these
  slots drive wet/dry detection and the positive Rusanov path.

**API reference**: see [zoomy_core](../api/_apidoc_zoomy_core/zoomy_core.rst).
