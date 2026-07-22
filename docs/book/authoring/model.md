# Model

A `Model` is a **symbolic PDE**: a state vector `Q`, some parameters, and
operator methods that return sympy expressions. Nothing in a `Model` is
numerical — no mesh, no timestep, no solver. That separation is the whole point:
one model definition drives every backend.

```
Model ──────────▶ SystemModel ──────────▶ NumericalSystemModel ──────▶ Solver
symbolic PDE      frozen operators        + scheme choices             NumPy · JAX
derivation        F, P, B, S, M           Riemann, reconstruction      AMReX · OpenFOAM
```

This page covers the first box. Then: [System Model](system-model.md) →
[Numerical System Model](numerics.md) → a [backend](../backends/index.md).

## The functions that actually matter

For everyday use there are four:

| Call | What it gives you |
| --- | --- |
| `SomeFamily(level=…, dimension=…, closures=[…])` | a ready model from a shipped family |
| `SystemModel.from_model(m)` | freeze it into operators |
| `NumericalSystemModel.from_system_model(sm, …)` | attach a scheme |
| `m.describe()` / `sm.describe()` | print what you actually built |

If you are subclassing rather than just using a family, four more:
`add_equation`, `remove_equation`, `apply`, and the `derive_model` hook. That
is the entire authoring surface — there are deliberately no
`multiply` / `resolve_dummy` convenience methods; every transformation goes
through `apply(SomeOperation(...))`.

```{warning}
`model.system_model` was removed. It raises `AttributeError` with a migration
message. The only path is `SystemModel.from_model(model)`.
```

## Your first model

You do not write shallow water by hand — you *derive* it, as `SME(level=0)`.
This runs as-is:

```python
import numpy as np
from zoomy_core.model.models import SME, Newtonian, NavierSlip, StressFree
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.systemmodel import SystemModel
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.mesh import BaseMesh
from zoomy_core.fvm.solver_numpy import HyperbolicSolver
from zoomy_core.fvm import timestepping

model = SME(
    level=0,                                    # level 0 == shallow water
    closures=[Newtonian(), NavierSlip(), StressFree()],
    boundary_conditions=BC.BoundaryConditions(
        [BC.Wall(tag="left"), BC.Wall(tag="right")]),
    initial_conditions=IC.RP(                   # dam break: h = 2 | 1
        high=lambda n: np.array([0.0, 2.0] + [0.0] * (n - 2)),
        low=lambda n: np.array([0.0, 1.0] + [0.0] * (n - 2)),
        jump_position_x=5.0),
)

sm = SystemModel.from_model(model)
print(sm.describe())                            # look at the operators first

sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
mesh = BaseMesh.create_1d(domain=(0.0, 10.0), n_inner_cells=50)
nsm = NumericalSystemModel.from_system_model(
    sm, reconstruction=ReconstructionSpec(order=1))

solver = HyperbolicSolver(time_end=0.5, compute_dt=timestepping.adaptive(CFL=0.9))
Q, _ = solver.solve(mesh, nsm, write_output=False)
```

`sm.state` is `[b, h, q_0]`, `sm.aux_state` is `[dq0dx, dhdx, dbdx]` — the
spatial derivatives were discovered and registered for you. In a closed (wall)
box, mass is conserved to round-off: `Q[1].sum()` stays at `75.0` to 1e-14.

```{tip}
`print(sm.describe())` before you run anything. A missing hydrostatic-pressure
entry or an empty NCP slot is visible on sight, and costs 30 seconds instead of
a day.
```

## The building blocks

### Model families

Every model derives from one of these; none is typed out by hand.

| Family | Import | Key kwargs | Status |
| --- | --- | --- | --- |
| `SME` | `zoomy_core.model.models` | `level=2`, `dimension=2`, `closures`, `small_slope`, `quadrature_order` | gate-tested |
| `VAM` | `zoomy_core.model.models` | `level=1`, `dimension=2`, `closures` | gate-tested |
| `MLSWE` | `zoomy_core.model.models` | `n_layers=2`, `dimension`, `closures` | gate-tested |
| `MLSME` | `zoomy_core.model.models` | `n_layers=2`, `level` (int **or per-layer list**), `dimension` | tested |
| `MLVAM` | `zoomy_core.model.models` | `n_layers=2`, `level=1`, `dimension` | tested |
| `ElderSME` | `zoomy_core.model.models` | SME + Elder viscosity defaults | tested |
| `Sigma3D` | `…model.models.sigma3d` | `dimension=2` only | tested |
| `SWE` | `zoomy_core.model.models` | `dimension` (1 or 2) | hand-built; see note |
| `MalpassetSWE` | `zoomy_core.model.models` | `n`, `nu`, `h_friction_floor`, … | hand-built |
| `KESME`, `QRKESME` | `zoomy_core.model.models` | `turbulence_level=1` | **unverified** — no test coverage |

```{warning}
**`dimension` means different things.** For `SME`, `VAM`, `ML*` and `Sigma3D` it
is the *total* dimension including the vertical: `dimension=2` gives **one**
horizontal direction, `dimension=3` gives two. For the hand-built `SWE` class it
is the *horizontal* dimension. This is a real inconsistency in the shipping API,
not a documentation slip.
```

Shallow water is `SME(level=0)`, not the `SWE` class. `SWE` is retained for the
Malpasset line, which needs a wet/dry momentum cap that no closure supplies yet.

### Closures — how physics gets in

Constitutive physics is **never** typed into a model. It is a list of `Closure`
objects passed to the constructor. Add physics by adding a closure, never by
editing the parent or adding a flag.

| Closure | Acts on | Physics |
| --- | --- | --- |
| `Newtonian` | bulk | `τ = ρν ∂_z u` |
| `KEpsilonViscosity` | bulk | k–ε eddy viscosity `ν_t = C_μ k²/ε` |
| `QRViscosity` | bulk | the same in `q`–`r` variables |
| `Bingham` | bulk | regularised viscoplastic |
| `ElderViscosity` | bulk | parabolic `ν_t = κ u_⋆ h ζ(1−ζ)` — closes analytically |
| `NavierSlip` | bottom | linear slip `τ_b = λ_s u_b` |
| `RoughWall` | bottom | quadratic drag `τ_b = ρ C_f u_b\|u_b\|` (the Chézy slot) |
| `WallFunctionBed` | bottom | k–ε rough-wall function |
| `ManningFriction` | bottom | `−g n²\|u\| / h^{1/3}` |
| `EddyViscosity` | horizontal | horizontal mixing `∇·(ν h ∇u)` |
| `ShallowInPlane` | horizontal | *drop* in-plane deviatoric stress |
| `NewtonianInPlane` | in-plane | *keep* in-plane `τ_de` |
| `StressFree` | surface | `τ(1) = 0` |
| `MeanInterface`, `UpwindInterface` | interface | multilayer transfer velocity |

`Bingham` and `KEpsilonViscosity` produce non-polynomial integrands — build the
host model with `quadrature_order > 0`.

```{note}
`ManningFriction`, `EddyViscosity`, `ShallowInPlane` and `NewtonianInPlane` are
**not** re-exported from `zoomy_core.model.models`; import them from
`zoomy_core.model.models.closures`. There is no `Chezy` class — that is
`RoughWall`.
```

### Bases

Modal models are written against the opaque `Basisfunction`, and only committed
to a concrete basis at resolve time. Shipped:
`Legendre_shifted` (what every family uses), `Monomials`, `Chebyshevu`,
`Chebyshevu_shifted`, `Legendre_DN`, `GalerkinBasis`, `SplineBasis`,
`LayeredBasis` — all in `zoomy_core.model.derivation.basisfunctions`.

### Boundary conditions

Attach per tag, and optionally per field with `on=`
(`"h"`, `"q"`, `"q_0"`, `"momentum"`, `"all"`). Unclaimed slots default to
`Extrapolation`.

`Extrapolation` · `ZeroNeumann` · `Dirichlet` · `Flux` · `Wall` · `RoughWall` ·
`Periodic` · `InflowOutflow` · `Lambda` · `FromModel` · `FromData` ·
`Characteristic` · `CharacteristicFarField` · `CharacteristicWall` ·
`CharacteristicReflective` · `WindStress` · `Coupled` (preCICE) —
all in `zoomy_core.model.boundary_conditions`.

`Periodic` and `Coupled` are whole-patch: they cannot be mixed with per-field
BCs on the same tag. Aux BCs are separate and offer only `Extrapolation` and
`Lambda`.

### Initial conditions

`Constant` · `RP` · `RP2d` · `RP3d` · `RadialDambreak` · `UserFunction` ·
`Project3D` · `RestartFromHdf5` — in `zoomy_core.model.initial_conditions`.

## Going beyond — subclass and compose

A new model subclasses the closest existing one and swaps the closure list. It
never re-derives from scratch and never patches the parent with a flag:

```python
from zoomy_core.model.models import SME, StressFree, RoughWall
from zoomy_core.model.models.closures import ManningFriction

class MyRiverSME(SME):
    def __init__(self, **kw):
        kw.setdefault("closures", [ManningFriction(), RoughWall(), StressFree()])
        super().__init__(**kw)
```

Writing your own `Closure` is the other extension point — declare what it
`closes` and what it `requires`, then supply `expression` (bulk) or `traction`
(boundary). See the [advanced SWE tutorial](../tutorials/swe.md).

For genuinely new structure, override `derive_model()` and build the derivation
with the operations below. `SME.derive_model` in
`zoomy_core/model/models/sme.py` is the canonical worked example: thirteen
tracked `apply(...)` steps from the 3-D incompressible balances to the final
operator form.

## Operation catalogue

These are the operations the shipping derivations use, from
`zoomy_core.model.derivation`.

**Coordinate transform** — `PDETransformation(coord_map, *, decorate="tilde")`
chain-rules a σ-map `z = b + h·ζ` through the whole model.

**Projection / integration** — `Integrate(var, bounds)`,
`Project(test_function, var, bounds)`, `ExpandSums()`, `EvaluateSums()`,
`PullConstants()`, `ExtractBrackets(basis, var)`, `ResolveBasis(basis, var)`.

**Modal** — `separation_of_variables(field, coeff, basis, order)`,
`TensorSeparationOfVariables(...)`, `ResolveModes(index, modes)`.

**Closure / algebra** — `Resolve(test_function, basis_cls, level)`,
`ResolveIntegral(basis_cls, *, method, level)`, `GaussQuadrature(var, order)`,
`DeferQuadrature`, `ResolveNumQuad`, `InvertMassMatrix(time)`,
`FoldConservative(...)`, `Split(variables)`, `Consolidate()`, `Simplify(sort)`,
`AutoTag(detect_ncp)`, `SortByTag()`, `Sort(detect_ncp)`.

**System solve / variable swaps** — `SolveFor(variable)`,
`SolveLinearSystem(equations, variables)`,
`ChangeOfVariables(old, new, relation)`.

**Leaf ops** from `zoomy_core.model.operations` — `Multiply(factor, outer)`,
`ProductRule(variables, direction)`, `Integrate(var, lower, upper, method)`
(the real-coordinate one, imported as `IntegrateZ`),
`KinematicBC(state, interface, *, at, mass_flux)`.

Every derivation ends with `InvertMassMatrix()`, so the solver always sees
`M = I`.

```{note}
`zoomy_core/model/operations.py` also contains a large older library
(`DepthIntegrate`, `AffineProjection`, `SigmaTransform`, `Recombine`,
`ResolveDummy`, `LayerMeanClosure`, `Symmetrize`, `EvaluateIntegrals`, …) with
no live call sites outside legacy notebooks. Do not build on it; the ops above
are the current surface.
```

## Imports, precisely

`zoomy_core/__init__.py` exports only `__version__`, and
`zoomy_core/model/__init__.py` and `zoomy_core/transformation/__init__.py` are
empty. Import by full path:

```python
from zoomy_core.model.models import SME, VAM, MLSME       # families + closures
from zoomy_core.model.models.sigma3d import Sigma3D        # NOT in __all__
from zoomy_core.model.models.closures import ManningFriction
from zoomy_core.systemmodel import SystemModel
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
```
