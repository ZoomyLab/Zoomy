# Authoring a `Model`

A `Model` is a parameterized symbolic PDE: a state vector `Q`, parameters
`p`, and operator methods (flux, source, nonconservative matrix, ...)
returning sympy expressions over `Q` and `p`. The class owns the
*derivation graph*: every assumption, substitution, and projection that
produced the operators is recorded by `.apply(...)` and replayed via
`.describe()`. The operator-form sibling that solvers and analysis
consume is the `SystemModel`; see [system-model.md](system-model.md).
`Model` and `SystemModel` are independent siblings, not inherited:
`SystemModel.from_model(m)` extracts the operators once and freezes them.

## The pipeline

```
Model ───────────▶ SystemModel ───────────────▶ NumericalSystemModel ──────▶ Solver
 symbolic           frozen operators             + numerics specs            fvm runtime
 derivation graph   F, P, B, S, M                Riemann / reconstruction    (NumPy is the
 .apply(...)        SystemModel.from_model(m)    .from_system_model(sm, …)    reference; other
                                                                             backends port from it)
```

Code printers (`transformation/to_c.py`, `to_amrex.py`, `to_ufl.py`, …) and
analysis (dispersion, hyperbolicity) both hang off `SystemModel`. The
NumericalSystemModel stage is in [numerics.md](numerics.md); the solver in
[`../backends/numpy.md`](../backends/numpy.md).

**New to Zoomy?** Read this page, then [SystemModel](system-model.md) →
[Numerics](numerics.md) → a backend ([NumPy](../backends/numpy.md) is the
reference).

## Build from the existing blocks — reuse, don't reinvent

Zoomy models are **composed from framework blocks**, not hand-rolled. A model
never carries its own solver, time loop, or ad-hoc parameter, and is never
patched with a flag, an `if`-branch, or a private attribute — those bypass the
GUI/CLI introspection and the tagging system (an untagged term silently returns
zero). Before adding new structure, be explicit about two things: (a) **where it
fits** — which existing block absorbs it — and (b) **what is genuinely new** and
cannot. Then:

- **A new model = subclass the closest existing one and compose closures.**
  e.g. `MalpassetSWE(SWE)` (adds Manning + eddy viscosity), `KESME(SME)`,
  `ElderSME(SME)`. Add physics with a `Closure` in the `closures=` list
  (`model/models/closures.py`), **not** by editing the parent.
- **A new scheme = a `Numerics`/`Closure` subclass, or a different spec on the
  NumericalSystemModel** — beside its siblings, not a fork.
- **Configure via `param` knobs and derivation-time tags**, not runtime
  branches: a plain attribute that bypasses `param` disappears from the GUI/CLI;
  an untagged term returns zero.
- **Fix root causes.** Hooks, adapters, aliases (`Old=New`), `from_X` factories,
  translation maps, and `hasattr`/`getattr(…, None)` "graceful no-op" fallbacks
  are the smell of dodging the real change. **Break, don't skip** — let missing
  wiring fail loudly.
- **Find the existing thing first:** `model/models/` for a model, the Riemann /
  reconstruction lists for numerics, `analysis/__init__.py` for analysis, a close
  `tutorials/` example, then `git log -- <area>` (many primitives were
  refactored; the old form may be in history).

## Class skeleton

```python
from zoomy_core.model.basemodel import Model

class MyModel(Model):
    dimension = 1
    variables = ["h", "qU"]
    aux_variables = ["b_x"]
    parameters = {"g": (9.81, "positive"), "nu": (1e-6, "positive")}
```

`variables`, `aux_variables`, `parameters` may be an `int` (auto-named
`q_0, q_1, ...`), a list of names, or a dict mapping name to
`(default, "positive" | "real")`. They are parsed into `Zstruct` containers
of real sympy `Symbol`s. See `library/zoomy_core/zoomy_core/model/basemodel.py`.

Operator methods return `ZArray` instances — build a `sympy.Matrix` of the right
shape, assign the non-zero entries, and wrap it:

```python
def flux(self):
    u = self.variables.u
    F = sympy.Matrix.zeros(self.n_variables, self.dimension)
    F[0, 0] = u**2 / 2          # inviscid Burgers
    return ZArray(F)
```

The shape contract:

| Method | Shape |
| --- | --- |
| `flux` | `(n_variables, dimension)` |
| `hydrostatic_pressure` | `(n_variables, dimension)` |
| `nonconservative_matrix` | `(n_variables, n_variables, dimension)` |
| `diffusion_matrix`, `diffusion_matrix_explicit` | `(n_variables, n_variables, dimension, dimension)` |
| `source`, `source_explicit` | `(n_variables,)` |
| `eigenvalues` | `(n_variables,)` |
| `update_variables`, `update_aux_variables` | `(n_variables,)`, `(n_aux_variables,)` |
| `initial_condition`, `initial_aux_condition` | `(n_variables,)`, `(n_aux_variables,)` |

## Override surface

Time evolution: `mass_matrix` only when non-canonical (default `I`); the
closure pipeline calls `InvertMassMatrix` before any solver sees the
system. `update_variables` / `update_aux_variables` are the per-step state
map (default: identity).

Spatial operators: `flux` (conservative divergence),
`nonconservative_matrix` `B · ∂_x Q`, `hydrostatic_pressure` (held separate
from `flux` so well-balanced reconstruction reads it off), and the two
diffusion slots `diffusion_matrix` (implicit, at `Qnp1`) /
`diffusion_matrix_explicit` (explicit, at `Qn`, parabolic CFL). All are
pure `(Q, Qaux, p)`; state derivatives enter through `Qaux`.

Source: `source` (implicit, at `Qnp1`) and `source_explicit` (explicit,
Forward-Euler at `Qn`); an IMEX backend evaluates each at its own stage.

Spectrum: `eigenvalues` is **optional** — the default derives the spectrum from
the flux Jacobian symbolically (for Burgers it returns `[n0·u]`, the
normal-projected wave speed, with no author effort). Override it only to supply a
closed form; or set `eigenvalue_mode = "numerical"` to defer to
`np.linalg.eigvals` at runtime ([numerics.md](numerics.md)).

Conditions — mind the singular/plural split. The overridable **methods**
`initial_condition(self)` / `boundary_conditions(self)` (no runtime args) are the
*symbolic-default* surface used by codegen; both default to a sensible no-op. The
**runtime** initial and boundary data are passed to the *constructor* as the
plural kwargs:

```python
m = MyModel(
    initial_conditions=IC.UserFunction(lambda x: ...),     # x: position array → state
    boundary_conditions=BC.BoundaryConditions([BC.Extrapolation(tag="left"), ...]),
)
```

`aux_boundary_conditions` defaults to per-tag `Extrapolation`. Full signatures
live in `library/zoomy_core/zoomy_core/model/basemodel.py`.

## `Zstruct` dot-access

```python
m.variables.h          # sympy Symbol
m.parameters.g         # sympy Symbol on the symbolic side
m.parameters.g = 9.81  # numeric override on the user-facing struct
m.aux_variables.b_x
```

`m.variables["h"]` works but is reserved for runtime-string keys.

## `Q` vs `Qaux` — primary state and everything derived

The solver advances **`Q`**, the primary conserved state (`variables`,
group `"model"`). **`Qaux`** holds everything the operators need that is *not*
itself integrated in time: eliminated fields, topography, and — critically —
**spatial derivatives of the state**.

- In a **declarative** model `Q` is the set of unknowns that still appear in a
  primary equation; a field leaves `Q` for `Qaux` the moment it has been fully
  substituted out (e.g. `p` after the hydrostatic reduction, `w̃` after the
  ω-elimination). This is automatic — do not hand-maintain the split.
- A hand-written model declares the split directly: `variables=[...]`,
  `aux_variables=[...]`.

**Runtime derivatives are aux, not magic.** When `SystemModel.from_model` freezes
the operators, `expose_aux_atoms()` scans every matrix and replaces each spatial
`Derivative(field)` atom with a fresh aux Symbol (named `dhdx`, `dq0dx`, …),
appended to `aux_state` and recorded in `sm.aux_registry` with a `multi_index`
(the per-axis derivative orders, matching `LSQMesh.compute_derivatives`). The
solver fills every derivative-aux in one least-squares gradient pass each step —
you never write a finite-difference stencil. Topography `b(t,x)` and TVD-limited
gradients (`limit(...)`, see [numerics.md](numerics.md)) are exposed the same way
(`kind = "function"` / `"limited_derivative"`). After `from_model`, solvers walk
`aux_registry` and never re-scan the operator matrices.

## 3-D lift — `interpolate_to_3d` / `project_from_3d`

A depth-averaged state is a *reduced* description of a 3-D column. The two
inverse maps make the reduction explicit, and they are what couples Zoomy to a
3-D solver (the preCICE exchange in [openfoam.md](../backends/openfoam.md) is
literally these two functions):

- `interpolate_to_3d()` reconstructs the canonical profile column
  `[b, h, u, v, w, p]` at a vertical coordinate `ζ` from the state.
- `project_from_3d()` is its Galerkin inverse: a sampled 3-D profile → state.

A derived model populates them in `derive_model` by stashing `_interpolate_rows` /
`_project_rows`, or component-wise via `register_group(slot, index, relation)`
(`model/derivation/model.py`). The wiring is **self-contained in the state**: SME
inlines its `ŵ_j` closure and the conservative change of variables so that
`interp[4] = w(ζ) = Σ_j ŵ_j(q, ∂_x q, ∂_x h, ∂_x b)·φ_j(ζ)` depends only on `Q`
and its derivatives.

## Jacobians

All Jacobians are **symbolic**, computed by `sp.derive_by_array` — you never
supply one:

- `quasilinear_matrix` = `∂F/∂Q + ∂P/∂Q + B` (the flux Jacobian is computed
  inline here; there is no separate `flux_jacobian` method).
- `source_jacobian` = `∂S/∂Q` (and a separate `∂S/∂Qaux`).

On the frozen `SystemModel` these recompute automatically in `__post_init__` and
after any `change_state_variables`. The one opt-out is `eigenvalues`: see
*User-supplied functions* in [numerics.md](numerics.md) for the
`eigenvalue_mode = "numerical"` path that defers the spectrum to `np.linalg.eigvals`
at runtime.

## Boundary conditions — per-field routing

`boundary_conditions` / `aux_boundary_conditions` are `BoundaryConditions`
containers (or a flat per-field list). Each tag is resolved into a
`PerFieldBoundary` by `resolve_and_attach(sm, bcs, aux_bcs)`: an `on=` selector
routes a BC to specific state slots (`"all"`, an exact name, a family base
`q → q_0,q_1,…`, or an alias like `momentum → q`); unclaimed slots default to
`Extrapolation`. The BC types are `Extrapolation`, `Dirichlet`, `Flux`
(Neumann/Robin `∂Q/∂n = gradient`), `Wall` (reflects only the *normal* momentum
component), `FromModel` (a model-derived `boundary:<name>` group), `Periodic`,
and `Coupled` (the preCICE patch). **Route per-field BCs through
`resolve_and_attach`, never a raw `Dirichlet(on=field)` inside a flat list** — the
latter zeroes the whole ghost cell including `h`. The moving-interface *kinematic* BC is not a BC
class — it is the derivation-time `KinematicBC` operation above.

## Parameters

The `parameters` dict becomes two parallel Zstructs: `self.parameters` (sympy
Symbols carrying the `"positive"`/`"real"` assumption) and
`self.parameter_values` (numeric defaults). Construction fails loudly if any free
symbol in the equations lacks a default. Override a value numerically with
`model.parameter_values.g = 12.0`; it is carried to the `SystemModel` and lifted
into the runtime `parameter` argument at the lambdify boundary. Declarative models
may declare lazily with `model.parameter("nu", 0.0)`.

## Derivation via `apply()`

`model.apply(operation)` records the operation for `.describe()` and transforms
the equations in place. A depth-averaged model is *derived*, not typed out: you
start from the 3-D balances, reduce them with a sequence of operations, and the
final operator form drops out. Two operator surfaces exist and it matters which
you use:

- **`model/derivation/` — the live surface.** Every current model (SME, VAM,
  ML-SME, ML-VAM, KE-SME, …) is built from these. Prefer them.
- **`model/operations.py` — the legacy surface.** A large older library
  (`DepthIntegrate`, `ApplyKinematicBCs`, `EvaluateIntegrals`, `AffineProjection`,
  `SigmaTransform`, `IntegralTransform`, `MapBasisToReference`,
  `SimplifyIntegrals`, `ProjectBasisIntegrals`, `Expand`). The current models do
  **not** call these — use the `model/derivation/` equivalents below. Only four
  leaf ops are still imported from here (`Multiply`, `ProductRule`, the vertical
  `Integrate`/`IntegrateZ`, and `KinematicBC`).

### Operation catalogue (`model/derivation/`)

**Coordinate / σ-transform** — `derivation/transformations.py`

- `PDETransformation(coord_map, *, decorate="tilde")` — the σ-map. Sends the
  physical vertical `z` to the reference `ζ ∈ [0,1]` through an explicit geometry
  `z = b(t,x) + h(t,x)·ζ`, chain-ruling every `∂_t, ∂_x, ∂_z` across the whole
  model. (Replaces the legacy `SigmaTransform` / `AffineProjection`.)

**Abstract integration / Galerkin projection** — `derivation/projection.py`

- `Integrate(var=None, bounds=(0,1))` — wrap each additive term in an
  *unevaluated* `∫`, and stop. No fundamental-theorem step, no evaluation.
- `Project(test_function, var=None, bounds=(0,1))` — Galerkin project:
  `Multiply(test_function)` then integrate over `(ζ, 0, 1)`.
- `ExpandSums()` — expand products/powers of unevaluated `Sum` factors into one
  multi-index `Sum` with distinct dummies (cross terms kept).
- `EvaluateSums()` — unroll every finite `Sum` to explicit modes.
- `PullConstants()` — pull every factor independent of a binder out of `∫` and `∂`.
- `ExtractBrackets(basis, var=None)` — replace `∫(φ-product)` with a named
  Galerkin bracket atom `⟨…⟩`, pulling ζ-independent factors out first.
- `ResolveBasis(basis, var=None)` — resolve every bracket to a number against a
  concrete basis (thin op over `Basisfunction.resolve`).

**Modal separation of variables** — `derivation/modal.py`

- `separation_of_variables(field, coeff, basis, order)` (op
  `SeparationOfVariables`) — replace a field by its modal sum
  `f = Σ_i f̂_i φ_i(ζ)` and swap the unknown family in `Q`.

**Mode-axis resolution** — `derivation/model.py`

- `ResolveModes(index, modes)` — the Galerkin moment-axis *shape bump*: turn an
  equation into an indexed `MomentFamily` over `modes`. Promotes `m.X` **in
  place**, so captured references to a moment family go stale — re-fetch after
  applying.

**Algebra / closure** — `derivation/closure.py`

- `Resolve(test_function, basis_cls, level)` — project-and-resolve in one step.
- `ResolveIntegral(basis_cls=None, *, method="auto", level=None)` — resolve the
  unevaluated `∫…dζ` left by an abstract projection, by a selectable method.
- `GaussQuadrature(var=None, order=8)` — replace surviving `Integral` atoms with
  an n-point Gauss–Legendre sum. The numerical escape hatch for non-analytic
  integrands (Bingham, k–ε eddy viscosity).
- `InvertMassMatrix(time=None)` — invert the Galerkin mass matrix by reading it
  off the equation. The canonical finaliser of every derivation. **Lives in
  `derivation/closure.py`, not `operations.py`.**
- `FoldConservative(flux_fields, *, h, b, x, gravity=None)` — re-bundle a
  momentum row's spatial part into the Kowalski–Torrilhon conservative form.
- `Split(variables=None)` — fold a self-product flux `c·f·∂_x f → ∂_x(c·f²/2)`.
- `Consolidate()` — fold `M·a·∂b + M·b·∂a → M·∂(ab)` (recover conservative
  grouping), also inside integrals.
- `Simplify(sort=True)` — canonicalise while keeping fluxes as fluxes.
- `AutoTag(detect_ncp=True)` — tag each untagged term with its physics category
  (time_derivative / flux / diffusion / nonconservative_flux / source).
- `SortByTag()` / `Sort(detect_ncp=True)` — reorder terms into the canonical
  `TAG_ORDER`; `Sort` = `AutoTag` then `SortByTag`. Apply last.

**System solve / variable swaps** — `derivation/operations.py`

- `SolveFor(variable)` — reorient an equation into `variable = solution`, in place.
- `SolveLinearSystem(equations, variables)` — assemble and solve a coupled
  linear system for a family of variables; `.solve()` returns the substitutable
  solution. This is how SME closes the vertical-velocity coefficients `ŵ_j`.
- `ChangeOfVariables(old, new, relation)` — coefficient-family change of
  variables that swaps the unknown family (the conservative CoV `û_i → q_i/h`).

**Leaf ops still imported from `model/operations.py`**

- `Multiply(factor, outer=False)` — scale every term; `outer=True` with a Zstruct
  of test functions promotes a leaf to a Zstruct of children.
- `ProductRule(variables=None, direction="inverse")` — product rule, auto-routed
  per term by the outermost shape (`inverse` folds `a∂b + b∂a → ∂(ab)`).
- `Integrate(var, lower, upper, method="auto")` (imported as `IntegrateZ`) —
  integrate each term over `[lower, upper]` in a *real* coordinate; used for the
  vertical hydrostatic-pressure integral.
- `KinematicBC(state, interface, *, at=None, mass_flux=None)` — the moving-interface
  kinematic BC. `at` defaults to `interface`; set it for σ-transformed systems and
  per-layer ML σ values. `mass_flux=None` is impermeable; a sympy expression adds
  the interface transfer flux `mass_flux/ρ` to the RHS (the multilayer `G`).

### Physics closures (`model/models/closures.py`)

Constitutive physics is **never** typed into a model — it is composed as a list
of `Closure` ops and injected at the projected-momentum stage by
`apply_stress_closures(closures, m, axes, state, horiz)`. Each `Closure` declares
the parameters it needs (`register`), validates its requirements (`check`), and
supplies either a bulk constitutive stress (`expression`) or a boundary traction
(`traction`). **Add physics by adding a closure to the list, never by editing the
parent model.**

| Closure | `closes` | Role |
| --- | --- | --- |
| `Newtonian` | bulk | `τ = ρν ∂_z u` |
| `KEpsilonViscosity` | bulk | k–ε eddy viscosity `τ = ρ(ν + C_μ k²/ε) ∂_z u` |
| `Bingham` | bulk | regularised viscoplastic `τ = (ρν + τ_y/√((∂_z u)²+ε²)) ∂_z u` |
| `ElderViscosity` | bulk | parabolic eddy viscosity `ν_t = κ u_⋆ h ζ(1−ζ)` |
| `NavierSlip` | bottom | linear bed slip `τ_b = λ_s u_b` |
| `RoughWall` | bottom | turbulent quadratic bed drag `τ_b = ρ C_f u_b|u_b|` |
| `ManningFriction` | bottom | Manning bed friction `−g n²|u|/h^{1/3}` |
| `EddyViscosity` | horizontal | horizontal eddy-viscosity mixing for depth-averaged SWE |
| `StressFree` | surface | free surface `τ(1) = 0` |
| `MeanInterface` / `UpwindInterface` | interface | multilayer transfer velocity `u*` (mean / donor-by-`G`) |

`Bingham` and `KEpsilonViscosity` produce non-polynomial integrands, so the host
model must be built with `quadrature_order > 0` (it routes through
`GaussQuadrature`). The multilayer variants use `apply_layer_stress_closures(...)`
and `interface_closure(...)` instead.

## `describe()`

```python
print(m.describe(derivation="mermaid"))   # graph TD block; verbose=True for one edge per apply()
print(m.describe(derivation="markdown"))  # parent / operations / self
```

The rendered diagram lives on the [SystemModel page](system-model.md).

## Worked derivation — SME step by step

`SME.derive_model` (`model/models/sme.py`) is the canonical example: the Shallow
Moment Equations derived from the incompressible 3-D balances. Read it top to
bottom — every reduction is one tracked `apply(...)`. The thirteen stages:

1. **Parameters & dimension.** Defaults `g, ρ, ν, λ_s, e_x`; user overrides merged.
   `dimension=2 → (t,x,z)` with one horizontal momentum; `dimension=3 → (t,x,y,z)`
   with two. Build `m = DModel(coords, parameters)`.
2. **Assemble the 3-D system from blueprints** (`model/models/equations.py`):
   `declare_state(h)`, a trivial bed equation `∂_t b = 0`, incompressible `Mass(m)`
   (registers `u[,v], w`), full-stress `Momentum(m)` (registers `p`, `τ_*`), then
   `moment_scaling(m, mom)` drops the in-plane stresses — the one shallow
   simplification — keeping only `τ_xz[, τ_yz]`.
3. **Kinematic BCs.** Build `kbc_top` at `b+h` and `kbc_bot` at `b` (`KinematicBC`).
4. **Hydrostatic pressure.** On the `z`-momentum row drop the inertial vertical
   terms, `IntegrateZ(z, z, b+h, "analytical")`, impose `p(b+h)=0`, `solve_for(p)`,
   broadcast the hydrostatic column `ρg(h+b−z)` into every horizontal momentum,
   remove the `z`-row, `Simplify()`.
5. **σ-map the whole model:** `PDETransformation({z: (ζ, Eq(z, b + h·ζ))})` — move
   to the fixed reference column `ζ ∈ [0,1]`.
6. **Set up the basis:** `Basis(symbol="phi", weight="c")`, reserved test index
   `k = test_index()`, shifted Legendre to level `N_u + 2`.
7. **Project mass + KBCs:** `Multiply(h)`, `Multiply(c(ζ)·φ_k)`, `ProductRule([ζ])`,
   `Integrate(ζ,(0,1))`, `ResolveIntegral()`, substitute `kbc_bot`/`kbc_top`, kill
   `∂_t b`, `Simplify()` → the depth-integrated moment form of mass.
8. **Project each horizontal momentum and close the stress:** the same
   project-and-resolve per axis, then `apply_stress_closures(self.closures, …)`
   injects the boundary tractions + bulk constitutive stress; `small_slope_scaling`
   if requested.
9. **Separation of variables:** `u_i → û_i` at order `N_u`, `w → ŵ` at order
   `N_u + 1` (`separation_of_variables`).
10. **Resolve mass → h-equation + ŵ closure:** `ExpandSums, PullConstants,
    ExtractBrackets, EvaluateSums, ResolveModes(index=k, modes=range(N_u+3)),
    ResolveBasis`; then `h_eq = mass[0].solve_for(∂_t h)` and the vertical-velocity
    closure `w_closure = SolveLinearSystem(mass[1:], [ŵ_j]).solve()`.
11. **Substitute ŵ into momentum, resolve:** per axis `ExpandSums … w_closure,
    ResolveModes(modes=range(N_u+1)), ResolveBasis`, plus `GaussQuadrature` if
    `quadrature_order > 0`.
12. **Finalise:** substitute `h_eq`, `Consolidate()`, conservative
    `ChangeOfVariables(û_d → q_d, λ q_i: q_i/h)`, then **`InvertMassMatrix()`**.
    The symbolic derivation ends here.
13. **Register reconstruction / projection:** build `_interpolate_rows`
    (the 3-D profile `[b,h,u,v,w,p]` with the ŵ closure inlined), the lateral-wall
    mirror BC via `register_group("boundary:wall", q_i, −q_i)`, the well-balanced
    `_reconstruction_rows`, and the Galerkin-inverse `_project_rows`.

Solver tagging is **not** done in `derive_model`: `SystemModel.from_model(m)`
routes the frozen terms into flux/source/NCP slots structurally (the hydrostatic
flux `P = g h²/2` is tagged explicitly in the model's `system_model` property).

## Author by inheritance + closures

A new model **subclasses the closest existing one** and supplies a different
closure list or overrides `derive_model()` — it never re-derives from scratch and
never patches the parent with a flag:

```python
from zoomy_core.model.models.swe import SWE
from zoomy_core.model.models.closures import ManningFriction, ElderViscosity

class MalpassetSWE(SWE):
    # bed friction + eddy viscosity added by COMPOSING closures — no flags on SWE
    def __init__(self, *args, **kwargs):
        super().__init__(*args, closures=[ManningFriction(), ElderViscosity()], **kwargs)
```

`KESME(SME)` adds two transported scalars (k, ε) through the same
project/resolve pipeline; `VAM` keeps the vertical-momentum row and the
non-hydrostatic pressure instead of eliminating them; `MLSME` / `MLVAM` build one
full sub-model **per moving layer** and couple them through the interface fluxes
`G`. In every case the new structure enters by subclassing + composing, never by
branching the parent.

## Authoring checklist

- `variables` declared as a list of names.
- `parameters` as a dict mapping name to `(default, "positive" | "real")`.
- At least one of `flux`, `nonconservative_matrix`, `source` overridden.
- Optional: `hydrostatic_pressure`, `diffusion_matrix(_explicit)`,
  `source_explicit`.
- For multi-route splitting (Chorin, IMEX, hyperbolic/parabolic):
  `solver_tag(...)` on every non-zero term. Catalogue of canonical tag
  names: [system-model.md](system-model.md).
- `initial_condition` over `(t, position, p)`; `initial_aux_condition`
  if `aux_variables` is non-empty.
- `boundary_conditions` covering every boundary tag;
  `aux_boundary_conditions` defaults to per-tag extrapolation.
- The model closes with `InvertMassMatrix` so the solver sees `M = I`;
  see [system-model.md](system-model.md) and [numerics.md](numerics.md).
  Reference solver: [`numpy.md`](../backends/numpy.md).
