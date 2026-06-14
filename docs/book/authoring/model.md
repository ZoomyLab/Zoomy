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
Model ──SystemModel.from_model(m)──▶ SystemModel ──NumericalSystemModel
 (symbolic derivation graph)         (frozen F,P,B,S,M)   .from_system_model(sm,…)──▶
NumericalSystemModel ──▶ Solver   (fvm/, param.Parameterized; numpy is the
 (frozen sm + numerics specs)            reference, backends port from it)
```

Code printers (`transformation/to_c.py`, `to_amrex.py`, `to_ufl.py`, …) and
analysis (dispersion, hyperbolicity) both hang off `SystemModel`. The
NumericalSystemModel stage is in [numerics.md](numerics.md); the solver in
[`../backends/numpy.md`](../backends/numpy.md).

## Build from the existing blocks — reuse, don't reinvent

**You build only from the framework's blocks. You never hand-roll a solver, a
time loop, or a parameter — and you never patch a model with a flag, an
`if`-branch, or a private attribute.** Before adding *any* new structure, state
to the user in writing: (a) **where it fits** — which existing block absorbs it
— and (b) **what is genuinely new**. Then:

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
  wiring fail loudly. (Full self-check: `~/.claude/agents/zoomy.md` → *solve root
  causes, not workarounds*.)
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

Operator methods return `ZArray` instances; the shape contract is:

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

Spectrum: `eigenvalues` solves the characteristic polynomial of the
normal-projected quasilinear matrix symbolically; set
`eigenvalue_mode = "numerical"` to defer to `np.linalg.eigvals` at runtime.

Conditions: `initial_condition` / `initial_aux_condition` are functions of
`(t, position, p)`. `boundary_conditions` / `aux_boundary_conditions` are
`BoundaryConditions` instances; aux BCs default to per-tag `Extrapolation`.

Full signatures live in `library/zoomy_core/zoomy_core/model/basemodel.py`.

## `Zstruct` dot-access

```python
m.variables.h          # sympy Symbol
m.parameters.g         # sympy Symbol on the symbolic side
m.parameters.g = 9.81  # numeric override on the user-facing struct
m.aux_variables.b_x
```

`m.variables["h"]` works but is reserved for runtime-string keys.

## Derivation via `apply()`

`model.apply(operation)` records the operation for `.describe()` and transforms
the equations. The operation catalogue lives in
`library/zoomy_core/zoomy_core/model/operations.py` and `…/model/derivation/`.

Physics closures are composed as a **list of `Closure` ops**
(`model/models/closures.py`), applied via
`apply_stress_closures(model, closures=[Newtonian(), NavierSlip(), …])` —
available closures include `Newtonian`, `NavierSlip`, `StressFree`,
`ManningFriction`, `ElderViscosity`, `KEpsilonViscosity`, `RoughWall`,
`Bingham`. **Add physics by adding a closure to the list, never by editing the
model.** Moving-interface kinematic BCs enter the upstream derivation via
`KinematicBC(...)`.

Operations transform each equation:

- `ProductRule()` — single-term-only; use via `apply_to_term(idx, ...)`.
- `Multiply(factor, outer=False)` — scale every term; `outer=True` with
  a Zstruct of test functions promotes a leaf to a Zstruct of children.
- `DepthIntegrate(...)` — Leibniz / fundamental-theorem integration in `z`.
- `ApplyKinematicBCs(state)` — substitutes kinematic BCs into surface
  terms produced by `DepthIntegrate`.
- `EvaluateIntegrals(state)` — single entry point closing `∫_0^1 dζ`;
  routes opaque `phi_k(ζ)` integrands to the basis cache.
- `Expand()` — distribute `Sum` / `Add` factors across products.
- `AffineProjection(state, lower, upper)` — FEM-style affine map
  `z = ζ·(upper − lower) + lower`; canonical reference-element step.
- `Integrate(var, lower, upper, method)` — explicit-bounds sympy
  integration (`analytical` / `direct` / `auto`).
- `IntegralTransform(ref_interval)` — generic remap to a reference interval.
- `MapBasisToReference(b, h)` — rewrite basis arguments from `(t, x, z)`
  to the reference-element coordinate.
- `SimplifyIntegrals(state)` — fold and cancel sibling integrals.
- `ProjectBasisIntegrals(...)` — internal helper of `EvaluateIntegrals`.
- `SigmaTransform(state)` — Kowalski–Torrilhon σ-mapping via chain rule.

The closure step that lifts to `SystemModel` is `derive_model()`; it ends
with `InvertMassMatrix` then per-term solver tagging. Tags are manual —
the author calls `solver_tag(flux=..., source=..., nonconservative_flux=...)`
on each equation; untagged terms return zero from the operator API.

## `describe()`

```python
print(m.describe(derivation="mermaid"))   # graph TD block; verbose=True for one edge per apply()
print(m.describe(derivation="markdown"))  # parent / operations / self
```

The rendered diagram lives on the [SystemModel page](system-model.md).

## Running example — author by inheritance + closures

Models live as **declarative classes** in `model/models/` (`sme.py:SME`,
`swe.py:SWE`, `vam.py:VAM`, `ml_swe.py`, `ml_sme.py`, …). A new model
**subclasses the closest one and supplies a different closure list / overrides
`derive_model()`** — it never re-derives from scratch and never patches the
parent with a flag:

```python
from zoomy_core.model.models.swe import SWE
from zoomy_core.model.models.closures import ManningFriction, ElderViscosity

class MalpassetSWE(SWE):
    # bed friction + eddy viscosity added by COMPOSING closures — no flags on SWE
    def __init__(self, *args, **kwargs):
        super().__init__(*args, closures=[ManningFriction(), ElderViscosity()], **kwargs)
```

`derive_model()` (overridden per model) builds the equations from the blueprints
in `model/models/equations.py` (`Mass`, `Momentum`), applies the operations
(`operations.py` / `derivation/`: `ProductRule`, `Integrate`, `EvaluateIntegrals`,
`AffineProjection`, `SigmaTransform`, …), applies the closures, and closes with
`InvertMassMatrix` + tagging. **`SME.derive_model`** (`model/models/sme.py`) is
the canonical worked example; `SystemModel.from_model(model)` then freezes it.

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
