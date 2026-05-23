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

`DerivedModel.apply(operation)` mutates `self._system` in place and records
the operation for `.describe()`. The operations catalogue lives in
`library/zoomy_core/zoomy_core/model/models/ins_generator.py`.

Relations / Materials / Assumptions substitute symbols across all
equations:

- `Newtonian(state, nu=...)` — `τ_ij = μ (∂_i u_j + ∂_j u_i)`.
- `Inviscid(state)` — every `τ_ij → 0`.
- `HydrostaticPressure(state)` — vertical momentum to `∂_z p + ρ g = 0`.
- `StressFreeSurface(state)` — drops surface tractions at `z = η`.
- `KinematicBC(state, interface, at, mass_flux=None)` —
  `w|_at = ∂_t interface + u · ∇ interface (+ mass_flux/ρ)`.

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

## Running example — SME with σ-coordinates

Full derivation:
`thesis/notebooks/modeling/transparent_derivations/kowalski_sigma_transform.py`.
Skeleton:

```python
from zoomy_core.model.models.ins_generator import (
    StateSpace, FullINS, SigmaTransform, KinematicBC,
    Integrate, Multiply, Expression,
)

state = StateSpace(dimension=2)
sys = FullINS(state)

sys.momentum.z.apply({state.tau.zx: 0, state.tau.zz: 0, state.w: 0})
sys.momentum.z.apply(
    Integrate(state.z, state.z, state.eta, method="analytical"))
sys.momentum.z.apply({state.p.subs(state.z, state.eta): 0}).simplify()
sys.momentum.x.apply(sys.momentum.z.solve_for(state.p)).simplify()

sys.apply(SigmaTransform(state))
sys.apply(KinematicBC(state, interface=state.b,   at=0))
sys.apply(KinematicBC(state, interface=state.eta, at=1))
```

Galerkin projection then multiplies each leaf by a test family
(`Multiply(phi_k, outer=True)`), applies `Integrate(zeta, 0, 1, method="analytical")`,
solver-tags every term, and wraps the result in a `Model` subclass
consumed by `SystemModel.from_model(model)`.

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
