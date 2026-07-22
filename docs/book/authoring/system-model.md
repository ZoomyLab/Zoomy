# System Model

A `SystemModel` is the **frozen operator form** of a `Model`. It holds the
matrices every solver and every code printer consumes, in one canonical balance
law:

$$
M(Q)\,\partial_t Q \;+\; \nabla\!\cdot\!\big(F(Q) + P(Q)\big)
\;+\; \sum_d B(Q)[:,:,d]\,\partial_d Q \;-\; S(Q) \;=\; 0.
$$

`SystemModel.from_model(m)` walks the model's operator methods once and freezes
the result. From then on the derivation is history — what remains is a fixed
set of tensors with a fixed shape contract.

```python
from zoomy_core.systemmodel import SystemModel

sm = SystemModel.from_model(model)
sm.describe()                     # the operator block
sm.describe(full=True)            # + Jacobians, spectrum, reconstruction maps
```

## The shape contract

This is the contract every backend relies on. `n_eq` is the number of equation
rows, `n_state` the number of state entries, `n_dim` the spatial dimension.

| Slot | Shape | Meaning |
| --- | --- | --- |
| `mass_matrix` | `(n_eq, n_state)` | $M(Q)$ — canonically $I$ |
| `flux` | `(n_eq, n_dim)` | $F(Q)$, conservative |
| `hydrostatic_pressure` | `(n_eq, n_dim)` | $P(Q)$, held separate from $F$ |
| `nonconservative_matrix` | `(n_eq, n_state, n_dim)` | $B(Q)$ in $B\,\partial_x Q$ |
| `source`, `source_explicit` | `(n_eq, 1)` | $S$, implicit / explicit |
| `diffusion_matrix[_explicit]` | `(n_eq, n_state, n_dim, n_dim)` | $A$ in $\nabla\!\cdot(A\!:\!\nabla Q)$ |
| `quasilinear_matrix` *(derived)* | `(n_eq, n_state, n_dim)` | $\partial F/\partial Q + \partial P/\partial Q + B$ |
| `source_jacobian_wrt_variables` *(derived)* | `(n_eq, n_state)` | $\partial S/\partial Q$ |
| `source_jacobian_wrt_aux_variables` *(derived)* | `(n_eq, n_aux)` | $\partial S/\partial Q_\text{aux}$ |
| `eigenvalues` *(derived)* | `(n_eq, 1)` or `None` | `None` ⇒ numerical wavespeed at runtime |
| `update_variables` | `(n_eq, 1)` | per-step state map |
| `update_aux_variables` | `(n_aux, 1)` | per-step aux map |
| `reconstruction_variables`, `state_from_reconstruction` | `(n_state,)` | the reconstruction pair |
| `interpolate_to_3d` | `(n_3d,)` | column profile `[b, h, u, v, w, p]` |
| `project_from_3d` | `(n_state,)` | its Galerkin inverse |

$P$ is kept out of $F$ so that well-balanced reconstruction can read it off
directly. The system is in general **rectangular**: `equation_to_state_index[r]`
records which state entry row `r` updates (identity for square systems,
non-identity for splitter sub-systems).

Hydrostatic pressure lives in its own slot precisely so a scheme can treat it
specially — if that slot is empty on a free-surface model, something went wrong
upstream, and `describe()` shows it immediately.

## Operator argument signatures

Every operator takes a fixed argument list. This table is machine-readable as
`OPERATOR_ARG_SLOTS`, and it is the single source of truth every backend reads
through `sm.operator_signature(name)`:

| Operators | Arguments |
| --- | --- |
| `flux`, `hydrostatic_pressure`, `nonconservative_matrix`, `quasilinear_matrix`, `diffusion_matrix[_explicit]`, `mass_matrix`, `source_explicit`, both `source_jacobian_*` | `(Q, Qaux, p)` |
| `eigenvalues` | `(Q, Qaux, p, n)` |
| `source` | `(Q, Qaux, p, t, dt, X)` |
| `update_variables` | `(Q, Qaux, p, dt)` |
| `update_aux_variables` | `(Q, Qaux, p, t, X)` |

`Q` is the state, `Qaux` the auxiliary state, `p` the parameters, `n` the face
normal, `X` a length-3 position.

## `Q` versus `Qaux`

The solver advances **`Q`**. **`Qaux`** carries everything the operators need
that is not itself integrated in time — eliminated fields, topography, and
critically the **spatial derivatives of the state**.

You never write a finite-difference stencil. During `from_model`,
`expose_aux_atoms()` scans every operator entry, replaces each spatial
`Derivative(field)` atom with a fresh aux symbol (`dhdx`, `dq0dx`, …), and
records it in `sm.aux_registry` with the per-axis derivative orders. The solver
fills every derivative-aux in one least-squares gradient pass per step. In the
shallow-water example from [Model](model.md), `sm.aux_state` comes back as
`[dq0dx, dhdx, dbdx]` without the author declaring anything.

## Solver tags — how terms reach slots

Each non-zero piece of a derivation carries a **solver tag** that routes it to a
slot. The routing is fixed (`model/derivation/tag_catalog.py`,
`SOLVER_TAG_TO_SLOT`):

| Tag | Slot |
| --- | --- |
| `flux` | `flux` |
| `nonconservative_flux` | `nonconservative_matrix` |
| `hydrostatic_pressure` | `hydrostatic_pressure` |
| `time_derivative` | `mass_matrix` |
| `implicit_source` / `explicit_source` | `source` / `source_explicit` |
| `implicit_diffusion` / `explicit_diffusion` | `diffusion_matrix[_explicit]` |

```{warning}
An **untagged sub-expression silently returns zero**. The derivation-time
`AutoTag` operation assigns physics categories, but the final expression → slot
routing is explicit. This is the single most common way a derived model comes
out quietly wrong — which is why printing `describe()` before running is a
standing rule.
```

## Post-freeze operations

`sm.apply(op)` works on a frozen system the same way `Model.apply` works on a
derivation. Three operations ship:

- **`InvertMassMatrix`** — divide each evolution row by its diagonal `M_ii`, so
  `M = I`. Guarded by `assert_diagonal_mass_matrix()`.
- **`RemoveNonDiagonalH`** — substitute the mass equation into rows with a
  non-zero `M[:, h]` column, pushing the cross-term into `B` and `S`.
- **`HydrostaticReconstruction`** — repackage chain-derived
  $g\,h\,\partial_x\eta$ into the standard $P = g h^2/2$ that Audusse-type
  Riemann solvers expect.

Regularisation helpers live in `zoomy_core.systemmodel`: `regularize_depth_aux`,
`regularize_depth_direct`, `regularize_pow`, `kp_hinv`, `register_aux`,
`map_operator_slots`, `normalize_face_normal`.

```{note}
Never floor or clip the depth `h`. Use the Kurganov–Petrova denominator scale
(`kp_hinv`) — its numerator still vanishes at `h = 0`, which a floor does not.
```

## Changing variables

```python
sm.change_state_variables(new_state, transform)
sm.refresh_derived_operators(eigenvalues=False)
```

`transform` maps each replaced *old* state symbol to its expression in the *new*
state. With $J[i,j] = \partial T_i/\partial Q_{\text{new},j}$ the operators
update as $F_\text{new} = F_\text{old}(T)$,
$B_\text{new}[i,k,d] = \sum_j B_\text{old}[i,j,d](T)\,J[j,k]$,
$M_\text{new} = M_\text{old}(T)\,J$. Aux-registry entries targeting replaced
variables are chain-rule propagated. Pass `eigenvalues=True` only when the
spectrum was symbolic and the change alters characteristic structure.

## Analysis without a solver

A `SystemModel` is useful on its own — it already carries `M`, the quasilinear
matrix and the source Jacobian symbolically, so linear stability and dispersion
analysis need no mesh and no runtime:

```python
from zoomy_core.analysis import (
    linearise, symbolic_eigenvalues_at,
    extract_quasilinear_pencil, sample_hyperbolicity,
)

evs    = symbolic_eigenvalues_at(sm, {h: h0, q: q0}, axis=0)
sm_lin = linearise(sm, {h: 1.0, q: 0.0})
M_t, M_x, _ = extract_quasilinear_pencil(sm_lin)
print(sample_hyperbolicity(M_x[0], M_t, {h0: (0.1, 5.0)}, n_samples=2000).summary())
```

Also available: `plane_wave_dispersion`, `generalised_eigenvalues`,
`sample_generalised_eigenvalues`, `is_hyperbolic_at`, `critical_parameter`,
`spatial_dispersion`, `plot_dispersion`, `plot_hyperbolic_region_2d`.
A hyperbolicity sweep is the standard check before trusting a derived model in
a hyperbolic solver.

## Splitting — the VAM pressure projection

A rectangular system can be split into sub-systems that different
discretisations consume. This is how the non-hydrostatic VAM chain is solved:

```python
from zoomy_core.model.splitter import split_for_pressure

result = split_for_pressure(sm, pressure_vars=["P_0", "P_1"], dt=dt)
sm_pred, sm_press, sm_corr = result.predictor, result.pressure, result.corrector
```

`split_for_pressure_structural` detects each constraint row as an
identically-zero `mass_matrix` row. Each sub-system is itself a `SystemModel`,
so the same printers and solvers consume them unchanged. `VAM` and `MLVAM` also
expose `chorin_split(dt)` directly.

## Checklist

- `describe()` printed and the slots sanity-checked — especially
  `hydrostatic_pressure` on a free-surface model.
- `assert_diagonal_mass_matrix()` passes; `M = I` on evolution rows.
- Every parameter has a default in `parameter_values`.
- After `change_state_variables`, `refresh_derived_operators` has run.

Next: [Numerical System Model](numerics.md).
