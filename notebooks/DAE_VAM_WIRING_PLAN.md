# DAE-VAM wiring plan (post-toy)

After the ARS232/ARS343 toy verification (`tests/scripts/dae_toy/test_ars_imex_dae.py` —
both schemes hit theoretical convergence orders on linear and nonlinear
index-1 DAEs), the next step is wiring the IMEX-ARK time integrator
onto the existing `IMEXSolver` and `VAMModel`.

This is more involved than the toy because **VAM's existing
`SplittingSolver` does not treat pressure as a state variable** — it
uses Chorin projection (`_pressure_correction`) on the velocity-only
state `(b, h, hu, hw)` after each timestep. A true DAE-direct approach
needs pressure in the state and continuity as a per-stage constraint.

## Three things to build

### 1. Generic IMEX-ARK time integrator wrapper

Promote the toy's `imex_ark_step` to a class method on `IMEXSolver`
(or a new mixin):

```
class IMEXARKMixin:
    def step_ark(self, dt, tableau):
        # 1. For each stage i = 1..s:
        #    a. Build explicit accumulator from previous stages' f_E, f_I.
        #    b. Newton-GMRES on the implicit residual:
        #       R_dyn(K) = M_dyn (K - rhs_E) - dt γ_ii f_I(K)[dyn]
        #       R_alg(K) = G(K)
        #    c. Store f_E(K_i), f_I(K_i) for next stages.
        # 2. Accumulate b̃_i, b_i to produce y^{n+1}.
```

Where `f_E` is **the existing hyperbolic flux operator** (Riemann FVM
+ non-conservative path integral) — already in
`HyperbolicSolver.get_flux_operator`. So `f_E` requires *no new code*.

`f_I` is **the existing implicit-source path** plus the new
constraint rows. The current `IMEXSolver.implicit_source` provides
the source half; the constraint half is a new function the model
must expose.

### 2. Extend `VAMModel` to expose constraint hooks

The `solver_dae_numpy.py` skeleton (already in tree, 469 LOC, no
callers) declares the right hook contract:

```
model.dae_dynamic_rows : np.ndarray of int
model.dae_constraint(Q, Qaux, params, mesh, t) -> (n_alg, n_cells)
model.dae_constraint_jac_q(Q, Qaux, params, mesh, t) -> (n_alg, n_vars, n_cells)
```

These don't yet exist on `VAMModel`. Two options:

**Option A — derive from the symbolic system.** The `PDESystem` returned
by the new transparent VAM derivation
(`tutorials/vam/vam_l1_physical_z.py`) already separates evolution rows
from algebraic constraints (cont_j_*, KBC_bot, KBC_top). A bridge:

```python
def dae_partition_from_pdesystem(pdesys, base_state):
    """Returns (dyn_rows, alg_rows, G_callable, G_jac_callable)."""
    M_t, M_xa, M_0 = extract_quasilinear_pencil(pdesys)
    dyn = [i for i in range(M_t.rows) if not all_zero_row(M_t, i)]
    alg = [i for i in range(M_t.rows) if all_zero_row(M_t, i)]
    G_expr = sp.Matrix([pdesys.equations[i] for i in alg])
    G_q_expr = G_expr.jacobian(pdesys.fields)
    return dyn, alg, lambdify(...), lambdify(...)
```

This is the model-agnostic path. It is what the user described
("a small new bridge where we can in the derivedmodel detect
which equations are constraints").

**Option B — hand-write the hooks for `VAMModel`.** Faster for a
proof-of-concept; less general.

### 3. Pressure as a state variable

Currently `VAMModel.variables = ['b', 'h', 'hu0', 'hu1', 'hu2',
'hw0', 'hw1', 'hw2']`. The DAE-direct version needs:

```
['b', 'h', 'hu0', 'hu1', 'hu2', 'hw0', 'hw1', 'hw2', 'p0', 'p1', 'p2']
                                                     ^^^^^^^^^^^^^^^^^^
                                                     new (algebraic)
```

The `p_i` rows have `M_t = 0` and the constraint
`G_i(Q) = (cont projection j=i + KBC bot/top)` determines them.

Mass and `(hu, hw)` momenta still evolve via flux + non-conservative
matrix. The existing flux operator is *unchanged* — we just don't
project pressure out at the end of the step; instead the implicit
ARS stage Newton solves for it.

## Why this isn't done in this autonomous session

The third piece (pressure-in-state) is a model-API change that
affects:
- `VAMModel.derive_model` (add `p_i` to state, expose constraint)
- `to_numpy.py` (compile constraint + Jacobian)
- Initial conditions (need `p_0` consistent with constraint)
- Boundary conditions (pressure BC at top/bottom)
- The Riemann solver (pressure flux contribution)

Each is small individually; together they constitute a
several-hundred-line refactor that needs careful testing and
review. **Better discussed with the user before committing.**

## Recommended next session

1. Decide on Option A vs Option B above (auto-detect from
   `PDESystem` vs hand-written hooks). Option A is the right
   long-term answer; Option B gets a working VAM-DAE example faster.
2. Either way, the **first concrete file to write** is

   ```
   tutorials/vam/vam_dae_direct_simulate.py
   ```

   that builds VAM-with-pressure, wraps the existing solver with
   the ARS343 stage loop, runs the same dam-break that
   `run_sme_vam_matrix.py` runs, and compares against
   `SplittingSolver`'s output.

3. Verification target: VAM-DAE-direct match VAM-Splitting at
   least to the splitting-error tolerance (~1e-3 relative on a
   small problem).

## Pinned references (per `notebooks/DAE_REFERENCES.md`)

- Ascher-Ruuth-Spiteri 1997 — DOI 10.1016/S0168-9274(97)00056-1
- Pareschi-Russo 2005 — DOI 10.1007/s10915-004-4636-4
- Gardner et al. 2018 — DOI 10.5194/gmd-11-1497-2018

Tableau used in toy: ARS232 + ARS343 from Ascher-Ruuth-Spiteri 1997
Tables 1 and 2.
