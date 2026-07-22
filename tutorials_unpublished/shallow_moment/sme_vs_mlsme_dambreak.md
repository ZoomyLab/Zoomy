# Tutorial: SME(2) vs ML-SME(0,2) — velocity profiles in a frictional dam break

**Files:** `sme_vs_mlsme_dambreak.py` (jupytext, py:percent) ↔ `sme_vs_mlsme_dambreak.ipynb` (executed, outputs embedded).
**Artifacts:** `_artifacts/velocity_profiles.png`, `_artifacts/velocity_profiles.gif`, `_artifacts/{sme2,mlsme02}.h5` (gitignored via local `.gitignore`).

## What it shows

Two depth-averaged **shallow-moment** models run on the *same* 1-D dam break and
are compared by their **reconstructed vertical velocity profile** `u(z)`:

| model | profile | state |
|---|---|---|
| **SME(2)** | one layer, 3 moments → smooth quadratic `u(z)` | `[b, h, q_0, q_1, q_2]` |
| **ML-SME(0,2)** | 2 constant layers → piecewise-constant `u(z)` | `[b, h, q_1_0, q_2_0]` |

The depth-averaged `h(x)` is nearly identical between the two; the difference
lives entirely in `u(z)`. **Friction is essential to the point of the tutorial:**
a Navier-slip bed + bulk viscosity drives vertical shear (surface faster than the
drag-retarded bed). Without friction the profiles are flat and the comparison is
empty.

## Setup

- Domain `(0,10)`, `NC=200`, dam at `x=5`, `h_L=2 / h_R=1`, `t_end=2 s`, CFL 0.4, 40 snapshots.
- Closures `[Newtonian(), NavierSlip(), StressFree()]`, params `nu=0.05, lambda_s=0.05`.
- Solver: `HyperbolicSolver` (numpy), `ReconstructionSpec(order=1)`, `write_output=True`.
- Probes at `x=3` and `x=7` (left/right of the discontinuity).

## How the profile reconstruction works (the load-bearing trick)

`interpolate_to_3d` lifts the depth-averaged state to the canonical 3-D profile
`[b, h, u, v, w, p](z)`; **row 2 is `u(z)`**, `z∈[0,1]` bed→surface. We
`sp.lambdify(state + aux_state + parameters + [z], row2, "numpy")` once per model,
pass `aux=0` (u doesn't use aux), and feed `parameter_values` in order. See
`make_u_of_z`. HDF5 stores fields by index (`h` = state index 1).

## Run it

```
MPLBACKEND=Agg JAX_PLATFORMS=cpu micromamba run -n zoomy \
  jupytext --to notebook --execute sme_vs_mlsme_dambreak.py
```

Or open the `.ipynb` in Jupyter (zoomy kernel) and Run-All. Requires the sqlite
fix in the `zoomy` env (`micromamba install -n zoomy -c conda-forge --force-reinstall libsqlite`).
