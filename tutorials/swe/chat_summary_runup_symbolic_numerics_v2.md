# Chat Summary: Symbolic Numerics V2 and Beach Runup Debugging

## Scope Covered

This chat focused on:

- introducing `symbolic_numerics_v2` with configurable field locations,
- adding runtime execution support (NumPy/JAX lambdas) for symbolic numerics,
- validating wet/dry hydrostatic reconstruction behavior and index handling,
- building a SWE vs GN beach-runup comparison workflow,
- debugging benchmark diagnostics and solver output visibility.

---

## Main Additions

### 1) New symbolic numerics v2 module

Added:

- `library/zoomy_core/zoomy_core/fvm/symbolic_numerics_v2.py`

Key features added:

- field-map driven access for `h`, `b`, and optional `hinv`:
  - supports `b` in `Q` or `Qaux`,
  - avoids hard-coded `Q[0]`, `Q[1]` assumptions.
- wet/dry hydrostatic reconstruction generalized to mapped fields.
- runtime callable generation for:
  - NumPy (`to_runtime_numpy()`),
  - JAX (`to_runtime_jax()`; dependent on `jax` install).
- stricter runtime error reporting for mixed scalar/vector outputs.

### 2) Step-1 validation script

Added:

- `web/tutorials/swe/check_symbolic_numerics_v2_step1.py`

Purpose:

- smoke-test v2 symbolic numerics on topo SWE-style states,
- verify alternative field map (`b` in `Qaux`) executes,
- compare backend behavior and ensure finite outputs.

### 3) Beach runup prototype script

Added:

- `web/tutorials/swe/beach_runup_swe_vs_gn_classical_v2.py`

Capabilities:

- compares `SWE` and `GN-classical` topo-aware models with shared setup,
- uses symbolic numerics v2 in the custom face operator path,
- computes/prints shoreline and rough Synolakis-style reference metrics,
- writes comparison plots for free surface, velocity, and raw fields.

### 4) IMEX output/logging improvements

Updated:

- `library/zoomy_core/zoomy_core/fvm/solver_imex_numpy.py`

Changes:

- added HDF5 snapshot writing path (`write_output=True`) compatible with existing io helpers,
- added periodic IMEX progress logging every 10 steps (`imex iteration ...`) to avoid long silent runs.

### 5) Multi-case benchmark isolation suite

Added:

- `web/tutorials/swe/benchmark_suite_symbolic_numerics_v2.py`

Cases:

1. flat, fully wet (no wet/dry front),
2. flat, wet/dry front,
3. current beach-runup setup.

Goal:

- isolate whether issues come from core numerics/solver coupling vs benchmark setup.

---

## Important Corrections Made During Chat

1. **Viscosity identity bug in v2**

- Incorrectly suppressed viscosity on `h` in fluctuations.
- Corrected to suppress viscosity on `b` (when `b` is in `Q`), matching topography handling intent.

2. **Wrong runup flux path variant**

- Temporary switch to `PositiveQuasilinearRusanov` was not aligned with intended split flux+NC path.
- Reverted to `PositiveNonconservativeRusanov` and fixed custom operator to use:
  - `numerical_flux` and
  - `numerical_fluctuations`
  together in the face update.

3. **IMEX timeline loading mismatch**

- Timeline reading failed for GN because IMEX solver initially did not write snapshots.
- Fixed by adding write-output path in `IMEXSourceSolver`.

---

## What Worked

- `symbolic_numerics_v2` compiles and runs in NumPy runtime path.
- Field mapping works for configured `h`/`b`/`hinv` locations.
- Flat wet/dry-front test case shows finite evolution and measurable shoreline/front shift.
- GN IMEX path now emits progress logs and can save HDF5 snapshots.

---

## What Did Not Work (or Was Inconclusive)

1. **Direct beach-runup `R_num/R_ref` match (current setup)**

- In tested quick/medium settings, shoreline-based runup metric stayed at or near zero.
- This remained true even after fixing core flux/fluctuation coupling.

2. **JAX runtime check**

- JAX backend check was skipped due to missing `jax` in the active environment.

---

## Open Problems to Continue Next Chat

1. **Benchmark fidelity for Synolakis-style comparison**

- Confirm case mapping (domain scaling, initial-wave placement, shoreline definition) against a concrete published setup.
- Ensure the selected metric (`R_num`) is computed in the same convention as the reference formula/data.

2. **Runup metric robustness**

- Improve shoreline detection and sensitivity studies (`h_threshold`, interpolation at wet-dry interface).
- Add explicit shoreline trajectory plots and gauge time-series outputs.

3. **Performance/accuracy sweep**

- Run systematic sweeps over:
  - `n_inner_cells`,
  - simulation horizon,
  - CFL,
  - and case parameters (`a/h0`, initial offset)
  to identify when physically meaningful runup emerges.

4. **SWE vs GN deficiency demonstration**

- Once benchmark mapping is validated, compare:
  - runup maxima,
  - phase/shape at gauges,
  - and dispersive behavior differences.

---

## Files Touched In This Chat

- `library/zoomy_core/zoomy_core/fvm/symbolic_numerics_v2.py`
- `library/zoomy_core/zoomy_core/fvm/solver_imex_numpy.py`
- `web/tutorials/swe/check_symbolic_numerics_v2_step1.py`
- `web/tutorials/swe/beach_runup_swe_vs_gn_classical_v2.py`
- `web/tutorials/swe/benchmark_suite_symbolic_numerics_v2.py`
- `tutorials/swe/chat_summary_runup_symbolic_numerics_v2.md`
