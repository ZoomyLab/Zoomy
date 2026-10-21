# Symbolic-solver tests: timing and the matrix

Two suites for the symbolic solvers (JAX, foam, AMReX, NumPy), over the cross-backend
matrix in `docs/dev/symbolic_solver/matrix/`.

| file | marker | what |
|---|---|---|
| `test_timing.py` | `benchmark` | march time of 7 representative runs per (backend, config), 3 repetitions each (the JAX Chorin run: ~1 min setup + compile) |
| `test_matrix.py` | `medium` | every row of `matrix.measure()` on a matrix store is one test |
| `timing_cases.py` | | the registry of timing runs (sizes, backends, configs, agreement checks) |
| `timing_run.py`, `timing_records.py`, `timing_paths.py` | | measurement, result files, paths |

## Run

```bash
source docs/dev/symbolic_solver/matrix/env.sh          # PYTHONPATH of the four libraries, x64, CPU JAX
export ZOOMY_AMREX_SYMCACHE=/scratch/amrex_cache       # env.sh puts it under the worktree
taskset -c 80 pytest tests/symsolver -m benchmark      # timings; ONE core for the serial config
pytest tests/symsolver/test_timing.py -m benchmark -k "swe_o2 and amrex"
MATRIX_STORE=path/to/store pytest tests/symsolver/test_matrix.py -m medium
```

Python is `micromamba run -n zoomy python -m pytest ...` where `pytest` is not on the path.
`-m benchmark` lifts the `ZOOMY_RUN_BENCHMARK_TESTS` gate of `tests/conftest.py`; a plain
`pytest` skips the timing tests and runs `test_matrix.py`.

**Timings come from an otherwise idle machine.**  Pin the process to one core
(`taskset -c N`), start nothing else on that core or its hyperthread sibling,
and run the tests one after the other.  The record keeps `cores_pinned`; a
serial timing with `cores_pinned > 1` lets JAX use several threads and is not a
one-core number.  AMReX is pinned by the test to the first core of the process
affinity (it overwrites `ZOOMY_AMREX_RUN_CPUS` / `ZOOMY_AMREX_BUILD_CPUS`).  A busy
machine shows as spread between the 3 march times: look at `min_s` vs `median_s`.

| variable | default | meaning |
|---|---|---|
| `ZOOMY_TIMING_REPS` | 3 | repetitions of the march |
| `ZOOMY_TIMING_RESULTS` | `tests/results/symsolver` | directory of the two result files |
| `ZOOMY_TIMING_WORK` | `tests/symsolver/work` (ignored) | case directories, JAX reference cache (`ref/`) |
| `ZOOMY_TIMING_CONFIGS` | `serial,gpu1` | configs to run; `gpu1` skips without a GPU (`ZOOMY_TIMING_GPU=0` forces the skip) |
| `ZOOMY_TIMING_SUBPROCESS` | unset | `1`: JAX runs in a child process also on CPU (the path of `gpu1`: `zoomy-cuda` env) |
| `MATRIX_STORE` | `matrix/store` | store read by `test_matrix.py` (never written) |

## The cases (`timing_cases.py`)

| case | matrix case | size, steps (constant dt, a power of 2) | notes |
|---|---|---|---|
| `swe_o1`, `swe_o2` | C1 bump | 200^2 cells, 50 steps | O2 minmod, no MOOD |
| `vam1_chorin` | C8 / L2P channel | 200 x 20 cells, 8 steps | VAM(1); JAX pressure solver `auto` (pyamg on CPU, AmgX on GPU; matrix-free GMRES is ~200x slower per step) |
| `imex_swe` | C14 | 100^2 cells, 50 steps | O2, Manning n 0.5 implicit |
| `euler3d_o1`, `euler3d_o2` | C20 box | 40^3 cells, 40 steps | O2 with MOOD; the pulse of C20 |
| `diffusion_impl` | C26 | 48^2 cells, 10 steps | IMEX, Newtonian with `stress=Stress.full`; foam differentiates the implicit residual by finite differences (~80x JAX per step) |

Configs: `serial` on every backend (NumPy: serial only, the pressure solver `auto` = pyamg; agreement with JAX serial as the matrix); `gpu1` on JAX (child process in the `zoomy-cuda` env, `JAX_PLATFORMS=cuda`)
and AMReX (the CUDA container); foam has no GPU.  Choose the GPU with `CUDA_VISIBLE_DEVICES`.  Sizes target a
march of about 1-20 s on one core; the backends differ by up to 100x per step, so the fastest run is the lower end.

## What a timing test does

Case, backend, config -> `cases.Run` of the matrix (constant dyadic `dt`, `steps` steps, so every
backend takes exactly `steps` steps) -> 3 marches.

- JAX: `setup` = model, mesh, printing; `compile` = first call minus one step; `march` = 3 marches of the compiled solver.
- foam / AMReX: 3 full runs; `march` = the solver's own "march wall time"; `compile` = the `build` call (kernel
  printing and a hash lookup: 1-2 s, ~40 s for Chorin; a cold binary cache adds the real build); `setup` = the rest (model,
  case files, container start, read-back).
  `setup_s` and `compile_s` are those of the first repetition.
- Asserted: finite final state, `t = t_end`, AMReX loop count = `steps`, JAX on a GPU for `gpu1`, and the
  agreement with the JAX serial run of the same case where the matrix expects it (`relmax` or `l1rel` and the
  threshold of `matrix.py`: `RO` 1e-12, `PRESS` 1e-9, `TH["C8 foam"]`; implicit solves 1e-8).  The JAX serial
  state is cached in `work/ref/` (key: library commits + source mtimes) and computed once if no JAX test ran first.
- A failed check or a raised error still appends a record, with `status` not `ok`.

## Results

`timings.jsonl`: one JSON object per test, appended (never rewritten).  `timings_latest.json`:
`{"schema": 1, "updated": date, "rows": [record, ...]}`, the last record of every `case/backend/config`,
rebuilt from the jsonl after each append (file lock, safe for concurrent pytest processes).

```
schema 1 | date (UTC) | host | cpu (lscpu) | gpu (nvidia-smi), n_gpus | cores_pinned, cpus ("80" or "80-95")
libs {zoomy_core, zoomy_jax, zoomy_foam, zoomy_amrex, superrepo: short commit, "+dirty" if modified}
wave_speed (`SystemModel.spectral_radius_method`; before ANALYSIS 61 `zoomy_core.symsolver.WAVE_SPEED`) | case, backend, config | device (JAX: cpu / gpu), detail (Chorin: pressure=pyamg|amgx)
cells, steps, dt, reps | march_s [3 s] | min_s, median_s | ms_step, us_cell_step (of min_s)
setup_s, compile_s | status ("ok" | "FAIL: ..." | "error: ...") | diff {metric, value, tol, source, ok} or null
```

A record of an `error:` status holds only the metadata, case, backend, config and status.

The docs page "Tests and timings" (`docs/book/developer/tests.md`) is rendered from
`timings_latest.json` and `matrix_latest.json` (the matrix rows, written by
`MATRIX_STORE=... python docs/scripts/generate_symsolver_results.py --from-store`) by
`python docs/scripts/generate_symsolver_results.py`, a step of the docs build.  Commit both
files after a run.

## test_matrix.py

`matrix.measure()` yields the rows of the matrix table `(case, check, backend, config, value, threshold, note)`;
a row is a test.  `value` against `threshold` is the verdict of `matrix.table()`.  FAIL fails, `n/a`,
`unsupported` and information rows (no threshold) skip.  A store with missing runs yields fewer rows, not errors.
