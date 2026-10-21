# Tests and timings

Generated from saved results by `docs/scripts/generate_symsolver_results.py`;
nothing on this page is typed by hand.

```
run the tests  ──▶  results saved under tests/results/symsolver/  ──▶  docs build regenerates this page
```

| step | command | writes |
|---|---|---|
| cross-backend matrix | `python matrix.py run BACKEND CASE [CONFIG]` in `docs/dev/symbolic_solver/matrix/` | the matrix store (`MATRIX_STORE`) |
| save the matrix | `MATRIX_STORE=... python docs/scripts/generate_symsolver_results.py --from-store` (zoomy env) | `tests/results/symsolver/matrix_latest.json` (rows, date, library commits) |
| timings | `taskset -c N pytest tests/symsolver -m benchmark` on an idle machine | `tests/results/symsolver/timings.jsonl`, `timings_latest.json` |
| render | `python docs/scripts/generate_symsolver_results.py` (part of the docs build) | `docs/_generated/symsolver_*.md` |

Commit the files under `tests/results/symsolver/`; the docs build reads only
those.  How to add a case: [Extending](extending.md).

## Cross-backend matrix

Every backend that prints the templates must pass the same physics checks:
convergence orders, exact solutions, conservation, symmetry, parallel = serial,
GPU = CPU.

```{include} ../../_generated/symsolver_matrix.md
```

## Timings

March time of representative runs (`tests/symsolver/timing_cases.py`), one
CPU core for `serial`.

```{include} ../../_generated/symsolver_timings.md
```
