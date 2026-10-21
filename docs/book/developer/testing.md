# Testing

[![CI](https://github.com/ZoomyLab/Zoomy/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/ZoomyLab/Zoomy/actions/workflows/ci.yml)
[![Weekly](https://github.com/ZoomyLab/Zoomy/actions/workflows/weekly.yml/badge.svg)](https://github.com/ZoomyLab/Zoomy/actions/workflows/weekly.yml)
[![Containers](https://github.com/ZoomyLab/Zoomy/actions/workflows/build-containers.yml/badge.svg?branch=main)](https://github.com/ZoomyLab/Zoomy/actions/workflows/build-containers.yml)

## Tiers

| tier | runs | what |
|---|---|---|
| push | every push and pull request to `main` and `develop` | the components the change touches, a few minutes each |
| weekly | Sunday 22:00 UTC, and by hand | every test of at most 30 minutes, all backends end to end, all containers rebuilt |
| manual | by hand, on a machine with many cores | cross-backend matrix and timing regressions, no workflow yet |

`#notest` in the head commit message of a push or in the pull request title skips all jobs. `[skip ci]` is GitHub's own and skips the whole workflow. Branch rules require the check `ci-ok`, which passes when every job succeeded or was skipped.

A change also runs what depends on it.

| changed | also runs |
|---|---|
| core | jax, server, prepost, plotting, superrepo tests, tutorials, QR cases, backend e2e |
| jax | superrepo tests, tutorials, QR cases, backend e2e |
| plotting | tutorials, QR cases |
| prepost | server |
| GUI, CLI | QR cases |
| `.github/` | everything |

## Markers

| marker | meaning | tiers |
|---|---|---|
| `large` | passes, too slow for the push budget | weekly |
| `legacy` | exercises superseded code | weekly |
| `broken` | fails today, `reason=` is the first error line | none |
| `benchmark` | timing regression | manual |

An unmarked test is in the push tier. `large`, `legacy` and `broken` are set by test id in `tests/ci_tiers.tsv`, one line per test, so no test file carries them. A fixed test loses its line.

## How to run

Push tier, from the folder of the component:

```bash
cd library/zoomy_core
pytest -n 4 -m "not large and not legacy and not broken and not benchmark"
```

The same in `library/zoomy_jax`. The other components run from the repository root with its configuration:

```bash
pytest -c pytest.ini --rootdir=. -n 4 -m "not large and not legacy and not broken and not benchmark" library/zoomy_server/tests
```

Replace the path by `library/zoomy_prepost/tests`, `library/zoomy_plotting/zoomy_plotting/tests`, or `tests --ignore=tests/qr_cases --ignore=tests/backend_e2e --ignore=tests/tutorials` for the superrepo tests.

Weekly tier, the same commands with the larger selection:

```bash
ZOOMY_RUN_LARGE_TESTS=1 pytest -n 4 -m "not broken and not benchmark" --timeout=1800
```

Manual tier:

```bash
pytest tests/symsolver -m benchmark
```

Tutorial notebooks, QR cases and the backend e2e have their own commands, see the READMEs in the table below.

## Where tests live

| tests | location | add one |
|---|---|---|
| cross-backend matrix | `docs/dev/symbolic_solver/matrix/` | [Extending](extending.md) |
| timings | `tests/symsolver/` | [Extending](extending.md) |
| tutorial notebooks | `tests/tutorials/`, `pytest tests/tutorials -m tutorial` | [Adding a tutorial](tutorials.md) |
| thesis QR cases | `tests/qr_cases/` | `tests/qr_cases/README.md` |
| backend containers | `tests/backend_e2e/` | `tests/backend_e2e/README.md` |
| libraries | `library/zoomy_*/tests/` | pytest, the root `conftest.py` sets the markers |
