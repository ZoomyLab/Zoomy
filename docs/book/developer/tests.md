# Tests

[![QR cases](https://github.com/ZoomyLab/Zoomy/actions/workflows/qr-cases.yml/badge.svg?branch=main)](https://github.com/ZoomyLab/Zoomy/actions/workflows/qr-cases.yml)
[![Backend e2e](https://github.com/ZoomyLab/Zoomy/actions/workflows/backend-e2e.yml/badge.svg?branch=main)](https://github.com/ZoomyLab/Zoomy/actions/workflows/backend-e2e.yml)
[![Tutorials](https://github.com/ZoomyLab/Zoomy/actions/workflows/tutorials.yml/badge.svg?branch=main)](https://github.com/ZoomyLab/Zoomy/actions/workflows/tutorials.yml)
[![Containers](https://github.com/ZoomyLab/Zoomy/actions/workflows/build-containers.yml/badge.svg?branch=main)](https://github.com/ZoomyLab/Zoomy/actions/workflows/build-containers.yml)

## Cross-backend matrix

```{include} ../../_generated/symsolver_matrix.md
```

## Timings

```{include} ../../_generated/symsolver_timings.md
```

## Where tests live

| tests | location | add one |
|---|---|---|
| cross-backend matrix | `docs/dev/symbolic_solver/matrix/` | [Extending](extending.md) |
| timings | `tests/symsolver/` | [Extending](extending.md) |
| tutorial notebooks | `tests/tutorials/` | [Adding a tutorial](tutorials.md) |
| thesis QR cases | `tests/qr_cases/` | `tests/qr_cases/README.md` |
| backend containers | `tests/backend_e2e/` | `tests/backend_e2e/README.md` |
| libraries | `library/zoomy_*/tests/` | pytest, the root `conftest.py` sets the markers |
