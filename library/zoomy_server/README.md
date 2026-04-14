# zoomy-server

FastAPI solver server for [Zoomy](https://github.com/mbd-rwth/Zoomy) — submits and manages simulation jobs across multiple backends.

## Installation

```bash
pip install zoomy-server
```

## Usage

```bash
zoomy-server --adapter numpy --port 8000
```

## Backends

- **NumPy** — pure Python FVM solver
- **JAX** — GPU-accelerated solver
- **AMReX** — block-structured AMR
- **DMPlex** — PETSc unstructured meshes
