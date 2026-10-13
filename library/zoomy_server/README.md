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

A case folder with a `run.py` runs that script; without one, the settings run the case on the
symbolic solvers (`zoomy_server.symbolic`):

- **NumPy** — `SymbolicHyperbolicSolverNumpy` / `SymbolicChorinSolverNumpy`, in-process
- **JAX** — `SymbolicHyperbolicSolverJax` / `SymbolicChorinSolverJax` (CPU or GPU)
- **AMReX** — `SymbolicHyperbolicSolverAmrex` on the uniform Cartesian grid; built and run natively in the AMReX image
- **OpenFOAM** — `SymbolicHyperbolicSolverFoam`; built and run natively in the OpenFOAM image
- **DMPlex**, **Firedrake** — their own solvers (not ported)
