# Installation

This page collects every supported installation path. For a one-line install
of just the core symbolic + NumPy solver, use `pip install zoomy_core`.

ZoomyLab is split into a base repository plus per-backend sub-repositories:

- **Base repository** (`Zoomy`): symbolic modeling layer, pre/post-processing,
  NumPy solver (`zoomy_core`).
- **Solver backends**: see [Backends](backends/index.md) for the list and
  links.
- **Utility repositories**: `meshes` (GMSH `.geo` / `.msh` catalogue), `data`
  (large-scale test cases).

## Pip (core only)

The fastest path. Symbolic layer + NumPy solver only:

```bash
pip install zoomy_core
```

For other backends see [Backends](backends/index.md); each one is its own pip
package, e.g. `pip install zoomy_jax`.

## Git clone with submodules

Zoomy pins **exact commits** for each submodule (reproducible checkouts).
`.gitmodules` also records `branch = main`, so you can optionally move
submodules to the latest `main` of their own repos.

### Full tree at the pins recorded by Zoomy (typical)

```bash
git clone --recurse-submodules https://github.com/ZoomyLab/Zoomy.git
cd Zoomy
```

If you already cloned without submodules:

```bash
git clone https://github.com/ZoomyLab/Zoomy.git
cd Zoomy
git submodule sync --recursive
git submodule update --init --recursive
```

### Full tree with every submodule on latest `main`

```bash
git clone https://github.com/ZoomyLab/Zoomy.git
cd Zoomy
git submodule sync --recursive
git submodule update --init --recursive
git submodule update --remote --merge --recursive
```

### Only selected submodules

```bash
git clone https://github.com/ZoomyLab/Zoomy.git
cd Zoomy
git submodule update --init meshes
git submodule update --init library/zoomy_core
git submodule update --init library/zoomy_jax
```

Paths match `.gitmodules` (e.g. `library/zoomy_firedrake`,
`library/zoomy_dmplex`, `data`, …).

### Bump one submodule to latest `main`

```bash
git submodule update --init library/zoomy_jax
git submodule update --remote --merge library/zoomy_jax
```

To persist the new commit in your Zoomy branch:

```bash
git add library/zoomy_jax
git commit -m "Bump zoomy_jax submodule to latest main"
```

`git pull` does not auto-advance submodules; use `git pull --recurse-submodules`
or follow up with `git submodule update --remote --merge --recursive`.

The individual sub-repositories are listed under
[ZoomyLab](https://github.com/ZoomyLab) on GitHub.

## Conda / Mamba / Micromamba

After cloning the repository:

**Core (NumPy + GMSH)**:

```bash
conda env create -f install/Zoomy.yml
conda activate zoomy
conda env update -f install/meshes.yml
pip install library/zoomy_core
```

**Core + JAX**:

```bash
conda env create -f install/Zoomy.yml
conda activate zoomy
conda env update -f install/zoomy_jax.yml
conda env update -f install/meshes.yml
pip install library/zoomy_core
pip install library/zoomy_jax
```

Other env files in `install/`: `zoomy_fenicsx.yml`, `zoomy_game.yml`,
`minimal.yml`, `jupyter-lite.yml`, `pyodide.yml`.

## Devcontainer (VS Code)

Open the repository in VS Code with the *Dev Containers* extension installed.
A popup will offer:

- Zoomy + JAX
- Zoomy + Firedrake

**Requires**: Docker, *Dev Containers* extension. On Windows, use Linux
containers.

## Docker

**Standalone** (locked, no library edits):

```bash
docker pull ghcr.io/zoomylab/zoomy_jax_standalone:latest
docker pull ghcr.io/zoomylab/zoomy_firedrake_standalone:latest
```

**Development** (allows editing the Zoomy libraries; `pip install -e` the
locals or use the devcontainers):

```bash
docker pull ghcr.io/zoomylab/zoomy_jax:latest
docker pull ghcr.io/zoomylab/zoomy_firedrake:latest
```

On Windows, use Linux containers.

## Apptainer / Singularity

Apptainer images can be built from the `ghcr.io/zoomylab/...` Docker images:

```bash
apptainer build zoomy_jax.sif docker://ghcr.io/zoomylab/zoomy_jax:latest
apptainer run --nv zoomy_jax.sif
```

This is the recommended path on HPC clusters where Docker is unavailable.

## Backend-specific setup

- **AMReX**: completely independent of the Conda/Mamba environment; follow
  [amrex-codes.github.io/amrex](https://amrex-codes.github.io/amrex/docs_html/Introduction.html).
- **OpenFOAM**: requires OpenFOAM 12+ and the preCICE adapter; see
  `install/setup_precice.sh`.
- **preCICE**: see `install/setup_precice.sh` and the example configurations
  in `tools/precice_configs/`.
- **PETSc** (needed by `zoomy_dmplex`): `install/install-petsc.sh` and
  `install/activate-petsc.sh`.

See the per-backend pages under [Backends](backends/index.md) for details.

## Environment variables

```bash
export ZOOMY_DIR=/path/to/Zoomy
export JAX_ENABLE_X64=True
export PETSC_DIR=/path/to/petsc/installation
export PETSC_ARCH=architecture-used-for-compiling-petsc
```

## Manual dependency list

See the `install/*.yml` files for the authoritative dependency list.
