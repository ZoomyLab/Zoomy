# Developer installation

Clone the repository with its submodules, install the libraries editable, build the images locally.
To use Zoomy without changing it, see [Install](../installation.md).

## Clone with submodules

Zoomy pins exact commits per submodule.

```bash
git clone --recurse-submodules https://github.com/ZoomyLab/Zoomy.git
cd Zoomy
```

Already cloned without submodules:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

Only what you need:

```bash
git submodule update --init library/zoomy_core library/zoomy_jax meshes
```

Move everything to the latest upstream `main`:

```bash
git submodule update --remote --merge --recursive
```

`git pull` does not advance submodules; use `git pull --recurse-submodules`.  The sub-repositories
are listed under [ZoomyLab](https://github.com/ZoomyLab) and are cloned anonymously over HTTPS.
To push over SSH from such a checkout, add once:

```bash
git config --global url."git@github.com:".pushInsteadOf https://github.com/
```

## Environment

Conda, mamba or micromamba, from the clone:

```bash
# core (NumPy + GMSH), plus the plotting and pre/post-processing packages
conda env create -f install/Zoomy.yml
conda activate zoomy
pip install -e library/zoomy_core -e library/zoomy_plotting -e library/zoomy_prepost

# add JAX
conda env update -f install/zoomy_jax.yml
pip install -e library/zoomy_jax              # reverse-mode AD: -e "library/zoomy_jax[ad]"
```

`zoomy_plotting` and `zoomy_prepost` live in the superrepo, not in a submodule, and need no packages
beyond `install/Zoomy.yml`.  Other environment files in `install/`: `zoomy_fenicsx.yml`,
`zoomy_game.yml`, `minimal.yml`, `jupyter-lite.yml`, `pyodide.yml`; they are the authoritative
dependency list.

Plain pip (NumPy only):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e library/zoomy_core -e library/zoomy_plotting
```

The scripts of the book (`docs/book/examples/`) run against the libraries of the checkout after

```bash
source docs/book/examples/env.sh      # PYTHONPATH of the checkout, CPU JAX, double precision
```

Environment variables:

```bash
export ZOOMY_DIR=/path/to/Zoomy
export JAX_ENABLE_X64=True
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=<arch-used-for-compiling-petsc>
```

## Devcontainer (VS Code)

Open the repository with the *Dev Containers* extension; it offers **Zoomy + JAX** and
**Zoomy + Firedrake + AMR** (plus a standalone variant of the latter).  Requires Docker (Linux
containers on Windows).

## Build the images locally

From the repository root, so the `%files` sources resolve:

```bash
apptainer build --fakeroot containers/<name>/<name>.sif containers/<name>/<name>.def
```

Recipes live in `containers/<name>/` (`Dockerfile` and `<name>.def`, kept in sync; see
`containers/README.md`).  CI publishes them on every container change
(`.github/workflows/build-containers.yml`).  `zoomy_core`, `zoomy_amrex_dummy` and
`zoomy_fenicsx_dummy` are CI placeholders without a backend toolchain;
`containers/zoomy_mesh`, `containers/basilisk` and `containers/zoomy_telemac` are experimental and
not published.

## Backend-specific setup

- **AMReX**: the container is the supported path.  To build against a local AMReX, see
  [amrex-codes.github.io/amrex](https://amrex-codes.github.io/amrex/docs_html/Introduction.html).
- **OpenFOAM**: OpenFOAM 13 (openfoam.org) and preCICE 3; `install/setup_precice.sh`.  The
  `zoomy_openfoam` container has both.
- **preCICE**: `install/setup_precice.sh`, example configs in `tools/precice_configs/`.
- **PETSc** (for `zoomy_dmplex`): `install/install-petsc.sh`, `install/activate-petsc.sh`.
