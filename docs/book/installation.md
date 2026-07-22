# Installation

Pick the lightest path that covers what you need:

| You want to… | Use |
| --- | --- |
| try the symbolic layer + NumPy solver | `pip install zoomy_core` |
| run in a browser, install nothing | the [GUI](user-guide-gui.md) or the <a href="jupyter-lite/_output/lab/index.html?path=pyodide.ipynb">Pyodide notebook</a> |
| run one backend (JAX, AMReX, OpenFOAM…) | a [prebuilt container](#prebuilt-containers) |
| develop Zoomy itself | [clone with submodules](#git-clone-with-submodules) + conda |

## Pip — core only

```bash
pip install zoomy_core
```

Symbolic layer, model families, and the reference NumPy solver. Other backends
are separate packages (`pip install zoomy_jax`), but the compiled ones (AMReX,
OpenFOAM, PETSc/DMPlex, Firedrake) are far easier via containers.

## Prebuilt containers

CI publishes every backend to GHCR on each container change — public, anonymous
pull. **Apptainer** users pull the ORAS-pushed SIF; **Docker** users pull the
image.

```bash
# Apptainer (recommended on HPC — no daemon, no root)
apptainer pull zoomy_numpy.sif oras://ghcr.io/zoomylab/zoomy_numpy_sif:latest
apptainer run zoomy_numpy.sif 8080          # solver API on :8080

# Docker
docker pull ghcr.io/zoomylab/zoomy_numpy:latest
docker run --rm -p 8080:8080 ghcr.io/zoomylab/zoomy_numpy:latest
```

Substitute any name from the table below.

### What ships

| Container | Backend tag | Docker | Apptainer SIF | Notes |
| --- | --- | :---: | :---: | --- |
| `zoomy_numpy` | `numpy` | ✅ | ✅ | reference solver; smallest image |
| `zoomy_jax` | `jax` | ✅ | ✅ | CPU and GPU (`--nv`) |
| `zoomy_amrex` | `amrex` | ✅ | ✅ | block-structured AMR; a GPU variant also builds |
| `zoomy_dmplex` | `dmplex` | ✅ | ✅ | builds PETSc from source |
| `zoomy_firedrake` | `firedrake` | ✅ | ✅ | |
| `zoomy_openfoam` | `foam` | — | ✅ | OpenFOAM 13 + preCICE |
| `zoomy_postprocess` | `postprocess` | ✅ | ✅ | plotting / HDF5 tooling |
| `zoomy_jax_dev`, `zoomy_firedrake_dev` | — | ✅ | — | heavy base layers the two above build on |

```{note}
`zoomy_core`, `zoomy_amrex_dummy` and `zoomy_fenicsx_dummy` are also published
but are **CI placeholders**: they install `zoomy_core` from PyPI and carry no
backend toolchain. Do not use them to run simulations. `containers/zoomy_mesh`,
`containers/basilisk` and `containers/zoomy_telemac` are experimental, are not
built by CI, and have no published image.
```

### Three modes per image

The same image serves three roles, chosen by the first argument:

```bash
apptainer run IMG [port]     # 1. solver API on :8080  (the GUI's "Connect" target)
apptainer run IMG jupyter    # 2. JupyterLab on :8888 with the backend's kernel
apptainer run IMG shell      # 3. interactive dev shell
```

Apptainer shares the host network namespace, so `:8080` / `:8888` are reachable
at `localhost` with no port publishing. Add `--nv` for GPU.

Use an image as a notebook kernel:

```bash
apptainer run --bind $PWD:/workspace containers/zoomy_jax/zoomy_jax.sif jupyter
# open the printed http://127.0.0.1:8888/?token=... , or in VS Code:
# "Jupyter: Connect to Existing Server" -> that URL
```

`import zoomy_core, zoomy_jax` just work; `$ZOOMY_ROOT=/workspace` is the served
root, so bind your repo there.

### Building locally instead

```bash
# from the repository root, so the %files sources resolve
apptainer build --fakeroot containers/<name>/<name>.sif containers/<name>/<name>.def
```

Recipes live in `containers/<name>/`; the Docker and Apptainer recipes for a
given backend are kept in sync.

## Git clone with submodules

Zoomy pins exact commits per submodule for reproducible checkouts.

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

`git pull` does not advance submodules — use `git pull --recurse-submodules`.
The sub-repositories are listed under
[ZoomyLab](https://github.com/ZoomyLab) on GitHub.

## Conda / Mamba / Micromamba

After cloning:

```bash
# core (NumPy + GMSH)
conda env create -f install/Zoomy.yml
conda activate zoomy
pip install -e library/zoomy_core

# add JAX
conda env update -f install/zoomy_jax.yml
pip install -e library/zoomy_jax
```

Other environment files in `install/`: `zoomy_fenicsx.yml`, `zoomy_game.yml`,
`minimal.yml`, `jupyter-lite.yml`, `pyodide.yml`. These files are the
authoritative dependency list.

## Devcontainer (VS Code)

Open the repository with the *Dev Containers* extension; it offers **Zoomy +
JAX** and **Zoomy + Firedrake**. Requires Docker (Linux containers on Windows).

## Backend-specific setup

- **AMReX** — the container is the supported path. To build against a local
  AMReX instead, see
  [amrex-codes.github.io/amrex](https://amrex-codes.github.io/amrex/docs_html/Introduction.html).
- **OpenFOAM** — OpenFOAM 12+ and the preCICE adapter; `install/setup_precice.sh`.
  The `zoomy_openfoam` container already has both.
- **preCICE** — `install/setup_precice.sh`, example configs in `tools/precice_configs/`.
- **PETSc** (for `zoomy_dmplex`) — `install/install-petsc.sh`,
  `install/activate-petsc.sh`.

## Environment variables

```bash
export ZOOMY_DIR=/path/to/Zoomy
export JAX_ENABLE_X64=True
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=<arch-used-for-compiling-petsc>
```
