# Installation

Pick the lightest path that covers what you need:

| You want to… | Use |
| --- | --- |
| try the symbolic layer + NumPy solver | `pip install zoomy_core` |
| run in a browser, install nothing | the [GUI](#gui-setup) or the <a href="jupyter-lite/_output/lab/index.html?path=pyodide.ipynb">Pyodide notebook</a> |
| run one backend (JAX, AMReX, OpenFOAM…) | a [prebuilt container](#prebuilt-containers) |
| develop Zoomy itself | [clone with submodules](#git-clone-with-submodules) + [conda](#conda--mamba--micromamba) |
| follow a QR code in the thesis | a browser — see [Reproducing the thesis](#reproducing-the-thesis) |

## Pip — core only

```bash
pip install zoomy_core
```

Symbolic layer, model families, and the reference NumPy solver. `zoomy_core`
is the only Zoomy package on PyPI. The JAX backend is installed from the
repository (`pip install -e library/zoomy_jax`, see
[Conda](#conda--mamba--micromamba)) or used through the `zoomy_jax` container;
the compiled backends (AMReX, OpenFOAM, PETSc/DMPlex, Firedrake) are far easier
via containers.

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
| `zoomy_openfoam` | `foam` | ✅ | ✅ | OpenFOAM 13 + preCICE 3 |
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
# open the printed http://127.0.0.1:8888/lab , or in VS Code:
# "Jupyter: Connect to Existing Server" -> that URL
```

The server starts without a token (research use on your own machine); set
`ZOOMY_JUPYTER_TOKEN=<secret>` in the environment to require one.

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
# core (NumPy + GMSH), plus the plotting and pre/post-processing packages
conda env create -f install/Zoomy.yml
conda activate zoomy
pip install -e library/zoomy_core -e library/zoomy_plotting -e library/zoomy_prepost

# add JAX (not on PyPI — installed from the clone)
conda env update -f install/zoomy_jax.yml
pip install -e library/zoomy_jax
```

`zoomy_plotting` (figures and animations from HDF5/VTK output) and
`zoomy_prepost` (case pre/post-processing) live in the superrepo, not in a
submodule, and need no packages beyond `install/Zoomy.yml`. Every thesis case
imports them, and every container ships them.

Other environment files in `install/`: `zoomy_fenicsx.yml`, `zoomy_game.yml`,
`minimal.yml`, `jupyter-lite.yml`, `pyodide.yml`. These files are the
authoritative dependency list.

## Devcontainer (VS Code)

Open the repository with the *Dev Containers* extension; it offers **Zoomy +
JAX** and **Zoomy + Firedrake + AMR** (plus a standalone variant of the
latter). Requires Docker (Linux containers on Windows).

## Backend-specific setup

- **AMReX** — the container is the supported path. To build against a local
  AMReX instead, see
  [amrex-codes.github.io/amrex](https://amrex-codes.github.io/amrex/docs_html/Introduction.html).
- **OpenFOAM** — OpenFOAM 13 (openfoam.org) and preCICE 3;
  `install/setup_precice.sh`. The `zoomy_openfoam` container already has both.
- **preCICE** — `install/setup_precice.sh`, example configs in `tools/precice_configs/`.
- **PETSc** (for `zoomy_dmplex`) — `install/install-petsc.sh`,
  `install/activate-petsc.sh`.

## GUI setup

There is nothing to set up: the <a href="gui/index.html">GUI</a> is a static web
application and runs entirely in the browser, including the NumPy solver on an
in-browser Pyodide kernel. Every other backend (JAX, AMReX, OpenFOAM, DMPlex)
is a solver container from the [table above](#what-ships) that you run on your
own machine and connect to the GUI — see
[Connecting your own backend](user-guide-gui.md#connecting-your-own-backend):

```bash
apptainer run zoomy_jax.sif 8080      # then GUI -> "Connect backend" -> http://localhost:8080
```

## Reproducing the thesis

Every QR code in the thesis opens in a browser and needs no installation:
the animations are GIF viewers, the interactive examples are GUI sessions that
run on the in-browser NumPy solver (bingham, derivation, system model, code
printer), and the code in the intro margin is this page.

The simulations behind the chapters used the following backends; the
one-liners above install each of them.

| Thesis chapters | Backend | Install |
| --- | --- | --- |
| verification (SWASHES smooth, Ritter, friction), curved channel 2-D reference | JAX | [conda](#conda--mamba--micromamba) or `zoomy_jax` container |
| Malpasset dam break, strong scaling | AMReX (JAX for the comparison) | `zoomy_amrex` container |
| SME↔SME and SME↔VOF coupling, hydropower spin-up | OpenFOAM 13 + preCICE 3 | `zoomy_openfoam` SIF (`export ZOOMY_OPENFOAM_SIF=/path/to/zoomy_openfoam.sif`) |
| VAM bump, Bingham roll waves, width-averaged curved channel, GUI examples | NumPy | `pip install zoomy_core` or the GUI |

## Environment variables

```bash
export ZOOMY_DIR=/path/to/Zoomy
export JAX_ENABLE_X64=True
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=<arch-used-for-compiling-petsc>
```
