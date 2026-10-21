# Install

Zoomy needs Python 3.11 or later.  This installs the NumPy and JAX backends from the published
packages; no clone is needed.

```{warning}
Not published yet: `zoomy-jax` is not on PyPI, and the PyPI `zoomy-core` (0.2.15) predates the
solver API these docs describe.  Until both are released, use a [container](containers.md) or the
[developer installation](developer/installation.md).
```

With micromamba (conda and mamba work the same):

```bash
micromamba create -n zoomy -c conda-forge python=3.12 jupyterlab
micromamba activate zoomy
pip install zoomy-core zoomy-jax zoomy-plotting
```

With plain pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install zoomy-core zoomy-jax zoomy-plotting jupyterlab
```

JAX on Windows runs on the CPU (PyPI wheels); a GPU needs Linux or WSL2.

## Run

JupyterLab with a tutorial notebook (more on the [tutorials page](tutorials/index.md)):

```bash
curl -LO \
  https://raw.githubusercontent.com/ZoomyLab/Zoomy/main/tutorials/swe/simple_numpy.ipynb
jupyter lab simple_numpy.ipynb
```

The GUI runs in the [browser](browser.md); to solve on your own machine, start the solver server
and connect it (GUI, "Connect backend", `http://localhost:8080`):

```bash
pip install zoomy-server
zoomy-server --adapter jax --port 8080
```

AMReX, OpenFOAM and GPU images: [Containers](containers.md).

## Reproducing the thesis

Every QR code in the thesis opens in a [browser](browser.md): the animations are GIF viewers, the
interactive examples are GUI sessions on the in-browser NumPy solver.  The simulations behind the
chapters used:

| Thesis chapters | Backend | Install |
| --- | --- | --- |
| verification (SWASHES smooth, Ritter, friction), curved channel 2-D reference | JAX | this page, or the `zoomy_jax` [container](containers.md) |
| Malpasset dam break, strong scaling | AMReX (JAX for the comparison) | `zoomy_amrex` container |
| SME↔SME and SME↔VOF coupling, hydropower spin-up | OpenFOAM 13 + preCICE 3 | `zoomy_openfoam` SIF (`export ZOOMY_OPENFOAM_SIF=/path/to/zoomy_openfoam.sif`) |
| VAM bump, Bingham roll waves, width-averaged curved channel, GUI examples | NumPy | this page, or the GUI |
