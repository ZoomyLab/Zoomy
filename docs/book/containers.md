# Containers

## Get an image

```bash
# Apptainer
apptainer pull zoomy_jax.sif oras://ghcr.io/zoomylab/zoomy_jax_sif:latest

# Docker
docker pull ghcr.io/zoomylab/zoomy_jax:latest
```

Every image below exists for both.  Replace `zoomy_jax` by the image you need.  Which backend for
what: [Backends](guide/backends.md).

| Image | Backend |
| --- | --- |
| `zoomy_numpy` | NumPy |
| `zoomy_jax` | JAX, CPU and GPU |
| `zoomy_amrex` | AMReX, CPU and GPU |
| `zoomy_openfoam` | OpenFOAM with preCICE |
| `zoomy_dmplex` | DMPlex |
| `zoomy_firedrake` | Firedrake |
| `zoomy_postprocess` | plotting and HDF5 |

## Run

The first argument chooses the mode.

```bash
apptainer run zoomy_jax.sif 8080      # solver for the GUI on localhost:8080
apptainer run zoomy_jax.sif jupyter   # JupyterLab on localhost:8888
apptainer run zoomy_jax.sif shell     # shell
```

Docker needs the ports and your files:

```bash
docker run --rm -p 8080:8080 ghcr.io/zoomylab/zoomy_jax:latest
docker run --rm -p 8888:8888 -v $PWD:/workspace ghcr.io/zoomylab/zoomy_jax:latest jupyter
```

## GUI

Start the solver on port 8080, open the [GUI](browser.md) and connect it (GUI, "Connect backend",
`http://localhost:8080`).  Details: [Connecting your own backend](guide/gui.md#connecting-your-own-backend).

## Jupyter

```bash
apptainer run --bind $PWD:/workspace zoomy_jax.sif jupyter
```

Open the printed address.  The kernel has the backend installed, your files are in `/workspace`.

## CPU and GPU

`zoomy_jax` and `zoomy_amrex` run on the CPU, and on the GPU when the host driver is brought in.

```bash
apptainer exec zoomy_jax.sif python run.py          # CPU
apptainer exec --nv zoomy_jax.sif python run.py     # GPU
docker run --rm --gpus all ghcr.io/zoomylab/zoomy_jax:latest   # GPU with Docker
```
