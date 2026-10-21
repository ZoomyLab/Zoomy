# Containers

Every backend ships as a Docker image and an Apptainer SIF on GHCR (public, anonymous pull).
Use them for AMReX, OpenFOAM, GPUs, or when the [Install](installation.md) route is not available.

```bash
# Apptainer (recommended on HPC: no daemon, no root)
apptainer pull zoomy_numpy.sif oras://ghcr.io/zoomylab/zoomy_numpy_sif:latest

# Docker
docker pull ghcr.io/zoomylab/zoomy_numpy:latest
```

| Container | Backend | Docker | Apptainer SIF | Notes |
| --- | --- | :---: | :---: | --- |
| `zoomy_numpy` | NumPy | ✅ | ✅ | reference solver; smallest image |
| `zoomy_jax` | JAX | ✅ | ✅ | CPU and GPU in one image |
| `zoomy_amrex` | AMReX | ✅ | ✅ | block-structured AMR; CPU and GPU in one image |
| `zoomy_openfoam` | OpenFOAM | ✅ | ✅ | OpenFOAM 13 + preCICE 3 |
| `zoomy_dmplex` | DMPlex | ✅ | ✅ | builds PETSc from source |
| `zoomy_firedrake` | Firedrake | ✅ | ✅ | |
| `zoomy_dune` | DUNE | ✅ | ✅ | DUNE-FEM 2.12 + ALUGrid + fem-dg; UFL weak forms, unstructured AMR |
| `zoomy_postprocess` | postprocessing | ✅ | ✅ | plotting / HDF5 tooling |

## Three modes

The first argument chooses what the image does:

```bash
apptainer run zoomy_jax.sif 8080      # solver API on :8080 (the GUI's "Connect backend" target)
apptainer run zoomy_jax.sif jupyter   # JupyterLab on :8888 with the backend's kernel
apptainer run zoomy_jax.sif shell     # interactive shell
```

Apptainer shares the host network, so `:8080` and `:8888` are `localhost`.  With Docker publish
the ports (`-p 8080:8080`, `-p 8888:8888`) and bind your files to `/workspace`:

```bash
docker run --rm -p 8080:8080 ghcr.io/zoomylab/zoomy_jax:latest
docker run --rm -p 8888:8888 -v $PWD:/workspace ghcr.io/zoomylab/zoomy_jax:latest jupyter
```

## GUI

Start the solver API, then open the [GUI](browser.md) and connect it (GUI, "Connect backend",
`http://localhost:8080`; see [Connecting your own backend](guide/gui.md#connecting-your-own-backend)).

## Jupyter

```bash
apptainer run --bind $PWD:/workspace zoomy_jax.sif jupyter
# open the printed http://127.0.0.1:8888/lab, or in VS Code: "Jupyter: Connect to Existing Server"
```

No token by default (your own machine); set `ZOOMY_JUPYTER_TOKEN=<secret>` to require one.
`import zoomy_core, zoomy_jax` just work; `$ZOOMY_ROOT=/workspace` is the served root.

## GPUs

`zoomy_jax` and `zoomy_amrex` use the GPU when the host's NVIDIA driver is brought in:

```bash
apptainer exec --nv zoomy_jax.sif python run.py                 # JAX picks the GPU
apptainer exec zoomy_jax.sif python run.py                      # the same image on the CPU
docker run --rm --gpus all ghcr.io/zoomylab/zoomy_jax:latest    # Docker: NVIDIA container toolkit
```

`JAX_PLATFORMS=cpu` keeps JAX on the CPU; ranks on GPUs: `mpiexec -n 2 python -m zoomy_jax.gpurank run.py`.
AMReX uses CUDA when the run asks for it (`gpu=True`, see [Backends](guide/backends.md)).  The CUDA
libraries make the SIFs larger: 4.3 GB (`zoomy_jax`) and 1.6 GB (`zoomy_amrex`).
