# Zoomy backend containers

Each backend ships **one triple-purpose image** — the same `.sif` serves three
roles, selected by the first argument to `apptainer run`:

| mode | command | what it does |
|------|---------|--------------|
| **1. Solver API** | `apptainer run IMG [port]` | FastAPI solver server on `:8080` (the GUI's "Connect" target). Default when no mode is given. |
| **2. Jupyter** | `apptainer run IMG jupyter [port]` | JupyterLab on `:8888` with the backend's Python env (zoomy_core + zoomy_&lt;backend&gt;) — a usable kernel. |
| **3. Dev shell** | `apptainer run IMG shell` | interactive `bash` in the backend env. |

Apptainer shares the host network namespace, so `:8080`/`:8888` are reachable at
`localhost` with no port publishing. GPU backends: add `--nv`.

The dispatch is baked into every image's `%runscript` (mode keyword → `zoomy-server`
/ `zoomy-jupyter` / `bash`); the adapter is baked per image (`$ZOOMY_ADAPTER`), so a
bare port still starts the server (back-compat).

## Using an image as a Jupyter kernel (VS Code / browser)

```bash
apptainer run --bind $PWD:/workspace containers/zoomy_jax/zoomy_jax.sif jupyter
# -> open the printed http://127.0.0.1:8888/?token=... , or in VS Code:
#    "Jupyter: Connect to Existing Server" -> that URL
```

The kernel runs the container's interpreter, so `import zoomy_core, zoomy_jax` (etc.)
just work. `$ZOOMY_ROOT=/workspace` is the served root; bind your repo there.

## Using an image as a dev container

- Lightweight: `apptainer run IMG shell` gives an interactive env with the full
  backend stack.
- VS Code Dev Containers (Docker): see `.devcontainer/` (jax, firedrake) — same
  `*_dev` bases these images build on, `/workspace` bind + editable installs.

## The server API

`zoomy-server --adapter <backend>` exposes:
`GET /api/v1/health`, `POST /api/v1/jobs {case_dir}`, `GET /api/v1/jobs/{id}`,
`GET /api/v1/jobs/{id}/results/hdf5` (gated on job completion),
`GET /api/v1/jobs/{id}/results` (JSON). A **case folder** is
`{model.py, mesh.py, settings.json[, numerics.py]}`; numpy/jax read `mesh.h5`,
dmplex/firedrake read `mesh.msh` (both emitted by the shared case via
`zoomy_prepost.mesh_to_gmsh`). Non-HDF5 backends convert their output with
`zoomy_prepost.vtk_to_hdf5` so the normal postprocessing reads it.

## Backend status (shared SWE case over the API)

| backend | image (3 modes) | shared case end-to-end |
|---------|:---:|---|
| numpy | ✅ | ✅ 1D + 2D |
| jax | ✅ | ✅ (CPU/GPU) |
| amrex | ✅ | ✅ 1D (REQ-89 run_case; adapter derives the structured spec from mesh.h5) |
| dmplex | ✅ | ✅ 2D (REQ-96 model-IC via symbolic IC.RP; .vtu.series -> simulation.h5) |
| firedrake | ✅ | ✅ 2D (REQ-95 fixed: source-integral domain pin + vtk-VTU fallback) |
| foam | (openfoam sif; host adapter) | ✅ 1D (REQ-93 run_case + REQ-110 case-dir wiring) |

## Pulling prebuilt images (no local build)

The Containers pipeline publishes every backend to ghcr (public, anonymous
pull). Apptainer users pull the ORAS-pushed SIF; Docker users pull the image:

```bash
# backend server, e.g. numpy (same pattern: zoomy_postprocess, zoomy_jax,
# zoomy_amrex, zoomy_dmplex, zoomy_firedrake)
apptainer pull zoomy_numpy.sif oras://ghcr.io/zoomylab/zoomy_numpy_sif:latest
apptainer run zoomy_numpy.sif 8080        # -> GUI "Connect" target

docker pull ghcr.io/zoomylab/zoomy_numpy:latest
docker run --rm -p 8080:8080 ghcr.io/zoomylab/zoomy_numpy:latest
```

All images share the `zoomy-entry` dispatch (`containers/common/zoomy-entry`):
bare `run` → server on `$ZOOMY_PORT`, `<port>` → server on that port,
`jupyter [port]`, `shell`, or any other command is exec'd as-is — identical
between the Docker image and the pipeline-converted SIF.

Build any image locally instead:
`apptainer build --fakeroot containers/<name>/<name>.sif containers/<name>/<name>.def`
(from the repo root, so the `%files` sources resolve).
