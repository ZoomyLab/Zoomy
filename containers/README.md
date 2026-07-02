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
| amrex | ✅ | needs `zoomy_amrex.run_case` (REQ-89) |
| dmplex | ✅ | codegen+build+mesh OK; needs IC-from-model in C++ (REQ-96) |
| firedrake | ✅ | adapter+mesh+`.pvd`→h5 OK; needs zero-source weak-form guard (REQ-95) |
| foam | (REQ-93) | needs `zoomy_foam.run_case` (REQ-93) |

Build any image: `apptainer build --fakeroot containers/<name>/<name>.sif containers/<name>/<name>.def`
(from the repo root, so the `%files` sources resolve).
