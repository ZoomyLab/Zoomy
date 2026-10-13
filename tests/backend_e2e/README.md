# Backend end-to-end tests

Every Zoomy GUI session and solver card runs through its backend **container**, the
`zoomy-server` inside it and the HTTP protocol the GUI uses; a subset of the cross-backend
matrix runs **inside** each container. Opt-in (`-m backend_e2e`), like the other heavy tiers.

## Run

```bash
# dev mode (default): this checkout's library/ is copied (tracked files) to $E2E_WORK/snapshots/
# and mounted at /zoomy, its packages first on PYTHONPATH -- the commit is tested before any
# image is published
taskset -c 56-63 python -m pytest tests/backend_e2e -m backend_e2e -rA

E2E_CHECKOUT=~/git/wt-port2 pytest tests/backend_e2e -m backend_e2e     # another checkout
E2E_MODE=published pytest tests/backend_e2e -m backend_e2e              # the images as they are
E2E_BACKENDS=jax,amrex pytest tests/backend_e2e -m backend_e2e -k malpasset
pytest tests/backend_e2e -m backend_e2e -k matrix                       # the matrix part only
```

Python env: the `zoomy` stack (numpy, scipy, sympy, h5py, matplotlib, jax for the in-process
JAX comparison), `node` (the GUI's own composition), `apptainer` or `docker`, `rsync`, `git`.

| variable | meaning (default) |
|---|---|
| `E2E_CHECKOUT` | the Zoomy checkout whose `library/` (and matrix definitions) is tested (this one) |
| `E2E_MODE` | `dev` (snapshot mounted, first on PYTHONPATH) or `published` (the image's own library) |
| `E2E_RUNTIME` | `apptainer` (if installed) or `docker` |
| `E2E_BACKENDS` | `numpy,jax,amrex,openfoam` |
| `E2E_IMAGE_<NAME>` | image of a backend: SIF path, or a docker ref |
| `E2E_IMAGES` | directory of pulled SIFs: `apptainer pull zoomy_jax.sif oras://ghcr.io/zoomylab/zoomy_jax_sif:latest` |
| `E2E_WORK` | scratch: snapshots, container work dirs (job stores, logs, build caches), `report-<checkout>-<mode>.json` (`$TMPDIR/zoomy_backend_e2e`) |
| `E2E_CPUS` | CPU list of the containers (this process's affinity) |
| `E2E_JOB_TIMEOUT`, `E2E_MATRIX_TIMEOUT` | per job / per container matrix run, s (840) |
| `E2E_SAME_TOL`, `E2E_MATRIX_TOL` | host run = container run: GUI cases, matrix (1e-9) |
| `E2E_MATRIX_HOST_STORE` | compare the container runs with this matrix store instead of fresh host runs |
| `E2E_LIVE_COUPLING=1` | also run the live coupling sessions (they write into the host checkout) |

Images (apptainer): `E2E_IMAGE_<NAME>`, `E2E_IMAGES/<container>.sif`, the checkout's
`containers/<c>/<c>.sif`, `~/git/Zoomy/containers/<c>/<c>.sif`; docker: `ghcr.io/zoomylab/<c>:latest`.
A backend without an image is skipped with the paths it looked at. The containers run with
`--contain` (no host home: the host's `~/.local` editable installs would shadow the image's
packages), `PYTHONNOUSERSITE=1`, a scratch HOME / TMPDIR under `$E2E_WORK`.

## What is covered

**GUI sessions and cards** (`test_gui_sessions.py`), per backend, the QR-wired ones first:

- Cases: every packed session of `library/zoomy_gui/projects/*.zip` (exact duplicates once;
  `zoomy-cases.zip` repacks the others) and every catalog solver card in a fresh GUI. A case
  goes to the backend whose adapter tag is its solver card's `requires_tag` (numpy sessions:
  the GUI runs them in the browser, here they are submitted to the numpy backend as well).
- Composition: `compose.mjs` builds the spec as `model-config-widget.ts` does
  (`materializeSessions`; `gatherSpec` for a card) and the case.py with
  `ZoomyCLI.exportCase` of `library/zoomy_cli` (the copy the deployed GUI loads).
  `time_end` is set where a quick look needs it (`sessions.TIME_LIMITS`: Malpasset 2 s), as a
  reader would in the Parameters panel.
- Protocol (`harness.GuiClient` = `HttpAdapter.submitCase`): `POST /api/v1/cases`, poll
  `GET /api/v1/jobs/<id>` every 2 s, `GET /api/v1/jobs/<id>/results/hdf5`.
- Checks: every float finite; the last snapshot at `time_end`; the conserved row's relative
  change within `sessions.EXPECT` (walled / periodic domains, 1e-12); the store opened by the
  GUI's `engine.open_hdf5` and rendered by the selected viewer card (`renderVizCard`); in dev
  mode the case also runs on the host through `engine.py` (the QR suite's `session_runner.py`;
  NumPy / JAX in-process, AMReX / foam with the host Python driving the build and run through
  apptainer in the same image) and the two last snapshots agree to 1e-9 (`max|a - b| / max|b|`,
  the matrix's relmax).

**Matrix in the container** (`test_matrix_in_container.py`): serial C1 (`o1_n20`, `o2_n20`),
C4 (`sub_o2_n25`, `trans_o1_n25`), C8 (`vam`), C26 (`visc`), C28 (`quad_expl_dir0`,
`quad_impl_neu`; JAX, numpy: `orders_quad_expl`), C31 (`swe_ssp2_k0`, `vam_ssp2_k0`), C32
(`noslip_n20`) and C33 (`imex_k0`), each backend the runs its `cases.py` lists it for, by
`matrix.py`'s own `run` (`matrix_driver.py`) inside each image: its Python, and for foam /
AMReX its OpenFOAM / AMReX toolchain (built natively, `ZOOMY_{FOAM,AMREX}_IN_IMAGE=1`;
`shim/apptainer` for older drivers that always call `apptainer exec <image>`). The case
builders import no backend (`gates/common.py` takes JAX if present), so every image sets up
every case. The same runs on the host (the same library; foam / AMReX through apptainer with
the same image) give the reference; every stored array must agree to the matrix's tolerance
for two runs of the case (`tol`: round-off 1e-12, or the implicit-solve tolerance: C8, C31
VAM, C28 implicit 1e-9; C26, C31 SWE, C32 1e-8; C33 1e-10; the image's numpy / JAX may
differ from the host's); the timing and provenance keys are ignored. JAX also runs C1
`o2_n40` on 2 ranks with the image's `mpiexec` (OpenMPI, mpi4jax), checked as matrix C7:
= the serial run in the same image, round-off.

## Run times

Measured 2026-10-13 on main 7ce6f03b (dev mode, apptainer, the GHCR images of that day, 8
cores via `taskset -c 56-63`); "host" = the comparison run on the host.

| backend | GUI cases | matrix subset (container + host) |
|---|---|---|
| numpy | Steffler 30 s (45 s with the host run); the four QR sessions 68 s; Bingham threshold (time_end 20 s) 254 s (492 s); all 11 cases ~12 min | C1 27 + 14 s, C32 41 + 27 s, C33 13 + 5 s; all five 3.5 min |
| jax | Malpasset (time_end 2 s) 68 s (133 s with the host run); the cards 4 s | C1 45 + 32 s, C8 87 + 77 s, C28 45 + 30 s, C32 65 + 50 s, C33 33 + 24 s; 8 min |
| amrex | Malpasset (time_end 2 s) 66 s with the build, 12 s cached (81 s with the host run) | C1 91 + 84 s, C32 67 + 54 s, C33 59 + 48 s; 7 min |
| openfoam | the two cards 28 s (wmake) / 6 s (43 / 35 s with the host run) | C33 22 + 40 s; 3 min |

One pytest command per backend (numpy: two) stays under 15 min.

## Results (2026-10-13, main 7ce6f03b, dev mode)

| backend | pass | fail (cause) |
|---|---|---|
| numpy | steffler-0, bingham-analytics-0, bingham-threshold-0 | bingham-0, zoomy-codeprinter-0, card-solver-{numpy,imex-numpy,split-numpy}: the code needs the notebook namespace (mesh.py uses the model cell's `wave_length`; the run cell uses a step's `nsm`, and `to_folder` drops steps; the card templates use `model` / `split`), which the folder form the server runs does not have; coupling-0: the Static plot viewer cannot draw its 3-D store (`add_collection3d`). The GUI runs all of these in the browser. |
| jax | (backend run of malpasset-0: store, viewer, mass 0, host run identical) | malpasset-0: the GUI's Run does not submit it (below); card-solver-{jax,imex-jax}: the composed run cell is the card's `class` construct snippet (`model = SymbolicHyperbolicSolverJax()`, `display(model.describe())`) |
| amrex | (backend run of malpasset-1: as JAX, mass 2.3e-16) | malpasset-1: as malpasset-0; card-solver-{amrex,split-amrex}: as the JAX cards |
| openfoam | card-solver-foam, card-solver-foam-vof (both run the generated symbolic foam case, SWE: the VOF card runs no VoF) | known gaps, not run: coupling-live-0, coupling-triple-live-0 (SME-VOF live coupling and the server's `/couple` path are not ported) |

The GUI's Run (`runAssembly`, `zoomy-theia-ext/src/browser/model-config-widget.ts`) runs a case
in the browser when `cardCode(solver, ...)` of the picked catalog solver card is non-empty, and
`cardCode` returns a construct snippet for every card with a `class`: the JAX and AMReX cards
have one, so their cases (both Malpasset QR sessions) never reach a connected backend.

Matrix in the container: equal to the host run (relmax 0; JAX C8 2.9e-10, the pressure
tolerance) for numpy C1 C32 C33, jax C1 C8 C28 C32 C33, amrex C1 C32 C33, foam C33. Fail:
C8 and C28 in the numpy / foam / AMReX images (builders import jax); foam C1 and C32 (the
OpenFOAM image pins `numpy<2`, `zoomy_core.mesh.geometry_xp` calls `np.linalg.pinv(..., rtol=)`,
numpy >= 2.0). Published images of 2026-10-13 (built before the port): Malpasset JAX and
AMReX fail (the images' libraries predate the sessions), Steffler passes.

## CI

`.github/workflows/backend-e2e.yml`: one job per backend on `ubuntu-latest`, docker images
from GHCR (`E2E_RUNTIME=docker`), dev mode by default (this commit's `library/` mounted).
Push to main: the QR-wired sessions of numpy, jax, amrex (`E2E_QR_ONLY=1`); manual dispatch:
backends, mode, QR-only and a `-k` selection. The images fit a hosted runner (0.25 / 1.34 /
0.75 / 1.12 GB compressed); the foam / AMReX matrix reference needs apptainer on the host and
is skipped there. Not run yet: the workflow and the docker path of `harness.Container` were
written without docker on the development machine.
