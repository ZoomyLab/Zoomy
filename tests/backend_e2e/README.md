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
  `zoomy-cases.zip` repacks the others) and every catalog solver card in a fresh GUI (with the
  model card it names: the Chorin split cards with VAM, `sessions.CARD_MODELS`). A case
  goes to the backend whose adapter tag is its solver card's `requires_tag` (numpy sessions:
  the GUI runs them in the browser, here they are submitted to the numpy backend as well).
- Composition: `compose.mjs` builds the spec as `model-config-widget.ts` does
  (`materializeSessions`; `gatherSpec` for a card) and the case.py with
  `ZoomyCLI.exportCase` of `library/zoomy_cli` (the copy the deployed GUI loads); a card's
  code and the route of Run come from the card rules the GUI loads (`zoomy_cli/src/cards.mjs`).
  `time_end` is set where a quick look needs it (`sessions.TIME_LIMITS`: Malpasset 2 s), as a
  reader would in the Parameters panel.
- Protocol (`harness.GuiClient` = `HttpAdapter.submitCase`): `POST /api/v1/cases`, poll
  `GET /api/v1/jobs/<id>` every 2 s, `GET /api/v1/jobs/<id>/results/hdf5`.
- Checks: every float finite; at least 2 snapshots and the last at `time_end` (the QR suite's
  `session_runner.check_final_store`; where the knob has other units or none, `sessions.END_TIME`:
  the Bingham roll wave's t', Steffler's own 20 s; a replayed store: the snapshots only; the
  Bingham analytics case writes a one-snapshot summary store, `sessions.SUMMARY_STORE`); the conserved row's relative
  change within `sessions.EXPECT` (walled / periodic domains, 1e-12); the store opened by the
  GUI's `engine.open_hdf5` and rendered by the selected viewer card (`renderVizCard`); in dev
  mode the case also runs on the host through `engine.py` (the QR suite's `session_runner.py`;
  NumPy / JAX in-process, AMReX / foam with the host Python driving the build and run through
  apptainer in the same image) and the two last snapshots (the store the GUI keeps of a
  browser run: the one the run opened, else its `simulation.h5`) agree to 1e-9
  (`max|a - b| / max|b|`, the matrix's relmax); the GUI's Run sends numpy cases to the browser
  and every other tag to its backend.

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
| numpy | Steffler 30 s (45 s with the host run); the four QR sessions 68 s; Bingham threshold (time_end 20 s) 254 s (492 s); all 11 cases ~12 min | C1 24 + 16 s, C4 24 + 14, C8 32 + 19, C26 28 + 24, C28 56 + 45, C31 45 + 31, C32 40 + 27, C33 12 + 8; 7 min |
| jax | Malpasset (time_end 2 s) 68 s (133 s with the host run); the cards 4 s | C1 45 + 31 s, C4 46 + 32, C8 88 + 71, C26 66 + 54, C28 45 + 27, C31 138 + 129, C32 66 + 51, C33 34 + 25; C1 mpi2 + serial in the image 70 s; 15 min |
| amrex | Malpasset (time_end 2 s) 66 s with the build, 12 s cached (81 s with the host run) | C1 114 + 114 s, C4 56 + 50, C8 134 + 117, C26 64 + 54, C28 82 + 85, C31 134 + 128, C32 63 + 50, C33 72 + 58; 20 min |
| openfoam | the card 22 s (44 s with the host run) | C1 59 + 63 s, C4 60 + 63, C8 77 + 64, C26 67 + 57, C28 41 + 46, C31 96 + 107, C32 74 + 67, C33 38 + 36; 18 min |

Matrix columns: 2026-10-13, branch dev/containers-matrix (dev mode, apptainer; numpy / AMReX
the GHCR images 0.1.1.163 / 0.1.163, JAX / OpenFOAM local builds of this branch's recipes;
4-8 cores each, several backends at once). A pytest command per backend and half of the
subset (`-k "matrix and (C1 or C4 or C8 or C26)"`, the rest) stays under 15 min.
For the GUI cases, one pytest command per backend (numpy: two) stays under 15 min.

## Results (2026-10-14, dev/qr-backend-routing, dev mode)

| backend | pass | not run |
|---|---|---|
| numpy | steffler-0, bingham-0, coupling-0, zoomy-codeprinter-0, bingham-analytics-0, bingham-threshold-0, card-solver-{numpy,imex-numpy,split-numpy} | |
| jax | malpasset-0 (mass 0, host run identical), card-solver-{jax,imex-jax} | |
| amrex | malpasset-1 (mass 2.3e-16), card-solver-{amrex,split-amrex} | |
| openfoam | card-solver-foam | coupling-live-0, coupling-triple-live-0: the SME-VOF live coupling and the server's `/couple` path are not ported |

Fixed against main 7ce6f03b, where the numpy sessions / cards failed on the server, the
Malpasset sessions were not submitted and the JAX / AMReX cards ran a `class` construct:

- Run routes by the solver card's backend (`zoomy_cli/src/cards.mjs` `solverRoute`): numpy in
  the browser, every other tag to its backend; a backend card without a template contributes
  no code, so its case runs the generated runner (`ZoomyCLI._runCode`) on that backend.
- The server runs a composed case folder as the GUI runs the case (`zoomy_server.case_run`):
  model.py (with the steps), mesh.py, settings.json, run.py in one namespace, with the GUI
  runtime's `display`, `typed_kwargs` and `open_hdf5` (the run's store: the job's
  `simulation.h5`); the Bingham roll-wave and code-printer runs leave `simulation.h5`.
- The Static plot viewer draws a 3-D store (coupling-0's foam store); the incompressibleVOF
  solver card is gone (the symbolic solvers have no VoF; it ran the symbolic foam SWE case).
- `SymbolicHyperbolicSolverJax.run` returns once the march and its snapshot writes are done:
  with asynchronous dispatch (seen with jax 0.4.38 on the CPU, the QR suite's CPython run)
  Malpasset's run cell copied `simulation.h5` before the first snapshot was written.

Matrix in the container (2026-10-13, branch dev/containers-matrix, dev mode; images as in Run
times): all 33 pass, each against the host run of the same library. Bit-identical (relmax 0)
but for the iterative solves: JAX C8 2.9e-10 (pressure, tolerance 1e-9), C26 5.4e-16, C31
`vam_ssp2_k0` 4.3e-15; foam C8 1.9e-14, C31 3.5e-15; AMReX C8 7.9e-15, C31 4.7e-15; numpy
bit-identical throughout (image numpy 2.4.6, host 2.3.2). JAX C1 `o2_n40` on 2 ranks in the
image = its serial run to 1.22e-13 (the host matrix's C7: 1.22e-13). Before (main 7ce6f03b,
the GHCR images of 2026-10-13): C8 / C28 could not be set up in the numpy / foam / AMReX
images (their builders imported JAX), foam C1 / C32 failed (the OpenFOAM image pinned
`numpy<2`; `zoomy_core.mesh.geometry_xp` calls `np.linalg.pinv(..., rtol=)`, numpy >= 2,
now declared in zoomy_core's pyproject), and the JAX image had no MPI.

## CI

`.github/workflows/backend-e2e.yml`: one job per backend on `ubuntu-latest`, docker images
from GHCR (`E2E_RUNTIME=docker`), dev mode by default (this commit's `library/` mounted).
Push to main: the QR-wired sessions of numpy, jax, amrex (`E2E_QR_ONLY=1`); manual dispatch:
backends, mode, QR-only and a `-k` selection. The images fit a hosted runner (0.25 / 1.34 /
0.75 / 1.12 GB compressed); the foam / AMReX matrix reference needs apptainer on the host and
is skipped there. Not run yet: the workflow and the docker path of `harness.Container` were
written without docker on the development machine.
