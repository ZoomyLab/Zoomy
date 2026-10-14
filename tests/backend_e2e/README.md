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
pytest tests/backend_e2e -m backend_e2e -k "matrix and jax and (C15- or C22-)"   # cases of a backend
pytest tests/backend_e2e -m backend_e2e -k C16                          # preCICE across containers
E2E_GPUS=0,1 pytest tests/backend_e2e -m backend_e2e -k "matrix and gpu"   # GPU configs (--nv)
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
| `E2E_GPUS` | the GPUs the gpuN configs may use (`CUDA_VISIBLE_DEVICES`, else every GPU `nvidia-smi -L` lists); a config needing more is skipped |

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
- Coupled sessions across images (`sessions.PEERS`: the live SME | VOF couplings): the case runs
  on its solver card's backend (OpenFOAM: the VOF tank) and its job submits the other
  participants (the SME arms) to the backends they need (JAX), as the GUI submits a case
  (`zoomy_server.peers`). Every server binds one directory at `/e2e/couple`
  (`ZOOMY_COUPLE_DIR`, preCICE's exchange directory) and finds the others in its
  `peers.json` (`ZOOMY_PEERS`, `{adapter tag: url}`, rewritten as servers come up); the test
  starts the peers first. No host run (`sessions.NO_HOST_RUN`): the comparison with the old
  solver is `library/zoomy_gui/projects/tools/symsolver/figures/coupling_live_old_vs_new.py`.
- Protocol (`harness.GuiClient` = `HttpAdapter.submitCase`): `POST /api/v1/cases`, poll
  `GET /api/v1/jobs/<id>` every 2 s, `GET /api/v1/jobs/<id>/results/hdf5`.
- Checks: every float finite; at least 2 snapshots and the last at `time_end` (the QR suite's
  `session_runner.check_final_store`; where the knob has other units or none, `sessions.END_TIME`:
  the Bingham roll wave's t'; a replayed store: the snapshots only; the
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
C4 (`sub_o2_n25`, `trans_o1_n25`), C5 (`n48`), C8 (`vam`), C14 (`o1_n20`, `o2_n20`), C23
(`o1_n8`, `o2_n8`), C26 (`visc`), C28 (`quad_expl_dir0`, `quad_impl_neu`; JAX, numpy:
`orders_quad_expl`), C31 (`swe_ssp2_k0`, `vam_ssp2_k0`), C32 (`noslip_n20`), C33 (`imex_k0`),
C35 (`c1_full`, `c1_half`, `c1_restart`, `c1_typo`); JAX also C15 (`hyperbolic`, `imex`; fwd
and rev) and C22 (`o1_dt`, `o2_dt`; fwd and rev), AMReX C17 (`cover_o2`, `box`, `lake`,
`wave16`, `wave32`); each backend the runs its `cases.py` lists it for, by `matrix.py`'s own
`run` (`matrix_driver.py`) inside each image: its Python, and for foam / AMReX its OpenFOAM /
AMReX toolchain (built natively, `ZOOMY_{FOAM,AMREX}_IN_IMAGE=1`; `shim/apptainer` for older
drivers that always call `apptainer exec <image>`). The case builders import no backend
(`gates/common.py`, `gates/euler_common.py` take JAX if present), so every image sets up every
case. The same runs on the host (the same library; foam / AMReX through apptainer with the
same image) give the reference; every stored array must agree to the matrix's tolerance for
two runs of the case (`tol`: round-off 1e-12, or the implicit-solve tolerance: C8, C31 VAM,
C28 implicit 1e-9; C14, C26, C31 SWE, C32 1e-8; C33 1e-10; the image's numpy / JAX may
differ from the host's); of a custom run's metrics the values (`VALUES`: J, jvp, reverse; the AMR wave's
L1; t), not the round-off measures (mass drift, |FD - jvp|, seconds, messages naming paths);
the timing and provenance keys are ignored. Then the matrix's own checks of the case
(`matrix_driver.py --checks`: `matrix.measure`, the table's thresholds, with C6 mass and C7
parallel = serial) must pass on the container's runs: C15 / C22 |FD - jvp| and |rev - jvp|,
C17 fine level = fine single level, mass and energy with reflux, lake at rest, the wave order,
C35 restarted = continuous bitwise and a misspelt `restart_from` failing at once, C5 h >= 0.
Distributed, with the image's MPI (JAX: OpenMPI 5 + mpi4jax, `mpiexec`; foam: OpenMPI 4.1,
`decomposePar` + `mpirun -parallel`; AMReX: OpenMPI 4.1, `mpiexec`): C1 `o2_n40` on 2 ranks = the serial run
in the same image (matrix C7, round-off), C35 `c1` restart on 2 ranks (bitwise), AMReX C17
`box` on 2 ranks = 1 rank.
On GPUs (`GPU`; the container started with `--nv` and `CUDA_VISIBLE_DEVICES` from `E2E_GPUS`;
JAX and AMReX, the images that carry CUDA): C1 `o2_n40` on 1 GPU (`gpu1`) and C5 `n48` on 2
GPUs, one rank each (`gpu2`; JAX: the image's OpenMPI 5 + mpi4jax with its CUDA bridge, halos
through host memory, `python -m zoomy_jax.gpurank`; AMReX: the CUDA build with the image's
nvcc, OpenMPI 4.1.6). Each = the host's run of the same config (JAX: the `zoomy-cuda` env, as
the matrix runs it; AMReX: `symsolver` through `apptainer exec --nv` with the same image) to
round-off, and the matrix's C9 (GPU = the serial CPU run, both in the image) and C6 (mass) on
the container's runs. Without enough GPUs the GPU configs are skipped (CI's hosted runners).

**preCICE across containers** (`test_couple_across_containers.py`, matrix C16): gate 3's 1-D
dam break split at x = 0.5, each participant in its own image, two `apptainer exec` processes
on the host network finding each other through m2n:sockets and an exchange directory bound at
`/e2e/couple` in both (`Container.shared`). Pairs (Left | Right): jax | jax, jax | openfoam,
openfoam | openfoam, amrex | amrex, amrex | jax, serial; jax | jax, openfoam | openfoam and
amrex | amrex also on 2 ranks a side. Participants: JAX `gates/participant.py`
(`participant_mpi.py` on ranks), foam / AMReX `gates/couple_side.py` (the setup gates 10, 26,
38 share, backend-neutral, so it runs in an image without JAX). Check (matrix C16, gate 26):
max|coupled - single domain| and the mass drift <= 1e-14, the single domain marched on the
host (`couple_side.py check`). Apptainer only: two docker containers would need
`--network host` (not wired; the docker path is untested here).

**Coverage of the image-dependent paths.** The physics is proven on the host (the full matrix,
`docs/dev/symbolic_solver/matrix`); the containers prove image = host where the result depends
on what the image ships.

| path (what the image ships) | in the container | images | host only, and why |
|---|---|---|---|
| Python stack (numpy, scipy, sympy, h5py, JAX versions) | the matrix subset above = host | all four | - |
| C++ toolchains (OpenFOAM 13 `wmake`; AMReX 25.02, g++) | every foam / AMReX run above builds and runs natively | openfoam, amrex | - |
| MPI | C1 `o2_n40` mpi2 = serial; C35 `c1` mpi2; C16 2 + 2 ranks; C17 `box` mpi2 | jax, openfoam, amrex | mpi4 / mpi8 (the same path, more ranks); multi-node / Slurm (untestable here); numpy is serial by design |
| preCICE | C16, five pairs across two images (serial; three pairs on 2 + 2 ranks) | jax, openfoam, amrex | jax <-> foam on ranks (no driver: gate 10 is serial, matrix n/a); amrex (2 ranks) <-> jax (1 rank), the matrix's `aj 2` (here both sides run the same ranks) |
| AMR build | C17 `cover_o2`, `box`, `lake`, `wave16`, `wave32`; `box` mpi2 | amrex | `cover_dry`, `boxnr`, `dry` (the same build path) |
| AD (the JAX version) | C15 `hyperbolic`, `imex`; C22 `o1_dt`, `o2_dt` | jax | C15 `chorin` (the same AD path; 3 min on the host for fwd + rev), C22 `o1_adaptive`, AD on ranks |
| 3-D compressible Euler | C23 `o1_n8`, `o2_n8` | all four | C18-C21, C24 (the same kernels, larger) |
| restart (each backend's restart state: h5, foam time directories, AMReX output) | C35 `c1` serial and mpi2 | all four (numpy serial) | `c8` (the Chorin restart with the lagged pressure; 3 min on the host for its four runs) |
| wet/dry + MOOD, IMEX, implicit diffusion | C5; C14, C31, C33; C26, C28, C32 | all four | - |
| GPU (JAX's CUDA plugin + NVIDIA's CUDA wheels, mpi4jax's CUDA bridge; nvcc + CUDA 12.6 libraries) | C1 `o2_n40` gpu1, C5 `n48` gpu2 (2 GPUs, one rank each): = host, C9, C6 | jax, amrex | C8 / C14 / C26 on GPUs (the same build paths; C8 AMReX: hypre-CUDA, built at run time), the GPU parts of C15, C17; CUDA-aware MPI (in no image: halos staged through the host); AmgX (`zoomy_jax_mpi` only) |
| large tier (L1-L5) | - | - | scaling, catchment: more than 8 cores, GPUs |
| the other cases (C2, C3, C10-C13, C18-C21, C24, C27, C29, C30, C34, C36) | - | - | physics on paths the subset already exercises; C29, C30, C34 compare with JAX in-process |

## Run times

GUI cases: measured 2026-10-13 on main 7ce6f03b (dev mode, apptainer, the GHCR images of that
day, 8 cores via `taskset -c 56-63`); "host" = the comparison run on the host. One pytest
command per backend (numpy: two) stays under 15 min.

| backend | GUI cases |
|---|---|
| numpy | Steffler 30 s (45 s with the host run); the four QR sessions 68 s; Bingham threshold (time_end 20 s) 254 s (492 s); all 11 cases ~12 min |
| jax | Malpasset (time_end 2 s) 68 s (133 s with the host run); the cards 4 s |
| amrex | Malpasset (time_end 2 s) 66 s with the build, 12 s cached (81 s with the host run) |
| openfoam | the card 22 s (44 s with the host run) |

Matrix and coupling: measured 2026-10-14, branch dev/container-matrix-coverage (dev mode,
apptainer, 8 cores via `taskset -c 40-47`, one command at a time; numpy / AMReX the GHCR
images of 2026-10-14, JAX / OpenFOAM local builds of this branch's recipes). Every command
below stays under 15 min (`pytest tests/backend_e2e -m backend_e2e -k "<expression>"`):

| `-k` | tests | time |
|---|---|---|
| `matrix and numpy` | 12 | 9:45 |
| `matrix and jax and (C1-serial or C4- or C8- or C26-)` | 4 | 7:07 |
| `matrix and jax and (C28- or C31- or C32- or C33-)` | 4 | 8:23 |
| `matrix and jax and (C5- or C14- or C23- or C35- or C1-mpi2)` | 6 | 7:21 |
| `matrix and jax and (C15- or C22-)` | 2 | 9:35 |
| `matrix and openfoam and (C1-serial or C4- or C8- or C26-)` | 4 | 9:25 |
| `matrix and openfoam and (C28- or C31- or C32- or C33-)` | 4 | 8:56 |
| `matrix and openfoam and (C5- or C14- or C23- or C35- or C1-mpi2)` | 6 | 9:18 |
| `matrix and amrex and (C1-serial or C4- or C8- or C26-)` | 4 | 9:45 |
| `matrix and amrex and (C28- or C31- or C32- or C33-)` | 4 | 8:14 |
| `matrix and amrex and (C5- or C14- or C23- or C35- or C1-mpi2)` | 6 | 9:02 |
| `matrix and amrex and C17-` | 2 | 6:12 |
| `C16` | 8 | 4:43 (the foam / AMReX builds cached; cold: + 1 / 1.5 min) |

Per test, container + host (s; mpiN: in the image only, with its serial run where the check
reads it). numpy: C1 24 + 13, C4 24 + 11, C5 16 + 11, C8 35 + 18, C14 23 + 10, C23 8 + 4,
C26 41 + 26, C28 57 + 46, C31 48 + 31, C32 41 + 26, C33 12 + 5, C35 36 + 23. JAX: C1 45 + 31,
C4 45 + 30, C5 31 + 20, C8 85 + 69, C14 46 + 35, C15 213 + 148, C22 111 + 101, C23 26 + 13,
C26 67 + 49, C28 46 + 28, C31 132 + 124, C32 64 + 46, C33 34 + 25, C35 74 + 58; C1 mpi2 67,
C35 mpi2 58. OpenFOAM: C1 73 + 67, C4 65 + 84, C5 17 + 15, C8 77 + 68, C14 60 + 73, C23
53 + 61, C26 66 + 59, C28 41 + 51, C31 109 + 112, C32 78 + 67, C33 37 + 36, C35 68 + 73; C1
mpi2 40, C35 mpi2 52. AMReX: C1 80 + 83, C4 52 + 55, C5 14 + 11, C8 110 + 89, C14 56 + 57,
C17 159 + 192, C23 86 + 90, C26 64 + 46, C28 34 + 36, C31 95 + 91, C32 65 + 57, C33 61 + 51,
C35 57 + 51; C1 mpi2 32, C17 mpi2 18, C35 mpi2 42. C16 (both participants, cached builds):
jax_jax 27, jax_foam 26, foam_foam 15, amrex_amrex 12, amrex_jax 27; mpi2: jax_jax 37,
foam_foam 16, amrex_amrex 12.

## GPU-capable images (2026-10-14, dev/gpu-images, dev mode)

One image per backend for CPU-only and GPU machines (`containers/README.md`): `zoomy_jax` adds
JAX's CUDA 12 plugin with NVIDIA's CUDA wheels (`containers/zoomy_jax/cuda-pins.txt`: the
versions of the host's `zoomy-cuda` env) to the unchanged conda-forge CPU jaxlib, and builds
mpi4jax with its CUDA bridge; `zoomy_amrex` adds nvcc and the CUDA 12.6 libraries the CUDA
build links (shared only). Sizes, before = the GHCR images of 2026-10-14
(`zoomy_jax:0.4.171`, `zoomy_amrex:0.1.171`), after = this branch's recipes built with
apptainer 1.5.2; unpacked = `du -sx /` in the image:

| image | SIF before | SIF after | unpacked before | unpacked after |
|---|---|---|---|---|
| zoomy_jax | 1.45 GB | 4.32 GB (+2.87) | 4.87 GB | 9.43 GB (+4.56) |
| zoomy_amrex | 0.72 GB | 1.61 GB (+0.89) | 2.29 GB | 3.70 GB (+1.41) |

Local builds (16 cores): JAX 8:01 on the cached `zoomy_jax_dev` base, AMReX 4:56. CI
(`build-containers.yml`) drops the hosted runner's preinstalled toolchains before building
these two and puts apptainer's temporaries on the larger disk: the JAX image's docker build
and SIF conversion need ~31 GB at the peak (docker store ~9.4 GB, its export, the unpacked
rootfs, the 4.3 GB SIF; AMReX ~12 GB), more than a runner leaves free. Estimated, not run in
CI (nothing pushed).

Runs on the new images (GPUs 0 and 1 of the 4 x L40S node, driver 580; `E2E_GPUS=0,1`):

| test | container = host | matrix checks (container) | s, container + host |
|---|---|---|---|
| jax C1 gpu1 | 5.8e-14 | C9 GPU vs serial CPU 1.5e-13, C6 mass 0 | 219 + 151 |
| jax C5 gpu2 | 5.2e-15 | C9 5.2e-15, C6 1.9e-16, h_min 0 | 150 + 61 |
| amrex C1 gpu1 | 0 | C9 0, C6 0 | 185 + 122 (the nvcc build in the image and on the host) |
| amrex C5 gpu2 | 0 | C9 0, C6 0, h_min 0 | 103 + 52 |

gpu2 binds one GPU per rank: AMReX's two ranks on GPU 0 and 1 (`nvidia-smi`), JAX's through
`zoomy_jax.gpurank`, which now picks one device per rank from a listed
`CUDA_VISIBLE_DEVICES` (before: every rank on the first) and runs the script as `python
script.py` does (its directory on `sys.path`; matrix.py imports `cases` from there).

CPU on the same images, no `--nv` (same as before the change): `matrix and jax and not gpu`
16/16 (36:42 on 8 cores; container = host bit-identical but C5 4.3e-15, C8 5.5e-11 (tolerance
1e-9), C15 3.7e-15, C22 1.6e-15, C23 3.5e-16, C26 5.3e-16, C31 1.9e-15, C32 2.2e-16; C1 `o2_n40`
2 ranks vs serial 1.4e-13), `matrix and amrex and not gpu` 16/16 (32:06 on 8 cores;
bit-identical but C8 2.5e-15, C23 3.5e-16, C31 2.6e-15), `C16` 8/8 (coupled vs single domain
3.3e-15 to 5.2e-15, mass drift 0; jax_foam with the published OpenFOAM SIF: the one in
`~/git/Zoomy/containers` still links preCICE 3.4.1 and refuses the 3.4.0 peers). The server
of both images reports `status=ok` without a GPU (CI's smoke). Without `--nv` JAX logs that
the CUDA plugin could not initialize and runs on the CPU (with `JAX_PLATFORMS=cpu`, as the
harness sets, nothing is logged); AMReX builds the CPU variant unless a run asks for the GPU.

## Results (2026-10-14, dev/qr-backend-routing, dev mode)

| backend | pass | not run |
|---|---|---|
| numpy | steffler-0, bingham-0, coupling-0, zoomy-codeprinter-0, bingham-analytics-0, bingham-threshold-0, card-solver-{numpy,imex-numpy,split-numpy} | |
| jax | malpasset-0 (mass 0, host run identical), card-solver-{jax,imex-jax} | |
| amrex | malpasset-1 (mass 2.3e-16), card-solver-{amrex,split-amrex} | |
| openfoam | card-solver-foam | (the server's legacy `/couple` template path: not ported; no session uses it) |

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

Matrix in the container and preCICE across containers (2026-10-14, branch
dev/container-matrix-coverage, dev mode; images as in Run times): all 58 matrix tests and all
8 coupling tests pass. Container = host bit-identical but for JAX C5 4.1e-15, C8 2.9e-10
(pressure, tolerance 1e-9), C15 `imex` 2.6e-15, C22 `o2_dt_fwd` 1.6e-15, C23 `o2_n8` 3.5e-16,
C26 5.4e-16, C31 4.3e-15; foam C8 1.9e-14, C31 3.5e-15; AMReX C8 7.9e-15, C23 3.5e-16, C31
4.7e-15. Matrix checks in the images: C15 |FD - jvp| 7.9e-9 / 3.7e-7, |rev - jvp| 7.8e-15 /
1.1e-14 (hyperbolic / imex); C22 |FD - jvp| 2.1e-8 / 4.4e-8, |rev - jvp| 2.1e-15 / 0; C17
fine level = fine single level 0, box mass 0 and energy 2.2e-16 with reflux, lake max|u|
1.3e-15, wave order 16->32 2.5, box on 2 ranks = 1 rank 0; C35 restarted = continuous 0
(every backend serial; JAX, foam, AMReX also on 2 ranks) and the misspelt `restart_from`
fails at once; C7 C1
`o2_n40` 2 ranks vs serial: JAX 1.2e-13, foam 9.4e-16, AMReX 0. C16 coupled vs single domain,
mass drift 0 everywhere: jax_jax 3.1e-15, jax_foam 4.3e-15, foam_foam 4.7e-15, amrex_amrex
3.4e-15, amrex_jax 3.1e-15; on 2 + 2 ranks the same values. The host's C16 (gates 10, 26, 28,
38 through `matrix.py run`, foam with this branch's image) gives the same numbers.

Before (the GHCR images of 2026-10-14, main 790dc047): no JAX pair could couple — the JAX image
had no pyprecice (`zoomy_jax.def` installed it, but CI builds the Dockerfile), nor the petsc4py
`runtime.PreciceAdapter` imports first; with it, the OpenFOAM image's libprecice 3.4.1 refused
the 3.4.0 peers ("Mixing preCICE versions ... is forbidden"; on the host zoomy_foam linked a
3.4.0 build under `~/of_build`, which no container sees). C23 could not be set up in the
numpy / foam / AMReX images (`gates/euler_common.py` imported JAX) and C17 not in the AMReX
image (gate 37 set up and referenced with JAX; now zoomy_core's NumPy backend, = JAX to
round-off).

## CI

`.github/workflows/backend-e2e.yml`: one job per backend on `ubuntu-latest`, docker images
from GHCR (`E2E_RUNTIME=docker`), dev mode by default (this commit's `library/` mounted).
Push to main: the QR-wired sessions of numpy, jax, amrex (`E2E_QR_ONLY=1`); manual dispatch:
backends, mode, QR-only and a `-k` selection. The images fit a hosted runner (numpy / JAX /
AMReX / OpenFOAM 0.25 / 1.34 / 0.75 / 1.12 GB compressed before the GPU-capable images; JAX
and AMReX SIFs now 4.32 / 1.61 GB); the foam / AMReX matrix reference needs apptainer on the
host and is skipped there; the GPU configs need a GPU and are skipped there. Not run yet: the
workflow and the docker path of `harness.Container` (`--gpus` too) were written without docker
on the development machine.

## Results (2026-10-14, dev/smevof-live, dev mode)

The live SME | VOF sessions on the symbolic solvers (OpenFOAM backend: the VOF tank and the
`swePreciceCoupling` function object built in the job; JAX backend: the SME arm(s),
`SymbolicHyperbolicSolverJax` with the column exchange and the characteristic coupled face),
the GHCR images of 2026-10-14, `taskset -c 16-31`:

| backend | pass | time |
|---|---|---|
| openfoam (+ jax peer) | coupling-live-0 | 120 s (46 s of it the function object's first build) |
| openfoam (+ jax peer) | coupling-triple-live-0 | 76 s (the build cached) |

Against the zoomyFoam participant on the same configuration (Fr* = 0.752, t_end 2.5 s),
rel. L2 / Linf of the depth at the coupled face over the snapshots: pair 8.6e-5 / 1.3e-4;
triple, arm with the dam 3.9e-4 / 6.7e-4, the arm downstream of the tank 5.4e-3 / 1.9e-2,
where the zoomyFoam run itself moves by 3.5e-3 / 8.4e-3 when the reservoir is 1e-9 m deeper
(`coupling_live_{pair,triple}_old_vs_new.json`).
