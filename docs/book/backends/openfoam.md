# OpenFOAM backend — zoomyFoam (via preCICE)

`zoomy_foam` runs a Zoomy depth-averaged model as a native **OpenFOAM** finite-volume
solver, and couples it across an interface to another solver (a second Zoomy
participant, or a 3-D OpenFOAM region) through the [preCICE](https://precice.org/)
library. It is the one backend where preCICE is in scope; the coupling is built on
the `interpolate_to_3d` / `project_from_3d` maps every Zoomy model already carries
(see [authoring/model.md](../authoring/system-model.md#the-shape-contract)).

## How it is built — code generation, then compile

Unlike NumPy/JAX, zoomyFoam is **generated, then compiled**. A C++ code printer
emits per-case OpenFOAM headers from a frozen `SystemModel` + `Numerics`
(`zoomy_core/transformation/to_openfoam.py`; see the
[printer inventory](../authoring/numerics.md#generating-code)):

| Printer | Emits | Contents |
| --- | --- | --- |
| `FoamSystemModelPrinter` | `Model.H` | `flux_*`, `nonconservative_matrix_*`, `quasilinear_matrix_*`, `eigenvalues`, `source`, plus `interpolate_to_3d` / `project_from_3d` and the `precice_*` patch metadata. |
| `FoamNumericsPrinter` | `NumericsKernels.H` | `numerical_flux`, `numerical_fluctuations`. |
| `FoamUpdateAuxPrinter` | `UpdateAuxVariables.H` | the runtime aux-refresh. |

`zoomy_foam/create_model.py` drives the emit (`build_system_model` / `emit`,
using `PositiveNonconservativeRusanov` and a `Coupled(tag="coupled",
mesh_name="interface")` boundary). The generated headers compile into the solver
binary via `compile.sh` / `wmake`, run inside the project's apptainer image.

This is the same symbolic operator surface the other backends consume — the
printer emits the *same* `numerical_flux` / `quasilinear_matrix_x` kernels as
C++ instead of lambdifying them, so a coupled run matches the
[NumPy](numpy.md) / [JAX](jax.md) reference. If a coupled run needs different
operators, fix the derivation, not the printer.

## The runtime — `zoomyFoam.C`

`zoomy_foam/zoomyFoam.C` is a `SystemModel`-driven explicit finite-volume solver.
`Q` and `Qaux` are `List<volScalarField*>`; reconstruction `W`/`gradW`, the source
field, and the cell-interior non-conservative integral are built once. Time
stepping is explicit forward-Euler or Heun SSP-RK2 — it deliberately bypasses
`fvm::ddt` so the update matches the NumPy/JAX reference exactly.

## preCICE coupling

The adapter is an in-tree preCICE-3 wrapper, `zoomy_foam/precice/PreciceManager.H`
(`precice::Participant`), driven from the `zoomyFoam.C` time loop. Per window:
`read` peer data → solve → `write` this participant's data → `advance`, with
checkpoint write/read around the step and `dt = min(dt, preciceDt)`.

**What is exchanged.** The canonical 6-field profile column
`[b, h, u, v, w, p]` — i.e. `interpolate_to_3d` — sampled on a uniform vertical
grid (`n_faces × nZ` vertices per coupled patch):

```
write:  Q on a coupled patch  --interpolate_to_3d(z_k)-->  6 fields
read:   6 fields  --project_from_3d (depth-average)-->  Q ghost
```

`write` lifts the interior cell to a ζ-column; `read` reduces the peer column
back to a state and imposes a ghost cell selected by `preciceGhost`:

- `fullstate` (default) — conservative: set the whole peer state.
- `characteristic` — only the incoming Riemann invariants, via the
  eigendecomposition of `A = ∂F/∂Q + B`.
- `froude` — a Froude-switched primitive Dirichlet–Neumann condition on `h, hu`.

`preciceFrozenMass yes` freezes the coupled-face mass-row flux per window for a
closed interface mass ledger.

**Participants.** The coupling is symmetric: the *same* model and binary run on
both sides. preCICE forbids reusing a `(data, mesh)` pair, so each participant
writes one data set and reads the peer's — e.g. `SmeA` writes `*_1` / reads
`*_2`, `SmeB` the reverse. Mapping is nearest-neighbour, mesh `dimensions="3"`,
and the coupling scheme is selectable (`parallel-explicit`, …).

## Setting up a coupled run

A coupled case is generated, not hand-written. The reference lives at
`thesis/notebooks/coupling/cases/sme_self/`:

```bash
# emits part1/, part2/, mono/ case dirs + precice-config.xml
python generate.py LEVEL SCHEME ZSAMPLES DT TEND

# blockMesh/setFields each case, launch both participants in parallel (apptainer)
./run.sh
```

`generate.py` writes the preCICE keys into each case's `system/controlDict`, e.g.:

```
preciceParticipant  SmeA;
preciceConfig       precice-config.xml;
preciceMeshes       ( MeshA );
preciceWriteData    ( b_1 h_1 u_1 v_1 w_1 p_1 );
preciceReadData     ( b_2 h_2 u_2 v_2 w_2 p_2 );
preciceGhost        fullstate;
preciceFrozenMass   yes;
preciceZSamples     16;
```

It produces `mono/` (a single uncoupled run) alongside the two participants so
the coupled result can be verified against the monolithic one — the standard
correctness check for a coupling change.

## Authoring checklist

- Generate the case with `generate.py`; never hand-edit the emitted headers.
- Run the `mono/` baseline and confirm the two coupled participants reproduce it.
- Choose `preciceGhost` (`fullstate` / `characteristic` / `froude`) to match the
  interface physics; set `preciceFrozenMass` if a closed mass ledger matters.
- Each participant must `write` one data set and `read` the peer's — never reuse
  a `(data, mesh)` pair.

## Install

Requires OpenFOAM 12+, preCICE 3, and the project apptainer image. See
`zoomy_foam/install/` and the case `run.sh` for the apptainer invocation.

**Repository**: [`library/zoomy_foam`](https://github.com/ZoomyLab/zoomy-foam)
**API reference**: see [zoomy_foam](https://github.com/ZoomyLab/zoomy-foam).
