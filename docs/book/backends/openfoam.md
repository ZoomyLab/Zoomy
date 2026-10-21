# OpenFOAM backend

`zoomy_foam` runs the solver templates as a native **OpenFOAM** 13 finite-volume solver,
`symFoam`.  The driver `zoomy_foam.symsolver` prints the template as C++, builds it with `wmake`
inside the project's apptainer image, writes the OpenFOAM case, runs it (serial or
`mpirun -parallel`) and reads the result back.  It takes unstructured meshes and, through
[preCICE](https://precice.org/), couples a depth-averaged model to another Zoomy participant or to
a resolved 3-D OpenFOAM region.

## How it is built: print, then compile

Unlike NumPy and JAX, the foam solver is **generated, then compiled**.  The C++ printer of
`zoomy_core.symsolver` (one printer, shared with AMReX) emits the build tree, and `wmake` compiles
it:

| File | Contents |
| --- | --- |
| `Solver.H` | the printed template: the step, the march, MOOD, IMEX, the pressure solve |
| `Kernels.H` | the NSM's kernels as C++: flux, nonconservative matrix, eigenvalues, source, boundary ghost states |
| `rt.H` | the foam runtime: the template's building blocks on OpenFOAM fields (gradients, Riemann problems on all faces, reductions, linear solves) |
| `rt_common.H` | the part of the runtime shared with AMReX: functors, `configure()` settings, effects, Newton skeletons |
| `symFoam.C` | the host program |

The binary is cached by a hash of the sources, the build flags and the image
(`ZOOMY_FOAM_BINCACHE`).  The kernels are the same operators the other backends consume, printed
as C++ instead of Python, so a foam run matches the [NumPy](numpy.md) and [JAX](jax.md) run to
round-off at first order and to the discretization error at second order (foam takes its second-order
gradients from OpenFOAM's own least-squares stencils).  If a run needs different operators, fix the
derivation, not the printer.

## Solvers

| Class | Template | Call |
| --- | --- | --- |
| `SymbolicHyperbolicSolverFoam` | `HyperbolicTemplate` | `time, Q = solver.run(mesh, nsm, Q0, workdir)` |
| `SymbolicChorinSolverFoam` | `ChorinTemplate` | `time, Q = solver.run(mesh, (pred, press, corr), Q0, workdir)` |

`Q0` is the initial state `(n_state, n_cells)` in the mesh's order and `workdir` the case
directory: absolute, recreated by the run.  `run` returns the final state in the mesh's order; the
snapshots of `dt_write` stay in `solver.snapshots`.  A structured mesh is written as a `blockMesh`
case; any other mesh goes through `gmshToFoam` on the mesh extruded by one layer.

## Settings

The settings shared by all backends are the tables of [Solvers and options](../guide/solvers.md).
foam adds:

| option | default | meaning |
|---|---|---|
| `nprocs` | `1` | MPI ranks: `decomposePar`, then `mpirun -parallel` |
| `decomposition` | `Decomposition.scotch` | `decomposePar` method: `Decomposition.scotch` (graph partition) or `Decomposition.simple` (`nprocs` slabs along x) |
| `linear_solver` (Chorin) | `LinearSolver.hypre` | `hypre`, `schur_gamg` or `gmres`, see [Solvers and options](../guide/solvers.md) |
| `couple`, `subcycling` | `None`, `False` | preCICE participant (dict below); adaptive steps inside a window |

Options are members of `Option` enums, so a plain string raises `TypeError` and names the member:

```python
from zoomy_foam.symsolver import Decomposition, SymbolicHyperbolicSolverFoam

solver = SymbolicHyperbolicSolverFoam(time_end=1.0, cfl=0.9, nprocs=4,
                                      decomposition=Decomposition.simple)
SymbolicHyperbolicSolverFoam(decomposition="simple")
# TypeError: decomposition='simple': use Decomposition.simple
```

Environment: `ZOOMY_OF_SIF` (the image, default `~/of_build/zoomy_openfoam.sif`),
`ZOOMY_FOAM_BINCACHE` (the binary cache).  Inside the image the driver builds and runs directly.

## End-to-end example: shallow water dam break

The case is `docs/book/examples/dambreak.py` of the [NumPy page](numpy.md).  From
`docs/book/examples/`:

```{literalinclude} ../examples/foam_dambreak.py
:language: python
```

```text
t = 0.500, cells = 300, h in [1.00000, 2.00000]
max |h_foam - h_numpy| = 1.73e-09
```

The first run builds `symFoam` (cached afterwards).  The 2-D case of the
[Quick start](../guide/quickstart.md) runs on foam the same way.

## preCICE coupling

`couple` makes the run one preCICE participant.  The data is the rows of the state: a participant
writes `write` and reads `read`, one preCICE data name per state row, preCICE forbids reusing a
`(data, mesh)` pair, so the two sides name them differently.  A dam break on `[0, 1]` split at
$x = 0.5$ into two foam participants (Gate 3 of the symbolic-solver gates, matrix case C16):

```python
from zoomy_foam.symsolver import SymbolicHyperbolicSolverFoam

left = dict(participant="Left", config="precice-config.xml", mesh="Left-Mesh",
            write=["L0", "L1", "L2"], read=["R0", "R1", "R2"], tag="right", side=0)
solver = SymbolicHyperbolicSolverFoam(cfl=0.9, nprocs=2, couple=left)
```

| key | meaning |
|---|---|
| `participant`, `config`, `mesh` | the preCICE participant, its configuration file, the mesh the data lives on |
| `write`, `read` | the data names this participant writes (its state rows) and reads (the peer's) |
| `tag` | the coupled boundary of this domain |
| `side` | 0 if this domain is the low side of the interface |

Per window the participant reads the peer's data, takes its step (one step per window with
`subcycling=False`, exactly conservative; adaptive steps inside the window with `True`), and
writes its own.  The two-participant reference setup is
`docs/dev/symbolic_solver/gates/couple_side.py` (the foam and AMReX sides, with the preCICE
configuration; `participant.py` is the JAX side): the coupled run reproduces the single-domain
march to round-off with exact mass.  Restarts and checkpoints are not available with coupling.

**A resolved 3-D peer.** `zoomy_foam.vof` is the other side of a depth-averaged | VOF coupling:
OpenFOAM 13 `incompressibleVoF` in a rectangular tank, with the `swePreciceCoupling` function
object imposing the depth-averaged arm's column on the coupled patches.  The exchange is the
canonical six-field column `[b, h, u, v, w, p]` (`<field>_S` written by the depth-averaged arm,
`<field>_V` by the VOF participant) at the cell-centre heights of the tank, that is the model's
`interpolate_to_3d` / `project_from_3d`.  The depth-averaged arm is a JAX participant
(`zoomy_jax.symsolver.coupling.column_participant`) that answers with the Riemann problem towards
the characteristic ghost (`CoupledFace.characteristic`), see the [JAX page](jax.md).

## Authoring checklist

- Run the single-domain march next to the coupled one and confirm the participants reproduce it:
  the standard correctness check of a coupling change.
- The coupled data names must be unique per `(data, mesh)` pair: each participant writes one data
  set and reads the peer's.
- `workdir` is absolute and is deleted and recreated; do not point it at a folder you keep.

## Install

Requires OpenFOAM 13 (openfoam.org) and, for coupling, preCICE 3: the `zoomy_openfoam`
[container](../containers.md) ships both.

**Repository**: [`library/zoomy_foam`](https://github.com/ZoomyLab/zoomy-foam)  
**API reference**: {doc}`zoomy_foam <../api/_autodoc2/zoomy_foam/zoomy_foam>`.
