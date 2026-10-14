# Distributed runs (MPI, mpi4jax)

`SymbolicHyperbolicSolverJax(comm=MPI.COMM_WORLD)`: one MPI rank per device,
each on its piece of the mesh (owned cells + 2 halo layers).  Halo exchange
and global reductions are mpi4jax collectives inside the compiled loop;
forward and reverse AD work through them (gates 16-18, ANALYSIS.md section 21).

```python
from mpi4py import MPI
from zoomy_core.mesh.distributed import write_mesh
write_mesh(mesh, "mesh.h5")                     # once (or write_mesh_parallel)
solver = SymbolicHyperbolicSolverJax(time_end=0.5, comm=MPI.COMM_WORLD)
Q, Qaux = solver.setup("mesh.h5", model)        # or a mesh object (small)
time, Q, Qaux = solver.run(Q, Qaux)
Q_all = solver.partition.gather(Q)              # global order, on rank 0
```

- Partition: cells along a space-filling curve (Hilbert 2-D, Morton 3-D),
  rank r owns a contiguous block (`zoomy_core.mesh.distributed`).
- A mesh file: every rank reads only its cells; a mesh object: every rank
  holds it and renumbers (fine up to a few million cells).
- Large meshes are written in parallel: `write_mesh_parallel(comm, path,
  type, X, v0, cv, c0, faces=, face_tags=, cell_data=)`, each rank passing
  its blocks of vertices, cells and boundary elements (example:
  `gates/bench_catchment.py mesh`, a 14.8M-cell DEM catchment in 25 s on 4
  ranks).  `cell_data` (bed, initial state) is read per rank with
  `partition.source.cell_data(name, partition.gids)`.
- AD: differentiate each rank's owned share `J_r` (e.g. a sum over
  `Q[:, :partition.n_own]`); the global derivative is the sum over ranks.
  Quantities every rank holds copies of (parameters, vertex coordinates,
  initial halo states) sum their copies: `partition.sum_to_global(g, ids, n)`.
- Output: `output=` writes one HDF5 file.  From a mesh file with h5py built
  with MPI (`zoomy-cuda-ompi`, the container): every rank writes its own block;
  otherwise gathered on rank 0.  The file's `/mesh` is a mesh file again.

Launch on GPUs: `mpiexec -n 4 python -m zoomy_jax.gpurank run.py` -- binds one
GPU per rank (node-local rank) before CUDA loads, for every CUDA user in the
process (JAX, AmgX, mpi4jax).  `setup` binds JAX's GPU as a fallback, too late
for AmgX (it then runs on GPU 0).  A launcher that sets `CUDA_VISIBLE_DEVICES`
per rank (`srun --gpus-per-task=1`) takes precedence.

Chorin (pressure systems): `SymbolicChorinSolverJax(comm=...)`.  Pressure
solves: 'gmres' (distributed FGMRES), 'pyamg' (FGMRES + per-rank AMG + coarse
correction), 'amgx' (one global AmgX solve in AmgX's MPI mode; AmgX and
JAX-AMG built with MPI, see AMG.md).  ANALYSIS.md section 25.

NumPy backend (`zoomy_core.symsolver.numpy_backend`, no JAX): the same `comm=`
API, partition and halo layers (`zoomy_core.mesh.distributed`), mpi4py only in a
distributed run (the serial backend and the browser import without it).

```python
from mpi4py import MPI
from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
s = SymbolicHyperbolicSolverNumpy(time_end=0.5, comm=MPI.COMM_WORLD)
Q, Qaux = s.setup(mesh, model)                  # the local piece
time, Q, Qaux = s.run(Q, Qaux)
mass = s.integral(Q)                            # a collective: every rank
Q_all = s.partition.gather(Q)                   # rank 0
```

- Halo: point-to-point (Isend / Irecv) of the exchange lists; reductions over
  the cells / faces each rank counts (`Partition.owned`, JAX's rule); min / max
  NaN-propagating (gathered, then `np.min`: MPI.MIN drops a NaN).
- Implicit solves: Newton norms global (every rank the same updates); the
  linear solves `mpi_linear_solver=MpiLinearSolver.auto | petsc | krylov`:
  PETSc via petsc4py (GMRES matrix-free for `gmres`, FGMRES + BoomerAMG on the
  colour-assembled rows for `pyamg`, MUMPS LU for the implicit diffusion's
  Jacobian), or without petsc4py a distributed FGMRES (block-Jacobi: the rank's
  LU / SA V-cycle).  ANALYSIS.md section 50.
- `mpiexec -n 4 python run.py`; keep `OMP_NUM_THREADS=1`.

GPU transport (`bench_halo.py`: 1000 halo exchanges in one compiled loop,
4 L40S, 4 rows; `bench_parallel.py`: whole step, 160k cells, 400 steps):

| MPI (env) | exchange path | 25 KiB | 63 KiB | step, 160k cells |
|---|---|---|---|---|
| MPICH (`zoomy-cuda`) | through host memory | | 119 us | 7.00 ms |
| OpenMPI 5 (`zoomy-cuda-ompi`) | through host memory | 59 us | 78 us | 6.73 ms |
| OpenMPI 5 + UCX, CUDA-aware | GPU -> GPU | 54 us | 52 us | |
| ... + `UCX_MEMTYPE_CACHE=n` (needed) | GPU -> GPU | | 74 us | 6.72 ms |
| OpenMPI 5, CUDA-aware, default pml (ob1) | GPU -> GPU | 172 us | 137 us | |
| container `zoomy_jax_mpi` (OpenMPI 5 + UCX) | GPU -> GPU | | 104 us | 7.06 ms |
| container, `MPI4JAX_USE_CUDA_MPI=0` | through host memory | | | 7.20 ms |

- On this node (PCIe, GPU pairs on two sockets, no NVLink) the transport
  does not change the step time at 160k cells; at 14.8M cells (catchment,
  messages up to 350 kB per peer) GPU -> GPU stalled: one simulated second
  did not finish in 3 min, through host memory it takes 1.3 s.  The
  container therefore exchanges through host memory by default
  (`MPI4JAX_USE_CUDA_MPI=0`); opt in with `--env MPI4JAX_USE_CUDA_MPI=1`.
- CUDA-aware needs all of: `OMPI_MCA_opal_cuda_support=true` (conda-forge
  ships it off), `OMPI_MCA_pml=ucx` (the default ob1 path is slower than
  staging), `UCX_MEMTYPE_CACHE=n` (else UCX copies small JAX device buffers
  as host memory: segfault), `MPI4JAX_USE_CUDA_MPI=1`.
- Expected to matter with NVLink / InfiniBand (GPUDirect RDMA) and many
  GPUs with small pieces; then also overlap exchange and interior work.

Install (credits: mpi4jax, Hafner & Vicentini 2021, doi:10.21105/joss.03419;
mpi4py, Dalcin & Fang 2021; Open MPI, Gabriel et al. 2004).  Three options,
side by side:

1. Container (deployable): `containers/zoomy_jax_mpi/zoomy_jax_mpi.def` --
   JAX 0.9.2 CUDA 12, OpenMPI 5 + UCX CUDA-aware, mpi4jax with CUDA, the
   transport settings above baked in.
   `apptainer exec --nv zoomy_jax_mpi.sif mpiexec -n 4 python run.py`
2. Conda, CUDA-aware (`zoomy-cuda-ompi`): same stack as the container.
3. Conda, existing envs: `zoomy` (CPU, MPICH) and `zoomy-cuda` (GPU, MPICH,
   exchange through host memory).

```bash
# 2. zoomy-cuda-ompi
micromamba create -n zoomy-cuda-ompi -c conda-forge python=3.12 "openmpi>=5" \
  ucx cuda-version=12.9 mpi4py numpy=2.3 scipy=1.16 sympy=1.14 param=2.2 \
  "h5py=*=mpi_openmpi*" meshio pyamg loguru matplotlib nanobind cython pip
pip install "jax[cuda12]==0.9.2" equinox==0.13.6
OMPI_CC=gcc OMPI_CXX=g++ PIP_CONSTRAINT=jaxpin.txt \
  pip install --no-build-isolation mpi4jax==0.9.1.post2
# run: export OMPI_MCA_opal_cuda_support=true OMPI_MCA_pml=ucx \
#        UCX_MEMTYPE_CACHE=n MPI4JAX_USE_CUDA_MPI=1

# 3. zoomy (CPU): mpi4py from conda-forge; mpi4jax from its sdist with the
#    CUDA probe in setup.py (get_cuda_info) disabled
MPICH_CC=/usr/bin/gcc MPICH_CXX=/usr/bin/g++ PIP_CONSTRAINT=jaxpin.txt \
  pip install ./mpi4jax-0.9.1.post2
#    zoomy-cuda (GPU, MPICH): mpi4py mpich=4.3.1 from conda-forge, then
CUDA_ROOT=$HOME/opt/cuda-12.9 MPICH_CC=/usr/bin/gcc MPICH_CXX=/usr/bin/g++ \
  PIP_CONSTRAINT=jaxpin.txt pip install mpi4jax==0.9.1.post2
```

`jaxpin.txt` pins `jax==0.9.2`, `jaxlib==0.9.2` (pip must not move JAX).
OpenMPI binds each rank to one core by default (MPICH does not); with
`taskset` or few cores use `--bind-to none` or `--map-by numa`.

## preCICE in distributed runs

`couple=dict(adapter=PreciceAdapter(name, cfg, mesh, write, read, comm=comm), ...)`
with `comm=MPI.COMM_WORLD`: each rank is a preCICE rank and registers its own
coupling faces; preCICE distributes the data.  Run each participant as its own
`mpiexec` (pyprecice 3.4 cannot take a sub-communicator):

    mpiexec -n 3 python participant_mpi.py Left 1 L.npy window 2d &
    mpiexec -n 2 python participant_mpi.py Right 1 R.npy window 2d

Gate 28 (`gate28_jax_couple_parallel.py`) runs both and compares with 1 + 1
ranks.  The foam participant runs with `-parallel` the same way
(`Participant(name, cfg, myProcNo, nProcs)`).
