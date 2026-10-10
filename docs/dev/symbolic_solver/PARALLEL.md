# Distributed runs (MPI, mpi4jax)

`SymbolicHyperbolicSolverJax(comm=MPI.COMM_WORLD)`: one MPI rank per device,
each on its piece of the mesh (owned cells + 2 halo layers).  Halo exchange
and global reductions are mpi4jax collectives inside the compiled loop;
forward and reverse AD work through them (gates 16-18, ANALYSIS.md section 21).

```python
from mpi4py import MPI
solver = SymbolicHyperbolicSolverJax(time_end=0.5, comm=MPI.COMM_WORLD)
Q, Qaux = solver.setup(global_mesh, model)     # local state (owned + halo)
time, Q, Qaux = solver.run(Q, Qaux)
Q_all = solver.partition.gather(Q)              # global order, on rank 0
```

- Partition: coordinate bisection (`parallel.bisection`); any owner array via
  `parallel.split(mesh, comm, part=...)`.
- AD: differentiate each rank's owned share `J_r` (e.g. a sum over
  `Q[:, :partition.n_own]`); the global derivative is the sum over ranks.
  Quantities every rank holds copies of (parameters, vertex coordinates,
  initial halo states) sum their copies: `partition.sum_to_global(g, ids, n)`.
- Output: `output=` writes one HDF5 file on rank 0, global mesh.

Launch, one GPU per rank (MPICH sets `PMI_RANK`):

```bash
mpiexec -n 4 bash -c 'CUDA_VISIBLE_DEVICES=$PMI_RANK python run.py'
```

Install (credits: mpi4jax, Hafner & Vicentini 2021, doi:10.21105/joss.03419;
mpi4py, Dalcin & Fang 2021)

```bash
# CPU env (zoomy): mpi4py from conda-forge; mpi4jax built from its sdist
# with the CUDA probe in setup.py (get_cuda_info) disabled
micromamba install -n zoomy -c conda-forge mpi4py
MPICH_CC=/usr/bin/gcc MPICH_CXX=/usr/bin/g++ PIP_CONSTRAINT=jaxpin.txt \
  pip install ./mpi4jax-0.9.1.post2
# GPU env (zoomy-cuda): mpi4py matching its MPICH, mpi4jax with CUDA
micromamba install -n zoomy-cuda -c conda-forge mpi4py mpich=4.3.1
CUDA_ROOT=$HOME/opt/cuda-12.9 MPICH_CC=/usr/bin/gcc MPICH_CXX=/usr/bin/g++ \
  PIP_CONSTRAINT=jaxpin.txt pip install mpi4jax==0.9.1.post2
```

`jaxpin.txt` pins `jax==0.9.2`, `jaxlib==0.9.2` (pip must not move JAX).
