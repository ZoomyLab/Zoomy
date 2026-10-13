# zoomy-jax

This repository is a submodule of the the [Zoomy Lab](https://github.com/ZoomyLab/Zoomy) repository.

## SPMD scaling (JAX-native multi-device)

`zoomy_jax` ships a JAX-native SPMD path for `HyperbolicSolver` —
no MPI, no `mpi4jax`, just `jax.lax.ppermute` inside `jax.shard_map`.
The existing solver code runs unchanged on a per-partition mesh; the
only required pieces are a halo-exchange primitive and a partition
utility that remaps LSQ stencils to local-padded indices.

### Pattern at a glance

```python
import jax, jax.numpy as jnp, numpy as np
from functools import partial
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.numerics.numerical_system_model import Limiter, ReconstructionSpec
from zoomy_core.model.models.system_model import SystemModel
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.mesh.partition_jax import (
    partition_1d_contiguous,         # 1D-only, custom slab layout
    partition_xaxis_structured,      # 1D / 2D / 3D structured, x-axis split
)
from zoomy_jax.fvm.halo_exchange_jax import halo_exchange_inplace

# 1. Build the solver once on a global mesh — extract the runtime.
mesh_global = LSQMesh.create_2d(domain=(0., 1., 0., 1.), nx=NX, ny=NY)
nsm = NumericalSystemModel.from_system_model(
    SystemModel.from_model(model),
    reconstruction=ReconstructionSpec(order=2, limiter=Limiter.venkatakrishnan),
)
solver = HyperbolicSolver()
Q, Qaux = solver.setup_simulation(mesh_global, nsm)
runtime = solver._rt_model

# 2. x-axis decomposition.  In 2D/3D, halo cells form strips/slabs
#    of `halo * ny` (or `halo * ny * nz`) cells in the flat cell axis.
parts = partition_xaxis_structured(
    solver._rt_mesh, n_parts=4, halo=2,
    domain=(0., 1., 0., 1.), shape=(NX, NY),
)
flux_op_part = solver.get_flux_operator(parts[1], runtime)

# 3. Wrap in shard_map with halo exchange before each stage.
spmd_mesh = Mesh(np.array(jax.devices()), axis_names=("cells",))
halo_cells = 2 * NY    # halo_x * x_stride; x_stride = NY in 2D, NY*NZ in 3D.

@partial(shard_map, mesh=spmd_mesh,
         in_specs=(P(None, "cells"), P(None, "cells")),
         out_specs=P(None, "cells"), check_rep=False)
def spmd_step(Q_pad, Qaux_pad):
    Q_pad = halo_exchange_inplace(Q_pad, halo=halo_cells,
                                  axis_name="cells", n_devices=4)
    dQ = flux_op_part(dt, time, Q_pad, Qaux_pad, parameters,
                      jnp.zeros_like(Q_pad))
    return Q_pad + dt * dQ
```

For SSP-RK2 / RK3, call halo exchange **before each stage** —
stage-1's local update writes nonsense at halo cells, so stage-2
must re-fetch from neighbors.

### 1D / 2D / 3D and which partition function to use

* **1D structured mesh** — either `partition_1d_contiguous(mesh,
  n_parts, halo)` (the original, hand-built slab layout) or
  `partition_xaxis_structured(mesh, n_parts, halo, domain=(x_min,
  x_max), shape=(nx,))` (the unified path).  Both produce
  bit-identical results.
* **2D structured mesh** — `partition_xaxis_structured(mesh,
  n_parts, halo, domain=(x_min, x_max, y_min, y_max), shape=(nx,
  ny))`.  Decomposes only along x; y stays whole.  Halo cells form
  a `halo * ny`-cell strip on each side of the owned slab in the
  flat cell-axis indexing.
* **3D structured mesh** — `partition_xaxis_structured(mesh,
  n_parts, halo, domain=(x_min, x_max, y_min, y_max, z_min,
  z_max), shape=(nx, ny, nz))`.  Halo slabs are `halo * ny * nz`
  cells each.

In every case the same `halo_exchange_inplace` primitive works —
the cell axis is flat, the halo is whatever number of cells is on
either side.

### Layout

Per device, `Q` is padded:

```
[  halo_left  |  n_local owned cells  |  halo_right  ]
   |← halo →|                          |← halo →|
```

`partition_1d_contiguous` produces a `MeshJAX` with
`n_inner_cells = n_local + 2*halo` (the full padded slab) and the
LSQ A matrix per owned cell copied verbatim from the global mesh —
the matrix is **geometry-only** and identical between global and
partitioned forms; only `lsq_neighbors` index pointers are remapped
to local-padded space (`g_idx - p*n_local + halo`).

Halo cells whose stencil fits in the padded slab (the inner-most
ones on each side when `halo >= 2` for radius-1 linear LSQ) get
real remapped LSQ data; outer halo cells fall back to a
self-stencil (gradient = 0).  **For bit-identical second-order
MUSCL across partition boundaries, use `halo = 2`.**  `halo = 1`
gives first-order at the partition boundary.

### Running

Single-process dev (fake devices via XLA):

```bash
XLA_FLAGS="--xla_force_host_platform_device_count=4" python myscript.py
```

Multi-process cluster (SLURM auto-detection):

```python
import jax
jax.distributed.initialize()  # once per process at startup
# Then proceed as above; jax.devices() lists this process's slice
# and shard_map collectives use NCCL / TPU-ICI across hosts.
```

### Validation suite

24 bit-identity tests in `tests/unit/zoomy_jax/`:

| File                                | What it pins                       |
|-------------------------------------|------------------------------------|
| `test_halo_exchange.py`             | `lax.ppermute` halo fill + zero-at-boundary |
| `test_partition_jax.py`             | Padded slab shapes + LSQ remap     |
| `test_spmd_advection.py`            | Upwind advection bit-identity, 2/4 dev × 16/32 cells |
| `test_spmd_solver_integration.py`   | `ConstantReconstruction` composes with SPMD |
| `test_spmd_lsq_muscl.py`            | LSQ MUSCL reconstruction composes (inter-partition + replicated) |
| `test_spmd_hyperbolic_solver.py`    | `HyperbolicSolver.get_flux_operator` composes (order=1 + order=2) |
| `test_spmd_rk2_step.py`             | SSP-RK2 step with per-stage halo exchange |
| `test_spmd_parameter_mutation.py`   | Parameter changes propagate (not constant-folded) — unblocks param sweeps + `jax.grad` |
| `test_spmd_xaxis_structured.py`     | 2D + 3D x-axis SPMD (order=1 + order=2) bit-identity |

Run all with:

```bash
XLA_FLAGS="--xla_force_host_platform_device_count=4" \
    pytest tests/unit/zoomy_jax/test_*spmd*.py tests/unit/zoomy_jax/test_halo_*.py \
           tests/unit/zoomy_jax/test_partition_*.py -q
```
