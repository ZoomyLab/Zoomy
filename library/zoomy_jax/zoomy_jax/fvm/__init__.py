"""Module `zoomy_jax.fvm`."""

from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax

# SPMD (multi-device) parallelisation — accessible entry points.  The machinery
# is verified bit-identical to single-device by tests/unit/zoomy_jax/test_spmd_*.
from zoomy_jax.fvm.halo_exchange_jax import (
    halo_exchange_inplace,
    halo_exchange_owned_first,
)
from zoomy_jax.fvm.spmd_jax import (
    spmd_device_mesh,
    shard_global_state,
    gather_owned,
    build_sharded_flux_run,
    halo_exchange_periodic,
    run_solver_sharded,
)

__all__ = [
    "IMEXSourceSolverJax",
    "halo_exchange_inplace",
    "halo_exchange_owned_first",
    "spmd_device_mesh",
    "shard_global_state",
    "gather_owned",
    "build_sharded_flux_run",
    "halo_exchange_periodic",
    "run_solver_sharded",
]
