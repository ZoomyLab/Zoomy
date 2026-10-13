"""Module `zoomy_jax.mesh` — SPMD mesh partitioning (accessible entry points).

Partitioners chop a global mesh into per-device padded slabs for the
`shard_map` + halo-exchange SPMD path (see `zoomy_jax.fvm.spmd_jax`),
verified by `tests/unit/zoomy_jax/test_spmd_*` / `test_partition_jax`.
"""

from zoomy_jax.mesh.partition_jax import (
    partition_1d_contiguous,
    partition_xaxis_structured,
)
from zoomy_jax.mesh.partition import (
    PartitionInfo,
    partition_mesh,
    extract_local_mesh,
)

__all__ = [
    "partition_1d_contiguous",
    "partition_xaxis_structured",
    "PartitionInfo",
    "partition_mesh",
    "extract_local_mesh",
]
