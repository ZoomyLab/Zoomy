"""One GPU per MPI rank, bound before anything loads CUDA.

    mpiexec -n 4 python -m zoomy_jax.gpurank run.py [args ...]

Sets ``CUDA_VISIBLE_DEVICES`` from the node-local rank (round robin over the
node's GPUs), then runs the script.  Every CUDA user in the process -- JAX,
AmgX, mpi4jax's copies -- sees only that GPU.  A launcher that binds GPUs
itself (``srun --gpus-per-task=1``, an existing ``CUDA_VISIBLE_DEVICES``) is
left alone.  Binding later, inside the run, is too late for libraries that
use the CUDA runtime directly: ``import jax`` already loads CUDA.
"""
import os
import runpy
import sys

_LOCAL_RANK = ("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID",
               "SLURM_LOCALID", "PMI_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK")


def bind():
    """Set ``CUDA_VISIBLE_DEVICES`` for this rank; True if it did."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return False
    local = next((int(os.environ[k]) for k in _LOCAL_RANK if k in os.environ), None)
    try:
        n_gpu = len(os.listdir("/proc/driver/nvidia/gpus"))
    except OSError:
        return False
    if local is None or n_gpu < 1:
        return False
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local % n_gpu)
    return True


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    bind()
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
