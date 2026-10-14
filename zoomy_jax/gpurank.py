"""One GPU per MPI rank, bound before anything loads CUDA.

    mpiexec -n 4 python -m zoomy_jax.gpurank run.py [args ...]

Sets `CUDA_VISIBLE_DEVICES` from the node-local rank (round robin over the
node's GPUs, or over the devices an existing `CUDA_VISIBLE_DEVICES` lists),
then runs the script.  Every CUDA user in the process -- JAX, AmgX, mpi4jax's
copies -- sees only that GPU.  A launcher that binds one GPU per rank itself
(`srun --gpus-per-task=1`: one device listed) is left alone.  Binding later,
inside the run, is too late for libraries that use the CUDA runtime directly:
`import jax` already loads CUDA.
"""
import os
import runpy
import sys

_LOCAL_RANK = ("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID",
               "SLURM_LOCALID", "PMI_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK")


def bind():
    """Set `CUDA_VISIBLE_DEVICES` for this rank; True if it did."""
    local = next((int(os.environ[k]) for k in _LOCAL_RANK if k in os.environ), None)
    listed = os.environ.get("CUDA_VISIBLE_DEVICES")
    if listed is not None:          # the devices this job may use: one per rank among them
        devices = [d.strip() for d in listed.split(",") if d.strip()]
        if local is None or len(devices) < 2:
            return False
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[local % len(devices)]
        return True
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
    # as `python script.py`: the script's directory first on sys.path (its sibling modules)
    sys.path[0] = os.path.dirname(os.path.abspath(sys.argv[0]))
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
