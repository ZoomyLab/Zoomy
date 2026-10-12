# source env.sh: the environment of every matrix command (README.md)
L=$HOME/git/Zoomy-solver-template/library
export PYTHONPATH=$L/zoomy_core:$L/zoomy_jax:$L/zoomy_foam:$L/zoomy_amrex
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export ZOOMY_AMREX_BUILD_CPUS=0-7 ZOOMY_AMREX_RUN_CPUS=0-7
export OMP_NUM_THREADS=1
alias mx='timeout 295 taskset -c 0-7 micromamba run -n zoomy python matrix.py'
export ZOOMY_AMREX_SYMCACHE=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/work/amrex_cache
export MATRIX_BUDGET=270     # stop before a run that would not fit the 295 s command
