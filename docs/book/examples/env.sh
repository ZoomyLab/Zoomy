# source docs/book/examples/env.sh: the libraries of this checkout, CPU JAX in double precision
L=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../../library" && pwd)
export PYTHONPATH=$L/zoomy_core:$L/zoomy_jax:$L/zoomy_foam:$L/zoomy_amrex
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
