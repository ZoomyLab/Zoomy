#!/bin/bash
# Build hypre (static libHYPRE.a) inside the AMReX apptainer, for the AMReX symbolic solvers'
# pressure solve (USE_HYPRE=TRUE; zoomy_amrex.symsolver links it).
#
#   build_hypre_cuda.sh          # CUDA build in zoomy_amrex.sif (--nv) -> ~/.cache/zoomy/hypre-cuda-3.2.0
#   build_hypre_cuda.sh cpu      # CPU build in zoomy_amrex.sif        -> ~/.cache/zoomy/hypre-cpu-3.2.0
#   build_hypre_cuda.sh cpu-env  # CPU shared library against the micromamba env's MPI (no
#                                # container)                         -> ~/.cache/zoomy/hypre-cpu-env-3.2.0
# Run inside the image (APPTAINER_CONTAINER set, or no apptainer on PATH) it builds natively.
#
# The CPU build also links into foam's symFoam (zoomy_foam.symsolver, linear_solver='hypre'): the
# OF13 image has the same Ubuntu 24.04, gcc 13.3 and OpenMPI 4.1.6 as zoomy_amrex.sif.  The
# cpu-env build (libHYPRE.so) is JAX's (zoomy_jax.symsolver.hypre, ctypes): it must link the MPI
# mpi4py uses (the zoomy env's MPICH).
#
# Env: HYPRE_VERSION (3.2.0), CUDA_ARCH (89), ZOOMY_HYPRE_CACHE (~/.cache/zoomy),
# ZOOMY_AMREX_CONTAINERS (the .sif directory), ZOOMY_AMREX_BUILD_CPUS (16-47), JOBS (16),
# ZOOMY_ENV (zoomy: the cpu-env build's micromamba env).
# Installs into a temporary prefix and renames it on success: an interrupted build leaves no
# half-installed tree.  hypre: Falgout, Yang 2002, https://doi.org/10.1007/3-540-47789-6_66
set -euo pipefail
VARIANT=${1:-cuda}
case "$VARIANT" in cuda|cpu|cpu-env) ;; *) echo "usage: $0 [cuda|cpu|cpu-env]" >&2; exit 2 ;; esac
VER=${HYPRE_VERSION:-3.2.0}
CACHE=${ZOOMY_HYPRE_CACHE:-$HOME/.cache/zoomy}
PREFIX=$CACHE/hypre-$VARIANT-$VER
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -n "${ZOOMY_AMREX_CONTAINERS:-}" ]; then SIFDIR=$ZOOMY_AMREX_CONTAINERS
elif [ -f "$HERE/zoomy_amrex.sif" ]; then SIFDIR=$HERE
else SIFDIR=$HOME/git/Zoomy/containers/zoomy_amrex; fi
SIF=$SIFDIR/zoomy_amrex.sif          # one image: the CPU and the CUDA toolchain
CPUS=${ZOOMY_AMREX_BUILD_CPUS:-16-47}
JOBS=${JOBS:-16}

LIB=libHYPRE.a; [ "$VARIANT" = cpu-env ] && LIB=libHYPRE.so
if [ -f "$PREFIX/lib/$LIB" ]; then echo "$PREFIX exists"; exit 0; fi
mkdir -p "$CACHE"
TAR=$CACHE/hypre-$VER.tar.gz
if [ ! -f "$TAR" ]; then
    URL="https://github.com/hypre-space/hypre/archive/refs/tags/v$VER.tar.gz"
    if command -v curl > /dev/null; then curl -fsSL "$URL" -o "$TAR.part"
    else wget -q "$URL" -O "$TAR.part"; fi       # the images have wget, not curl
    mv "$TAR.part" "$TAR"
fi
WORK=$(mktemp -d "$CACHE/build-hypre-$VARIANT.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
tar -xzf "$TAR" -C "$WORK"
if [ "$VARIANT" = cuda ]; then
    # hypre 3.x enables Umpire for GPU builds by default: off (not in the image)
    CONF="--with-cuda --with-cuda-home=/usr/local/cuda --with-gpu-arch=${CUDA_ARCH:-89} \
          --without-umpire --enable-device-malloc-async"
    NV=--nv
elif [ "$VARIANT" = cpu-env ]; then
    CONF="--enable-shared"
    NV=""
else
    CONF=""
    NV=""
fi
BUILD="
    set -e
    cd $WORK/hypre-$VER/src
    taskset -c $CPUS ./configure --prefix=$WORK/install --disable-fortran $CONF \
        CC=mpicc CXX=mpicxx > $WORK/configure.log 2>&1 || { tail -30 $WORK/configure.log; exit 1; }
    taskset -c $CPUS make -j$JOBS install > $WORK/make.log 2>&1 || { tail -30 $WORK/make.log; exit 1; }"
if [ "$VARIANT" = cpu-env ]; then
    micromamba run -n "${ZOOMY_ENV:-zoomy}" bash -c "$BUILD"
elif [ -n "${APPTAINER_CONTAINER:-}" ] || ! command -v apptainer > /dev/null; then
    bash -c "$BUILD"            # already inside the image: build natively
else
    apptainer exec --cleanenv $NV --bind "$CACHE" "$SIF" bash -c "$BUILD"
fi
rm -rf "$PREFIX"
mv "$WORK/install" "$PREFIX"
echo "hypre $VER ($VARIANT) -> $PREFIX"
