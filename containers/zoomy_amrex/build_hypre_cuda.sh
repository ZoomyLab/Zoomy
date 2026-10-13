#!/bin/bash
# Build hypre (static libHYPRE.a) inside the AMReX apptainer, for the AMReX symbolic solvers'
# pressure solve (USE_HYPRE=TRUE; zoomy_amrex.symsolver links it).
#
#   build_hypre_cuda.sh          # CUDA build in zoomy_amrex_gpu.sif -> ~/.cache/zoomy/hypre-cuda-3.2.0
#   build_hypre_cuda.sh cpu      # CPU build in zoomy_amrex.sif      -> ~/.cache/zoomy/hypre-cpu-3.2.0
#
# The CPU build also links into foam's symFoam (zoomy_foam.symsolver, linear_solver='hypre'): the
# OF13 image has the same Ubuntu 24.04, gcc 13.3 and OpenMPI 4.1.6 as zoomy_amrex.sif.
#
# Env: HYPRE_VERSION (3.2.0), CUDA_ARCH (89), ZOOMY_HYPRE_CACHE (~/.cache/zoomy),
# ZOOMY_AMREX_CONTAINERS (the .sif directory), ZOOMY_AMREX_BUILD_CPUS (16-47), JOBS (16).
# Installs into a temporary prefix and renames it on success: an interrupted build leaves no
# half-installed tree.  hypre: Falgout, Yang 2002, https://doi.org/10.1007/3-540-47789-6_66
set -euo pipefail
VARIANT=${1:-cuda}
case "$VARIANT" in cuda|cpu) ;; *) echo "usage: $0 [cuda|cpu]" >&2; exit 2 ;; esac
VER=${HYPRE_VERSION:-3.2.0}
CACHE=${ZOOMY_HYPRE_CACHE:-$HOME/.cache/zoomy}
PREFIX=$CACHE/hypre-$VARIANT-$VER
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -n "${ZOOMY_AMREX_CONTAINERS:-}" ]; then SIFDIR=$ZOOMY_AMREX_CONTAINERS
elif [ -f "$HERE/zoomy_amrex.sif" ]; then SIFDIR=$HERE
else SIFDIR=$HOME/git/Zoomy/containers/zoomy_amrex; fi
if [ "$VARIANT" = cuda ]; then SIF=$SIFDIR/zoomy_amrex_gpu.sif; else SIF=$SIFDIR/zoomy_amrex.sif; fi
CPUS=${ZOOMY_AMREX_BUILD_CPUS:-16-47}
JOBS=${JOBS:-16}

if [ -f "$PREFIX/lib/libHYPRE.a" ]; then echo "$PREFIX exists"; exit 0; fi
mkdir -p "$CACHE"
TAR=$CACHE/hypre-$VER.tar.gz
if [ ! -f "$TAR" ]; then
    curl -fsSL "https://github.com/hypre-space/hypre/archive/refs/tags/v$VER.tar.gz" -o "$TAR.part"
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
else
    CONF=""
    NV=""
fi
apptainer exec --cleanenv $NV --bind "$CACHE" "$SIF" bash -c "
    set -e
    cd $WORK/hypre-$VER/src
    taskset -c $CPUS ./configure --prefix=$WORK/install --disable-fortran $CONF \
        CC=mpicc CXX=mpicxx > $WORK/configure.log 2>&1 || { tail -30 $WORK/configure.log; exit 1; }
    taskset -c $CPUS make -j$JOBS install > $WORK/make.log 2>&1 || { tail -30 $WORK/make.log; exit 1; }"
rm -rf "$PREFIX"
mv "$WORK/install" "$PREFIX"
echo "hypre $VER ($VARIANT) -> $PREFIX"
