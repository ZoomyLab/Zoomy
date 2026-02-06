#!/bin/bash

# --- 1. Path Setup ---
TARGET_DIR="${1:-$(pwd)}"
INSTALL_PATH="$TARGET_DIR/petsc-3.24"

echo "Installing PETSc to: $INSTALL_PATH"

# --- 2. Minimal Module Environment ---
# We ONLY load the base toolchain. No HDF5/MUMPS/SuperLU modules.
module purge
module load foss
module load Python/3.12.3
module load CMake  # Required for several external packages

# --- 3. Source Retrieval ---
if [ ! -d "$INSTALL_PATH" ]; then
    git clone -b v3.24.0 https://gitlab.com/petsc/petsc.git "$INSTALL_PATH"
fi
cd "$INSTALL_PATH" || exit

# --- 4. PETSc Configuration ---
export PETSC_DIR=$(pwd)
export PETSC_ARCH=arch-linux-gcc-opt

# Deep clean
rm -rf "$PETSC_ARCH"
rm -f configure.log

echo "Running PETSc Configure (Full Download Mode)..."

python3 ./configure \
  --with-cc=mpicc \
  --with-cxx=mpicxx \
  --with-fc=mpif90 \
  COPTFLAGS="-O3 -march=native" \
  CXXOPTFLAGS="-O3 -march=native" \
  FOPTFLAGS="-O3 -march=native" \
  --with-debugging=0 \
  --with-shared-libraries=1 \
  --download-netlib-lapack \
  --with-netlib-lapack-c-bindings \
  --download-fblaslapack \
  --download-scalapack \
  --download-mumps \
  --download-hdf5 \
  --download-superlu_dist \
  --download-hwloc \
  --download-eigen \
  --download-metis \
  --download-parmetis \
  --download-pastix \
  --download-ptscotch \
  --download-parmmg \
  --download-mmg \
  --download-pragmatic \
  --download-bison \
  --with-fortran-bindings=0 \
  --with-python-bindings=0

# --- 5. Build ---
if [ $? -eq 0 ]; then
    echo "Configuration successful. Starting build..."
    make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all
    make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH check
else
    echo "--------------------------------------------------------"
    echo "ERROR: Configure failed. Check $(pwd)/configure.log"
    echo "--------------------------------------------------------"
    exit 1
fi
