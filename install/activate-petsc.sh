#!/bin/bash

# --- 1. Define Paths ---
# We use the path established during your successful build
export PETSC_DIR="/home/is086873/Git/Zoomy/petsc-3.24"
export PETSC_ARCH="arch-linux-gcc-opt"

# --- 2. Load Required Modules ---
# Since we "downloaded all", we only need the base toolchain 
# and Python to satisfy the runtime links (MPI, etc.)
module purge
module load foss
module load Python/3.12.3
module load CMake

source ./.venv/bin/activate

# --- 3. Update System Paths ---
# This ensures that any helper tools built by PETSc are in your PATH
export PATH="$PETSC_DIR/$PETSC_ARCH/bin:$PATH"
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:$PKG_CONFIG_PATH"

# --- 4. Visual Confirmation ---
echo "--------------------------------------------------------"
echo "PETSc Environment Activated"
echo "PETSC_DIR:  $PETSC_DIR"
echo "PETSC_ARCH: $PETSC_ARCH"
echo "Toolchain:  foss (GCC/OpenMPI)"
echo "--------------------------------------------------------"

