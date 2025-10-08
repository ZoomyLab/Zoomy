#!/bin/bash
# Stop if any command fails
set -e

export HOME=$HOME
export PATH=/usr/bin:/bin

# Load OpenFOAM environment
source /home/ingo/Git/OpenFOAM/OpenFOAM-12/etc/bashrchome/ingo/OpenFOAM/ingo-12/applications/solvers/zoomyFoam


cd /home/ingo/Git/Zoomy/library/zoomyFoam
# Clean & build
wclean
wmake
cd -

zoomyFoam
