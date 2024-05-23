#!/bin/zsh
conda activate sms
export SMS=$(pwd)
export PYTHONPATH=${PYTHONPATH}:$(pwd)
export PETSC_DIR=/home/ingo/Git/petsc/.venv/bin/python
export PETSC_ARCH=linux-gnu