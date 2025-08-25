#!/bin/bash
mkdir -p $CONDA_PREFIX/etc/micromamba/activate.d

echo 'export ZOOMY_DIR="${PWD}"' > $CONDA_PREFIX/etc/micromamba/activate.d/env_vars.sh
echo 'export PYTHONPATH=":${PWD}"' >> $CONDA_PREFIX/etc/micromamba/activate.d/env_vars.sh
echo 'export JAX_ENABLE_X64=True' >> $CONDA_PREFIX/etc/micromamba/activate.d/env_vars.sh



