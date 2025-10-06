#!/bin/bash
mkdir -p $CONDA_PREFIX/etc/micromamba/activate.d

echo 'export ZOOMY_DIR="${PWD}"' > $CONDA_PREFIX/etc/micromamba/activate.d/env_vars.sh
echo 'export PYTHONPATH=":${PWD}"' >> $CONDA_PREFIX/etc/micromamba/activate.d/env_vars.sh
echo 'export JAX_ENABLE_X64=True' >> $CONDA_PREFIX/etc/micromamba/activate.d/env_vars.sh
export CPATH="$CONDA_PREFIX/include:$CPATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"




