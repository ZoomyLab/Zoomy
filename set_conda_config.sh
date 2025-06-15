mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'export SMS="${PWD}"' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export PYTHONPATH=":${PWD}"' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export JAX_ENABLE_X64=True' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh



