# Release/test image: conda JAX stack + Zoomy libs from this repo (zoomy_jax is not necessarily on PyPI).
ARG JAX_DEV_IMAGE=ghcr.io/zoomylab/zoomy_jax_dev:latest
FROM ${JAX_DEV_IMAGE}

SHELL ["/bin/bash", "-c"]
ENV PATH=/opt/conda/envs/zoomy/bin:$PATH
ENV CONDA_PREFIX=/opt/conda/envs/zoomy

COPY library/zoomy_core     /tmp/zoomy_src/zoomy_core
COPY library/zoomy_jax      /tmp/zoomy_src/zoomy_jax
COPY library/zoomy_server   /tmp/zoomy_src/zoomy_server
COPY library/zoomy_prepost  /tmp/zoomy_src/zoomy_prepost
COPY library/zoomy_plotting /tmp/zoomy_src/zoomy_plotting

RUN /opt/conda/envs/zoomy/bin/python -m pip install --no-cache-dir \
      /tmp/zoomy_src/zoomy_core \
      /tmp/zoomy_src/zoomy_jax \
      /tmp/zoomy_src/zoomy_server \
      /tmp/zoomy_src/zoomy_prepost \
      /tmp/zoomy_src/zoomy_plotting \
      pytest-html \
      jupytext \
      nbformat \
      nbclient \
      jupyterlab \
      jupyter-server \
      ipykernel \
    && rm -rf /tmp/zoomy_src

# MPI for distributed runs (`comm=`; the matrix's mpiN / gpu2 configs): OpenMPI 5 and mpi4py
# from conda-forge (mpi4jax: below). python / numpy / scipy / jax / jaxlib pinned to the
# installed builds (--freeze-installed cannot solve: OpenMPI needs another libhwloc).
# GPU -> GPU MPI and AmgX: containers/zoomy_jax_mpi.
# preCICE (pyprecice 3.4.0 + conda-forge's libprecice 3.4.0) for coupling: the version of the
# other participants (the AMReX and OpenFOAM images' libprecice); preCICE refuses to couple
# different versions. In the MPI solve: conda-forge's precice comes in mpich and openmpi
# variants, and this picks the openmpi one. petsc4py (the PETSc precice links): the JAX
# participant (`runtime.PreciceAdapter`) imports it before precice. The import fails the
# build, not a coupled run.
RUN pins=$(/opt/conda/bin/conda list -n zoomy --export '^(python|numpy|scipy|jax|jaxlib)$' \
           | grep -v '^#') \
    && /opt/conda/bin/mamba install -n zoomy -y -c conda-forge $pins \
      "openmpi>=5,<6" mpi4py "pyprecice=3.4.0" petsc4py \
    && /opt/conda/bin/conda clean -afy \
    && /opt/conda/envs/zoomy/bin/python -c "import petsc4py, precice; v = precice.get_version_information().split(b';')[0]; assert v == b'3.4.0', v; print('preCICE', v.decode())"

# One image for CPU and GPU machines: JAX's CUDA 12 plugin for the installed jaxlib 0.9.2 with
# NVIDIA's CUDA libraries as pip wheels (a GPU run needs only the host's driver:
# `docker run --gpus all`, `apptainer --nv`). conda-forge's CPU jaxlib stays, so without a GPU
# (or with JAX_PLATFORMS=cpu) JAX runs exactly as before. mpi4jax 0.9.1.post2 is built here
# with its CUDA bridge (conda-forge's is CPU-only), against this OpenMPI and the wheels'
# cudart; gcc and nanobind only for that build.
# Versions: cuda-pins.txt (the zoomy-cuda env the GPU matrix ran in).
COPY containers/zoomy_jax/cuda-pins.txt /opt/cuda-pins.txt
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ \
    && PIP_CONSTRAINT=/opt/cuda-pins.txt /opt/conda/envs/zoomy/bin/python -m pip install \
      --no-cache-dir "jax-cuda12-plugin[with-cuda]" \
    && /opt/conda/envs/zoomy/bin/python -m pip install --no-cache-dir nanobind \
    && OMPI_CC=gcc OMPI_CXX=g++ PIP_CONSTRAINT=/opt/cuda-pins.txt /opt/conda/envs/zoomy/bin/python -m pip \
      install --no-cache-dir --no-build-isolation --no-deps "mpi4jax==0.9.1.post2" \
    && /opt/conda/envs/zoomy/bin/python -m pip uninstall -y nanobind \
    && apt-get purge -y --auto-remove gcc g++ && rm -rf /var/lib/apt/lists/* /opt/cuda-pins.txt \
    && /opt/conda/envs/zoomy/bin/python -c "import glob, mpi4jax, os; d = os.path.dirname(mpi4jax.__file__); assert glob.glob(d + '/_src/xla_bridge/mpi_xla_bridge_cuda*.so'), 'mpi4jax without CUDA'; import jax_cuda12_plugin, jaxlib; print('jaxlib', jaxlib.__version__, '+ CUDA plugin, mpi4jax CUDA bridge')"

# Server env + triple-purpose dispatch (parity with zoomy_jax.def): the GUI's
# jax backend runs `apptainer run img [port]` on this image too.
ENV ZOOMY_ADAPTER=jax \
    ZOOMY_PORT=8080 \
    ZOOMY_ROOT=/workspace \
    MPLBACKEND=Agg \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    MPI4JAX_USE_CUDA_MPI=0 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
COPY containers/common/zoomy-entry /usr/local/bin/zoomy-entry
RUN chmod +x /usr/local/bin/zoomy-entry && mkdir -p /workspace

EXPOSE 8080 8888
ENTRYPOINT ["/usr/local/bin/zoomy-entry"]
CMD []
