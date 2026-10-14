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

# MPI on the CPU for distributed runs (`comm=`; the matrix's mpiN configs): OpenMPI 5,
# mpi4py and mpi4jax from conda-forge. python / numpy / scipy / jax / jaxlib pinned to the
# installed builds (--freeze-installed cannot solve: OpenMPI needs another libhwloc).
# GPU-aware MPI and AmgX: containers/zoomy_jax_mpi.
# preCICE (pyprecice 3.4.0 + conda-forge's libprecice 3.4.0) for coupling: the version of the
# other participants (the AMReX and OpenFOAM images' libprecice); preCICE refuses to couple
# different versions. In the MPI solve: conda-forge's precice comes in mpich and openmpi
# variants, and this picks the openmpi one. petsc4py (the PETSc precice links): the JAX
# participant (`runtime.PreciceAdapter`) imports it before precice. The import fails the
# build, not a coupled run.
RUN pins=$(/opt/conda/bin/conda list -n zoomy --export '^(python|numpy|scipy|jax|jaxlib)$' \
           | grep -v '^#') \
    && /opt/conda/bin/mamba install -n zoomy -y -c conda-forge $pins \
      "openmpi>=5,<6" mpi4py "mpi4jax=0.9.1.post2" "pyprecice=3.4.0" petsc4py \
    && /opt/conda/bin/conda clean -afy \
    && /opt/conda/envs/zoomy/bin/python -c "import petsc4py, precice; v = precice.get_version_information().split(b';')[0]; assert v == b'3.4.0', v; print('preCICE', v.decode())"

# Server env + triple-purpose dispatch (parity with zoomy_jax.def): the GUI's
# jax backend runs `apptainer run img [port]` on this image too.
ENV ZOOMY_ADAPTER=jax \
    ZOOMY_PORT=8080 \
    ZOOMY_ROOT=/workspace \
    MPLBACKEND=Agg \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
COPY containers/common/zoomy-entry /usr/local/bin/zoomy-entry
RUN chmod +x /usr/local/bin/zoomy-entry && mkdir -p /workspace

EXPOSE 8080 8888
ENTRYPOINT ["/usr/local/bin/zoomy-entry"]
CMD []
