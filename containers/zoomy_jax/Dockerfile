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
    && rm -rf /tmp/zoomy_src

# Server env + triple-purpose dispatch (parity with zoomy_jax.def): the GUI's
# jax backend runs `apptainer run img [port]` on this image too.
ENV ZOOMY_ADAPTER=jax \
    ZOOMY_PORT=8080 \
    ZOOMY_ROOT=/workspace \
    MPLBACKEND=Agg
COPY containers/common/zoomy-entry /usr/local/bin/zoomy-entry
RUN chmod +x /usr/local/bin/zoomy-entry && mkdir -p /workspace

EXPOSE 8080 8888
ENTRYPOINT ["/usr/local/bin/zoomy-entry"]
CMD []
