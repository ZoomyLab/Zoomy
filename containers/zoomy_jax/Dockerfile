# Release/test image: conda JAX stack + Zoomy libs from this repo (zoomy_jax is not necessarily on PyPI).
ARG JAX_DEV_IMAGE=ghcr.io/zoomylab/zoomy_jax_dev:latest
FROM ${JAX_DEV_IMAGE}

SHELL ["/bin/bash", "-c"]
ENV PATH=/opt/conda/envs/zoomy/bin:$PATH
ENV CONDA_PREFIX=/opt/conda/envs/zoomy

COPY library/zoomy_core /tmp/zoomy_src/zoomy_core
COPY library/zoomy_jax /tmp/zoomy_src/zoomy_jax

RUN /opt/conda/envs/zoomy/bin/python -m pip install --no-cache-dir \
      /tmp/zoomy_src/zoomy_core \
      /tmp/zoomy_src/zoomy_jax \
      pytest-html \
      jupytext \
      nbformat \
      nbclient \
    && rm -rf /tmp/zoomy_src

CMD ["/bin/bash"]
