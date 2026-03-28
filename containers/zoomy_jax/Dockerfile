# Release/test image: conda JAX stack + zoomy_core / zoomy_jax from PyPI.
ARG JAX_DEV_IMAGE=ghcr.io/zoomylab/zoomy_jax_dev:latest
FROM ${JAX_DEV_IMAGE}

SHELL ["/bin/bash", "-c"]
ENV PATH=/opt/zoomy/envs/zoomy/bin:$PATH
ENV CONDA_PREFIX=/opt/zoomy/envs/zoomy

RUN /opt/zoomy/envs/zoomy/bin/python -m pip install --upgrade pip && \
    /opt/zoomy/envs/zoomy/bin/python -m pip install --no-cache-dir \
      zoomy_core \
      zoomy_jax \
      pytest-html \
      jupytext \
      nbformat \
      nbclient

CMD ["/bin/bash"]
