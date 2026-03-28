# CI image: Firedrake dev base + Zoomy wheels from PyPI when published.
ARG FD_DEV_IMAGE=ghcr.io/zoomylab/zoomy_firedrake_dev:latest
FROM ${FD_DEV_IMAGE}

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
      zoomy_core \
      pytest-html \
      jupytext \
      nbformat \
      nbclient

# Optional until published to PyPI:
RUN python3 -m pip install --no-cache-dir zoomy_firedrake firedrake_animate zoomy_tests 2>/dev/null || true

CMD ["/bin/bash"]
