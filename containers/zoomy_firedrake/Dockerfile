# CI image: Firedrake dev base + Zoomy wheels from PyPI when published.
ARG FD_DEV_IMAGE=ghcr.io/zoomylab/zoomy_firedrake_dev:latest
FROM ${FD_DEV_IMAGE}

# Do not `pip install --upgrade pip` here: Firedrake uses Debian's python3-pip (no RECORD), which breaks uninstall.
RUN python3 -m pip install --no-cache-dir \
      zoomy_core \
      meshio \
      pytest-html \
      jupytext \
      nbformat \
      nbclient

# Optional until published to PyPI:
RUN python3 -m pip install --no-cache-dir zoomy_firedrake firedrake_animate zoomy_tests 2>/dev/null || true

CMD ["/bin/bash"]
