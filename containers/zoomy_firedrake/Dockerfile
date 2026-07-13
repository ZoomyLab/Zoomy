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

# zoomy_firedrake / firedrake_animate / zoomy_tests are NOT on PyPI, so install
# them from source. The build context is the repo ROOT and the Containers
# workflow checks out `submodules: recursive`, so these submodule sources are
# present in the context. (Was: an optional `pip install … || true` that silently
# no-op'd, leaving `import zoomy_firedrake` broken in the image.)
COPY library/zoomy_firedrake  /tmp/pkgs/zoomy_firedrake
COPY library/firedrake_animate /tmp/pkgs/firedrake_animate
COPY library/zoomy_tests       /tmp/pkgs/zoomy_tests
RUN python3 -m pip install --no-cache-dir \
      /tmp/pkgs/zoomy_firedrake \
      /tmp/pkgs/firedrake_animate \
      /tmp/pkgs/zoomy_tests \
 && rm -rf /tmp/pkgs

# Server env + triple-purpose dispatch (parity with the backend .def images):
# the GUI's firedrake backend runs `apptainer run img [port]` on this image too.
COPY library/zoomy_server   /tmp/pkgs2/zoomy_server
COPY library/zoomy_prepost  /tmp/pkgs2/zoomy_prepost
COPY library/zoomy_plotting /tmp/pkgs2/zoomy_plotting
RUN python3 -m pip install --no-cache-dir \
      /tmp/pkgs2/zoomy_server \
      /tmp/pkgs2/zoomy_prepost \
      /tmp/pkgs2/zoomy_plotting \
 && rm -rf /tmp/pkgs2

ENV ZOOMY_ADAPTER=firedrake \
    ZOOMY_PORT=8080 \
    ZOOMY_ROOT=/workspace \
    MPLBACKEND=Agg
COPY containers/common/zoomy-entry /usr/local/bin/zoomy-entry
RUN chmod +x /usr/local/bin/zoomy-entry && mkdir -p /workspace

EXPOSE 8080 8888
# ENTRYPOINT (not bare CMD): survives the pipeline's docker->sif conversion as
# the runscript; arbitrary commands (CI `docker run img bash -lc ...`) pass through.
ENTRYPOINT ["/usr/local/bin/zoomy-entry"]
CMD []
