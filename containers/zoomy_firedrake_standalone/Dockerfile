# Use your prebuilt Firedrake+PETSc+Zoomy environment
FROM ghcr.io/zoomylab/zoomy_firedrake:latest

# Copy your local library folder into the image
COPY library /tmp/library

# Install your local packages
RUN python -m pip install /tmp/library/zoomy_core && \
    python -m pip install /tmp/library/zoomy_tests && \
    python -m pip install /tmp/library/firedrake_animate && \
    python -m pip install /tmp/library/zoomy_firedrake && \
    rm -rf /tmp/library

CMD ["/bin/bash"]
