# Prebuilt Zoomy Firedrake stack (PyPI zoomy_* on top of dev base); use zoomy_firedrake_dev for editable-only workflows.
FROM ghcr.io/zoomylab/zoomy_firedrake:latest

# Install minimal X11 libs so Firedrake's PETSc/Xlib detection works
RUN apt-get update && apt-get install -y \
    libx11-6 \
    libxext6 \
    libxt6 \
    libxrender1 \
    libxft2 \
    libxpm4 \
    && rm -rf /var/lib/apt/lists/*


