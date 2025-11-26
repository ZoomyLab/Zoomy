FROM ghcr.io/zoomylab/zoomy_firedrake_standalone:latest

# Install minimal X11 libs so Firedrake's PETSc/Xlib detection works
RUN apt-get update && apt-get install -y \
    libx11-6 \
    libxext6 \
    libxt6 \
    libxrender1 \
    libxft2 \
    libxpm4 \
    && rm -rf /var/lib/apt/lists/*


