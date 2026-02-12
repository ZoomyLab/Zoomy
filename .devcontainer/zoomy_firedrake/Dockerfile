FROM ghcr.io/zoomylab/zoomy_firedrake:latest

# 1. Install system-level X11 libs
RUN apt-get update && apt-get install -y \
    libx11-6 libxext6 libxt6 libxrender1 libxft2 libxpm4 \
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*

# 2. Install PyYAML. We rely on the PATH to find the right python.
RUN python3 -m pip install PyYAML

# 3. Setup workspace structure for the build
WORKDIR /tmp/build
COPY .devcontainer/install_conda_with_pip.py .
COPY install/Zoomy.yml install/Zoomy.yml

# 4. Run the helper script
RUN python3 install_conda_with_pip.py