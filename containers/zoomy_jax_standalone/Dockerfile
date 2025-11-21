# ------------------------------------------------------------
# Base image: your published zoomy_jax environment
# ------------------------------------------------------------
FROM ghcr.io/zoomylab/zoomy_jax:latest
SHELL ["/bin/bash", "-c"]

# Make sure we use the micromamba/conda environment Python
ENV PATH=/opt/zoomy/envs/zoomy/bin:$PATH
ENV CONDA_PREFIX=/opt/zoomy/envs/zoomy

# ------------------------------------------------------------
# Copy and install local Zoomy Python packages
# ------------------------------------------------------------
# Adjust these paths to match your layout on the host
COPY library /tmp/library

# Install into the zoomy conda environment
RUN python -m pip install /tmp/library/zoomy_core && \
    python -m pip install /tmp/library/zoomy_jax && \
    rm -rf /tmp/library

# ------------------------------------------------------------
# Default entrypoint
# ------------------------------------------------------------
CMD ["/bin/bash"]

