#########################################
# Stage 1: Builder
#########################################

FROM ubuntu:22.04 AS builder

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    curl bzip2 ca-certificates git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install micromamba
RUN curl -Ls https://micro.mamba.pm/install.sh | bash
ENV PATH=/root/.local/bin:$PATH
ENV MAMBA_ROOT_PREFIX=/root/micromamba
ARG MAMBA_NO_BASHRC=1

# Copy env YAMLs
COPY install/Zoomy.yml /tmp/Zoomy.yml
COPY install/zoomy_jax.yml /tmp/zoomy_jax.yml
COPY install/meshes.yml /tmp/meshes.yml

# Create environment
RUN micromamba create -y -n zoomy -f /tmp/Zoomy.yml && \
    micromamba env update -n zoomy -f /tmp/zoomy_jax.yml && \
    micromamba env update -n zoomy -f /tmp/meshes.yml && \
    micromamba clean --all --yes


#########################################
# Stage 2: Runtime Image
#########################################

FROM debian:bookworm-slim

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    bash ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the environment folder directly
COPY --from=builder /root/micromamba/envs/zoomy /opt/zoomy

# Strip unnecessary files to reduce size
RUN find /opt/zoomy -type d -name "__pycache__" -exec rm -rf {} + || true && \
    find /opt/zoomy -type f -name "*.pyc" -delete && \
    find /opt/zoomy -type d -name "tests" -exec rm -rf {} + || true && \
    rm -rf /opt/zoomy/share || true && \
    rm -rf /opt/zoomy/doc || true && \
    rm -rf /opt/zoomy/lib/python*/site-packages/*/tests || true

# Add environment to PATH
ENV PATH=/opt/zoomy/bin:$PATH

CMD ["/bin/bash"]

