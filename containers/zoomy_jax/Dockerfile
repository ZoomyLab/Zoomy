#########################################
# Stage 1: Builder
#########################################

FROM ubuntu:22.04 AS builder
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl bzip2 ca-certificates git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------
# Install micromamba with a custom root prefix (/opt/zoomy)
# -------------------------------------------------------
RUN curl -Ls https://micro.mamba.pm/install.sh | bash
ENV PATH=/root/.local/bin:$PATH

# IMPORTANT: Install micromamba environment directly into /opt/zoomy
ENV MAMBA_ROOT_PREFIX=/opt/zoomy
ARG MAMBA_NO_BASHRC=1

# Copy environment YAMLs
COPY install/Zoomy.yml /tmp/Zoomy.yml
COPY install/zoomy_jax.yml /tmp/zoomy_jax.yml
COPY install/meshes.yml /tmp/meshes.yml

# Create the environment directly in /opt/zoomy/envs/zoomy
RUN micromamba create -y -n zoomy -f /tmp/Zoomy.yml && \
    micromamba env update -n zoomy -f /tmp/zoomy_jax.yml && \
    micromamba env update -n zoomy -f /tmp/meshes.yml && \
    micromamba clean --all --yes


#########################################
# Stage 2: Runtime Image
#########################################

FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime libs needed for scientific conda packages
RUN apt-get update && apt-get install -y \
    bash ca-certificates \
    libgfortran5 libstdc++6 libc6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the entire micromamba root (already in correct path)
COPY --from=builder /opt/zoomy /opt/zoomy

# Add the environment to PATH
ENV PATH=/opt/zoomy/envs/zoomy/bin:$PATH

# Set CONDA_PREFIX for safety (some packages check it)
ENV CONDA_PREFIX=/opt/zoomy/envs/zoomy

CMD ["/bin/bash"]
