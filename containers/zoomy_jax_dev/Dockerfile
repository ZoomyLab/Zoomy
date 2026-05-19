#########################################
# JAX dev image: miniforge + mamba env from install/*.yml (JAX + jraph GNN stack via zoomy_jax.yml).
# No Zoomy packages — use zoomy_jax image or pip install -e in devcontainer.
#########################################

FROM ubuntu:22.04 AS builder
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl bzip2 ca-certificates git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -L -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:$PATH

COPY install/Zoomy.yml /tmp/Zoomy.yml
COPY install/zoomy_jax.yml /tmp/zoomy_jax.yml

RUN mamba env create -y -n zoomy \
        -f /tmp/Zoomy.yml \
        -f /tmp/zoomy_jax.yml && \
    conda clean -afy

FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    bash ca-certificates \
    libgfortran5 libstdc++6 libc6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/conda /opt/conda

ENV PATH=/opt/conda/envs/zoomy/bin:$PATH
ENV CONDA_PREFIX=/opt/conda/envs/zoomy

CMD ["/bin/bash"]
