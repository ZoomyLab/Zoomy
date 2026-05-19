#########################################
# JAX dev image: micromamba env from install/*.yml (JAX + jraph GNN stack via zoomy_jax.yml).
# No Zoomy packages — use zoomy_jax image or pip install -e in devcontainer.
#########################################

FROM ubuntu:22.04 AS builder
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl bzip2 ca-certificates git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://micro.mamba.pm/install.sh | bash
ENV PATH=/root/.local/bin:$PATH

ENV MAMBA_ROOT_PREFIX=/opt/zoomy
ARG MAMBA_NO_BASHRC=1

COPY install/Zoomy.yml /tmp/Zoomy.yml
COPY install/zoomy_jax.yml /tmp/zoomy_jax.yml

RUN micromamba create -y -n zoomy \
        -f /tmp/Zoomy.yml \
        -f /tmp/zoomy_jax.yml && \
    micromamba clean --all --yes

FROM ubuntu:22.04
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    bash ca-certificates \
    libgfortran5 libstdc++6 libc6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/zoomy /opt/zoomy

ENV PATH=/opt/zoomy/envs/zoomy/bin:$PATH
ENV CONDA_PREFIX=/opt/zoomy/envs/zoomy

CMD ["/bin/bash"]
