FROM ubuntu:latest

SHELL ["/bin/bash", "-c"]

# ---------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    curl bzip2 vim wget git ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------
# 2. Install micromamba
# ---------------------------------------------------------------------

RUN mkdir -p /opt/micromamba/bin && \
    curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj --strip-components=1 -C /opt/micromamba/bin bin/micromamba

ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

RUN echo ". /opt/micromamba/etc/profile.d/micromamba.sh" >> /etc/bash.bashrc && \
    echo "micromamba activate zoomy" >> /etc/bash.bashrc


# Copy environment files into the image
COPY install/zoomy.yml /tmp/zoomy.yml
COPY install/env-jax.yml /tmp/env-jax.yml
COPY install/env-mesh.yml /tmp/env-mesh.yml

# ---------------------------------------------------------------------
# 3. Create and update the environment (zoomy)
# ---------------------------------------------------------------------
RUN micromamba create -y -n zoomy -f /tmp/zoomy.yml && \
    micromamba env update -n zoomy -f /tmp/env-jax.yml && \
    micromamba env update -n zoomy -f /tmp/env-mesh.yml

# ---------------------------------------------------------------------
# 4. Auto-activate environment on container start
# ---------------------------------------------------------------------
# Add activation to bashrc
RUN echo "micromamba activate zoomy" >> ~/.bashrc

# For interactive shells, ensure non-login shells also activate
ENV MAMBA_DEFAULT_ENV=zoomy
ENV MAMBA_EXE="$MAMBA_ROOT_PREFIX/bin/micromamba"

# Default command
CMD ["/bin/bash"]
