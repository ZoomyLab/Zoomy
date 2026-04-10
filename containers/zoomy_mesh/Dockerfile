# Lightweight mesh converter: gmsh + meshio + zoomy_core (no PETSc, no JAX).
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.description="Zoomy mesh converter service"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# gmsh CLI (the Debian package also pulls in the shared lib).
RUN apt-get update && apt-get install -y --no-install-recommends \
        gmsh \
        libglu1-mesa \
        libxrender1 \
        libxcursor1 \
        libxinerama1 \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps: gmsh Python API, meshio, and the core Zoomy stack.
COPY library/zoomy_core  /tmp/src/zoomy_core
COPY library/zoomy_mesh  /tmp/src/zoomy_mesh
COPY library/zoomy_server /tmp/src/zoomy_server

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
        /tmp/src/zoomy_core \
        /tmp/src/zoomy_mesh \
        /tmp/src/zoomy_server \
        gmsh \
        meshio \
    && rm -rf /tmp/src

WORKDIR /workspace
EXPOSE 8080
CMD ["zoomy-server", "--adapter", "mesh", "--port", "8080"]
