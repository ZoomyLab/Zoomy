# Placeholder until dolfinx/FEniCSx is wired into CI.
FROM python:3.11-slim-bookworm
LABEL org.opencontainers.image.description="Placeholder for FEniCSx CI; extend with conda/dolfinx when tests exist."

ENV PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir \
        pytest pytest-html nbformat nbclient jupytext zoomy_core

WORKDIR /workspace
CMD ["/bin/bash"]
