# Slim CI image: released zoomy_core from PyPI + pytest reporting stack.
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
      zoomy_core \
      pytest \
      pytest-html \
      nbformat \
      nbclient \
      jupytext

WORKDIR /workspace
CMD ["/bin/bash"]
