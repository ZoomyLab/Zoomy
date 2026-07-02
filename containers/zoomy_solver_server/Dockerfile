FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY library/zoomy_core /tmp/src/zoomy_core
COPY library/zoomy_server /tmp/src/zoomy_server

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
      h5py meshio \
      /tmp/src/zoomy_core \
      /tmp/src/zoomy_server \
    && rm -rf /tmp/src

EXPOSE 8080
CMD ["zoomy-server", "--adapter", "numpy", "--port", "8080"]
