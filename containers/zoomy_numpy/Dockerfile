# Triple-purpose NumPy backend image (formerly zoomy_solver_server):
#   docker run ... -p 8080:8080            -> solver API (zoomy-server --adapter numpy)
#   docker run ... zoomy-jupyter --port .. -> JupyterLab with the backend env
# Mirrors ./zoomy_numpy.def — keep the two in sync.
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ZOOMY_ADAPTER=numpy \
    ZOOMY_PORT=8080 \
    ZOOMY_ROOT=/workspace

COPY library/zoomy_core    /tmp/src/zoomy_core
COPY library/zoomy_server  /tmp/src/zoomy_server
COPY library/zoomy_prepost /tmp/src/zoomy_prepost

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
      h5py meshio jupyterlab jupyter-server ipykernel jupytext nbformat \
      /tmp/src/zoomy_core \
      /tmp/src/zoomy_server \
      /tmp/src/zoomy_prepost \
    && rm -rf /tmp/src && mkdir -p /workspace

EXPOSE 8080 8888
CMD ["zoomy-server", "--adapter", "numpy", "--port", "8080"]
