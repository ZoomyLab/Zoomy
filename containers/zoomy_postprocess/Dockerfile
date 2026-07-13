# Post-processing backend (tag: postprocess): VTK<->H5, 2D->3D lift, visualize.py
# execution, zoomy_plotting. Mirrors ./zoomy_postprocess.def — keep in sync.
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ZOOMY_ADAPTER=postprocess \
    ZOOMY_PORT=8080 \
    ZOOMY_ROOT=/workspace \
    MPLBACKEND=Agg

COPY library/zoomy_core     /tmp/src/zoomy_core
COPY library/zoomy_plotting /tmp/src/zoomy_plotting
COPY library/zoomy_server   /tmp/src/zoomy_server
COPY library/zoomy_prepost  /tmp/src/zoomy_prepost

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
      h5py meshio matplotlib jupytext nbformat jupyterlab jupyter-server ipykernel \
      /tmp/src/zoomy_core /tmp/src/zoomy_plotting /tmp/src/zoomy_server /tmp/src/zoomy_prepost \
    && rm -rf /tmp/src && mkdir -p /workspace

EXPOSE 8080 8888
CMD ["zoomy-server", "--adapter", "postprocess", "--port", "8080"]
