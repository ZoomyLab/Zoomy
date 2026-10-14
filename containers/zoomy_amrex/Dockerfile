FROM ubuntu:24.04

LABEL org.opencontainers.image.description="AMReX development container for Zoomy (CPU, and CUDA with a GPU)"

ENV DEBIAN_FRONTEND=noninteractive \
    AMREX_HOME=/opt/amrex \
    ZOOMY_AMREX_HOME=/opt/amrex \
    LC_ALL=C \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        git \
        ca-certificates \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        pkg-config \
        libopenmpi-dev \
        openmpi-bin \
        libhdf5-openmpi-dev \
        wget \
    && rm -rf /var/lib/apt/lists/*

# One image for CPU and GPU machines (as zoomy_amrex.def): CUDA 12.6 from NVIDIA's apt
# repository, nvcc and the libraries the CUDA build links (cudart, curand, cusparse: AMReX;
# cublas, cusolver: hypre-CUDA; nvtx: TINY_PROFILE; the driver stub libcuda.so for the link,
# the real one comes with `docker run --gpus all` / `apptainer --nv`), shared libraries only:
# the static archives (1.5 GB) are deleted in the same layer, but cudart's (nvcc links it).
# The generated solver is compiled for the CPU or with CUDA as the run asks
# (zoomy_amrex.symsolver `gpu=`).
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
        -O /root/cuda-keyring.deb \
    && dpkg -i /root/cuda-keyring.deb && rm -f /root/cuda-keyring.deb \
    && apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-12-6 cuda-cudart-dev-12-6 cuda-driver-dev-12-6 cuda-cccl-12-6 \
        cuda-nvtx-12-6 cuda-profiler-api-12-6 libcurand-dev-12-6 libcusparse-dev-12-6 \
        libcublas-dev-12-6 libcusolver-dev-12-6 \
    && rm -rf /var/lib/apt/lists/* \
    && find /usr/local/cuda-12.6 -name 'lib*_static*.a' ! -name libcudart_static.a -delete \
    && ln -sfn /usr/local/cuda-12.6 /usr/local/cuda \
    && nvcc --version | tail -1

RUN python3 -m pip install --no-cache-dir --break-system-packages \
        numpy scipy sympy attrs meshio pytest rasterio h5py

# fastapi/uvicorn = the server; jupyterlab/jupyter-server/ipykernel back
# zoomy-entry's `jupyter` mode, which execs `zoomy-jupyter` -> `jupyter lab`.
# Without them the image prints Jupyter's help text and exits (the .def had
# them, the Dockerfile did not, and the published .sif is converted from the
# Docker image).
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        "fastapi>=0.100" "uvicorn[standard]>=0.20" \
        jupyterlab jupyter-server ipykernel

# preCICE 3.4.0 for the coupled AMReX participant (zoomy_amrex symsolver couple=...), as in
# zoomy_amrex.def: the published .sif is converted from THIS image, so without it no AMReX
# coupling runs.  3.4.0 = the Python peers' version (preCICE refuses to couple different ones).
RUN wget -q "https://github.com/precice/precice/releases/download/v3.4.0/libprecice3_3.4.0_noble.deb" \
        -O /root/libprecice3.deb \
    && apt-get update && apt-get install -y --no-install-recommends /root/libprecice3.deb \
    && rm -f /root/libprecice3.deb && rm -rf /var/lib/apt/lists/* \
    && pkg-config --modversion libprecice

RUN git clone --depth 1 --branch 25.02 \
        https://github.com/AMReX-Codes/amrex.git /opt/amrex \
    && find /opt/amrex -name "*.o" -delete 2>/dev/null || true

COPY library/zoomy_core /tmp/src/zoomy_core
COPY library/zoomy_server /tmp/src/zoomy_server
COPY library/zoomy_prepost /tmp/src/zoomy_prepost
COPY library/zoomy_plotting /tmp/src/zoomy_plotting
# zoomy_amrex EDITABLE from a persistent tree: run_case resolves the C++
# templates as <pkg>/../Source (repo layout) — a plain pip install ships only
# the python package and codegen dies with ".../dist-packages/Source missing".
COPY library/zoomy_amrex /opt/src/zoomy_amrex
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        /tmp/src/zoomy_core /tmp/src/zoomy_server \
        /tmp/src/zoomy_prepost /tmp/src/zoomy_plotting \
        -e /opt/src/zoomy_amrex \
    && rm -rf /tmp/src

ENV ZOOMY_ROOT=/workspace \
    ZOOMY_AMREX_IN_IMAGE=1 \
    ZOOMY_ADAPTER=amrex \
    ZOOMY_PORT=8080

COPY containers/common/zoomy-entry /usr/local/bin/zoomy-entry
RUN chmod +x /usr/local/bin/zoomy-entry

WORKDIR /workspace
EXPOSE 8080
# ENTRYPOINT (not bare CMD): survives the pipeline's docker->sif conversion as
# the runscript, so `apptainer run img [port|jupyter|shell]` keeps working.
ENTRYPOINT ["/usr/local/bin/zoomy-entry"]
CMD []
