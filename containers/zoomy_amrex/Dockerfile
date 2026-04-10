FROM ubuntu:24.04

LABEL org.opencontainers.image.description="AMReX development container for Zoomy"

ENV DEBIAN_FRONTEND=noninteractive \
    AMREX_HOME=/opt/amrex \
    ZOOMY_AMREX_HOME=/opt/amrex \
    LC_ALL=C

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

RUN python3 -m pip install --no-cache-dir --break-system-packages \
        numpy scipy sympy attrs meshio pytest rasterio h5py

RUN python3 -m pip install --no-cache-dir --break-system-packages \
        "fastapi>=0.100" "uvicorn[standard]>=0.20"

RUN git clone --depth 1 --branch 25.02 \
        https://github.com/AMReX-Codes/amrex.git /opt/amrex \
    && find /opt/amrex -name "*.o" -delete 2>/dev/null || true

COPY library/zoomy_core /tmp/src/zoomy_core
COPY library/zoomy_amrex /tmp/src/zoomy_amrex
COPY library/zoomy_server /tmp/src/zoomy_server
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        /tmp/src/zoomy_core /tmp/src/zoomy_amrex /tmp/src/zoomy_server \
    && rm -rf /tmp/src

ENV ZOOMY_ROOT=/workspace

WORKDIR /workspace
EXPOSE 8080
CMD ["zoomy-server", "--adapter", "amrex", "--port", "8080"]
