FROM ubuntu:24.04

LABEL org.opencontainers.image.description="PETSc/DMPlex development container for Zoomy"

ENV DEBIAN_FRONTEND=noninteractive \
    PETSC_DIR=/opt/petsc \
    PETSC_ARCH=arch-container-opt \
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
        cmake \
        libopenmpi-dev \
        openmpi-bin \
        wget \
        flex \
        bison \
        valgrind \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --break-system-packages \
        numpy scipy sympy attrs "cython<3.1" pytest h5py meshio

RUN git clone --depth 1 --branch v3.22.3 \
        https://gitlab.com/petsc/petsc.git "$PETSC_DIR" \
    && cd "$PETSC_DIR" \
    && python3 ./configure \
        --with-cc=mpicc \
        --with-cxx=mpicxx \
        --with-fc=mpif90 \
        COPTFLAGS="-O2" \
        CXXOPTFLAGS="-O2" \
        FOPTFLAGS="-O2" \
        --with-debugging=0 \
        --with-shared-libraries=1 \
        --with-fortran-bindings=0 \
        --download-hdf5 \
        --download-metis \
        --download-parmetis \
        --download-eigen \
        --download-ptscotch \
        --download-netlib-lapack \
        --download-fblaslapack \
        --download-scalapack \
        --download-mumps \
        --download-superlu_dist \
        --download-triangle \
        --with-petsc4py=0 \
    && make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all -j$(nproc) \
    && python3 -m pip install --no-cache-dir --break-system-packages "setuptools<80" "cython==3.0.11" \
    && cd "$PETSC_DIR/src/binding/petsc4py" \
    && python3 -m pip install --no-cache-dir --break-system-packages --no-build-isolation . \
    && find "$PETSC_DIR/$PETSC_ARCH" -name "*.o" -delete 2>/dev/null || true \
    && rm -rf "$PETSC_DIR/.git" 2>/dev/null || true \
    && rm -rf "$PETSC_DIR/$PETSC_ARCH/externalpackages" 2>/dev/null || true

RUN python3 -m pip install --no-cache-dir --break-system-packages \
        "fastapi>=0.100" "uvicorn[standard]>=0.20"

COPY library/zoomy_core /tmp/src/zoomy_core
COPY library/zoomy_server /tmp/src/zoomy_server
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        /tmp/src/zoomy_core /tmp/src/zoomy_server \
    && rm -rf /tmp/src

ENV LD_LIBRARY_PATH="/opt/petsc/arch-container-opt/lib:$LD_LIBRARY_PATH" \
    PYTHONPATH="/opt/petsc/arch-container-opt/lib:$PYTHONPATH" \
    ZOOMY_ROOT=/workspace

WORKDIR /workspace
EXPOSE 8080
CMD ["zoomy-server", "--adapter", "dmplex", "--port", "8080"]
