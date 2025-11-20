# Use firedrake vanilla image (default architecture, main branch)
FROM firedrakeproject/firedrake-vanilla-default:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

# Install additional build tools and dependencies for PETSc
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    cmake \
    git \
    sudo \
    libblas-dev \
    liblapack-dev \
    libboost-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Rebuild PETSc with mesh adaptivity frameworks enabled
RUN cd /opt/petsc && \
    make clean && rm -rf arch-* && \
    ./configure \
        --download-ptscotch \
        --with-make-np=8 \
        --download-hwloc \
        --download-hdf5 \
        --with-hdf5=1 \
        --with-hdf5-fortran-bindings=0 \
        --download-netlib-lapack \
        --with-netlib-lapack-c-bindings \
        --download-mumps \
        --download-scalapack \
        --download-parmetis \
        --download-superlu_dist \
        --download-pastix \
        --download-metis \
        --download-mmg \
        --download-parmmg \
        --download-pragmatic \
        --download-eigen \
        --download-slepc \
        --with-shared-libraries \
        PETSC_ARCH=arch-firedrake-default && \
    make PETSC_DIR=/opt/petsc PETSC_ARCH=arch-firedrake-default all

# Re-install Python PETSc bindings
RUN pip install --no-cache-dir --force-reinstall \
    /opt/petsc/src/binding/petsc4py

# --- Install Python dependencies ---
RUN python3 -m pip install --no-cache-dir \
      jupyterlab \
      notebook \
      ipywidgets \
      matplotlib \
      pandas


# Clean up
RUN rm -rf /var/lib/apt/lists/*

