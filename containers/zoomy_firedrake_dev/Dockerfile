# Firedrake + PETSc dev base (no Zoomy). Build zoomy_firedrake image on top for CI.
FROM firedrakeproject/firedrake-vanilla-default:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

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
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

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

RUN pip install --no-cache-dir --force-reinstall \
    /opt/petsc/src/binding/petsc4py

RUN python3 -m pip install --no-cache-dir \
      jupyterlab \
      notebook \
      ipywidgets \
      matplotlib \
      pandas \
      pytest \
      pytest-html \
      nbformat \
      nbclient \
      jupytext \
      param \
      loguru \
      attrs \
      "sympy>=1.14"

RUN rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]
