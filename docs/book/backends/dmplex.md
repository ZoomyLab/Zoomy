# PETSc DMPlex backend

`zoomy_dmplex` is a C++ finite-volume solver built on
[PETSc DMPlex](https://petsc.org/release/manual/dmplex/) for unstructured
distributed meshes.

**Highlights**
- MUSCL reconstruction with Venkatakrishnan limiter.
- Generic SWE / SME model support via C++ headers generated from the
  Zoomy symbolic layer.
- MPI-parallel via PETSc.

**Install**: requires PETSc (see `install/install-petsc.sh`) and a C++
compiler. Build with the usual CMake workflow inside `library/zoomy_dmplex`.

**Repository**: [`library/zoomy_dmplex`](https://github.com/ZoomyLab/zoomy-dmplex)  
**API reference**: see [dmplex](../api/dmplex.rst).
