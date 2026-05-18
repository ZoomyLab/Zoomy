# AMReX backend

`zoomy_amrex` couples Zoomy models to [AMReX](https://amrex-codes.github.io/)
for block-structured adaptive mesh refinement (AMR) on CPU and GPU.

**Highlights**
- Adaptive mesh refinement out of the box.
- GPU acceleration (CUDA, HIP, SYCL via AMReX).
- 3D voxelised topography ingest from DEM rasters.

**Install**: AMReX is independent of the Conda/Mamba Zoomy environment.
Follow the AMReX
[Getting Started guide](https://amrex-codes.github.io/amrex/docs_html/Introduction.html),
then build `library/zoomy_amrex`.

**Repository**: [`library/zoomy_amrex`](https://github.com/ZoomyLab/zoomy-amrex)  
**API reference**: see [amrex](../api/amrex.rst).
