# OpenFOAM backend (via preCICE)

`zoomy_foam` couples Zoomy solvers to [OpenFOAM](https://openfoam.org/) using
the [preCICE](https://precice.org/) coupling library, enabling mixed-fidelity
free-surface / two-phase simulations.

**Highlights**
- Bidirectional coupling with OpenFOAM via preCICE.
- Example configurations live in `tools/precice_configs/`.

**Install**: requires OpenFOAM 12+, preCICE, and the preCICE OpenFOAM adapter.
See `install/setup_precice.sh`.

**Repository**: [`library/zoomy_foam`](https://github.com/ZoomyLab/zoomy-foam)
