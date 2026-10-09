# Acknowledgements

Zoomy is built on open-source software and on published numerical methods. We thank their authors. This page lists what we use and how to cite it. The full BibTeX for every entry is in [`docs/book/references/third_party.bib`](https://github.com/ZoomyLab/Zoomy/blob/main/docs/book/references/third_party.bib). If you publish results computed with Zoomy, please also cite the components your run used. If we missed a project, please [open an issue](https://github.com/ZoomyLab/Zoomy/issues).

## Software we build on

### Symbolic and numerical core

| Project | Role in Zoomy | Citation |
|---|---|---|
| [SymPy](https://github.com/sympy/sympy) | symbolic modelling layer (`Model`, `SystemModel`, code generation) | Meurer et al. 2017, [doi:10.7717/peerj-cs.103](https://doi.org/10.7717/peerj-cs.103) |
| [NumPy](https://github.com/numpy/numpy) | arrays, NumPy reference solver | Harris et al. 2020, [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2) |
| [SciPy](https://github.com/scipy/scipy) | Krylov/sparse solvers, stability analysis | Virtanen et al. 2020, [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2) |
| [param](https://github.com/holoviz/param) | declarative solver and model parameters | repository |
| [loguru](https://github.com/Delgan/loguru) | logging | repository |
| [attrs](https://github.com/python-attrs/attrs) | data classes in several backends | repository |

### Solver backends, parallelism and coupling

| Project | Role in Zoomy | Citation |
|---|---|---|
| [JAX](https://github.com/jax-ml/jax) | `zoomy_jax` backend (JIT, GPU, autodiff) | Bradbury et al. 2018, repository |
| [PETSc](https://gitlab.com/petsc/petsc) | DMPlex backend, linear solvers, Firedrake base | Balay et al. 2025, [doi:10.2172/2998643](https://doi.org/10.2172/2998643) |
| PETSc DMPlex | unstructured mesh management | Knepley & Karpeev 2009, [doi:10.1155/2009/948613](https://doi.org/10.1155/2009/948613); Lange et al. 2016, [doi:10.1137/15M1026092](https://doi.org/10.1137/15M1026092) |
| [petsc4py](https://gitlab.com/petsc/petsc) | Python access to PETSc | Dalcin et al. 2011, [doi:10.1016/j.advwatres.2011.04.013](https://doi.org/10.1016/j.advwatres.2011.04.013) |
| [Firedrake](https://github.com/firedrakeproject/firedrake) | `zoomy_firedrake` backend | Ham et al. 2023, [doi:10.25561/104839](https://doi.org/10.25561/104839) |
| [UFL](https://github.com/FEniCS/ufl) | weak forms in the Firedrake and FEniCSx backends | Alnæs et al. 2014, [doi:10.1145/2566630](https://doi.org/10.1145/2566630) |
| [Animate](https://github.com/mesh-adaptation/animate) | anisotropic mesh adaptation for Firedrake (vendored, MIT) | Wallwork et al. 2024, repository |
| [DOLFINx](https://github.com/FEniCS/dolfinx) | `zoomy_fenicsx` backend | Baratta et al. 2023, [doi:10.5281/zenodo.10447666](https://doi.org/10.5281/zenodo.10447666) |
| [Basix](https://github.com/FEniCS/basix) | finite element bases in the FEniCSx backend | Scroggs et al. 2022, [doi:10.21105/joss.03982](https://doi.org/10.21105/joss.03982); [doi:10.1145/3524456](https://doi.org/10.1145/3524456) |
| [OpenFOAM](https://github.com/OpenFOAM/OpenFOAM-13) | `zoomyFoam` backend (OpenFOAM 13, openfoam.org)[^openfoam] | Weller et al. 1998, [doi:10.1063/1.168744](https://doi.org/10.1063/1.168744) |
| [AMReX](https://github.com/AMReX-Codes/amrex) | `zoomy_amrex` backend (block-structured AMR, GPU) | Zhang et al. 2019, [doi:10.21105/joss.01370](https://doi.org/10.21105/joss.01370); Zhang et al. 2021, [doi:10.1177/10943420211022811](https://doi.org/10.1177/10943420211022811) |
| [preCICE](https://github.com/precice/precice) + [pyprecice](https://github.com/precice/python-bindings) | multi-physics coupling (JAX and OpenFOAM participants) | Chourdakis et al. 2022, [doi:10.12688/openreseurope.14445.2](https://doi.org/10.12688/openreseurope.14445.2) |
| [Eigen](https://gitlab.com/libeigen/eigen) | dense eigensolver in the DMPlex backend | Guennebaud, Jacob et al. 2010, repository |
| [JSON for Modern C++](https://github.com/nlohmann/json) | settings parser in the DMPlex backend (vendored, MIT) | Lohmann 2025, repository |
| [Open MPI](https://github.com/open-mpi/ompi) | MPI runtime in the AMReX and DMPlex containers | Gabriel et al. 2004, [doi:10.1007/978-3-540-30218-6_19](https://doi.org/10.1007/978-3-540-30218-6_19) |
| [mpi4py](https://github.com/mpi4py/mpi4py) | MPI from Python (Firedrake, FEniCSx) | Dalcin & Fang 2021, [doi:10.1109/MCSE.2021.3083216](https://doi.org/10.1109/MCSE.2021.3083216) |
| [METIS](https://github.com/KarypisLab/METIS) via [PyMetis](https://github.com/inducer/pymetis) | mesh partitioning in `zoomy_jax` | Karypis & Kumar 1998, [doi:10.1137/S1064827595287997](https://doi.org/10.1137/S1064827595287997) |
| [Equinox](https://github.com/patrick-kidger/equinox) | checkpointed reverse-mode while loops (`solver_jax`, `symsolver`); GNN prototypes | Kidger & Garcia 2021, [doi:10.48550/arXiv.2111.00254](https://doi.org/10.48550/arXiv.2111.00254) |
| [Optax](https://github.com/google-deepmind/optax) | optimisers for the GNN prototypes | DeepMind et al. 2020, repository |
| [Jraph](https://github.com/google-deepmind/jraph) | graph networks for the GNN prototypes | Godwin et al. 2020, repository |

### Meshing, I/O and visualisation

| Project | Role in Zoomy | Citation |
|---|---|---|
| [Gmsh](https://gitlab.onelab.info/gmsh/gmsh) | mesh generation | Geuzaine & Remacle 2009, [doi:10.1002/nme.2579](https://doi.org/10.1002/nme.2579) |
| [meshio](https://github.com/nschloe/meshio) | mesh format conversion | Schlömer 2024, [doi:10.5281/zenodo.1173115](https://doi.org/10.5281/zenodo.1173115) |
| [h5py](https://github.com/h5py/h5py) / [HDF5](https://github.com/HDFGroup/hdf5) | mesh and result files | repository |
| [rasterio](https://github.com/rasterio/rasterio) | raster topography import | repository |
| [Matplotlib](https://github.com/matplotlib/matplotlib) | plotting | Hunter 2007, [doi:10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55) |
| [PyVista](https://github.com/pyvista/pyvista) | mesh plots, OpenFOAM pre-processing | Sullivan & Kaszynski 2019, [doi:10.21105/joss.01450](https://doi.org/10.21105/joss.01450) |
| [VTK](https://github.com/Kitware/VTK) | VTU result reading | Schroeder, Martin & Lorensen 2006, ISBN 978-1-930934-19-1 |
| [ParaView](https://github.com/Kitware/ParaView) | post-processing macros | Ahrens, Geveci & Law 2005, [doi:10.1016/B978-012387582-2/50038-1](https://doi.org/10.1016/B978-012387582-2/50038-1) |
| [Plotly](https://github.com/plotly/plotly.py) | GUI previews | Kruchten, Seier & Parmer 2024, [doi:10.5281/zenodo.14503524](https://doi.org/10.5281/zenodo.14503524) |
| [seaborn](https://github.com/mwaskom/seaborn) | benchmark figures | Waskom 2021, [doi:10.21105/joss.03021](https://doi.org/10.21105/joss.03021) |
| [imageio](https://github.com/imageio/imageio) | animation export | repository |
| [tol-colors](https://github.com/Descanonge/tol_colors) | Paul Tol's colour schemes in figures | repository |
| [tqdm](https://github.com/tqdm/tqdm) | progress bars | da Costa-Luis et al., [doi:10.5281/zenodo.595120](https://doi.org/10.5281/zenodo.595120) |

### Server, GUI, containers and documentation

| Project | Role in Zoomy | Citation |
|---|---|---|
| [FastAPI](https://github.com/fastapi/fastapi), [Uvicorn](https://github.com/encode/uvicorn), [Pydantic](https://github.com/pydantic/pydantic) | `zoomy_server` solver API | repository |
| [Apptainer](https://github.com/apptainer/apptainer) | backend container images | Kurtzer, Sochat & Bauer 2017, [doi:10.1371/journal.pone.0177459](https://doi.org/10.1371/journal.pone.0177459) |
| [Docker](https://github.com/docker/cli) | CI and development containers | repository |
| [Eclipse Theia](https://github.com/eclipse-theia/theia) | GUI workbench | repository |
| [Pyodide](https://github.com/pyodide/pyodide) | in-browser Python for the GUI and JupyterLite | Pyodide development team 2021, [doi:10.5281/zenodo.5156931](https://doi.org/10.5281/zenodo.5156931) |
| [KaTeX](https://github.com/KaTeX/KaTeX), [marked](https://github.com/markedjs/marked), [JSZip](https://github.com/Stuk/jszip) | GUI math, Markdown and archive handling | repository |
| [Jupyter](https://github.com/jupyter/notebook) | tutorials, container Jupyter mode | Kluyver et al. 2016, [doi:10.3233/978-1-61499-649-1-87](https://doi.org/10.3233/978-1-61499-649-1-87) |
| [Jupytext](https://github.com/mwouts/jupytext) | notebooks as scripts | repository |
| [JupyterLite](https://github.com/jupyterlite/jupyterlite) | in-browser notebook tour | repository |
| [Jupyter Book](https://github.com/jupyter-book/jupyter-book) | this documentation | Executable Books Community 2020, [doi:10.5281/zenodo.2561065](https://doi.org/10.5281/zenodo.2561065) |
| [Sphinx](https://github.com/sphinx-doc/sphinx), [MyST-Parser](https://github.com/executablebooks/MyST-Parser), [sphinx-book-theme](https://github.com/executablebooks/sphinx-book-theme) | documentation toolchain | repository |
| [Doxygen](https://github.com/doxygen/doxygen), [Breathe](https://github.com/breathe-doc/breathe), [Exhale](https://github.com/svenevs/exhale) | C++ API reference | repository |

### Reference solvers and benchmark data

| Project | Role in Zoomy | Citation |
|---|---|---|
| [SWASHES](https://sourcesup.renater.fr/projects/swashes/) via [pyswashes](https://github.com/lrntct/pyswashes) | analytic shallow-water reference solutions | Delestre et al. 2013, [doi:10.1002/fld.3741](https://doi.org/10.1002/fld.3741) |
| [TELEMAC-MASCARET](https://gitlab.pam-retd.fr/otm/telemac-mascaret) | reference solver; Malpasset case and meshes | repository |
| [Basilisk](http://basilisk.fr) | reference solver for benchmarks | repository |

[^openfoam]: This offering is not approved or endorsed by OpenCFD Limited, producer and distributor of the OpenFOAM software via www.openfoam.com, and owner of the OPENFOAM® and OpenCFD® trade marks.

## Numerical methods

Methods implemented in `zoomy_core`, with the references the code follows.

### Fluxes and Riemann solvers

| Method | Use in Zoomy | Reference |
|---|---|---|
| Rusanov (local Lax–Friedrichs) | default numerical flux and its well-balanced variants | Rusanov 1962, [doi:10.1016/0041-5553(62)90062-9](https://doi.org/10.1016/0041-5553(62)90062-9) |
| HLL | numerical flux | Harten, Lax & van Leer 1983, [doi:10.1137/1025002](https://doi.org/10.1137/1025002) |
| Davis wave-speed estimates | HLL signal speeds | Davis 1988, [doi:10.1137/0909030](https://doi.org/10.1137/0909030) |
| HLLC | contact-resolving flux for free-surface models | Toro, Spruce & Speares 1994, [doi:10.1007/BF01414629](https://doi.org/10.1007/BF01414629) |
| Roe linearisation | nonconservative and quasilinear Roe fluxes | Roe 1981, [doi:10.1016/0021-9991(81)90128-5](https://doi.org/10.1016/0021-9991(81)90128-5) |

### Nonconservative products, well-balancing and positivity

| Method | Use in Zoomy | Reference |
|---|---|---|
| Path-conservative products | segment-path integral of the nonconservative matrix | Dal Maso, LeFloch & Murat 1995 (J. Math. Pures Appl. 74:483–548) |
| Path-conservative schemes | nonconservative fluxes, Roe matrix | Parés 2006, [doi:10.1137/050628052](https://doi.org/10.1137/050628052); Parés & Castro 2004, [doi:10.1051/m2an:2004041](https://doi.org/10.1051/m2an:2004041) |
| Hydrostatic reconstruction | well-balanced, positive wet/dry fluxes | Audusse et al. 2004, [doi:10.1137/S1064827503431090](https://doi.org/10.1137/S1064827503431090); Audusse & Bristeau 2005, [doi:10.1016/j.jcp.2004.12.016](https://doi.org/10.1016/j.jcp.2004.12.016) |
| Desingularised inverse depth | velocities near dry states | Kurganov & Petrova 2007, [doi:10.4310/CMS.2007.v5.n1.a6](https://doi.org/10.4310/CMS.2007.v5.n1.a6) |
| Cell-mean positivity limiter | a-priori positivity of reconstructed depth | Xing, Zhang & Shu 2010, [doi:10.1016/j.advwatres.2010.08.005](https://doi.org/10.1016/j.advwatres.2010.08.005); Xing & Zhang 2013, [doi:10.1007/s10915-013-9695-y](https://doi.org/10.1007/s10915-013-9695-y) |

### Reconstruction and limiting

| Method | Use in Zoomy | Reference |
|---|---|---|
| MUSCL reconstruction | second-order least-squares reconstruction | van Leer 1979, [doi:10.1016/0021-9991(79)90145-1](https://doi.org/10.1016/0021-9991(79)90145-1); LeVeque 2002, [doi:10.1017/CBO9780511791253](https://doi.org/10.1017/CBO9780511791253) |
| Barth–Jespersen limiter | slope limiter | Barth & Jespersen 1989, [doi:10.2514/6.1989-366](https://doi.org/10.2514/6.1989-366) |
| Venkatakrishnan limiter | default smooth slope limiter | Venkatakrishnan 1995, [doi:10.1006/jcph.1995.1084](https://doi.org/10.1006/jcph.1995.1084) |
| MOOD | a-posteriori order reduction | Clain, Diot & Loubère 2011, [doi:10.1016/j.jcp.2011.02.026](https://doi.org/10.1016/j.jcp.2011.02.026) |

### Time integration, splitting and linear solvers

| Method | Use in Zoomy | Reference |
|---|---|---|
| SSP Runge–Kutta (Shu–Osher form) | explicit time stepping | Shu & Osher 1988, [doi:10.1016/0021-9991(88)90177-5](https://doi.org/10.1016/0021-9991(88)90177-5) |
| IMEX ARS(2,3,2) and ARS(3,4,3) | implicit-explicit stepping for DAEs | Ascher, Ruuth & Spiteri 1997, [doi:10.1016/S0168-9274(97)00056-1](https://doi.org/10.1016/S0168-9274(97)00056-1) |
| Strang splitting | operator splitting | Strang 1968, [doi:10.1137/0705041](https://doi.org/10.1137/0705041) |
| Chorin projection | non-hydrostatic pressure split | Chorin 1968, [doi:10.1090/S0025-5718-1968-0242392-2](https://doi.org/10.1090/S0025-5718-1968-0242392-2) |
| GMRES | elliptic pressure and Newton–Krylov solves | Saad & Schultz 1986, [doi:10.1137/0907058](https://doi.org/10.1137/0907058) |

### Models and closures

| Model | Use in Zoomy | Reference |
|---|---|---|
| Shallow moment equations | `SME` model family | Kowalski & Torrilhon 2019, [doi:10.4208/cicp.OA-2017-0263](https://doi.org/10.4208/cicp.OA-2017-0263) |
| Hyperbolic SME analysis | SME eigenvalue structure | Koellermeier & Rominger 2020, [doi:10.4208/cicp.OA-2019-0065](https://doi.org/10.4208/cicp.OA-2019-0065) |
| Vertically averaged moments (VAM) | `VAM` / `ML-VAM` non-hydrostatic models | Escalante et al. 2024, [doi:10.1016/j.jcp.2024.112882](https://doi.org/10.1016/j.jcp.2024.112882) |
| Two-layer non-hydrostatic model | constraint scaling of the pressure system | Escalante et al. 2019, [doi:10.1007/s10915-018-0849-9](https://doi.org/10.1007/s10915-018-0849-9) |
| Layer-averaged dispersive hierarchy | experimental `ML-VAM-WD` constraints | Fernández-Nieto et al. 2018, [doi:10.4310/CMS.2018.v16.n5.a1](https://doi.org/10.4310/CMS.2018.v16.n5.a1) |
| Multilayer mass-exchange closure | `ML-SWE` interface fluxes | Aguillon, Hörnschemeyer & Sainte-Marie 2026, [arXiv:2601.16709](https://arxiv.org/abs/2601.16709) |
| Parabolic eddy viscosity | turbulence closure | Elder 1959, [doi:10.1017/S0022112059000374](https://doi.org/10.1017/S0022112059000374) |
| Depth-averaged k–ε | bed friction velocity in the k–ε sources | Rastogi & Rodi 1978, [doi:10.1061/JYCEAJ.0004962](https://doi.org/10.1061/JYCEAJ.0004962) |
| q–r form of k–ε (√k, √ε) | positivity-preserving `QRKESME` model | Finnie & Jeppson 1991, [doi:10.1061/(ASCE)0733-9429(1991)117:11(1513)](https://doi.org/10.1061/(ASCE)0733-9429(1991)117:11(1513)) |
