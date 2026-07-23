[![Meshes](https://github.com/ZoomyLab/meshes/actions/workflows/build-meshes.yml/badge.svg)](https://github.com/ZoomyLab/meshes/actions/workflows/build-meshes.yml)
[![Containers](https://github.com/ZoomyLab/Zoomy/actions/workflows/build-containers.yml/badge.svg)](https://github.com/ZoomyLab/Zoomy/actions/workflows/build-containers.yml)

![](https://gitlab.git.nrw/rwth-mbd/containers/-/raw/main/rwth_mbd_de_rgb.png)

# Zoomy

Flexible modeling and simulation software for free-surface flows.

![](docs/images/overview2.png)

Zoomy's main objective is to provide a convenient modeling interface for complex free-surface flow models. Zoomy transitions from a **symbolic** modeling layer to a **numerical** layer, compatible with a multitude of numerical solvers, e.g. NumPy, JAX, Firedrake, FEniCSx, OpenFOAM, and AMReX. Additionally, we support the preCICE coupling framework for some numerical backends, enabling convenient integration of our solver with your existing code.

## Documentation

See the [Zoomy documentation](https://zoomylab.github.io/Zoomy/) for the user guide, tutorials, API reference, and full installation / development setup.

## Installation

- **Try it now in your browser** — interactive GUI at [zoomylab.github.io/Zoomy/gui](https://zoomylab.github.io/Zoomy/gui/).
- **Try a notebook in your browser** — open the [shallow-water Pyodide tour](https://zoomylab.github.io/Zoomy/jupyter-lite/_output/lab/index.html?path=pyodide.ipynb) in JupyterLite (walks `Model` → `SystemModel` → `NumericalSystemModel` → C++ codegen → solve & plot, all running in the browser).
- **Pip install the core**:
  ```bash
  pip install zoomy_core
  ```
- **Everything else** (full repo clone, submodules, Apptainer / Docker, devcontainer, Conda/Mamba, backend-specific setup, development workflow) — see [Documentation → Installation](https://zoomylab.github.io/Zoomy/installation.html).

## Backends

Zoomy's symbolic layer compiles to several solver backends. Each one has its own page in the documentation:

- [NumPy](https://zoomylab.github.io/Zoomy/backends/numpy.html)
- [JAX](https://zoomylab.github.io/Zoomy/backends/jax.html)
- [Firedrake](https://zoomylab.github.io/Zoomy/backends/firedrake.html)
- [PETSc DMPlex](https://zoomylab.github.io/Zoomy/backends/dmplex.html)
- [AMReX](https://zoomylab.github.io/Zoomy/backends/amrex.html)
- [OpenFOAM](https://zoomylab.github.io/Zoomy/backends/openfoam.html)
- [FEniCSx](https://zoomylab.github.io/Zoomy/backends/fenicsx.html)

Note: Many backends are in an very early development status. If you are interested in a particular solver and you find no information, please contact us.

## Testing

CI test reports for each backend are published with the docs site: see [CI Reports](https://zoomylab.github.io/Zoomy/ci-reports.html).

## License

The Zoomy source code is free open-source software, licensed under version 3 or later of the GNU Lesser General Public License (LGPLv3+). See [LICENSE](LICENSE) for full copying permissions.

## Citation

```bibtex
@misc{steldermann_dimensionally_2025,
	author       = {Steldermann, Ingo and Bünning, Jonas and Pummer, Elena and Kowalski, Julia},
	editor       = {Pummer, Elena and Albayrak, Ismail},
	title        = {Dimensionally Reduced Shallow Moment Simulations of a Pumped-Storage Plant Tunnel System},
	publisher    = {ETH Zurich},
	date         = {2025},
	doi          = {10.3929/ETHZ-B-000727194},
	url          = {http://hdl.handle.net/20.500.11850/727194},
}

```

## Zoomy or Shallow Moment Method related publication

```bibtex
@incollection{steldermann_shallow_2024,
	author       = {Steldermann, Ingo and Torrilhon, Manuel and Kowalski, Julia},
	editor       = {Parés, Carlos and Castro, Manuel J. and Morales De Luna, Tomás and Muñoz-Ruiz, María Luz},
	title        = {Shallow Moment Equations—Comparing Legendre and Chebyshev Basis Functions},
	booktitle    = {Hyperbolic Problems: Theory, Numerics, Applications. Volume {II}},
	volume       = {35},
	pages        = {457--466},
	publisher    = {Springer Nature Switzerland},
	location     = {Cham},
	date         = {2024},
	isbn         = {978-3-031-55264-9},
	doi          = {10.1007/978-3-031-55264-9_39},
	url          = {https://link.springer.com/10.1007/978-3-031-55264-9_39},
}

@article{steldermann_shallow_2023,
	author       = {Steldermann, Ingo and Torrilhon, Manuel and Kowalski, Julia},
	title        = {Shallow Moments to Capture Vertical Structure in Open Curved Shallow Flow},
	journaltitle = {Journal of Computational and Theoretical Transport},
	volume       = {52},
	number       = {7},
	pages        = {475--505},
	date         = {2023-11-10},
	issn         = {2332-4309},
	doi          = {10.1080/23324309.2023.2284202},
	url          = {https://www.tandfonline.com/doi/full/10.1080/23324309.2023.2284202},
}
```

## Acknowledgements

Zoomy builds on, integrates with, or uses logos from the following open-source projects. We gratefully acknowledge their authors and the licenses under which we use their work:

| Project | Role in Zoomy | License | Notes |
|---|---|---|---|
| [SymPy](https://www.sympy.org/) | symbolic modeling layer | BSD-3-Clause | logo by Fredrik Johansson, free use under the same terms as SymPy |
| [NumPy](https://numpy.org/) | reference solver, arrays | BSD-3-Clause | |
| [JAX](https://jax.readthedocs.io/) | `zoomy_jax` backend | Apache-2.0 | |
| [Firedrake](https://www.firedrakeproject.org/) | `zoomy_firedrake` backend | LGPLv3+ | |
| [FEniCSx / DOLFINx](https://fenicsproject.org/) | `zoomy_fenicsx` backend | LGPLv3+ | |
| [PETSc](https://petsc.org/) | DMPlex backend, Firedrake | BSD-2-Clause | |
| [AMReX](https://amrex-codes.github.io/) | `zoomy_amrex` backend | BSD-3-Clause | |
| [OpenFOAM](https://openfoam.org/) | OpenFOAM backend | GPL-3.0 | logo: *Carnby, CC BY-SA 4.0, via Wikimedia Commons* |
| [preCICE](https://precice.org/) | coupling framework | LGPLv3+ | |
| [Gmsh](https://gmsh.info/) | mesh generation | GPLv2+ | |
| [ParaView](https://www.paraview.org/) | post-processing | BSD-3-Clause | macros in `tools/paraview_macros/` |
| [Jupyter Book](https://jupyterbook.org/) | this documentation site | BSD-3-Clause | |
| [JupyterLite](https://jupyterlite.readthedocs.io/) / [Pyodide](https://pyodide.org/) | in-browser notebook execution | BSD-3-Clause / MPL-2.0 | |

If you spot a project we use but haven't credited, please open an issue.
