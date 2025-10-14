# Zoomy

Flexible modeling and simulation software for free-surface flows.

![](web/images/overview2.png)

Zoomy's main objective is to provide a convenient modeling interface for complex free-surface flow models. Zoomy transitions from a **symbolic** modeling layer to **numerical** layer, compatible with a multitude of numerical solvers, e.g. Numpy, Jax, Firedrake, FenicsX, OpenFOAM and AMReX. Additionally, we support the PreCICE coupling framework in many of our numerical implementations, to allow for a convenient integration of our solver with your existing code.

## Documentation

See our [Documentation](https://mbd-rwth.github.io/Zoomy/) for details on

-   how to get started
-   tutorials
-   examples
-   API
-   ...

## License

The Zoomy code is free open-source software, licensed under version 3 or later of the GNU General Public License. See the file [LICENSE](LICENSE) for full copying permissions.

## BibTex Citation

T.b.d.

## Installation

### Conda/Mamba

The project is composed out of different environment files. We start by installing the base and than adding 'flavors', depdening on the solver backend that you want to use.

#### Default solver (Linux / Mac)

**Base Installation**

```         
cd install
conda env create -f install/zoomy.yml
./conda_config_setup.sh
```

**Mesh support for Numpy/Jax solver**

```         
conda env update -f install/env-mesh.yml
```

\*\* CPU/GPU solver with JAX\*\*

```         
conda env update -f install/env-jax.yml
```

#### FenicsX (Linux / Mac)

**Base Installation**

```         
cd install
conda env create -f install/zoomy.yml
./conda_config_setup.sh
```

**FenicsX**

```         
conda env update -f install/env-fenicsx.yml
```

#### Firedrake (Linux / Mac)

**Base Installation**

```         
cd install
conda env create -f install/zoomy.yml
./conda_config_setup.sh
```

**Firedrake**

Activate the environment before installing any Firedrake dependencies.

```         
conda activate zoomy
```

Mainly follow the instructions on the [Firedrake Webpage](https://www.firedrakeproject.org/install.html#install-firedrake).

Deviating from the instructions on the webpage, we use

```         
python3 ../firedrake-configure --show-petsc-configure-options --with-pnetcdf=0 | xargs -L1 ./configure
```

to compile PetSc without PNetCDF and then install Firedrake inside our conda environment

```         
pip install --no-binary h5py 'firedrake[check]'
```

#### AMReX (Linux / Mac)

**Base Installation**

```         
cd install
conda env create -f install/zoomy.yml
./conda_config_setup.sh
```

**AMReX**

Note that the AMReX installation is *completely indepdenent* and the AMReX solver does not depend on the Conda/Mamba environment. Follow the instructions on the [AMReX Webpage](https://amrex-codes.github.io/amrex/docs_html/Introduction.html)

#### OpenFOAM 12 (Linux / Mac)

T.b.d

**Activation**

```         
conda activate zoomy
```

### Docker

T.b.d

### Apptainer

T.b.d

### Manual installation

See the `install/zoomy.yml` for a complete list of requirements. Once the requirements are fulfilled, simply clone the repository.

The following environment variables need to be set

```{bash}
PYTHONPATH=/path/to/Zoomy
ZOOMY_DIR=/path/to/Zoomy
JAX_ENABLE_X64=True
PETSC_DIR=/path/to/petsc/installation
PETSC_ARCH=architecture used for compiling petsc
```

### External dependencies

#### PreCICE

T.b.d.

### Working in Jupyter Notebooks

Make sure to export the environment variables

```{bash}
PYTHONPATH=/path/to/Zoomy
ZOOMY_DIR=/path/to/Zoomy
JAX_ENABLE_X64=True //(if you use JAX)
PETSC_DIR=/path/to/petsc/installation
PETSC_ARCH=architecture used for compiling petsc
```

## Testing

T.b.d.

## Publications

T.b.d.

## Dependencies and acknowledgements

This
