# Zoomy

Flexible modeling and simulation software for free-surface flows.

## Documentation

See our [Documentation](https://mbd-rwth.github.io/Zoomy/) for details on 

* how to get started
* installation instructions
* tutorials
* examples
* ...

## License
The Zoomy code is free open-source software,
licensed under version 3 or later of the GNU General Public License.
See the file [LICENSE](LICENSE) for full copying permissions.

## BibTex Citation

T.b.d.


## Installation

### Conda/Mamba

#### CPU/GPU (Linux / Mac)

**Installation**

```
cd install
conda env create -f install/zoomy.yml
./conda_config_setup.sh
```

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

#### DolfinX

T.b.d.

#### OpenFoam

T.b.d.

#### AMReX

T.b.d.

#### PreCICE

T.b.d.



## Publications

T.b.d.


## Dependencies and acknowledgements

This 


