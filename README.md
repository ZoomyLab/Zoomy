# Zoomy

Flexible modeling and simulation software for free-surface flows.

## Installation

### Conda/Mamba


#### CPU (Linux / Mac)

**Installation**

```
conda env create -f env-zoomy.yml
./conda_config_setup.sh
```

**Activation**

```
conda activate zoomy
```

#### GPU (Linux / Mac)

```
conda env create -f env-zoomy-gpu.yml
./conda_config_setup.sh
```

**Activation**

```
conda activate zoomy-gpu
```

### Docker

T.b.d

### Apptainer

T.b.d

### Manual installation

See the `install/environment.yml`or the `install/environment-gpu.yml` for a complete list of requirements. Once the requirements are fulfilled, simply clone the repository.

The following environment variables need to be set 

```{bash}
PYTHONPATH=/path/to/Zoomy
SMS=/path/to/Zoomy
JAX_ENABLE_X64=True
PETSC_DIR=/path/to/petsc/installation
PETSC_ARCH=architecture used for compiling petsc
```

## Getting started

see our `tutorials` folder.

## License
The bryne code is free open-source software,
licensed under version 3 or later of the GNU General Public License.
See the file [LICENSE](LICENSE) for full copying permissions.

## Documentation

T.b.d.

## Publications

T.b.d.

## BibTex Citation

T.b.d.


