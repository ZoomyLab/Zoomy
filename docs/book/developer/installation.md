# Developer installation

## Clone

```bash
git clone --recurse-submodules https://github.com/ZoomyLab/Zoomy.git
cd Zoomy
```

After a plain clone:

```bash
git submodule update --init --recursive
```

## Editable install

The environment of [Install](../installation.md) from the files of the clone, with the libraries editable:

```bash
micromamba env create -f install/Zoomy.yml
micromamba activate zoomy
pip install -e library/zoomy_core -e library/zoomy_plotting -e library/zoomy_prepost

micromamba env update -f install/zoomy_jax.yml
pip install -e library/zoomy_jax
```

The files in `install/` are the dependency lists of the other environments.

## Images

```bash
apptainer build --fakeroot containers/<name>/<name>.sif containers/<name>/<name>.def
```

The recipes are in `containers/<name>/`, see `containers/README.md`.
