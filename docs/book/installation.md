# Install

## Minimal

```bash
pip install zoomy-core zoomy-plotting jupyterlab
curl -LO \
  https://raw.githubusercontent.com/ZoomyLab/Zoomy/main/tutorials/swe/simple_numpy.ipynb
jupyter lab simple_numpy.ipynb
```

- More notebooks: [Tutorials](tutorials/index.md)
- The GUI: [Using the GUI](guide/gui.md)
- Nothing to install: [In the browser](browser.md)

## Backends

JAX:

```bash
pip install zoomy-jax
curl -LO \
  https://raw.githubusercontent.com/ZoomyLab/Zoomy/main/tutorials/swe/bump_2d.ipynb
jupyter lab bump_2d.ipynb
```

JAX on an NVIDIA GPU:

```bash
pip install "zoomy-jax[cuda]"
```

OpenFOAM, AMReX, MPI and GPU images: [Containers](containers.md).

Which backend for what: [Backends](guide/backends.md).
