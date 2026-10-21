# Quick start

```bash
curl -LO \
  https://raw.githubusercontent.com/ZoomyLab/Zoomy/main/tutorials/swe/bump_2d.ipynb
jupyter lab bump_2d.ipynb
```

- [Quick start: a 2-D bump on NumPy and JAX](../tutorials/ipynb/swe/bump_2d.ipynb): model, SystemModel, NumericalSystemModel, two solvers
- [OpenFOAM](../tutorials/ipynb/foam/minimal.ipynb) and [AMReX](../tutorials/ipynb/amrex/minimal.ipynb): the same NumericalSystemModel in a container
- [Chorin projection: a VAM channel](../tutorials/ipynb/vam/chorin_channel.ipynb): a pressure-split model

## Next

- [Model](model.md) to change the physics, [NumericalSystemModel](numerical-system-model.md) for the scheme.
- [Solvers and options](solvers.md) for `dt`, time order, MOOD, IMEX and the pressure solver.
- [Tutorials](../tutorials/index.md) for the other notebooks.
