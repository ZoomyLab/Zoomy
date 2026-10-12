# Quick start

The C1 case of the [test matrix](../testing/symsolver.md): shallow water
(`SME(level=0)`) on $[0, 10]^2$ with walls, at rest with a Gaussian hump in the
depth, second order, until $t = 2.5$.  The scripts are in
`docs/book/examples/`; run them from there after `source env.sh` (it puts this
checkout's libraries on `PYTHONPATH`).

## The case

```{literalinclude} ../examples/bump.py
:language: python
```

## JAX

```{literalinclude} ../examples/quickstart_jax.py
:language: python
```

```text
t = 2.500, cells = 1600, h in [0.49671, 0.51884]
```

About 20 s on 8 CPU cores, most of it printing and compiling the template.

## foam and AMReX: the same NSM

The C++ backends take the NSM, the mesh and the initial state; they print the
same template as C++, build it in their container (cached by a hash of the
sources) and run it.

```{literalinclude} ../examples/quickstart_foam.py
:language: python
```

```text
t = 2.500, cells = 1600, h in [0.49669, 0.51856]
max |h_foam - h_jax| = 3.51e-04
```

```{literalinclude} ../examples/quickstart_amrex.py
:language: python
```

```text
t = 2.500, cells = 1600, h in [0.49671, 0.51884]
max |h_amrex - h_jax| = 1.82e-14
```

AMReX runs the same arithmetic as JAX and agrees to round-off.  foam computes
second-order gradients with OpenFOAM's own stencils, so it agrees with JAX to
the discretization error (at first order: to round-off; matrix case C1).  The
first AMReX run compiles AMReX itself (about a minute here); later runs reuse
the build.

Outputs above: library commits core 09d1584, jax 5fd4252, foam e17026f,
amrex ce3f72a; containers OpenFOAM 13 and AMReX as in
[Installation](../installation.md).

## Next

- [Model](model.md) to change the physics, [NumericalSystemModel](numerical-system-model.md) for the scheme.
- [Solvers and options](solvers.md) for `dt`, time order, MOOD, IMEX and the pressure solver.
- `docs/book/examples/chorin_vam.py`: the Chorin projection solver on VAM(1).
