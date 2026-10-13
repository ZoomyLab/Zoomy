# Shallow water

Runs on the reference **NumPy** backend and needs nothing but `zoomy_core` and
`zoomy_plotting` ([pip](../installation.md#pip-numpy))
— no container, no compiler.

| | What it covers |
|---|---|
| [Simple](ipynb/swe/simple_numpy.ipynb) | The whole pipeline once: model → operators → scheme → solve → plot, in 1-D. Start here. |

Shallow water is not a separate hand-written model in Zoomy — it is
`SME(level=0)`, the depth-averaged member of the Shallow Moment family. Raising
`level` resolves a vertical velocity profile: see
[Shallow moments](sme.md).
