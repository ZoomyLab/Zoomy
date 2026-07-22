# Shallow water

Two tutorials, in order. Both run on the reference **NumPy** backend and need
nothing but `pip install zoomy_core` — no container, no compiler.

| | What it covers |
|---|---|
| [Simple](ipynb/swe/simple_numpy.ipynb) | The whole pipeline once: model → operators → scheme → solve → plot, in 1-D. Start here. |
| [Advanced](ipynb/swe/advanced_numpy.ipynb) | Write your **own closure**, compose it into a **child model**, run in **2-D**, and measure that the new physics does what it claims. |

Shallow water is not a separate hand-written model in Zoomy — it is
`SME(level=0)`, the depth-averaged member of the Shallow Moment family. Raising
`level` resolves a vertical velocity profile; nothing else changes.

Both notebooks assert mass conservation on a closed domain, so they fail loudly
if a boundary condition leaks.
