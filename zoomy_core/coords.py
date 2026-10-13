"""Canonical independent coordinates for the derivation framework.

These are *the* coordinate symbols every `Model`
and the `zoomy_core.derivatives` helper share, so that `d.t(f)`
differentiates with respect to the very same `t` the fields depend on.

    from zoomy_core import coords
    t, x, z = coords.t, coords.x, coords.z

Only the independent variables are fixed here — fields (`u`, `p`, `h` …)
are declared freely by the user as ordinary `sympy` Functions.
"""

import sympy as sp

# Time + space.  ``y`` exists for 3-D (2 horizontal directions) runs.
t = sp.Symbol("t", real=True)
x = sp.Symbol("x", real=True)
y = sp.Symbol("y", real=True)
z = sp.Symbol("z", real=True)

# Reference (mapped) vertical coordinate on [0, 1] used after a
# :class:`~zoomy_core.model.derivation.PDETransformation` (``z = b + h·zeta``).
zeta = sp.Symbol("zeta", real=True)

#: The independent coordinates in canonical order.
ALL = (t, x, y, z)

__all__ = ["t", "x", "y", "z", "zeta", "ALL"]
