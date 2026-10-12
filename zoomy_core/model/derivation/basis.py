"""Opaque modal basis (a phi-family head plus a weight head, no polynomials) for the symbolic
derivation. Sits at the start of Model -> SystemModel; a concrete Basisfunction is swapped in later
by ResolveBasis.
"""

from __future__ import annotations

import sympy as sp


__all__ = ["Basis"]


class Basis:
    """Symbolic modal basis: opaque Function heads ``phi(k, zeta)`` and ``weight(zeta)``.

    Args:
        symbol: name of the two-argument trial/test head.
        weight: name of the one-argument weight head.
    """

    def __init__(self, symbol="phi", weight="c"):
        """Mint the opaque ``phi`` and ``weight`` Function classes."""
        self.symbol = symbol
        self.weight_name = weight
        # ONE opaque Function class per family — exactly the production
        # ``Basisfunction.phi_fn`` shape (``(index, ζ)`` arity).  The
        # single-class design is what makes ``Σ a(k,…)·φ(k, ζ)`` a clean
        # unexpanded ``sp.Sum`` and lets ``ExtractBrackets`` match by head.
        #
        # ``_is_basis_head`` marks these as basis machinery so the model's
        # field collector never mistakes ``φ`` / ``c`` for an unknown.
        self.phi = type(symbol, (sp.Function,),
                        {"nargs": 2, "_is_basis_head": True, "_basis": self})
        self.weight = type(weight, (sp.Function,),
                           {"nargs": 1, "_is_basis_head": True, "_basis": self})

    # ── compatibility shim for the bracket/resolve machinery ───────────
    @property
    def phi_fn(self):
        """Alias of ``phi``, matching the attribute name of a concrete Basisfunction."""
        return self.phi

    def __repr__(self):
        """Return ``Basis(symbol=..., weight=...)``."""
        return f"Basis(symbol={self.symbol!r}, weight={self.weight_name!r})"
