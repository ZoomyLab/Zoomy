"""MalpassetSWE — pure-SymPy ``Model`` subclass extracted from
``tutorials/firedrake/malpasset_viscous_v2.py`` (lines 134-383).

Zero Firedrake / PETSc imports — drops cleanly into JAX or NumPy
pipelines.  Used by SPMD scaling tests against the Malpasset
viscous SWE physics on a STRUCTURED 2D approximation of the
domain.

Geometry comes from
``data/malpasset/geo_malpasset-small.msh`` (the actual unstructured
triangular mesh).  Running on that mesh requires graph-partitioning
SPMD (future work — task #26); this module is the pure model so
the SPMD path can exercise the real Malpasset physics on a
structured grid that captures the same scales.
"""
from __future__ import annotations

import os

import sympy as sp
from sympy import Matrix, sqrt

from zoomy_core.model.basemodel import Model
from zoomy_core.misc.misc import ZArray


# Default parameter values, mirror the v2 script's env-tunable knobs.
MANNING_N = float(os.environ.get("MALPASSET_MANNING", "0.033"))
EPS_WD = float(os.environ.get("MALPASSET_EPS_WD", "1e-2"))
H_FRICTION_FLOOR = float(os.environ.get("MALPASSET_H_FRICTION", "0.5"))
NU = float(os.environ.get("MALPASSET_NU", "1.0"))


class MalpassetSWE(Model):
    """SWE with ``[b, h, hu, hv]`` state and ``[hinv]`` aux.

    Verbatim port of the Firedrake tutorial's MalpassetSWE.  See
    ``tutorials/firedrake/malpasset_viscous_v2.py`` for the physics
    notes (Audusse-style well-balanced hydrostatic-NCP split,
    Manning friction with floored ``h^{-1/3}``, depth-averaged
    eddy viscosity, KP desingularisation of ``1/h``, wet/dry
    eigenvalue gate).
    """

    def __init__(self, *, g=9.81, n=MANNING_N, nu=NU, eps=EPS_WD, **kw):
        super().__init__(
            dimension=2,
            variables=["b", "h", "hu", "hv"],
            aux_variables=["hinv"],
            parameters={
                "g": (float(g), "positive"),
                "n": (float(n), "non-negative"),
                "nu": (float(nu), "non-negative"),
                "eps": (float(eps), "positive"),
            },
            eigenvalue_mode="symbolic",
            **kw,
        )

    def _primitives(self):
        v = self.variables
        a = self.aux_variables
        return v.b, v.h, v.hu, v.hv, a.hinv

    def flux(self):
        _, h, hu, hv, hinv = self._primitives()
        F = Matrix.zeros(4, 2)
        F[1, 0] = hu
        F[1, 1] = hv
        F[2, 0] = hu * hu * hinv
        F[2, 1] = hu * hv * hinv
        F[3, 0] = hu * hv * hinv
        F[3, 1] = hv * hv * hinv
        return ZArray(F)

    def nonconservative_matrix(self):
        _, h, _, _, _ = self._primitives()
        g = self._parameter_symbols.g
        N = ZArray.zeros(4, 4, 2)
        N[2, 0, 0] = g * h
        N[2, 1, 0] = g * h
        N[3, 0, 1] = g * h
        N[3, 1, 1] = g * h
        return N

    def source(self):
        _, h, hu, hv, hinv = self._primitives()
        p = self._parameter_symbols
        u = hu * hinv
        w = hv * hinv
        u_mag = sqrt(u * u + w * w + 1e-12)
        h_safe = sp.Max(h, sp.Float(H_FRICTION_FLOOR))
        friction_div = h_safe ** (-sp.Rational(1, 3))
        factor = -p.n ** 2 * p.g * friction_div * u_mag
        S_b = sp.S.Zero
        S_h = sp.S.Zero
        S_hu = factor * u
        S_hv = factor * w
        return ZArray([S_b, S_h, S_hu, S_hv])

    def diffusion_matrix_explicit(self):
        _, h, hu, hv, hinv = self._primitives()
        nu = self._parameter_symbols.nu
        u = hu * hinv
        w = hv * hinv
        A = sp.MutableDenseNDimArray.zeros(4, 4, 2, 2)
        for d in (0, 1):
            A[2, 2, d, d] = nu
            A[3, 3, d, d] = nu
            A[2, 1, d, d] = -nu * u
            A[3, 1, d, d] = -nu * w
        return ZArray(A)

    def update_variables(self):
        v = self.variables
        h, hu, hv = v.h, v.hu, v.hv
        u_max = sp.Float(30.0)
        h_dry = sp.Float(EPS_WD)
        h_wet = sp.Max(h - h_dry, sp.S.Zero)
        max_hu = h_wet * u_max

        def cap(c):
            return sp.Max(-max_hu, sp.Min(c, max_hu))

        return ZArray([v.b, h, cap(hu), cap(hv)])

    def update_variables_jacobian_wrt_variables(self):
        n = self.n_variables
        return ZArray.zeros(n, n)

    def eigenvalues(self):
        _, h, hu, hv, hinv = self._primitives()
        p = self._parameter_symbols
        n = self.normal
        u = hu * hinv
        w = hv * hinv
        un = u * n.n0 + w * n.n1
        c = sqrt(p.g * sp.Max(h, p.eps))
        raw_ev = [sp.S.Zero, un, un - c, un + c]
        cond = sp.Function("conditional")
        gated = [cond(h > p.eps, e, sp.S.Zero) for e in raw_ev]
        return ZArray(gated)

    def update_aux_variables(self):
        v = self.variables
        p = self._parameter_symbols
        h = v.h
        eps = p.eps
        h_floor = sp.Max(h, eps)
        denom = sqrt(h ** 4 + h_floor ** 4)
        hinv_kp = sqrt(2) * h / denom
        return ZArray([hinv_kp])
