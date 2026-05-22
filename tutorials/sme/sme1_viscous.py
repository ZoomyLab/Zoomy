"""SME(1) with Newtonian viscous stress — SystemModel in 1D.

Materialises the symbolic derivation from
``notebooks/010_sme1_derivation.py`` as a runnable Model subclass.

State (1D):     ``[b, h, hu_0, hu_1]``
Aux:            ``[hinv]``  (= 1/h regularization)
Dimension:      1

Equations (K&T 2019 eq (4.14), augmented with Newtonian viscous
diffusion + bed-shear coupling — see ``notebooks/010``):

    ∂_t h     + ∂_x(h u_0)                              = 0
    ∂_t(hu_0) + ∂_x(hu_0² + ½ g h² + h u_1²/3)
              + g h · ∂_x b                             = ∂_x(2ν h ∂_x u_0) + τ_b
    ∂_t(hu_1) + ∂_x(2 hu_0 u_1) - u_0 ∂_x(hu_1)         = ∂_x(2ν h ∂_x u_1)
              -                                          12 ν u_1 / h - 3 τ_b

The hydrostatic pressure ½ g h² is wired through
``nonconservative_matrix`` so the well-balancing handshake works the
same way as ``MalpassetSWE`` (matches the Audusse/Castro-Parés route
the Firedrake backend uses).

The viscous flux ``2ν h ∂_x u_j`` is rewritten in conservative form
as ``2ν · ∂_x(hu_j) - 2ν u_j · ∂_x h`` and entered into
``diffusion_matrix_explicit`` — same trick as ``MalpassetSWE`` for
the depth-averaged SWE viscous term.

τ_b is held as a tunable bed-shear source.  Default ``tau_b = 0`` so
the model reduces to inviscid + viscous-diffusion only.
"""
from __future__ import annotations

import sympy as sp
from sympy import Matrix, sqrt

from zoomy_core.misc.misc import ZArray
from zoomy_core.model.basemodel import Model


class SME1Viscous(Model):
    """Shallow Moment Equation L=1 + Newtonian viscous stress (1D)."""

    def __init__(self, *, g=9.81, nu=1.0, eps=1e-3, tau_b=0.0, **kw):
        super().__init__(
            dimension=1,
            variables=["b", "h", "hu_0", "hu_1"],
            aux_variables=["hinv"],
            parameters={
                "g":     (float(g),     "positive"),
                "nu":    (float(nu),    "non-negative"),
                "eps":   (float(eps),   "positive"),
                "tau_b": (float(tau_b), None),
            },
            eigenvalue_mode="symbolic",
            **kw,
        )

    def _primitives(self):
        v = self.variables
        a = self.aux_variables
        return v.b, v.h, v.hu_0, v.hu_1, a.hinv

    # ---------------- Operators ----------------

    def flux(self):
        """Convective flux only — hydrostatic pressure ½ g h² lives in
        ``nonconservative_matrix`` (Audusse handshake).
        """
        _, h, hu_0, hu_1, hinv = self._primitives()
        F = Matrix.zeros(4, 1)
        # mass:  F[1] = h u_0 = hu_0
        F[1, 0] = hu_0
        # j=0 momentum:  F[2] = h u_0² + h u_1²/3 = hu_0·u_0 + (hu_1)·u_1/3
        F[2, 0] = hu_0 * hu_0 * hinv + sp.Rational(1, 3) * hu_1 * hu_1 * hinv
        # j=1 momentum:  F[3] = 2 h u_0 u_1 = 2 (hu_0)(hu_1)/h
        F[3, 0] = 2 * hu_0 * hu_1 * hinv
        return ZArray(F)

    def nonconservative_matrix(self):
        """NCP: gravity bathymetry/free-surface coupling on j=0
        momentum (g h ∂_x b + g h ∂_x h) and K&T row 3 cross term
        ``-u_0 · ∂_x(hu_1)`` on j=1.
        """
        _, h, hu_0, hu_1, hinv = self._primitives()
        g = self._parameter_symbols.g
        N = ZArray.zeros(4, 4, 1)
        # j=0 momentum row (index 2)
        N[2, 0, 0] = g * h       # coefficient of ∂_x b
        N[2, 1, 0] = g * h       # coefficient of ∂_x h
        # j=1 momentum row (index 3): -u_0 · ∂_x(hu_1)
        N[3, 3, 0] = -hu_0 * hinv  # coefficient of ∂_x(hu_1)
        return N

    def diffusion_matrix_explicit(self):
        """Newtonian viscous diffusion of velocity per moment.

        Per-moment flux is ``2ν h ∂_x u_j``.  Rewrite in conservative
        form using ``u_j = (hu_j)/h``:

            2ν h ∂_x u_j = 2ν ∂_x(hu_j) - 2ν u_j ∂_x h

        so the diffusion-matrix entries are

            A[hu_j, hu_j, 0, 0] = 2ν    (diffuses momentum)
            A[hu_j, h,    0, 0] = -2ν u_j   (cross-state on h)

        Explicit treatment — folded into the convective step at Qn,
        bounded by the parabolic CFL ``dt ≤ h²/(2ν)``.
        """
        _, h, hu_0, hu_1, hinv = self._primitives()
        nu = self._parameter_symbols.nu
        A = ZArray.zeros(4, 4, 1, 1)
        # j=0 row (state index 2 = hu_0):
        A[2, 2, 0, 0] = 2 * nu
        A[2, 1, 0, 0] = -2 * nu * hu_0 * hinv
        # j=1 row (state index 3 = hu_1):
        A[3, 3, 0, 0] = 2 * nu
        A[3, 1, 0, 0] = -2 * nu * hu_1 * hinv
        return A

    def source(self):
        """Bed-shear coupling on the moment equations.

        Per ``notebooks/010`` σ_xz projection: each Galerkin row picks
        up a ``φ_j(0) · τ_b`` boundary term plus the ``-ν/h K_ji u_i``
        local damping.  With shifted Legendre on [0, 1]:

            φ_0(0) = +1,  φ_1(0) = -1
            K_{00} = 0,  K_{01} = K_{10} = 0,  K_{11} = 4

        Equation-scaled by (2j+1):

            row j=0:  +τ_b
            row j=1:  -3 τ_b - 3 · (ν/h) · 4 u_1 = -3 τ_b - 12 ν u_1 / h
        """
        _, h, hu_0, hu_1, hinv = self._primitives()
        p = self._parameter_symbols
        S_b = sp.S.Zero
        S_h = sp.S.Zero
        S_hu0 = p.tau_b
        # j=1 viscous-shear damping: -12 ν u_1 / h.  ``u_1 = hu_1 · hinv``.
        S_hu1 = -3 * p.tau_b - 12 * p.nu * hu_1 * hinv * hinv
        return ZArray([S_b, S_h, S_hu0, S_hu1])


# ---- Quick reduction sanity check ----

def reduces_to_swe(model: "SME1Viscous") -> bool:
    """At hu_1 ≡ 0 the j=0 momentum flux + NCP must coincide with SWE.

    Symbolic check: F[2] and N[2, *] with hu_1 → 0 give the SWE
    momentum flux + bathymetry NCP exactly.
    """
    v = model.variables
    flux = model.flux()
    ncp = model.nonconservative_matrix()
    zero_hu1 = {v.hu_1: sp.S.Zero}
    F2 = sp.expand(sp.sympify(flux[2, 0]).xreplace(zero_hu1))
    N2_0 = sp.expand(sp.sympify(ncp[2, 0, 0]).xreplace(zero_hu1))
    N2_1 = sp.expand(sp.sympify(ncp[2, 1, 0]).xreplace(zero_hu1))
    g = model._parameter_symbols.g
    swe_F2 = v.hu_0 * v.hu_0 * model.aux_variables.hinv
    if sp.simplify(F2 - swe_F2) != 0:
        return False
    if sp.simplify(N2_0 - g * v.h) != 0 or sp.simplify(N2_1 - g * v.h) != 0:
        return False
    return True


if __name__ == "__main__":
    m = SME1Viscous()
    print("SME(1) viscous SystemModel:")
    print(f"  variables    : {list(m.variables.keys())}")
    print(f"  aux_variables: {list(m.aux_variables.keys())}")
    print(f"  parameters   : {list(m.parameters.keys()) if hasattr(m, 'parameters') else 'n/a'}")
    print()
    print("flux F[i, 0]:")
    F = m.flux()
    for i in range(4):
        print(f"  F[{i}] = {sp.sympify(F[i, 0])}")
    print()
    print("nonconservative_matrix B[i, j, 0]:")
    N = m.nonconservative_matrix()
    for i in range(4):
        for j in range(4):
            val = sp.sympify(N[i, j, 0])
            if val != 0:
                print(f"  B[{i},{j}] = {val}")
    print()
    print("diffusion_matrix_explicit A[i, j, 0, 0]:")
    A = m.diffusion_matrix_explicit()
    for i in range(4):
        for j in range(4):
            val = sp.sympify(A[i, j, 0, 0])
            if val != 0:
                print(f"  A[{i},{j}] = {val}")
    print()
    print("source S[i]:")
    S = m.source()
    for i in range(4):
        val = sp.sympify(S[i])
        if val != 0:
            print(f"  S[{i}] = {val}")
    print()
    print("SWE reduction (hu_1=0):", "✓ PASS" if reduces_to_swe(m) else "✗ FAIL")
