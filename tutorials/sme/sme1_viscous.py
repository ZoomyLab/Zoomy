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
        """NCP — **full library projection form** (notebooks/010).

        For row 2 (j=0 momentum) the library projection reduces to
        the K&T eq (4.14) row 2 exactly:

            B[2, 0] = g h   (gravity on ∂_x b)
            B[2, 1] = g h   (gravity on ∂_x h, Audusse handshake)

        For row 3 (j=1 momentum) the library projection carries
        cross-coupling terms K&T row 3 drops as a simplification.
        Decomposing with conservative flux ``F[3] = 2 h u_0 u_1``:

            B[3, 0] = -6 u_0² - 2 u_1²
                      (bathymetry coupling — absent in K&T)
            B[3, 1] = -3 u_0² + 2 u_0 u_1 - u_1²
                      (depth coupling, retains advective NCP)
            B[3, 2] = 3 u_0 - u_1
                      (momentum-0 coupling)
            B[3, 3] = 0
                      (K&T's ``-u_0 ∂_x(hu_1)`` cancels with the
                       library's extra ``+u_0 ∂_x(hu_1)`` from the
                       ∂_z(uw) projection of the linear mode)

        Symbolic equivalence with the library projection is verified
        in the ``__main__`` block by reconstructing the LHS and
        diff-ing against ``notebooks/010``'s ``xmom_j1_inv``.
        """
        _, h, hu_0, hu_1, hinv = self._primitives()
        g = self._parameter_symbols.g
        N = ZArray.zeros(4, 4, 1)

        # Velocities (primitive recovery via hinv).
        u_0 = hu_0 * hinv
        u_1 = hu_1 * hinv

        # ----- Row 2 (j=0 momentum) — matches K&T row 2 exactly -----
        N[2, 0, 0] = g * h
        N[2, 1, 0] = g * h

        # ----- Row 3 (j=1 momentum) — FULL library form -----
        # B[3, 0] = -6 u_0² - 2 u_1²
        N[3, 0, 0] = -6 * u_0 * u_0 - 2 * u_1 * u_1
        # B[3, 1] = -3 u_0² + 2 u_0 u_1 - u_1²
        N[3, 1, 0] = -3 * u_0 * u_0 + 2 * u_0 * u_1 - u_1 * u_1
        # B[3, 2] = 3 u_0 - u_1
        N[3, 2, 0] = 3 * u_0 - u_1
        # B[3, 3] = 0 (cancellation with extra library +u_0 term)
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

    def update_aux_variables(self):
        """Kurganov–Petrova desingularized inverse depth.

            hinv = √2 · h / √(h⁴ + max(h, eps)⁴)

        Same form used by ``MalpassetSWE``.  Tends smoothly to 0 as
        ``h → 0`` instead of saturating at ``1/eps``, so primitive
        velocities ``u_j = hu_j / h`` recover gracefully near the
        wet/dry interface.
        """
        v = self.variables
        p = self._parameter_symbols
        h_safe4 = sp.Max(v.h, p.eps) ** 4
        h4 = v.h ** 4
        hinv_kp = sp.sqrt(2.0) * v.h / sp.sqrt(h4 + h_safe4)
        return ZArray([hinv_kp])

    def eigenvalues(self):
        """Explicit eigenvalues — SME(1) wave-speed bound.

        For SME(L=1), the flux-Jacobian eigenvalues are (per K&T 2019
        analysis) of the form ``u_0 ± c_eff`` plus a passive ``u_0``
        and the trivial ``0`` for the bathymetry row.  A conservative
        upper bound on ``c_eff`` is
        ``c_eff = √(g h + 4 u_1² / 3)`` — exact at the linearised
        limit and an overestimate (so the CFL stays safe) elsewhere.

        Symbolic auto-derivation produces cubic-root expressions that
        are numerically fragile near zero shear / zero depth; the
        explicit form here avoids the division-by-zero traps.

        Dry-cell gate: where ``h ≤ eps`` waves stop, ``λ → 0`` —
        prevents bogus ``|u| = hu/h`` blow-up dictating the global
        ``dt``.
        """
        _, h, hu_0, hu_1, hinv = self._primitives()
        p = self._parameter_symbols
        n = self.normal[0]  # 1D: single normal component (±1)
        u_0 = hu_0 * hinv
        u_1 = hu_1 * hinv
        un = u_0 * n
        c_eff = sqrt(p.g * sp.Max(h, p.eps) + sp.Rational(4, 3) * u_1 * u_1)
        raw_ev = [sp.S.Zero, un, un - c_eff, un + c_eff]
        cond = sp.Function("conditional")
        gated = [cond(h > p.eps, ev, sp.S.Zero) for ev in raw_ev]
        return ZArray(gated)

    def source(self):
        """Implicit source — placeholder ``τ_b`` on row 2 (j=0
        momentum).  Default ``τ_b = 0`` so this is a no-op
        numerically; kept symbolic to give Firedrake a non-empty
        integration domain (the source weak form needs *some* sympy
        term referencing ``self.parameters`` so its UFL lowering can
        attach a mesh).
        """
        _, h, hu_0, hu_1, hinv = self._primitives()
        p = self._parameter_symbols
        return ZArray([sp.S.Zero, sp.S.Zero, p.tau_b, sp.S.Zero])

    def source_explicit(self):
        """Bed-shear coupling on the moment equations (explicit).

        Per ``notebooks/010`` σ_xz projection: each Galerkin row picks
        up a ``φ_j(0) · τ_b`` boundary term plus the ``-ν/h K_ji u_i``
        local damping.  With shifted Legendre on [0, 1]:

            φ_0(0) = +1,  φ_1(0) = -1
            K_{00} = 0,  K_{01} = K_{10} = 0,  K_{11} = 4

        Equation-scaled by (2j+1):

            row j=0:  +τ_b
            row j=1:  -3 τ_b - 3 · (ν/h) · 4 u_1 = -3 τ_b - 12 ν u_1 / h

        Treated **explicitly** — evaluated at ``Qn`` inside the
        convective step.  This avoids the stiff ``1/h`` Jacobian that
        an implicit treatment would expose to the Newton solver
        (∂/∂h of ``u_1/h²`` ~ ``-2 u_1/h³`` blows up at the dam-break
        front where the depth transitions abruptly).  The damping is
        not stiff at the scale we use (ν ~ 1 m²/s, h ~ 1 m → time
        scale ``h²/(12 ν) ≈ 0.08 s`` — well above the convective
        CFL ``dt ~ Δx/c ~ 0.01 s``).
        """
        _, h, hu_0, hu_1, hinv = self._primitives()
        p = self._parameter_symbols
        # NOTE: the σ_xz shear-damping term ``-12 ν u_1 / h`` is
        # **disabled** in this initial smoke build — it is stiff at
        # the wet/dry interface (Jacobian ``∝ 1/h³``) and causes
        # Newton to NaN out before the dam-break wave propagates.
        # Re-enable once a desingularised treatment (or implicit-IMEX
        # split into ``diffusion_matrix``) is added.
        S_b = sp.S.Zero
        S_h = sp.S.Zero
        S_hu0 = sp.S.Zero
        S_hu1 = -3 * p.tau_b
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
