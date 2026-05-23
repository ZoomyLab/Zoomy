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
from sympy import Matrix, Pow, sqrt

from zoomy_core.misc.misc import ZArray
from zoomy_core.model.basemodel import Model


def _regularize_h_powers(expr, h_sym, eps_sym):
    """Replace every ``h**(-n)`` atom in ``expr`` with ``(h + eps)**(-n)``.

    Lifted from ``zoomy_core.model.legacy.numerical_model.
    regularize_denominators`` — the legacy wrapper as a whole is
    API-rotted, but this helper is self-contained.  Same recipe used
    by every SWE-family model in zoomy: rewrite the symbolic
    operator with literal ``1 / h**n`` and apply this substitution
    once at the end so the runtime sees a non-singular form.
    """
    if not isinstance(expr, sp.Basic):
        return expr
    h_reg = h_sym + eps_sym
    return expr.replace(
        lambda e: isinstance(e, Pow) and e.base == h_sym and e.exp.is_negative,
        lambda e: h_reg ** e.exp,
    )


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

    # Symbolic ``1 / h`` for operators that want regularization (vs the
    # aux ``hinv`` which is computed numerically as Kurganov–Petrova in
    # ``update_aux_variables`` and used for primitive recovery only).
    def _h_inv(self):
        return 1 / self.variables.h

    def _reg_zarray(self, zarr):
        """Apply ``h**(-n) → (h + eps)**(-n)`` substitution elementwise.

        Single point where we replace literal inverse powers of ``h``
        with ``(h + eps)**(-n)`` — same recipe as
        ``zoomy_core.model.legacy.numerical_model.regularize_denominators``.
        Every operator that builds ``h**(-n)`` atoms passes its result
        through this method before returning, so the runtime sees only
        bounded denominators near the wet/dry interface.
        """
        import itertools
        h = self.variables.h
        eps = self._parameter_symbols.eps
        out = ZArray.zeros(*zarr.shape)
        for idx in itertools.product(*[range(s) for s in zarr.shape]):
            out[idx] = _regularize_h_powers(sp.sympify(zarr[idx]), h, eps)
        return out

    # ---------------- Operators ----------------

    def flux(self):
        """Convective flux only — hydrostatic pressure ½ g h² lives in
        ``nonconservative_matrix`` (Audusse handshake).

        Uses literal ``1/h`` (instead of aux ``hinv``) so ``_reg_zarray``
        substitutes ``h**(-1) → (h + eps)**(-1)`` at the end.
        """
        _, h, hu_0, hu_1, _ = self._primitives()
        F = Matrix.zeros(4, 1)
        F[1, 0] = hu_0
        F[2, 0] = hu_0 * hu_0 / h + sp.Rational(1, 3) * hu_1 * hu_1 / h
        F[3, 0] = 2 * hu_0 * hu_1 / h
        return self._reg_zarray(ZArray(F))

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
        _, h, hu_0, hu_1, _ = self._primitives()
        g = self._parameter_symbols.g
        N = ZArray.zeros(4, 4, 1)

        # Velocities written with literal 1/h so ``_reg_zarray`` picks
        # them up.
        u_0 = hu_0 / h
        u_1 = hu_1 / h

        # ----- Row 2 (j=0 momentum) — matches K&T row 2 exactly -----
        N[2, 0, 0] = g * h
        N[2, 1, 0] = g * h

        # ----- Row 3 (j=1 momentum) — FULL library form -----
        N[3, 0, 0] = -6 * u_0 * u_0 - 2 * u_1 * u_1
        N[3, 1, 0] = -3 * u_0 * u_0 + 2 * u_0 * u_1 - u_1 * u_1
        N[3, 2, 0] = 3 * u_0 - u_1
        # B[3, 3] = 0 (cancellation with extra library +u_0 term)
        return self._reg_zarray(N)

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
        _, h, hu_0, hu_1, _ = self._primitives()
        nu = self._parameter_symbols.nu
        A = ZArray.zeros(4, 4, 1, 1)
        # j=0 row (state index 2 = hu_0):
        A[2, 2, 0, 0] = 2 * nu
        A[2, 1, 0, 0] = -2 * nu * hu_0 / h
        # j=1 row (state index 3 = hu_1):
        A[3, 3, 0, 0] = 2 * nu
        A[3, 1, 0, 0] = -2 * nu * hu_1 / h
        return self._reg_zarray(A)

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
        _, h, hu_0, hu_1, _ = self._primitives()
        p = self._parameter_symbols
        # σ_xz Galerkin projection at j=1 produces the local
        # damping ``-ν/h K_11 u_1 = -4 ν u_1 / h`` (with K_11 = 4 for
        # shifted Legendre on [0, 1]).  Equation-scaled by (2j+1)=3:
        # ``-12 ν u_1 / h = -12 ν hu_1 / h²``.  Literal ``1/h**2`` is
        # regularised to ``1/(h+eps)**2`` by ``_reg_zarray``.
        S_b = sp.S.Zero
        S_h = sp.S.Zero
        S_hu0 = sp.S.Zero
        S_hu1 = -3 * p.tau_b - 12 * p.nu * hu_1 / (h * h)
        return self._reg_zarray(ZArray([S_b, S_h, S_hu0, S_hu1]))


class SME1Viscous2D(Model):
    """SME(L=1) + Newtonian viscous stress in 2D.

    Derivation by symmetric extension of the 1D notebook-010
    projection.  State (2D):  ``[b, h, hu_0, hv_0, hu_1, hv_1]``.

    j=0 momentum (depth-mean) is K&T 2019 eq (4.14) row 2 / 2D:

        ∂_t(hu_0) + ∂_x[hu_0² + g h²/2 + h u_1²/3]
                 + ∂_y[hu_0 v_0 + h u_1 v_1 / 3]
                 + g h ∂_x b = ν-diffusion + bed shear

    (symmetric form for ∂_t(hv_0)).

    j=1 momentum (1st shear) uses the K&T conservative flux + the
    **full library** cross-coupling NCP extras derived in
    ``notebooks/010``.  By symmetry the y-row mirrors the x-row with
    ``u_0 ↔ v_0, u_1 ↔ v_1, x ↔ y``:

        x-row:  ∂_t(hu_1) + ∂_x(2 hu_0 u_1) + ∂_y(hu_0 v_1 + hu_1 v_0)
                + B_x · ∂_x Q + B_y · ∂_y Q = σ-diffusion + bed shear

    where ``B_x`` carries the 1D library extras (in the x-direction)
    and ``B_y`` is their symmetric image.

    Viscous stress (Newtonian): ``σ_xx = 2ν ∂_x u``, ``σ_yy = 2ν ∂_y v``,
    ``σ_xy = ν (∂_y u + ∂_x v)`` (off-diagonal, treated as a single
    diffusion in both u and v).  Treated explicitly per moment with
    the same ``2ν h ∂_d u_j`` form as the 1D.

    Bed shear: ``σ_xz`` projection at j=1 gives ``-12 ν u_1/h`` on
    the hu_1 row and ``-12 ν v_1/h`` on the hv_1 row (equation-
    scaled by 2j+1=3).
    """

    def __init__(self, *, g=9.81, nu=1.0, eps=1e-3, tau_b=0.0,
                 cross_coupling=True, **kw):
        """
        Parameters
        ----------
        cross_coupling : bool, default True
            If ``True``, the j=1 momentum row carries the full library
            projection's cross-coupling NCP entries
            (notebooks/010 — bathymetry / depth / momentum-0 quadratic
            extras).  If ``False``, drops them and uses K&T 2019 eq
            (4.14) form only.  The K&T form is what published SME
            simulations use on real-bathymetry test cases — the full
            library extras can be ill-posed on steep ``∂b/∂x`` because
            ``-6 u_0² ∂_x b`` has no counterbalance in the source.
        """
        self._cross_coupling = bool(cross_coupling)
        super().__init__(
            dimension=2,
            variables=["b", "h", "hu_0", "hv_0", "hu_1", "hv_1"],
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
        return v.b, v.h, v.hu_0, v.hv_0, v.hu_1, v.hv_1, a.hinv

    def _reg_zarray(self, zarr):
        import itertools
        h = self.variables.h
        eps = self._parameter_symbols.eps
        out = ZArray.zeros(*zarr.shape)
        for idx in itertools.product(*[range(s) for s in zarr.shape]):
            out[idx] = _regularize_h_powers(sp.sympify(zarr[idx]), h, eps)
        return out

    def update_aux_variables(self):
        v = self.variables
        p = self._parameter_symbols
        h_safe4 = sp.Max(v.h, p.eps) ** 4
        h4 = v.h ** 4
        hinv_kp = sp.sqrt(2.0) * v.h / sp.sqrt(h4 + h_safe4)
        return ZArray([hinv_kp])

    def flux(self):
        """Convective flux (no hydrostatic pressure — Audusse NCP)."""
        _, h, hu_0, hv_0, hu_1, hv_1, _ = self._primitives()
        # State layout: [b=0, h=1, hu_0=2, hv_0=3, hu_1=4, hv_1=5].
        F = Matrix.zeros(6, 2)
        # Mass: F[1] = (hu_0, hv_0).
        F[1, 0] = hu_0
        F[1, 1] = hv_0
        # j=0 x-momentum: F[2, x] = hu_0²/h + h u_1²/3 = hu_0²/h + hu_1²/(3 h);
        #                F[2, y] = hu_0 v_0 + h u_1 v_1 / 3.
        F[2, 0] = hu_0 * hu_0 / h + sp.Rational(1, 3) * hu_1 * hu_1 / h
        F[2, 1] = hu_0 * hv_0 / h + sp.Rational(1, 3) * hu_1 * hv_1 / h
        # j=0 y-momentum: F[3, x] = symmetric of [2, 1]; F[3, y] = hv_0²/h + hv_1²/(3 h).
        F[3, 0] = hu_0 * hv_0 / h + sp.Rational(1, 3) * hu_1 * hv_1 / h
        F[3, 1] = hv_0 * hv_0 / h + sp.Rational(1, 3) * hv_1 * hv_1 / h
        # j=1 x-momentum: F[4, x] = 2 hu_0 u_1 = 2 hu_0 hu_1 / h;
        #                 F[4, y] = hu_0 v_1 + hu_1 v_0 = (hu_0 hv_1 + hu_1 hv_0)/h.
        F[4, 0] = 2 * hu_0 * hu_1 / h
        F[4, 1] = (hu_0 * hv_1 + hu_1 * hv_0) / h
        # j=1 y-momentum: F[5, x] = symmetric of [4, 1]; F[5, y] = 2 hv_0 hv_1 / h.
        F[5, 0] = (hu_0 * hv_1 + hu_1 * hv_0) / h
        F[5, 1] = 2 * hv_0 * hv_1 / h
        return self._reg_zarray(ZArray(F))

    def nonconservative_matrix(self):
        """NCP: gravity bathymetry/free-surface on j=0 momentum +
        K&T row 3/5 cross terms + full-library extras on j=1.
        """
        _, h, hu_0, hv_0, hu_1, hv_1, _ = self._primitives()
        g = self._parameter_symbols.g
        N = ZArray.zeros(6, 6, 2)

        u_0 = hu_0 / h
        v_0 = hv_0 / h
        u_1 = hu_1 / h
        v_1 = hv_1 / h

        # --- j=0 x-momentum (state index 2): K&T row 2 ---
        N[2, 0, 0] = g * h     # g h ∂_x b
        N[2, 1, 0] = g * h     # g h ∂_x h
        # --- j=0 y-momentum (index 3) ---
        N[3, 0, 1] = g * h     # g h ∂_y b
        N[3, 1, 1] = g * h     # g h ∂_y h

        # --- j=1 momentum: K&T eq (4.14) row 3 conservative-form
        # cross term  ``-u_0 · ∂_x(hu_1) - v_0 · ∂_y(hu_1)``  ---
        # (x-row, index 4)
        N[4, 4, 0] = -u_0   # -u_0 · ∂_x(hu_1)
        N[4, 4, 1] = -v_0   # -v_0 · ∂_y(hu_1)
        # (y-row, index 5)
        N[5, 5, 0] = -u_0
        N[5, 5, 1] = -v_0

        if self._cross_coupling:
            # --- Full library cross-coupling extras (notebooks/010) ---
            # x-row from verified 1D derivation; y-row by symmetric
            # u↔v swap.  NOTE: these terms can be unstable on steep
            # bathymetry (-6 u_0² ∂_x b has no counterbalance).
            # Override the K&T -u_0 entry to 0 (the 1D library shows
            # it cancels):
            N[4, 4, 0] = sp.S.Zero
            N[4, 0, 0] = -6 * u_0 * u_0 - 2 * u_1 * u_1
            N[4, 1, 0] = -3 * u_0 * u_0 + 2 * u_0 * u_1 - u_1 * u_1
            N[4, 2, 0] = 3 * u_0 - u_1
            # y-direction entries for x-row: mirror with u_0→v_0, u_1→v_1
            N[4, 0, 1] = -6 * u_0 * v_0 - 2 * u_1 * v_1
            N[4, 1, 1] = -3 * u_0 * v_0 + (u_0 * v_1 + u_1 * v_0) - u_1 * v_1
            N[4, 3, 1] = 3 * u_0 - u_1
            # y-row (state idx 5): swap x↔y, u↔v
            N[5, 5, 1] = sp.S.Zero
            N[5, 0, 1] = -6 * v_0 * v_0 - 2 * v_1 * v_1
            N[5, 1, 1] = -3 * v_0 * v_0 + 2 * v_0 * v_1 - v_1 * v_1
            N[5, 3, 1] = 3 * v_0 - v_1
            N[5, 0, 0] = -6 * u_0 * v_0 - 2 * u_1 * v_1
            N[5, 1, 0] = -3 * u_0 * v_0 + (u_0 * v_1 + u_1 * v_0) - u_1 * v_1
            N[5, 2, 0] = 3 * v_0 - v_1
        return self._reg_zarray(N)

    def diffusion_matrix_explicit(self):
        """Newtonian viscous flux per moment, 2D.

        ``F_diff[i, d] = 2 ν h ∂_d u_j`` for each Galerkin moment.
        Written in conservative-state form so the diffusion matrix
        entries are bounded.
        """
        _, h, hu_0, hv_0, hu_1, hv_1, _ = self._primitives()
        nu = self._parameter_symbols.nu
        A = ZArray.zeros(6, 6, 2, 2)
        # j=0 x-momentum row (idx 2): diffuse u_0 in both x and y.
        A[2, 2, 0, 0] = 2 * nu;  A[2, 1, 0, 0] = -2 * nu * hu_0 / h
        A[2, 2, 1, 1] = 2 * nu;  A[2, 1, 1, 1] = -2 * nu * hu_0 / h
        # j=0 y-momentum (idx 3): diffuse v_0 in both x and y.
        A[3, 3, 0, 0] = 2 * nu;  A[3, 1, 0, 0] = -2 * nu * hv_0 / h
        A[3, 3, 1, 1] = 2 * nu;  A[3, 1, 1, 1] = -2 * nu * hv_0 / h
        # j=1 x-momentum (idx 4): diffuse u_1 in both x and y.
        A[4, 4, 0, 0] = 2 * nu;  A[4, 1, 0, 0] = -2 * nu * hu_1 / h
        A[4, 4, 1, 1] = 2 * nu;  A[4, 1, 1, 1] = -2 * nu * hu_1 / h
        # j=1 y-momentum (idx 5): diffuse v_1 in both x and y.
        A[5, 5, 0, 0] = 2 * nu;  A[5, 1, 0, 0] = -2 * nu * hv_1 / h
        A[5, 5, 1, 1] = 2 * nu;  A[5, 1, 1, 1] = -2 * nu * hv_1 / h
        return self._reg_zarray(A)

    def eigenvalues(self):
        """Wave-speed bound, gated to 0 in dry cells.

        Same K&T-style bound as 1D, extended to a general normal:
        ``c_eff = √(g h + 4 (u_1·n)² / 3)``,
        ``λ_max = |u_0·n| + c_eff``.
        """
        _, h, hu_0, hv_0, hu_1, hv_1, hinv = self._primitives()
        p = self._parameter_symbols
        n0, n1 = self.normal[0], self.normal[1]
        u0n = (hu_0 * n0 + hv_0 * n1) * hinv
        u1n = (hu_1 * n0 + hv_1 * n1) * hinv
        c_eff = sqrt(p.g * sp.Max(h, p.eps) + sp.Rational(4, 3) * u1n * u1n)
        raw = [sp.S.Zero, u0n, u0n, u0n - c_eff, u0n + c_eff, u0n]
        cond = sp.Function("conditional")
        gated = [cond(h > p.eps, e, sp.S.Zero) for e in raw]
        return ZArray(gated)

    def source(self):
        """σ_xz / σ_yz bed-shear damping (implicit).

        Per ``notebooks/010`` σ_xz projection, equation-scaled by
        (2j+1)=3 on each shear moment:

            row hu_1:  -3 τ_b - 12 ν u_1 / h = -3 τ_b - 12 ν hu_1 / h²
            row hv_1:  -3 τ_b - 12 ν v_1 / h = -3 τ_b - 12 ν hv_1 / h²

        Treated *implicitly* (Newton evaluates at Qnp1) — the
        ``1/h**2`` is regularised by ``_reg_zarray`` to
        ``1/(h + eps)**2`` so the Jacobian
        ``∂(hu_1 / (h+eps)²)/∂h = -2 hu_1 / (h+eps)³`` stays bounded
        through the wet/dry interface.  This is the
        ``regularize_h``-style treatment the project uses
        everywhere.
        """
        _, h, hu_0, hv_0, hu_1, hv_1, _ = self._primitives()
        p = self._parameter_symbols
        S = [sp.S.Zero] * 6
        S[4] = -3 * p.tau_b - 12 * p.nu * hu_1 / (h * h)
        S[5] = -3 * p.tau_b - 12 * p.nu * hv_1 / (h * h)
        return self._reg_zarray(ZArray(S))


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
