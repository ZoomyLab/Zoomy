"""Predictor / pressure / corrector splitter for non-hydrostatic SystemModels.

=========================================================================
CANONICAL SPLITTER:  ``split_for_pressure_structural``
=========================================================================
This is the ONE generic, supported pressure splitter — use it for every
non-hydrostatic model (VAM, ML-VAM, …).  It is what ``model.chorin_split(dt)``
calls (see ``VAM.chorin_split`` / ``MLVAM.chorin_split``).  It is "structural":
it discovers the row roles purely from the OPERATORS — the divergence
constraints are exactly the rows with an identically-zero mass-matrix row, the
rest are evolution rows — so it needs NO ``equation_names`` convention and works
for any level / dimension / basis straight out of a declarative derivation.

    >>> split = VAM(level=1).chorin_split(dt)      # (SM_pred, SM_press, SM_corr)
    >>> split = split_for_pressure_structural(sm, P_syms, dt)   # explicit form

DEPRECATED — do NOT use for new work (each raises a ``DeprecationWarning``):
  * :func:`split_for_pressure` — old chain-DAE splitter; dispatches rows by the
    fragile ``equation_names`` convention (``mass`` / ``xmom_j*`` / ``zmom_j*`` /
    ``cont_j*``).  A declarative SystemModel carries no such names.
  * :func:`split_simple` — minimal hand-rolled splitter; used only by the
    hand-built ``escalante_vam_bump/run.py`` oracle (also name-based).
  * :func:`build_pressure_elliptic_block` — internal helper for the two
    deprecated splitters; the structural splitter does NOT use it.

The deprecated three are kept only so the legacy hand-built reference cases keep
running; new models and agents must go through ``chorin_split`` /
``split_for_pressure_structural``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import sympy as sp

from zoomy_core.misc.misc import ZArray
from zoomy_core.systemmodel.system_model import SystemModel


# ---------------------------------------------------------------------------
# Internal: pull row residuals from a SystemModel via reconstruct_residuals.
# ---------------------------------------------------------------------------


def _residuals_by_name(sm):
    """Return ``{equation_name: residual_expr}`` for the SystemModel."""
    if not hasattr(sm, "equation_names"):
        raise ValueError(
            "SystemModel must carry equation_names for the splitter to "
            "dispatch rows by name."
        )
    residuals = sm.reconstruct_residuals()
    return dict(zip(sm.equation_names, residuals))


def _state_symbol_by_name(sm):
    """Map state-entry name (str) → Symbol."""
    return {str(s): s for s in sm.state}


# ---------------------------------------------------------------------------
# Build pressure elliptic block.
# ---------------------------------------------------------------------------


def build_pressure_elliptic_block(
    sm,
    pressure_vars: Sequence,
    dt: sp.Symbol,
    *,
    bottom=None,
):
    """Build the pressure-stage elliptic block for ``SM_press``.

    Parameters
    ----------
    sm : SystemModel
        Chain-DAE SystemModel (e.g. ``SystemModel.from_model(
        VAMModelGalerkin(level=1))``).  Must carry ``equation_names``
        with the canonical mass / xmom_j / zmom_j / cont_j layout.
    pressure_vars : Sequence
        Pressure mode state Symbols (e.g. ``[P_0, P_1]`` for
        VAM(1,2,2)).  Internally converted to Function form to match
        the residual representation produced by
        :meth:`SystemModel.reconstruct_residuals`.
    dt : sp.Symbol
        Symbolic time-step.
    bottom : optional
        Bottom-topography Function ``b(x)``.  If None, located by
        scanning equation residuals for a Function named ``b``.

    Returns
    -------
    dict with keys ``rows``, ``U_tilde``, ``W_tilde``, ``T_u``, ``T_w``,
    ``U_corr``, ``W_corr``, ``pressure_vars``, ``h``, ``bottom``, ``t``,
    ``x``.  All entries are in Function form (coordinate-dependent).
    """
    residuals = _residuals_by_name(sm)
    name_to_sym = _state_symbol_by_name(sm)

    if "h" not in name_to_sym:
        raise ValueError("SystemModel must include the ``h`` state entry.")
    t = sm.time
    coords = list(sm.space)
    x = coords[0]

    # Convert state Symbols to coordinate-dependent Functions to match
    # the reconstruct_residuals representation.
    def _to_fn(sym):
        return sp.Function(str(sym), real=True)(t, *coords)

    h_fn = _to_fn(name_to_sym["h"])

    # Discover momentum-mode state entries by name.  Supports both
    # primitive form (``U_k``, ``W_k``) and conservative form
    # (``q_Uk``, ``q_Wk``) — the splitter is agnostic, picks whichever
    # naming the SystemModel uses.  Pressure modes are excluded
    # explicitly (``pressure_vars`` is passed by the caller).
    def _collect(prefixes):
        out = []
        k = 0
        while True:
            for pref in prefixes:
                key = f"{pref}{k}"
                if key in name_to_sym:
                    out.append(_to_fn(name_to_sym[key]))
                    break
            else:
                break
            k += 1
        return out

    coeffs_u = _collect(("U_", "q_U"))
    coeffs_w = _collect(("W_", "q_W"))
    M_M = len(coeffs_u) - 1
    N_w_active = len(coeffs_w)
    if M_M < 0:
        raise ValueError(
            "SystemModel has no U_k / q_Uk state entries."
        )
    if N_w_active < 1:
        raise ValueError(
            "SystemModel has no W_k / q_Wk state entries."
        )

    # Pressure Functions in residual form.
    pressure_funcs = [_to_fn(p) for p in pressure_vars]

    # The bottom topography is a coordinate-dependent Function ``b``;
    # locate it by scanning residual atoms (it isn't a state entry).
    if bottom is None:
        for eq in residuals.values():
            for atom in eq.atoms(sp.Function):
                if atom.func.__name__ == "b":
                    bottom = atom
                    break
            if bottom is not None:
                break
    if bottom is None:
        raise ValueError(
            "Could not locate bottom topography 'b' in residuals."
        )

    def mu(j):
        return sp.Rational(1, 2 * j + 1)

    # Per-momentum-row pressure sources T_k(P).  ``_extract_pressure_T``
    # walks the row residual for pressure-dependent atoms.  Unified
    # treatment — U and W rows go through the same machinery.
    T_u = [
        _extract_pressure_T(residuals, f"xmom_j{k}", pressure_funcs, mu(k))
        for k in range(M_M + 1)
    ]
    T_w = [
        _extract_pressure_T(residuals, f"zmom_j{k}", pressure_funcs, mu(k))
        for k in range(N_w_active)
    ]

    # Corrector update.  Universal formula derived from the row residual
    # ``M[k, e2s] · ∂_t state[e2s] + ... + p_terms = 0`` ⇒
    # ``state_new = state - dt · p_terms / M[k, e2s]``.  This handles
    # both primitive (M = μ_k · h, after ``remove_non_diagonal_h``) and
    # modal-conservative (M = 1, after ``InvertMassMatrix``) states
    # without branching: the mass-matrix entry self-selects the right
    # ratio.  The previous form ``state - (dt/h) · T_u[k]`` baked in the
    # primitive choice and was off by a factor ``h · μ_k`` for
    # modal-conservative state — invisible when h ≈ 1, fatal for the
    # dam-break test (h ranges 0.015–0.34).
    name_to_eq_idx = {n: i for i, n in enumerate(sm.equation_names)}
    state_syms = list(sm.state)
    sym_to_fn   = dict(zip(state_syms, [_to_fn(s) for s in state_syms]))

    def _pressure_part(eq_expr):
        e    = sp.expand(sp.sympify(eq_expr).doit())
        no_p = sp.expand(e.subs({p: sp.S.Zero for p in pressure_funcs}).doit())
        return sp.expand(e - no_p)

    def _mass_entry_in_func_form(row_idx, state_idx):
        return sp.sympify(sm.mass_matrix[row_idx, state_idx]).xreplace(sym_to_fn)

    def _corrector_for(coeffs, prefix):
        out = []
        for k, c in enumerate(coeffs):
            eq_name = f"{prefix}_j{k}"
            if eq_name not in name_to_eq_idx:
                # Row doesn't exist (defensive — coeffs comes from state
                # names; the row may not match by convention).
                out.append(c)
                continue
            row_idx = name_to_eq_idx[eq_name]
            e2s     = sm.equation_to_state_index[row_idx]
            M_kk    = _mass_entry_in_func_form(row_idx, e2s)
            if M_kk == 0:
                raise ValueError(
                    f"Row '{eq_name}' has zero mass-matrix diagonal — "
                    "cannot build corrector update."
                )
            p_terms = _pressure_part(residuals[eq_name])
            out.append(sp.expand(c - dt * p_terms / M_kk))
        return out

    U_corr = _corrector_for(coeffs_u, "xmom")
    W_corr = _corrector_for(coeffs_w, "zmom")

    # Elliptic block: substitute U_k → U_k - (dt/h)·T_u_k(P) (and
    # similarly W_k) into every algebraic continuity row.  Sympy's
    # ``.subs`` is single-pass: the substituted-in expression's U_k
    # remains the state Function, not a tilde alias.
    repl = {coeffs_u[k]: U_corr[k] for k in range(M_M + 1)}
    repl.update({coeffs_w[k]: W_corr[k] for k in range(N_w_active)})

    elliptic_rows = {}
    for name, eq in residuals.items():
        if name.startswith("cont_j"):
            j = int(name[len("cont_j"):])
            # ``.doit()`` distributes the compound ``Derivative`` atoms
            # introduced by the substitution down to atomic derivatives.
            substituted = sp.expand(eq.subs(repl).doit())
            elliptic_rows[j] = substituted

    return {
        "rows": elliptic_rows,
        "T_u": T_u,
        "T_w": T_w,
        "U_corr": U_corr,
        "W_corr": W_corr,
        "pressure_vars": list(pressure_funcs),
        "h": h_fn,
        "bottom": bottom,
        "t": t,
        "x": x,
    }


def verify_p_linearity(rows, pressure_vars, x):
    """Verify each row is linear in ``(P_l, ∂_x P_l, ∂_xx P_l)``."""
    P_VARS = []
    for p in pressure_vars:
        name = str(p)
        P_VARS.append((name, p))
        P_VARS.append((f"d_x {name}", sp.Derivative(p, x)))
        P_VARS.append((f"d_xx {name}", sp.Derivative(p, x, x)))

    dummies = [sp.Dummy(n) for n, _ in P_VARS]
    repl = {atom: d for d, (_, atom) in zip(dummies, P_VARS)}

    coeffs_out = {}
    constants_out = {}
    for j, row in rows.items():
        e = sp.expand(row).doit()
        e = sp.expand(e)
        e_poly = e.xreplace(repl)
        leftover_p = [p for p in pressure_vars if e_poly.has(p)]
        leftover_d = [
            d for d in e_poly.atoms(sp.Derivative)
            if d.args[0] in pressure_vars
        ]
        if leftover_p or leftover_d:
            raise AssertionError(
                f"row {j}: not in span of (P, ∂_x P, ∂_xx P); "
                f"unexpected atoms: {leftover_p + leftover_d}"
            )
        try:
            poly = sp.Poly(e_poly, *dummies)
        except sp.PolynomialError as exc:
            raise AssertionError(
                f"row {j}: pressure dependence is not polynomial: {exc}"
            )
        if poly.total_degree() > 1:
            raise AssertionError(
                f"row {j}: pressure dependence is degree "
                f"{poly.total_degree()}, expected ≤ 1."
            )
        constants_out[j] = poly.nth(*([0] * len(dummies)))
        coeffs_out[j] = {}
        for i, (name, _atom) in enumerate(P_VARS):
            idx = [0] * len(dummies)
            idx[i] = 1
            coeffs_out[j][name] = sp.simplify(poly.nth(*idx))

    return {"coefficients": coeffs_out, "constants": constants_out}


# ---------------------------------------------------------------------------
# Internal: pressure-source extraction.
# ---------------------------------------------------------------------------


def _extract_pressure_T(residuals, equation_name, pressure_vars, mu_k):
    """Extract ``T_*[k]`` from a chain-DAE residual.

    ``residual = μ_k ∂_t(h·Q_k) + … + pressure_terms = 0``.  The
    pressure-linear part divided by ``μ_k`` is the source per
    conserved variable ``h·Q_k``.

    ``.doit()`` is called first so compound ``Derivative(A+P*B, x)``
    atoms distribute via the product / sum rule; otherwise the
    eq-vs-no_p subtraction leaves spurious cancelling pairs.
    """
    eq = sp.expand(residuals[equation_name].doit())
    no_p = sp.expand(eq.subs({p: sp.S.Zero for p in pressure_vars}).doit())
    pressure_part = sp.expand(eq - no_p)
    return sp.expand(pressure_part / mu_k)


# ---------------------------------------------------------------------------
# Three-sub-system splitter.
# ---------------------------------------------------------------------------


def _resolve_subs(arr):
    """Collapse every bed/surface-trace ``Subs(f(ζ), ζ, 0|1)`` node in a
    sub-system operator array into its concrete ``Σ αₖ φₖ(0|1)`` value.

    Same discipline as REQ-130's ``_resolve_boundary_traces`` at the NSM
    lowering seam, but applied HERE at chorin-split construction so the
    SM_pred / SM_press / SM_corr sub-SystemModels are Subs-free for ALL
    consumers — the amrex Chorin header generator reads their operators
    directly (``write_chorin_headers`` → ``AmrexSystemModelPrinter``), never
    through ``NumericalSystemModel.from_system_model``, so the central resolve
    never reaches them and a leftover ``Subs`` prints as an undeclared C++
    symbol.  Only ``sp.Subs`` nodes are ``.doit()``-ed (never opaque Galerkin
    bracket integrals), matching REQ-130."""
    if arr is None or not hasattr(arr, "applyfunc"):
        return arr
    is_subs = lambda n: isinstance(n, sp.Subs)
    resolve = lambda n: n.doit()

    def _fix(e):
        e = sp.sympify(e)
        return e.replace(is_subs, resolve) if e.has(sp.Subs) else e

    return arr.applyfunc(_fix)


def _zero_ndim(*shape):
    """A zero rank-3 NCP array whose entries are sympy ``Zero`` (not the
    Python ``int`` 0 ``MutableDenseNDimArray.zeros`` fills with).

    ``from_model`` operators are sympy expressions throughout; the C++ /
    foam code printers walk them with ``expr.replace(...)``.  A bare
    Python ``int`` entry (as produced by ``...zeros(...)`` directly) has
    no ``.replace`` and crashes the printer — so the corrector's all-zero
    flux/NCP must be sympified to print standalone like any other model."""
    return sp.MutableDenseNDimArray.zeros(*shape).applyfunc(sp.sympify)


def _partition_pressure_aux(sm_press, pressure_vars):
    """Split ``sm_press.aux_registry`` into LIVE vs INPUT derivative aux.

    A Chorin pressure stage solves an elliptic block for the pressure
    modes ``P_l``.  During that solve the ONLY aux that changes is the
    pressure derivatives (``P_l_x``, ``P_l_xx``); every other derivative
    in the elliptic source (``q_k_x``, ``h_x``, ``h_xx``, ``b_x`` …) is a
    derivative of the *frozen* predictor state and is therefore a constant
    INPUT to the pressure solve — re-deriving it inside the Krylov loop is
    wasted work and, worse, misrepresents the model: a standalone Model
    printer reading ``aux_registry`` would emit compute-derivative calls
    for aux this sub-model does not actually re-derive.

    This routine makes ``sm_press`` self-describing:

    * ``aux_registry``        — keeps EXACTLY the pressure-mode derivative
      entries (``target_name`` is a pressure variable).  This is the
      minimal "what I re-derive" list a Model printer emits as the natural
      ``update_aux_variables`` — no ``state_index_filter`` needed.
    * ``aux_input_registry``  — the frozen predictor-produced derivative /
      function aux this stage READS but does not re-derive.  ``aux_state``
      is left whole (both groups remain inputs the kernels reference); a
      runtime that keeps a private pressure aux-pool fills it once per step
      from ``aux_registry + aux_input_registry`` (the input group is
      constant across the Krylov iterations).

    Idempotent and a no-op when there is no live pressure derivative.
    """
    registry = list(getattr(sm_press, "aux_registry", None) or [])
    live_names = {str(p) for p in pressure_vars}

    def _is_pressure_derivative(entry):
        return (entry.get("kind") in ("derivative", "limited_derivative")
                and entry.get("target_name") in live_names)

    live = [e for e in registry if _is_pressure_derivative(e)]
    inputs = [e for e in registry if not _is_pressure_derivative(e)]
    sm_press.aux_registry = live
    sm_press.aux_input_registry = inputs
    return sm_press


@dataclass
class PressureLinearOperator:
    """Explicit coefficient fields of the affine pressure operator.

    ``SM_press.source`` is AFFINE in the pressure modes and their spatial
    derivatives (that is why the matrix-free probe path works):

    .. math::

        \\mathrm{source}_k(P)
          = \\sum_l \\Big[ A0_{kl}\\,P_l
              + \\sum_a Ad^{(a)}_{kl}\\,\\partial_{x_a} P_l
              + \\sum_{a\\le b} Add^{(ab)}_{kl}\\,\\partial_{x_a x_b} P_l \\Big]
            - \\mathrm{RHS}_k .

    This carrier exposes those coefficients as printable sympy expressions in
    the SAME symbol universe as ``SM_press.source`` (frozen predictor state,
    params, and the input-aux derivative symbols), so the existing code
    printers emit them unchanged.  It is produced lazily by
    :meth:`SystemModel.pressure_operator` (attached to ``SM_press`` by the
    splitter) — it does NOT re-derive the model: it symbolically
    differentiates the already-split ``source``.

    Fields
    ------
    P_modes : list[Symbol]
        The ``NP`` pressure state symbols in ``SM_press.equation_to_state_index``
        order (the code's canonical pressure-mode ordering, ``e2s_press``).
    A0 : list[list]                 (NP×NP)   ``d source_k / d P_l``.
    first_derivative : dict[int, list[list]]
        ``axis → (NP×NP)`` coefficient of ``∂_{x_axis} P_l``.
    second_derivative : dict[tuple[int,int], list[list]]
        ``(a, b) with a ≤ b → (NP×NP)`` coefficient of ``∂_{x_a x_b} P_l``.
    RHS : list                      (length NP)   ``-source_k |_{P=0, dP=0}``.

    Convenience named accessors (``None`` when the axis is absent for the
    model's dimension): ``Ax``/``Ay``/``Az`` (first) and
    ``Axx``/``Ayy``/``Azz``/``Axy``/``Axz``/``Ayz`` (second).
    """
    P_modes: list
    A0: list
    first_derivative: dict
    second_derivative: dict
    RHS: list

    def _first(self, axis):
        return self.first_derivative.get(axis)

    def _second(self, a, b):
        return self.second_derivative.get((min(a, b), max(a, b)))

    @property
    def Ax(self):
        return self._first(0)

    @property
    def Ay(self):
        return self._first(1)

    @property
    def Az(self):
        return self._first(2)

    @property
    def Axx(self):
        return self._second(0, 0)

    @property
    def Ayy(self):
        return self._second(1, 1)

    @property
    def Azz(self):
        return self._second(2, 2)

    @property
    def Axy(self):
        return self._second(0, 1)

    @property
    def Axz(self):
        return self._second(0, 2)

    @property
    def Ayz(self):
        return self._second(1, 2)


def build_pressure_linear_operator(sm_press):
    """Extract the explicit affine pressure operator from a Chorin
    ``SM_press`` (see :class:`PressureLinearOperator`).

    Mechanical — no re-derivation.  ``SM_press.source`` is differentiated
    w.r.t. the bare pressure symbols (→ ``A0``) and w.r.t. the
    pressure-derivative aux symbols, which the existing aux metadata
    (``aux_registry`` / ``aux_input_registry`` entries whose ``state_index``
    is a pressure slot) sorts into first / second-derivative coefficient
    matrices by their ``multi_index``.  Asserts ``source`` is affine in
    ``P`` and its pressure derivatives (raises clearly otherwise).

    General over any chorin-split model (VAM any level, ML_VAM), any number
    of pressure modes ``NP``, and any dimension (2-D / 3-D).
    """
    state = list(sm_press.state)
    e2s = list(getattr(sm_press, "equation_to_state_index", None) or [])
    if not e2s:
        raise ValueError(
            "build_pressure_linear_operator: SM_press carries no "
            "equation_to_state_index — cannot locate the pressure modes.")
    P_modes = [state[i] for i in e2s]
    NP = len(P_modes)
    pmode_col = {si: l for l, si in enumerate(e2s)}  # state slot → P column l
    pmode_set = set(P_modes)

    n_dim = len(sm_press.space)

    # ── pressure-derivative aux: aux_symbol → (P column l, multi_index) ──
    # The canonical source of this metadata is the aux registry.  Live
    # pressure derivatives sit in ``aux_registry`` (kept by
    # ``_partition_pressure_aux``); scan ``aux_input_registry`` too so the
    # extractor is robust to an un-partitioned SM_press.
    registry = list(getattr(sm_press, "aux_registry", None) or [])
    registry += list(getattr(sm_press, "aux_input_registry", None) or [])
    deriv_aux = {}
    for e in registry:
        if e.get("kind") not in ("derivative", "limited_derivative"):
            continue
        si = e.get("state_index")
        if si not in pmode_col:
            continue
        deriv_aux[e["aux_symbol"]] = (pmode_col[si], tuple(e["multi_index"]))
    deriv_aux_syms = set(deriv_aux)

    def _row_scalar(i):
        src = sm_press.source
        try:
            return sp.sympify(src[i, 0])
        except (TypeError, IndexError):
            return sp.sympify(src[i])

    # Anything a coefficient must NOT still depend on (else non-affine).
    forbidden = pmode_set | deriv_aux_syms

    A0 = [[sp.S.Zero] * NP for _ in range(NP)]
    first = {}          # axis → NP×NP
    second = {}         # (a,b) a<=b → NP×NP
    RHS = [sp.S.Zero] * NP

    zero_p = {p: sp.S.Zero for p in P_modes}
    zero_p.update({s: sp.S.Zero for s in deriv_aux_syms})

    def _check_affine(coeff, k, what):
        bad = forbidden & coeff.free_symbols
        if bad:
            raise AssertionError(
                f"build_pressure_linear_operator: source_{k} is NOT affine "
                f"in the pressure — coefficient of {what} still depends on "
                f"pressure atoms {sorted(map(str, bad))}.")

    for k in range(NP):
        src_k = sp.expand(_row_scalar(k))

        # A0: coefficient of the bare pressure symbol.
        for l, Pl in enumerate(P_modes):
            c = sp.diff(src_k, Pl)
            if c != 0:
                _check_affine(c, k, str(Pl))
            A0[k][l] = c

        # First / second-derivative coefficients.
        for s, (l, mi) in deriv_aux.items():
            c = sp.diff(src_k, s)
            if c == 0:
                continue
            _check_affine(c, k, str(s))
            deg = sum(mi)
            axes = [a for a in range(n_dim) for _ in range(mi[a])]
            if deg == 1:
                a = axes[0]
                first.setdefault(a, [[sp.S.Zero] * NP for _ in range(NP)])
                first[a][k][l] += c
            elif deg == 2:
                a, b = axes[0], axes[1]
                key = (min(a, b), max(a, b))
                second.setdefault(key, [[sp.S.Zero] * NP for _ in range(NP)])
                second[key][k][l] += c
            else:
                raise AssertionError(
                    f"build_pressure_linear_operator: source_{k} carries a "
                    f"pressure derivative of order {deg} (aux {s}); only "
                    "first / second derivatives are supported.")

        # RHS = -source|_{P=0, dP=0}.  Everything left must be P-free.
        const_k = sp.expand(src_k.xreplace(zero_p))
        _check_affine(const_k, k, "the P-independent part")
        RHS[k] = sp.expand(-const_k)

    return PressureLinearOperator(
        P_modes=P_modes,
        A0=A0,
        first_derivative=first,
        second_derivative=second,
        RHS=RHS,
    )


def _attach_pressure_operator(sm_press):
    """Attach the lazy, cached ``pressure_operator()`` method to an
    ``SM_press`` instance (non-breaking; ``source`` is untouched)."""
    import types

    def _pressure_operator(self):
        cached = getattr(self, "_pressure_operator_cache", None)
        if cached is None:
            cached = build_pressure_linear_operator(self)
            self._pressure_operator_cache = cached
        return cached

    sm_press.pressure_operator = types.MethodType(_pressure_operator, sm_press)
    return sm_press


class Stage(NamedTuple):
    """One stage of a split solver's march (REQ-173).

    A split solver is supplied a *list* of stages and marches over it; a
    backend supplies one executor per stage ``kind``.

    * ``label`` — a STABLE, deterministic per-stage identifier
      (``"predictor"`` / ``"pressure"`` / ``"corrector"`` for the Chorin
      split).  Code-generating backends key generated names off it
      (amrex emits ``Model_<label>.H``); it MUST NOT be derived from the
      list position, or any reordering silently renames every header and
      invalidates every build cache.
    * ``kind`` — the executor kind: ``"hyperbolic" | "elliptic" |
      "pointwise"`` (extensible).  It selects which per-kind executor the
      backend runs for this stage.
    * ``sm`` — the :class:`SystemModel` for the stage.
    """
    label: str
    kind: str
    sm: object


@dataclass
class SplitForPressureResult:
    """Three sub-SystemModels produced by :func:`split_for_pressure`.

    For VAM(1, 2, 2) on ``Q = [h, U_0, U_1, W_0, W_1, P_0, P_1]``:

    * ``SM_pred`` — predictor: 5 evolution rows
      (``mass`` + 2 ``xmom`` + 2 ``zmom``) updating
      ``Q[0..4] = h, U_k, W_k`` with ``P_k`` frozen at ``P_k^n``.
    * ``SM_press`` — pressure stage: 2 algebraic rows (the elliptic
      block in ``(P_0, P_1)``) updating ``Q[5..6]``.  Mass matrix
      all-zero.
    * ``SM_corr`` — corrector: 4 algebraic update rows.
    """
    SM_pred: object
    SM_press: object
    SM_corr: object

    @property
    def stages(self):
        """The split as a canonical stage list (REQ-173).

        A split solver marches over this list; a backend supplies one
        executor per :class:`Stage` ``kind``.  The Chorin split is exactly
        this data — predictor (``hyperbolic``) → pressure (``elliptic``)
        → corrector (``pointwise``).  ``.SM_pred`` / ``.SM_press`` /
        ``.SM_corr`` remain the primary attributes; this is a view of the
        same three objects with their stable labels and kinds attached.
        """
        return [
            Stage("predictor", "hyperbolic", self.SM_pred),
            Stage("pressure",  "elliptic",   self.SM_press),
            Stage("corrector", "pointwise",  self.SM_corr),
        ]


def _assert_subsystem_symbols_declared(sm):
    """Raise if any free symbol of ``sm``'s hyperbolic operators is not a
    declared state / aux / parameter / coordinate / time / ``dt`` (REQ-151).

    A sub-system referencing an aux it does not declare is the failure the
    prefix fix prevents; this guard turns a would-be silent wrong-row read (or
    a lambdify ``NameError`` three layers down) into a loud split-time error.
    """
    allowed = {str(s) for s in sm.state}
    allowed |= {str(a) for a in sm.aux_state}
    allowed |= {str(p) for p in sm.parameters.values()}
    allowed |= {str(c) for c in sm.space}
    allowed |= {str(sm.time), "dt"}
    ops = [sm.flux, sm.hydrostatic_pressure, sm.nonconservative_matrix,
           sm.source, sm.mass_matrix]
    seen = set()
    for op in ops:
        if op is None:
            continue
        for entry in sp.flatten(op):
            for atom in sp.sympify(entry).free_symbols:
                seen.add(str(atom))
    undeclared = seen - allowed
    if undeclared:
        raise ValueError(
            f"split sub-system references symbols it does not declare: "
            f"{sorted(undeclared)}.  Declared aux = "
            f"{[str(a) for a in sm.aux_state]}; state = "
            f"{[str(s) for s in sm.state]}.  This is the REQ-151 aux-layout "
            f"divergence — the sub-system's aux pool must carry every row its "
            f"own operators reference.")


def _inherit_parent_aux(sm, sm_parent):
    """REQ-151: keep the PARENT's aux layout a PREFIX of every sub-system's.

    ``expose_aux_atoms`` only routes Function / Derivative atoms, so a
    plain-Symbol aux the parent owns — e.g. the KP-desingularized ``hinv``
    that every velocity ``u = q·hinv`` depends on — is SILENTLY dropped from
    the sub-system.  Then two things break: (A) the sub-system's own flux /
    eigenvalues reference an aux row that is absent (``NameError: hinv`` on
    jax, an out-of-range aux index on numpy, or a sympy expression that
    survives into ``np.asarray`` → ``TypeError: Cannot convert expression to
    float``); (B) the parent's BC kernel, forwarded verbatim, indexes aux by
    the PARENT's row numbers against the sub-system's DIFFERENT layout (an
    out-of-range crash, or worse a silent wrong-row read).  In 1-D the parent
    rows happen to be a prefix, so this never fired.  Restore that invariant
    by construction: prepend the parent rows (in parent order), then reindex
    the registries to the new layout.

    Applies to EVERY sub-system, including the hand-built corrector — that one
    bypassed ``_build_subsystem`` and so missed this, which is exactly why
    ``desingularize_hinv`` before ``chorin_split`` used to kill the corrector
    with failure mode (A).
    """
    parent_aux = list(sm_parent.aux_state)
    if not parent_aux:
        return
    pnames = {str(a) for a in parent_aux}
    extra = [a for a in sm.aux_state if str(a) not in pnames]
    sm.aux_state = parent_aux + extra
    names = [str(a) for a in sm.aux_state]
    for reg in ("aux_registry", "aux_input_registry"):
        for e in (getattr(sm, reg, None) or []):
            if e.get("name") in names:
                e["row"] = names.index(e["name"])
    # Carry the parent's LOCAL aux formula (``hinv = kp_hinv(h)``) so the
    # desingularized rows are actually COMPUTED on the sub-system pool
    # (defect D — the aux would otherwise sit at 0, zeroing every velocity).
    # The parent rows are now a prefix, so the parent's per-row column aligns
    # 1:1; extra rows are identity passthrough.
    parent_upd = getattr(sm_parent, "update_aux_variables", None)
    if parent_upd is not None and getattr(parent_upd, "shape", (0,))[0]:
        n_parent = parent_upd.shape[0]
        rows = [parent_upd[i, 0] if i < n_parent else a
                for i, a in enumerate(sm.aux_state)]
        sm.update_aux_variables = ZArray([[r] for r in rows])
    # REQ-169: ``aux_state`` just GREW (parent rows prepended) AFTER
    # ``expose_aux_atoms`` already sized ``source_jacobian_wrt_aux_variables``
    # to the smaller sub-system aux vector.  Rebuild the derived jacobians so
    # their columns match the FINAL aux_state layout — otherwise a printer
    # that iterates ``c in range(len(aux_state))`` (e.g. the amrex Chorin
    # header generator's ``_expr_source_jac_aux``) runs off the end of the
    # stale ``(n_eq, n_aux_before)`` jacobian → ``Index (0, k) out of border``.
    # Mirrors what ``expose_aux_atoms`` / ``expose_*_as_aux`` do after every
    # ``aux_state`` mutation.
    sm.refresh_derived_operators(eigenvalues=False)


def _inherit_equilibrium_reconstruction(sm_pred, sm_parent):
    """Carry the parent's ``equilibrium_reconstruction`` onto the PREDICTOR.

    ``ChorinSplitVAMSolver`` inherits :class:`HyperbolicSolver` precisely so the
    predictor substep reuses the same well-balanced reconstruction machinery as
    every other hyperbolic solver.  That hook reads the flag off the sub-system
    it is handed (``getattr(self.sm, "equilibrium_reconstruction", "none")``) —
    but the sub-systems are built FRESH here and default the field to
    ``"none"``, so a model declaring ``equilibrium_reconstruction='audusse'``
    silently lost it at the split and the inherited hook never fired.  The
    inheritance was real; the flag just never arrived.

    Only the predictor gets it: it is the hyperbolic flux step, and it is the
    only substep the reconstruction is defined for.  The pressure and corrector
    substeps are not flux steps and must stay ``"none"``.
    """
    eqr = getattr(sm_parent, "equilibrium_reconstruction", "none")
    if eqr and eqr != "none":
        sm_pred.equilibrium_reconstruction = eqr


def _inherit_roles(sm, sm_parent):
    """Carry the parent's DECLARED state roles (depth / bed / stationary /
    positive / momentum / gravity) onto a split stage — the stages share the
    parent's state vector, so the declaration applies verbatim.  Without it a
    stage would lose its free-surface identity (and the role-gated NSM
    default operations) at the split."""
    from zoomy_core.systemmodel.roles import (
        declared_role_record, declare_roles)
    from zoomy_core.systemmodel.tags import declare_tags, declared_tags
    rec = declared_role_record(sm_parent)
    if rec:
        declare_roles(sm, description="roles inherited from the split parent",
                      **rec)
    tags = declared_tags(sm_parent)
    if tags:
        declare_tags(sm, tags, "tags inherited from the split parent")
    if getattr(sm_parent, "admissible", None) is not None \
            and getattr(sm, "admissible", None) is None:
        sm.admissible = sm_parent.admissible


def _build_subsystem(*, eq_names, eq_residuals, sm_parent, state,
                     equation_to_state_index, history_entry,
                     source_only=False):
    """Build a rectangular SystemModel from a list of (name, residual)
    pairs.

    Parameters
    ----------
    source_only : bool, default False
        Default (False) auto-tags each residual into flux / NCP /
        source operators and a mass matrix — appropriate for sub-
        systems whose runtime uses the symbolic Riemann / NCP path-
        integral framework (predictor, corrector).  Set True for sub-
        systems that should use **central discretisation through the
        aux-derivative stencil**: the entire non-time-derivative
        residual is packed into ``source`` (sign-flipped to SystemModel
        ``... − S = 0`` convention), ``F = P_hydro = B = 0``, and any
        spatial derivatives become aux entries registered by
        ``expose_aux_atoms``.  Use this for elliptic / Poisson-type
        sub-systems (e.g. SM_press from :func:`split_simple`) where
        a centrally-differenced stencil — not Roe / path-integral
        upwinding — is the natural numerics.
    """
    from zoomy_core.systemmodel.system_model import SystemModel
    from zoomy_core.model.derivation.tag_extraction import (
        auto_solver_tag, collect_solver_tag,
    )

    t = sm_parent.time
    coords = list(sm_parent.space)
    g_param = (sm_parent.parameters.g
               if sm_parent.parameters.contains("g") else None)

    # State Symbols and coordinate-dependent Functions for tag classifier.
    state_syms = list(state)
    name_to_sym = {str(s): s for s in state_syms}
    state_funcs = [sp.Function(str(s), real=True)(t, *coords)
                   for s in state_syms]
    sym_to_func = dict(zip(state_syms, state_funcs))
    func_to_sym = dict(zip(state_funcs, state_syms))

    # Auto-tag every residual (converting Symbol state to Function form
    # for the tag classifier).  ``auto_solver_tag`` returns a SolverTagged
    # carrier keyed under ``name`` in ``tagged`` — the carrier's own ``.name``
    # is unused downstream (collect_solver_tag keys off the dict).
    tagged: dict = {}
    for idx, (name, res) in enumerate(zip(eq_names, eq_residuals)):
        res_func = sp.sympify(res).xreplace(sym_to_func)
        # The row's OWN evolved state Function — lets the re-tagger tell a
        # genuine viscous self-diffusion ``∂_x(ν·∂_x own)`` (→ central-stencil
        # source) apart from an off-diagonal ``∂_x(A·∂_x field)`` of a FOREIGN
        # field such as the bed ``∂_x b`` (→ conservative flux).  Mirrors
        # ``system_extract._classify_row``'s ``_is_self_diffusion`` (REQ-80).
        own_func = state_funcs[equation_to_state_index[idx]]
        tagged[name] = auto_solver_tag(
            res_func,
            state_funcs=state_funcs,
            gravity_param=g_param,
            t=t, coords=coords,
            own_func=own_func,
        )

    class _Holder:
        pass

    holder = _Holder()
    holder.equations = tagged

    n_eq = len(eq_names)
    n_state = len(state_syms)
    n_dim = len(coords)
    variable_map = {n: [i] for i, n in enumerate(eq_names)}

    if source_only:
        F = sp.zeros(n_eq, n_dim)
        P = sp.zeros(n_eq, n_dim)
        B = sp.MutableDenseNDimArray.zeros(n_eq, n_state, n_dim)
        S_list = []
        for i, name in enumerate(eq_names):
            res_func = sp.sympify(eq_residuals[i]).xreplace(sym_to_func)
            td_part = tagged[name].get_solver_tag("time_derivative")
            non_td = sp.expand(
                (res_func - (td_part if td_part is not None else sp.S.Zero)).doit()
            )
            S_list.append(non_td)
        S_mat = sp.Matrix(n_eq, 1, lambda i, _j: -S_list[i])
    else:
        F = collect_solver_tag(
            holder, "flux", variable_map=variable_map,
            n_variables=n_eq, n_directions=n_dim, coords=coords,
            state_variables=state_funcs, policy="strict",
        )
        P = collect_solver_tag(
            holder, "hydrostatic_pressure", variable_map=variable_map,
            n_variables=n_eq, n_directions=n_dim, coords=coords,
            state_variables=state_funcs, policy="strict",
        )
        # ``collect_solver_tag`` allocates a square (N × N × n_dim) NCP
        # array.  Our sub-systems are rectangular (n_eq < n_state); pass
        # ``n_variables=max(n_eq, n_state)`` and then take the first n_eq
        # rows / n_state columns.
        n_max = max(n_eq, n_state)
        B_raw = collect_solver_tag(
            holder, "nonconservative_flux", variable_map=variable_map,
            n_variables=n_max, n_directions=n_dim, coords=coords,
            state_variables=state_funcs, policy="strict",
        )
        S_list = collect_solver_tag(
            holder, "source", variable_map=variable_map,
            n_variables=n_eq, policy="strict",
        )
        # ``collect_solver_tag`` ALREADY flips source terms to the RHS
        # convention (sign = −1 for implicit/explicit_source) — store
        # verbatim.  The previous extra negation here double-flipped the
        # sign, so every split predictor ran with ANTI-DAMPED friction
        # (the inverted VAM dam-break shear profile).
        S_mat = sp.Matrix(n_eq, 1, lambda i, _j: S_list[i])

        B = sp.MutableDenseNDimArray.zeros(n_eq, n_state, n_dim)
        for i in range(n_eq):
            for j in range(n_state):
                for d in range(n_dim):
                    B[i, j, d] = B_raw[i, j, d]

    # Mass matrix: rebuild from the time-derivative tag on each row.
    # ``.doit()`` expands compound ``Derivative(c·F, t)`` atoms.
    M_mat = sp.zeros(n_eq, n_state)
    for i, name in enumerate(eq_names):
        eq = tagged[name]
        td_part = eq.get_solver_tag("time_derivative")
        if td_part is None or td_part == 0:
            continue
        td = sp.expand(td_part.doit())
        for j, fj in enumerate(state_funcs):
            dot_sym = sp.Symbol(f"_dot_{str(state_syms[j])}", real=True)
            dt_atom = sp.Derivative(fj, t)
            replaced = td.subs(dt_atom, dot_sym)
            coeff = sp.diff(replaced, dot_sym)
            if coeff != 0:
                M_mat[i, j] = sp.expand(coeff)

    # Map Function-form atoms back to Symbol state.
    def _to_sym(matrix):
        if isinstance(matrix, sp.Matrix):
            return matrix.xreplace(func_to_sym)
        out = sp.MutableDenseNDimArray.zeros(*matrix.shape)
        shape = matrix.shape
        ndim = len(shape)

        def _iter(shape):
            if not shape:
                yield ()
                return
            for i in range(shape[0]):
                for rest in _iter(shape[1:]):
                    yield (i,) + rest

        for idx in _iter(shape):
            entry = sp.sympify(matrix[idx])
            out[idx] = entry.xreplace(func_to_sym)
        return out

    F = _to_sym(F)
    P = _to_sym(P)
    B = _to_sym(B)
    S_mat = _to_sym(S_mat)
    M_mat = _to_sym(M_mat)

    # Resolve any bed/surface-trace Subs(f(ζ),ζ,0|1) node the moment expansion
    # left in the residual (e.g. the VAM r_k bottom-KBC trace
    # ``Subs(2·q_1/h, ζ, 0)``) BEFORE this sub-SystemModel reaches any printer.
    # SM_pred / SM_press are consumed directly by the amrex Chorin header
    # generator, which bypasses from_system_model's REQ-130 resolve.
    F = _resolve_subs(F)
    P = _resolve_subs(P)
    B = _resolve_subs(B)
    S_mat = _resolve_subs(S_mat)
    M_mat = _resolve_subs(M_mat)

    # ── diffusion tensors (REQ-111) ──────────────────────────────────────
    # The horizontal-eddy-viscosity slot ``diffusion_matrix_explicit`` (and
    # the implicit ``diffusion_matrix``) has shape (n_eq, n_state, n_dim,
    # n_dim); the REQ-50 jax flux operator reads it off the PREDICTOR runtime.
    # Sub-systems SHARE the parent's state/coord layout (``state=state`` →
    # same n_state, n_dim, and the COLUMNS reference the shared state), but
    # carry a REDUCED equation set (the predictor keeps only the evolution
    # rows).  So the parent's ROW axis must be remapped, not forwarded whole:
    # row ``i`` of the sub-system evolves state slot ``e2s[i]``; copy the
    # parent's diffusion row that evolves that SAME slot.  Rows with no parent
    # diffusion (pressure/constraint rows, bed) stay zero; an all-zero result
    # collapses to None so plain models keep ``diffusion_matrix_explicit is
    # None`` and the corrector/pressure stages carry no spurious tensor.
    def _remap_diffusion(A_parent):
        if A_parent is None:
            return None
        if hasattr(A_parent, "todense"):
            Ap = sp.MutableDenseNDimArray(A_parent.todense())
        elif hasattr(A_parent, "tolist"):
            Ap = sp.MutableDenseNDimArray(A_parent.tolist())
        else:
            Ap = sp.MutableDenseNDimArray(A_parent)
        parent_e2s = list(sm_parent.equation_to_state_index)
        slot_to_prow = {parent_e2s[p]: p for p in range(len(parent_e2s))}
        out = sp.MutableDenseNDimArray.zeros(n_eq, n_state, n_dim, n_dim)
        any_nonzero = False
        for i in range(n_eq):
            p = slot_to_prow.get(equation_to_state_index[i])
            if p is None:
                continue
            for a in range(n_state):
                for b in range(n_dim):
                    for c in range(n_dim):
                        val = Ap[p, a, b, c]
                        out[i, a, b, c] = val
                        if val != 0:
                            any_nonzero = True
        return out if any_nonzero else None

    diff_impl = _remap_diffusion(sm_parent.diffusion_matrix)
    diff_expl = _remap_diffusion(sm_parent.diffusion_matrix_explicit)

    sm = SystemModel(
        time=t,
        space=coords,
        state=state_syms,
        aux_state=[],
        # Propagate the 3-D position Zstruct so the sub-system prints
        # standalone (the C++/foam Model printers read ``model.position``
        # for the ``X`` register map); without it the field defaults to
        # None and the printer crashes on ``position.values()``.
        position=sm_parent.position,
        parameters=sm_parent.parameters,
        parameter_values=sm_parent.parameter_values,
        flux=F,
        hydrostatic_pressure=P,
        nonconservative_matrix=B,
        source=S_mat,
        mass_matrix=M_mat,
        # Row-remapped from the parent (REQ-111): reaches the predictor's
        # REQ-50 jax flux runtime, which reads ``runtime.diffusion_matrix_*``.
        diffusion_matrix=diff_impl,
        diffusion_matrix_explicit=diff_expl,
        equation_to_state_index=list(equation_to_state_index),
        # Indexed BC kernels carry across unchanged — the sub-system
        # shares the parent's state Symbols, so the parent's
        # ``boundary_conditions(idx, ..., Q, Qaux, p, normal) → Q_face``
        # Function still resolves against the same Q layout.  Without
        # this propagation the runtime build (which lambdifies these
        # Functions unconditionally — "prefer breaking over silent
        # skip") would fail.
        boundary_conditions=sm_parent.boundary_conditions,
        aux_boundary_conditions=sm_parent.aux_boundary_conditions,
        boundary_gradients=sm_parent.boundary_gradients,
        initial_conditions=sm_parent.initial_conditions,
        aux_initial_conditions=sm_parent.aux_initial_conditions,
        update_variables=sm_parent.update_variables,
        reconstruction_variables=sm_parent.reconstruction_variables,
        state_from_reconstruction=sm_parent.state_from_reconstruction,
    )
    # Auto-scan: route every non-state Function / Derivative atom in
    # this sub-system's operators into ``aux_state`` + ``aux_registry``
    # — the same machinery ``SystemModel.from_model`` runs on the
    # monolithic chain.  So the predictor carries ``b`` / ``b_x`` /
    # ``h_x``, the pressure stage carries the pressure derivatives
    # ``P_l_x`` / ``P_l_x_x``, etc.
    sm.expose_aux_atoms()
    _inherit_parent_aux(sm, sm_parent)
    # Guard that would have caught all of the above: every free symbol of the
    # sub-system's hyperbolic operators must be a state, an aux, a parameter,
    # a coordinate, time, or ``dt`` — anything else means a kernel references
    # something the sub-system does not declare.  Fail LOUD at split time, not
    # at lambdify/index time three layers down.
    _assert_subsystem_symbols_declared(sm)
    sm.equation_names = list(eq_names)
    # Share the parent's BC source objects so the sub-system prints
    # standalone — the C++ Model printer reads ``sm._bc_source`` (tag
    # dict) for its boundary-id header, and the runtime re-resolves the
    # indexed kernel from it.  Sub-systems share the parent state layout,
    # so the same source applies verbatim.
    sm._bc_source = getattr(sm_parent, "_bc_source", None)
    sm._aux_bc_source = getattr(sm_parent, "_aux_bc_source", None)
    sm.history.append(history_entry)
    _inherit_roles(sm, sm_parent)
    return sm


def split_for_pressure(sm, pressure_vars, dt, *, bottom=None):
    """DEPRECATED — use :func:`split_for_pressure_structural` (via
    ``model.chorin_split(dt)``).

    Old chain-DAE splitter that dispatches rows by the ``equation_names``
    convention (``mass`` / ``xmom_j*`` / ``zmom_j*`` / ``cont_j*``); a
    declarative SystemModel carries no such names.  The structural splitter
    discovers the same roles from the operators (zero mass-matrix rows ⇒
    constraints) and is the only supported path for new models.
    """
    warnings.warn(
        "split_for_pressure is deprecated; use split_for_pressure_structural "
        "(model.chorin_split(dt)) — it discovers row roles from the operators "
        "and needs no equation_names convention.",
        DeprecationWarning, stacklevel=2)
    block = build_pressure_elliptic_block(sm, pressure_vars, dt,
                                          bottom=bottom)
    residuals = _residuals_by_name(sm)
    name_to_sym = _state_symbol_by_name(sm)
    state = list(sm.state)
    state_names = [str(s) for s in state]

    pressure_indices = [state.index(p) for p in pressure_vars]
    t = sm.time
    coords = list(sm.space)
    x = coords[0]

    # Use the Function-form pressure variants from the block to match
    # the Function-form residuals.
    pressure_funcs = block["pressure_vars"]

    # ── SM_pred ──────────────────────────────────────────────────────────
    # No pred_p_freeze rename — the predictor reads pressure straight
    # from the state Functions, and at runtime ``Q[pressure_indices]``
    # holds the start-of-step pressure (the predictor's substep does
    # not write there, so it stays at P^n implicitly).
    pred_eq_names = []
    pred_eq_residuals = []
    pred_e2s_index = []
    def _state_for_equation(eq_name):
        """Resolve the state slot updated by an equation row.  Tries
        primitive-form names first (``U_k`` / ``W_k`` / ``h``), then
        conservative-form (``q_Uk`` / ``q_Wk`` / ``h``).  Returns the
        first match in ``state_names`` or ``None``."""
        if eq_name == "mass":
            return "h"
        if eq_name == "bathymetry":
            return "b" if "b" in state_names else None
        if eq_name.startswith("xmom_j"):
            k = eq_name[len("xmom_j"):]
            for cand in (f"U_{k}", f"q_U{k}"):
                if cand in state_names:
                    return cand
        if eq_name.startswith("zmom_j"):
            k = eq_name[len("zmom_j"):]
            for cand in (f"W_{k}", f"q_W{k}"):
                if cand in state_names:
                    return cand
        return None

    for name in sm.equation_names:
        if name.startswith("cont_j"):
            continue
        target = _state_for_equation(name)
        if target is None:
            continue
        eq_pred = sp.expand(residuals[name])
        pred_eq_names.append(name)
        pred_eq_residuals.append(eq_pred)
        pred_e2s_index.append(state_names.index(target))

    SM_pred = _build_subsystem(
        eq_names=pred_eq_names,
        eq_residuals=pred_eq_residuals,
        sm_parent=sm,
        state=state,
        equation_to_state_index=pred_e2s_index,
        history_entry={
            "name": "split_for_pressure[pred]",
            "description": f"predictor: {len(pred_eq_names)} rows updating "
                           f"{[state_names[i] for i in pred_e2s_index]}",
        },
    )

    # ── SM_press ─────────────────────────────────────────────────────────
    press_eq_names = [f"elliptic_j{j}" for j in sorted(block["rows"])]
    press_eq_residuals = [
        sp.expand(block["rows"][j]) for j in sorted(block["rows"])
    ]
    SM_press = _build_subsystem(
        eq_names=press_eq_names,
        eq_residuals=press_eq_residuals,
        sm_parent=sm,
        state=state,
        equation_to_state_index=list(pressure_indices),
        history_entry={
            "name": "split_for_pressure[press]",
            "description": f"pressure: {len(press_eq_names)} algebraic rows "
                           f"determining {[str(p) for p in pressure_vars]}",
        },
    )

    # ── SM_corr ──────────────────────────────────────────────────────────
    # Explicit-update operator: ``Q[corr_idx] ← update_variables(Q, Qaux, p, dt)``.
    # The expression for each updated slot is the closed-form
    # ``coeffs_u[k] - (dt/h)·T_u_k(P)`` (and W analogue) in pure state
    # Functions — no P_new rename, no tilde rename.  The "tilde" /
    # "new-pressure" semantics live in execution order: at SM_corr
    # apply time ``Q[1:5]`` holds the predictor output and ``Q[5:6]``
    # holds the new pressure.
    M_M = len(block["U_corr"]) - 1
    N_w_active = len(block["W_corr"])
    corr_e2s_index = []
    update_exprs = []
    update_names = []  # for SM_corr.equation_names — diagnostic only
    def _state_index_for(prefixes_and_k):
        for pref, k in prefixes_and_k:
            cand = f"{pref}{k}"
            if cand in state_names:
                return state_names.index(cand), cand
        return None, None
    for k in range(M_M + 1):
        idx, _ = _state_index_for([("U_", k), ("q_U", k)])
        if idx is None:
            continue
        update_exprs.append(sp.expand(block["U_corr"][k]))
        corr_e2s_index.append(idx)
        update_names.append(f"corr_U_{k}")
    for k in range(N_w_active):
        idx, _ = _state_index_for([("W_", k), ("q_W", k)])
        if idx is None:
            continue
        update_exprs.append(sp.expand(block["W_corr"][k]))
        corr_e2s_index.append(idx)
        update_names.append(f"corr_W_{k}")

    # Map state Functions back to state Symbols — the SystemModel
    # convention: operators are in Symbol form.
    state_funcs = [sp.Function(str(s), real=True)(t, *coords) for s in state]
    func_to_sym = dict(zip(state_funcs, state))
    update_exprs_sym = [e.xreplace(func_to_sym) for e in update_exprs]

    # Build SM_corr with zero residual fields + the corrector formula
    # carried on ``update_variables``.  ``_build_subsystem`` autotags an
    # empty residual list to all-zero operators; the per-cell
    # ``update_variables(Q, Qaux, p, dt)`` field then carries the explicit
    # projection update (scattered to ``equation_to_state_index``).
    n_eq = len(corr_e2s_index)
    n_dim = sm.n_dim
    n_st = sm.n_state
    SM_corr = SystemModel(
        time=sm.time,
        space=list(sm.space),
        state=list(state),
        aux_state=[],
        position=sm.position,
        parameters=sm.parameters,
        parameter_values=sm.parameter_values,
        flux=sp.zeros(n_eq, n_dim),
        hydrostatic_pressure=sp.zeros(n_eq, n_dim),
        nonconservative_matrix=_zero_ndim(n_eq, n_st, n_dim),
        source=sp.zeros(n_eq, 1),
        mass_matrix=sp.zeros(n_eq, n_st),
        equation_to_state_index=list(corr_e2s_index),
        boundary_conditions=sm.boundary_conditions,
        aux_boundary_conditions=sm.aux_boundary_conditions,
        boundary_gradients=sm.boundary_gradients,
        initial_conditions=sm.initial_conditions,
        aux_initial_conditions=sm.aux_initial_conditions,
        update_variables=sp.Matrix(update_exprs_sym),
        reconstruction_variables=sm.reconstruction_variables,
        state_from_reconstruction=sm.state_from_reconstruction,
    )
    SM_corr.equation_names = list(update_names)
    SM_corr.expose_aux_atoms()
    # SM_corr is the ONE sub-system built directly instead of through
    # ``_build_subsystem``, so it used to miss the parent-aux prefix — and with
    # it any plain-Symbol aux the parent owns (``hinv``).  The corrector's
    # ``update_variables`` then referenced ``hinv`` without declaring it and the
    # symbol survived into ``np.asarray`` (``TypeError: Cannot convert
    # expression to float``).  Inherit exactly like the other two stages.
    _inherit_parent_aux(SM_corr, sm)
    _inherit_equilibrium_reconstruction(SM_pred, sm)
    SM_corr._bc_source = getattr(sm, "_bc_source", None)
    SM_corr._aux_bc_source = getattr(sm, "_aux_bc_source", None)
    _inherit_roles(SM_corr, sm)
    SM_corr.history.append({
        "name": "split_for_pressure[corr]",
        "description": (
            f"corrector: explicit update on "
            f"{[state_names[i] for i in corr_e2s_index]} "
            f"via update_variables field"
        ),
    })

    return SplitForPressureResult(
        SM_pred=SM_pred,
        SM_press=SM_press,
        SM_corr=SM_corr,
    )


def split_for_pressure_structural(sm, pressure_vars, dt):
    """Structural predictor / pressure / corrector splitter — no
    equation-name conventions.

    The chain-DAE :func:`split_for_pressure` dispatches rows by the legacy
    naming scheme (``xmom_j*``, ``cont_j*``, ``U_k``/``q_Uk``).  SystemModels
    extracted from a DECLARATIVE derivation carry no such names, but the row
    roles are fully visible in the operators:

    * **constraint rows** — identically-zero mass-matrix row (no ``∂_t``):
      these are the divergence/incompressibility projections → the elliptic
      block determining the pressure modes;
    * **evolution rows** — everything else; rows whose residual carries a
      pressure variable get the corrector update
      ``Q[s] ← Q[s] − dt·(pressure part)/M[row, s]``;
    * ``equation_to_state_index`` supplies the row→slot map (identity for a
      square ``from_model`` extraction).

    The predictor is the PRESSURE-FREE hydrostatic system (the evolution
    residuals with every pressure mode set to zero) — Escalante et al. 2024
    eq (12a): the hyperbolic step carries no non-hydrostatic terms; the FULL
    pressure impulse ``−dt·T(P^{n+1})`` is applied by the corrector.
    (Keeping the pressure force at ``P^n`` in the predictor double-applies
    it and turns the pressure into an undamped alternating iteration
    ``P^{n+1} + P^n ≈ P_phys`` that diverges at shocks — the historic VAM
    dam-break NaN.)  The elliptic rows are the constraint residuals with
    every corrected velocity substituted; the corrector applies the
    closed-form update via ``update_variables``.

    Parameters
    ----------
    sm : SystemModel
        Square (or rectangular) extraction with ``equation_to_state_index``.
    pressure_vars : Sequence
        Pressure-mode state Symbols (must be entries of ``sm.state``).
    dt : sp.Symbol
        Symbolic time-step.
    """
    n_eq = sm.n_equations
    state = list(sm.state)
    state_names = [str(s) for s in state]
    t = sm.time
    coords = list(sm.space)
    e2s = list(sm.equation_to_state_index)
    residuals = sm.reconstruct_residuals()

    def _to_fn(sym):
        return sp.Function(str(sym), real=True)(t, *coords)

    state_funcs = [_to_fn(s) for s in state]
    sym_to_fn = dict(zip(state, state_funcs))
    pressure_indices = [state.index(p) for p in pressure_vars]
    pressure_funcs = [state_funcs[i] for i in pressure_indices]

    def _row_is_constraint(i):
        return all(sp.sympify(sm.mass_matrix[i, j]) == 0
                   for j in range(sm.n_state))

    constraint_rows = [i for i in range(n_eq) if _row_is_constraint(i)]
    evolution_rows = [i for i in range(n_eq) if not _row_is_constraint(i)]
    if len(constraint_rows) != len(pressure_vars):
        raise ValueError(
            f"split_for_pressure_structural: {len(constraint_rows)} "
            f"constraint rows (zero mass-matrix rows) vs "
            f"{len(pressure_vars)} pressure variables — the elliptic "
            "block must be square.  Constraint rows found: "
            f"{[state_names[e2s[i]] for i in constraint_rows]}.")

    def _pressure_part(e):
        e = sp.expand(sp.sympify(e).doit())
        no_p = sp.expand(
            e.subs({p: sp.S.Zero for p in pressure_funcs}).doit())
        return sp.expand(e - no_p)

    # ── corrector updates (and the substitution for the elliptic rows) ──
    corr_e2s, update_exprs, update_names = [], [], []
    repl = {}
    for i in evolution_rows:
        p_part = _pressure_part(residuals[i])
        if p_part == 0:
            continue
        s = e2s[i]
        M_ss = sp.sympify(sm.mass_matrix[i, s]).xreplace(sym_to_fn)
        if M_ss == 0:
            raise ValueError(
                f"evolution row {i} (state {state_names[s]}) carries "
                "pressure terms but a zero mass-matrix diagonal — cannot "
                "build the corrector update.")
        corrected = sp.expand(state_funcs[s] - dt * p_part / M_ss)
        repl[state_funcs[s]] = corrected
        corr_e2s.append(s)
        update_exprs.append(corrected)
        update_names.append(f"corr_{state_names[s]}")

    # ── elliptic rows: corrected velocities into the constraints ──
    press_names = [f"elliptic_{state_names[pressure_indices[k]]}"
                   for k in range(len(constraint_rows))]
    press_res = [sp.expand(sp.sympify(residuals[i]).subs(repl).doit())
                 for i in constraint_rows]

    # ── predictor: the evolution rows, PRESSURE-FREE (hydrostatic part) ──
    # A blanket ``.doit()`` here is WRONG: the reconstructed residual carries the
    # conservative fluxes as COMPOUND outer derivatives ``∂_x(F)``, and for the
    # VAM r_k vertical-momentum rows ``F`` holds the bed-slope integrand
    # ``A·∂_x b`` (A=(2q₀q₁−2q₁²)/(5h), from the bottom KBC ``w|_bed=u·∂_x b``).
    # ``.doit()`` product-rule-expands that outer ``∂_x`` into a bed-CURVATURE
    # source ``A·∂²_x b`` (``b_x_x``) plus cross terms, which then blow up on the
    # bump near x=0.  Keeping the ``∂_x`` compound lets ``_build_subsystem``'s
    # re-tagger route ``∂_x(A·∂_x b)`` to the conservative FLUX (inner ``∂_x b``
    # frozen to a gradient-aux), matching ``system_extract._classify_row``
    # (9f7f52e / REQ-80).
    #
    # But the blanket ``.doit()`` ALSO collapsed benign NON-horizontal derivative
    # artifacts left by the vertical (ζ) moment derivation — e.g. ``∂_ζ(ζ) → 1``
    # in the source — which the runtime code-printer cannot lower.  So doit
    # ONLY the derivatives that touch no horizontal coord (the trivial vertical /
    # time ones), via ``xreplace`` so it reaches inside the surviving ``∂_x``
    # compounds without expanding them.  ``.subs`` zeroes the pressure modes
    # inside the unevaluated derivatives too, so no horizontal doit is needed.
    coords_set = set(coords)

    def _collapse_nonhorizontal(e):
        e = sp.sympify(e)
        repl = {d: d.doit() for d in e.atoms(sp.Derivative)
                if not any(v in coords_set for v in d.variables)}
        return e.xreplace(repl) if repl else e

    pred_names = [f"pred_{state_names[e2s[i]]}" for i in evolution_rows]
    pred_res = [sp.expand(_collapse_nonhorizontal(
                    sp.sympify(residuals[i])
                    .subs({p: sp.S.Zero for p in pressure_funcs})))
                for i in evolution_rows]
    pred_e2s = [e2s[i] for i in evolution_rows]

    SM_pred = _build_subsystem(
        eq_names=pred_names, eq_residuals=pred_res, sm_parent=sm,
        state=state, equation_to_state_index=pred_e2s,
        history_entry={
            "name": "split_for_pressure_structural[pred]",
            "description": f"predictor: {len(pred_names)} evolution rows "
                           f"updating {[state_names[i] for i in pred_e2s]}",
        })
    SM_press = _build_subsystem(
        eq_names=press_names, eq_residuals=press_res, sm_parent=sm,
        state=state, equation_to_state_index=list(pressure_indices),
        source_only=True,
        history_entry={
            "name": "split_for_pressure_structural[press]",
            "description": f"pressure: {len(press_names)} elliptic rows "
                           f"determining {[str(p) for p in pressure_vars]}",
        })
    # Keep only the LIVE pressure-mode derivatives in SM_press.aux_registry;
    # the frozen predictor-produced aux move to aux_input_registry (still in
    # aux_state as kernel inputs).  Lets a Model printer emit a minimal,
    # correct update_aux_variables with no split-awareness.
    _partition_pressure_aux(SM_press, pressure_vars)
    # Opt-in explicit affine pressure operator (coefficient fields for
    # MG/MLABecLaplacian backends).  Lazy + cached — extracted by symbolic
    # differentiation of the (unchanged) ``SM_press.source`` on first call.
    _attach_pressure_operator(SM_press)

    func_to_sym = dict(zip(state_funcs, state))
    update_exprs_sym = [e.xreplace(func_to_sym) for e in update_exprs]
    update_variables = _resolve_subs(sp.Matrix(update_exprs_sym)) \
        if update_exprs_sym else sp.Matrix(update_exprs_sym)
    n_corr = len(corr_e2s)
    SM_corr = SystemModel(
        time=sm.time,
        space=list(sm.space),
        state=list(state),
        aux_state=[],
        position=sm.position,
        parameters=sm.parameters,
        parameter_values=sm.parameter_values,
        flux=sp.zeros(n_corr, sm.n_dim),
        hydrostatic_pressure=sp.zeros(n_corr, sm.n_dim),
        nonconservative_matrix=_zero_ndim(n_corr, sm.n_state, sm.n_dim),
        source=sp.zeros(n_corr, 1),
        mass_matrix=sp.zeros(n_corr, sm.n_state),
        equation_to_state_index=list(corr_e2s),
        boundary_conditions=sm.boundary_conditions,
        aux_boundary_conditions=sm.aux_boundary_conditions,
        boundary_gradients=sm.boundary_gradients,
        initial_conditions=sm.initial_conditions,
        aux_initial_conditions=sm.aux_initial_conditions,
        update_variables=update_variables,
        reconstruction_variables=sm.reconstruction_variables,
        state_from_reconstruction=sm.state_from_reconstruction,
    )
    SM_corr.equation_names = list(update_names)
    SM_corr.expose_aux_atoms()
    # SM_corr is the ONE sub-system built directly instead of through
    # ``_build_subsystem``, so it used to miss the parent-aux prefix — and with
    # it any plain-Symbol aux the parent owns (``hinv``).  The corrector's
    # ``update_variables`` then referenced ``hinv`` without declaring it and the
    # symbol survived into ``np.asarray`` (``TypeError: Cannot convert
    # expression to float``).  Inherit exactly like the other two stages.
    _inherit_parent_aux(SM_corr, sm)
    _inherit_equilibrium_reconstruction(SM_pred, sm)
    SM_corr._bc_source = getattr(sm, "_bc_source", None)
    SM_corr._aux_bc_source = getattr(sm, "_aux_bc_source", None)
    _inherit_roles(SM_corr, sm)
    SM_corr.history.append({
        "name": "split_for_pressure_structural[corr]",
        "description": f"corrector: explicit update on "
                       f"{[state_names[i] for i in corr_e2s]}",
    })

    # ── cid=50: the NSM default-operation sweep must reach EVERY stage ──
    # The stages above are (re)built from the RAW residuals, so no rewrite
    # applied to the PARENT can reach them (sympy slots are rebound, never
    # mutated).  Backends that consume the split directly (amrex / foam
    # ``write_chorin_*_headers`` print ``split.SM_*`` verbatim, bypassing the
    # ``to_numerical_system_model`` front door) therefore emitted a bare,
    # unregularized ``Pow(h, -2)`` in the predictor quasilinear / source
    # Jacobians; on a dry-bed case the numerical eigenvalue kernel receives a
    # non-finite matrix at ``h = 0``, returns ``+inf``, and the CFL min pins
    # ``dt = 0`` at step 1.  Fix: route every stage through the SAME front
    # door the parent uses, so the NSM default operations —
    # ``normalize_face_normal`` + the KP ``desingularize_hinv`` (with its
    # internal eigenvalue ``eps_h`` floor) — reach each stage exactly as they
    # reach the parent.  The ``_is_transport_system`` gate decides per stage:
    # the predictor (a real flux system) is swept; the elliptic pressure /
    # pointwise corrector stages carry no wavespeeds and stay clean by design.
    #
    # Promotion RETURNS A NEW OBJECT — bind it.  This loop used to discard the
    # return value, which was correct only while ``from_system_model`` re-classed
    # its argument in place.  It no longer does: it opens with
    # ``sm = construct_numerical(sm)`` (a ``copy.deepcopy`` — THE Model → SM →
    # NSM boundary, deliberately introduced so the numerical layer gets
    # assumption-free symbol twins instead of relabelling the symbolic one), so
    # it promotes a COPY and leaves the caller's object untouched.  The loop was
    # therefore a total no-op and the cid=50 fix above was dead: measured on
    # ``chorin_split(VAM(1) Escalante-bump)`` without a pre-applied
    # ``desingularize_hinv``, all three stages came out plain ``SystemModel``
    # with no ``hinv`` in ``aux_state`` and a bare ``1/h`` + ``h**(-2)`` still in
    # ``SM_pred.quasilinear_matrix`` — exactly the ``Pow(h, -2)`` this block
    # claims to remove.  Golden S02 hid it by applying ``desingularize_hinv``
    # BEFORE the split.
    #
    # Rebinding the three fields is sufficient to keep the ``SM_pred is
    # stages[0].sm`` identity, because ``SplitForPressureResult.stages`` is a
    # derived property that reads these same attributes rather than a stored
    # list.  Each stage freeze-materializes its OWN quasilinear matrix from its
    # raw operators before the substitution (never recomputed from the swept
    # flux — that would drop the exact ``∂(q²/h)/∂h = −q²/h²`` wavespeed term;
    # see ``regularize_pow``).  A later ``from_system_model(...)`` on a stage
    # (the numpy/jax solvers re-promote at setup) is idempotent in effect:
    # ``hinv`` is already registered and no bare negative depth power remains to
    # rewrite.
    from zoomy_core.numerics.numerical_system_model import (  # lazy: no cycle
        to_numerical_system_model)
    SM_pred = to_numerical_system_model(SM_pred)
    SM_press = to_numerical_system_model(SM_press)
    SM_corr = to_numerical_system_model(SM_corr)

    return SplitForPressureResult(
        SM_pred=SM_pred, SM_press=SM_press, SM_corr=SM_corr)


def split_simple(sm, pressure_vars, dt, *, bottom=None):
    """DEPRECATED — use :func:`split_for_pressure_structural` (via
    ``model.chorin_split(dt)``).

    Minimal hand-rolled splitter retained ONLY for the hand-built
    ``escalante_vam_bump/run.py`` reference oracle; it dispatches rows by the
    ``equation_names`` convention (``cont_j*`` ⇒ constraint).  New models /
    agents must use the structural splitter, which discovers the same roles
    from the operators (zero mass-matrix rows ⇒ constraints).

    Copies the SystemModel, then deletes rows and terms.

    Algorithm:

    1. **Copy** the parent SystemModel three times → ``(SM_pred,
       SM_press, SM_corr)``.

    2. **SM_pred**: keep only the rows whose ``equation_name`` does
       NOT start with ``cont_j`` (evolution rows including any
       trivial ones like ``b_eq`` for bathymetry-as-state).  Then
       ZERO out every pressure-state-dependent term:
       ``F.xreplace({P_k: 0})``, same for ``P``, ``B``, ``S``.
       The predictor thus advances with kinetic + gravity + NCP
       and ANY non-pressure source — exactly like the OLD
       ``PredictorCorrectorSolver`` did with its flux-only step.

    3. **SM_press**: keep only the ``cont_j*`` rows, with the
       ``dt``-baked ``U_corr`` substitution applied (reuses
       :func:`build_pressure_elliptic_block` for the substitution
       machinery).  This is the elliptic block in
       ``(P_0, P_1)`` that the implicit Newton-Krylov solver
       targets.

    4. **SM_corr**: the closed-form ``update_variables`` formula
       ``Q[k] ← Q[k] − (dt/h)·T_u_k(P)`` (also from
       :func:`build_pressure_elliptic_block`).  This is the
       projection-corrector that applies the freshly-solved
       pressure to the momentum modes.

    Parameters
    ----------
    sm : SystemModel
        Parent system.  Must carry ``equation_names`` whose values
        flag algebraic rows by the ``cont_j`` prefix.
    pressure_vars : Sequence[sp.Symbol]
        Pressure state symbols (e.g. ``[P_0, P_1]``).
    dt : sp.Symbol
        Symbolic time-step (gets baked into ``SM_press`` /
        ``SM_corr`` operators).
    bottom : optional
        Forwarded to :func:`build_pressure_elliptic_block`.

    Returns
    -------
    :class:`SplitForPressureResult`
    """
    warnings.warn(
        "split_simple is deprecated; use split_for_pressure_structural "
        "(model.chorin_split(dt)).  split_simple is kept only for the "
        "hand-built escalante_vam_bump reference oracle.",
        DeprecationWarning, stacklevel=2)
    # ── 0. preliminaries
    state = list(sm.state)
    state_names = [str(s) for s in state]
    if "h" not in state_names:
        raise ValueError("SystemModel must include 'h' state entry.")
    pressure_indices = [state.index(p) for p in pressure_vars]
    pressure_subs = {p: sp.S.Zero for p in pressure_vars}

    # ── 1. SM_pred: copy + drop algebraic rows + zero pressure terms ──
    # Identify evolution rows by equation_name (cont_j* are algebraic).
    eq_names = list(sm.equation_names)
    evo_rows = [i for i, name in enumerate(eq_names)
                if not name.startswith("cont_j")]
    pred_eq_names = [eq_names[i] for i in evo_rows]

    # Build ``equation_to_state_index`` for SM_pred from the parent.
    parent_e2s = list(sm.equation_to_state_index) if (
        sm.equation_to_state_index is not None
    ) else list(range(sm.n_equations))
    pred_e2s = [parent_e2s[i] for i in evo_rows]

    n_pred = len(evo_rows)
    n_st = sm.n_state
    n_dim = sm.n_dim

    def _ndim(matrix):
        """Number of array dimensions.  Distinguishes sp.Matrix
        (ndim=2 always; .rank() = linear-algebra rank) from
        NDimArray (.rank() = number of indices)."""
        return len(matrix.shape)

    def _row_slice(matrix, rows):
        """Slice rows from a (n_eq, ...) array; return sp.Matrix
        for 2D or NDimArray for 3D."""
        nd = _ndim(matrix)
        if nd == 2:
            return sp.Matrix(
                len(rows), matrix.shape[1],
                lambda i, j, rs=rows: sp.sympify(matrix[rs[i], j]),
            )
        if nd == 3:
            n_a, n_b, n_c = matrix.shape
            out = sp.MutableDenseNDimArray.zeros(len(rows), n_b, n_c)
            for i_new, i_old in enumerate(rows):
                for j in range(n_b):
                    for k in range(n_c):
                        out[i_new, j, k] = sp.sympify(matrix[i_old, j, k])
            return out
        raise ValueError(f"Unsupported matrix ndim {nd}")

    def _zero_pressure(matrix):
        """Substitute every pressure-state symbol → 0 in every entry."""
        nd = _ndim(matrix)
        if nd == 2:
            return matrix.xreplace(pressure_subs)
        if nd == 3:
            n_a, n_b, n_c = matrix.shape
            out = sp.MutableDenseNDimArray.zeros(n_a, n_b, n_c)
            for i in range(n_a):
                for j in range(n_b):
                    for k in range(n_c):
                        out[i, j, k] = sp.sympify(matrix[i, j, k]).xreplace(
                            pressure_subs
                        )
            return out
        raise ValueError(f"Unsupported matrix ndim {nd}")

    F_pred = _zero_pressure(_row_slice(sm.flux, evo_rows))
    P_pred = _zero_pressure(_row_slice(sm.hydrostatic_pressure, evo_rows))
    B_pred = _zero_pressure(_row_slice(sm.nonconservative_matrix, evo_rows))
    S_pred = _zero_pressure(_row_slice(sm.source, evo_rows))
    M_pred = _row_slice(sm.mass_matrix, evo_rows)    # mass matrix is unchanged

    from zoomy_core.systemmodel.system_model import SystemModel as _SM

    SM_pred = _SM(
        time=sm.time,
        space=list(sm.space),
        state=list(state),
        aux_state=list(sm.aux_state),
        parameters=sm.parameters,
        parameter_values=sm.parameter_values,
        flux=F_pred,
        hydrostatic_pressure=P_pred,
        nonconservative_matrix=B_pred,
        source=S_pred,
        mass_matrix=M_pred,
        equation_to_state_index=list(pred_e2s),
        boundary_conditions=sm.boundary_conditions,
        aux_boundary_conditions=sm.aux_boundary_conditions,
        boundary_gradients=sm.boundary_gradients,
        initial_conditions=sm.initial_conditions,
        aux_initial_conditions=sm.aux_initial_conditions,
        update_variables=sm.update_variables,
        reconstruction_variables=sm.reconstruction_variables,
        state_from_reconstruction=sm.state_from_reconstruction,
    )
    SM_pred.equation_names = list(pred_eq_names)
    # Copy the aux_registry so the runtime knows how to compute each
    # aux row.  The aux_state list (Symbol order) is already shared.
    SM_pred.aux_registry = list(sm.aux_registry) if sm.aux_registry else None
    SM_pred.history.append({
        "name": "split_simple[pred]",
        "description": (
            f"copied parent + dropped rows {[eq_names[i] for i in range(sm.n_equations) if i not in evo_rows]}"
            f" + zeroed pressure-state symbols {[str(p) for p in pressure_vars]} in F/P/B/S"
        ),
    })

    # ── 2. SM_press: dt-baked elliptic block on the cont_j* rows ─────
    # Reuse build_pressure_elliptic_block — it does the U_corr
    # substitution and returns rows ready to use as algebraic
    # constraints in (P_0, P_1).
    block = build_pressure_elliptic_block(sm, pressure_vars, dt,
                                          bottom=bottom)
    press_eq_names = [f"elliptic_j{j}" for j in sorted(block["rows"])]
    press_eq_residuals = [
        sp.expand(block["rows"][j]) for j in sorted(block["rows"])
    ]
    SM_press = _build_subsystem(
        eq_names=press_eq_names,
        eq_residuals=press_eq_residuals,
        sm_parent=sm,
        state=state,
        equation_to_state_index=list(pressure_indices),
        source_only=True,
        history_entry={
            "name": "split_simple[press]",
            "description": (
                f"elliptic block (source-only / central-discretised): "
                f"{len(press_eq_names)} rows determining "
                f"{[str(p) for p in pressure_vars]}.  The full elliptic "
                "residual is packed into ``source``; spatial derivatives "
                "become aux entries evaluated by a centrally-differenced "
                "LSQ stencil at runtime — the natural numerics for an "
                "elliptic problem, in contrast to the path-integral / "
                "Riemann decomposition used by the hyperbolic predictor."
            ),
        },
    )
    # Minimal live aux: only the pressure derivatives stay in aux_registry;
    # the frozen predictor-produced aux move to aux_input_registry.
    _partition_pressure_aux(SM_press, pressure_vars)
    _attach_pressure_operator(SM_press)

    # ── 3. SM_corr: update_variables = U − (dt/h)·T_u(P) ─────────────
    # Identical pattern to split_for_pressure's SM_corr.
    t = sm.time
    coords = list(sm.space)
    M_M = len(block["U_corr"]) - 1
    N_w_active = len(block["W_corr"])
    corr_e2s_index = []
    update_exprs = []
    update_names = []

    def _state_index_for(prefixes_and_k):
        for pref, k in prefixes_and_k:
            cand = f"{pref}{k}"
            if cand in state_names:
                return state_names.index(cand), cand
        return None, None

    for k in range(M_M + 1):
        idx, _ = _state_index_for([("U_", k), ("q_U", k)])
        if idx is None:
            continue
        update_exprs.append(sp.expand(block["U_corr"][k]))
        corr_e2s_index.append(idx)
        update_names.append(f"corr_U_{k}")
    for k in range(N_w_active):
        idx, _ = _state_index_for([("W_", k), ("q_W", k)])
        if idx is None:
            continue
        update_exprs.append(sp.expand(block["W_corr"][k]))
        corr_e2s_index.append(idx)
        update_names.append(f"corr_W_{k}")

    state_funcs = [sp.Function(str(s), real=True)(t, *coords) for s in state]
    func_to_sym = dict(zip(state_funcs, state))
    update_exprs_sym = [e.xreplace(func_to_sym) for e in update_exprs]

    n_eq_corr = len(corr_e2s_index)
    SM_corr = _SM(
        time=sm.time,
        space=list(sm.space),
        state=list(state),
        aux_state=[],
        position=sm.position,
        parameters=sm.parameters,
        parameter_values=sm.parameter_values,
        flux=sp.zeros(n_eq_corr, n_dim),
        hydrostatic_pressure=sp.zeros(n_eq_corr, n_dim),
        nonconservative_matrix=_zero_ndim(n_eq_corr, n_st, n_dim),
        source=sp.zeros(n_eq_corr, 1),
        mass_matrix=sp.zeros(n_eq_corr, n_st),
        equation_to_state_index=list(corr_e2s_index),
        boundary_conditions=sm.boundary_conditions,
        aux_boundary_conditions=sm.aux_boundary_conditions,
        boundary_gradients=sm.boundary_gradients,
        initial_conditions=sm.initial_conditions,
        aux_initial_conditions=sm.aux_initial_conditions,
        update_variables=sp.Matrix(update_exprs_sym),
        reconstruction_variables=sm.reconstruction_variables,
        state_from_reconstruction=sm.state_from_reconstruction,
    )
    SM_corr.equation_names = list(update_names)
    SM_corr.expose_aux_atoms()
    # SM_corr is the ONE sub-system built directly instead of through
    # ``_build_subsystem``, so it used to miss the parent-aux prefix — and with
    # it any plain-Symbol aux the parent owns (``hinv``).  The corrector's
    # ``update_variables`` then referenced ``hinv`` without declaring it and the
    # symbol survived into ``np.asarray`` (``TypeError: Cannot convert
    # expression to float``).  Inherit exactly like the other two stages.
    _inherit_parent_aux(SM_corr, sm)
    _inherit_equilibrium_reconstruction(SM_pred, sm)
    SM_corr._bc_source = getattr(sm, "_bc_source", None)
    SM_corr._aux_bc_source = getattr(sm, "_aux_bc_source", None)
    SM_corr.history.append({
        "name": "split_simple[corr]",
        "description": (
            f"corrector: explicit update on "
            f"{[state_names[i] for i in corr_e2s_index]} via update_variables"
        ),
    })

    return SplitForPressureResult(
        SM_pred=SM_pred,
        SM_press=SM_press,
        SM_corr=SM_corr,
    )


__all__ = [
    "build_pressure_elliptic_block",
    "build_pressure_linear_operator",
    "PressureLinearOperator",
    "verify_p_linearity",
    "split_for_pressure",
    "split_simple",
    "SplitForPressureResult",
    "Stage",
]
