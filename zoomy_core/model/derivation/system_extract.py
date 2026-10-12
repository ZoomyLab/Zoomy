"""Classifies the additive terms of a derivation Model's residuals into SystemModel slots (mass
matrix, flux, hydrostatic pressure, nonconservative matrix, source, diffusion matrix). It is the
bridge from the derivation Model to ``SystemModel.from_model``.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.misc.misc import Zstruct


__all__ = ["extract_system_operators", "HydrostaticPressure", "ViscousDiffusion"]


class ViscousDiffusion(sp.Function):
    """Opaque provenance marker wrapping a viscous diffusive flux ``d_x(ViscousDiffusion(D d_x
    Q_j))``. It routes the whole rank-4 compound, including foreign-state cross pieces, into
    ``diffusion_matrix`` rather than ``flux``.
    """

    nargs = 1

    @classmethod
    def eval(cls, arg):
        """Leave the marker unevaluated; a zero argument collapses to zero."""
        if arg == 0:
            return sp.S.Zero
        return None


class HydrostaticPressure(sp.Function):
    """Opaque marker wrapping a pressure-flux argument; ``d_x(HydrostaticPressure(P))`` routes to
    ``hydrostatic_pressure`` while any other conservative ``d_x(F)`` routes to ``flux``. The model
    applies it explicitly, never inferred from gravity.
    """

    nargs = 1

    @classmethod
    def eval(cls, arg):
        """Leave the marker unevaluated."""
        return None


def _field_name(field):
    """Name of a field's Function head (or of the field itself)."""
    head = field.func if hasattr(field, "func") else field
    return getattr(head, "__name__", str(head))


#: Sign predicates a model author can DECLARE on a Function head, strongest
#: first.  Exactly ONE is re-asserted on the extracted Symbol and sympy derives
#: the rest of the closure — asserting the derived predicates too (nonzero,
#: negative=False, …) over-specifies the symbol and sympy rejects it.
_SIGN_PREDICATES = ("zero", "positive", "negative",
                    "nonnegative", "nonpositive", "nonzero")


def head_assumptions(field) -> dict:
    """Assumptions declared on a field's Function head (``real`` plus its strongest sign predicate),
    so that e.g. ``h > 0`` survives into the SystemModel. Only declared fields inherit; derived aux
    symbols do not.
    """
    head = field.func if hasattr(field, "func") else field
    out = {"real": True}                  # every state field is real-valued
    for k in _SIGN_PREDICATES:
        if getattr(head, f"is_{k}", None) is True:
            out[k] = True
            break
    return out


def _state_symbol(field):
    """State/aux Symbol of a modal field: ``q(0,t,x)`` becomes ``q_0``, ``h(t,x)`` becomes ``h``,
    with the head's assumptions.
    """
    name = _field_name(field)
    args = getattr(field, "args", ())
    assumptions = head_assumptions(field)
    if args and getattr(args[0], "is_Integer", False):
        return sp.Symbol(f"{name}_{int(args[0])}", **assumptions)
    return sp.Symbol(name, **assumptions)


def _collect_fields(expr, param_names):
    """All undefined-Function field applications in ``expr``, excluding parameters, basis heads and
    sympy builtins.
    """
    from sympy.core.function import AppliedUndef
    out = set()
    for atom in expr.atoms(sp.Function):
        if not isinstance(atom, AppliedUndef):
            func = getattr(atom, "func", None)
            if (func is None
                    or sp.Function not in getattr(func, "__bases__", ())
                    or func.__dict__.get("eval") is not None):
                continue
        if getattr(atom.func, "_is_basis_head", False):
            continue
        if atom.func.__name__ in param_names:
            continue
        out.add(atom)
    return out


def extract_system_operators(model, Q, Qaux=None, dae=False):
    """Extract the operator matrices (mass, flux, hydrostatic pressure, nonconservative, source,
    diffusion) from a Model's residuals, one evolution row per state field. Raises ``ValueError``
    for fields outside ``Q`` and ``Qaux``, unresolved integrals, unroutable terms, and (unless
    ``dae``) a mass matrix that is not the identity.

    Args:
        model: Model after the full closure pipeline.
        Q: Applied state fields, one per evolution row.
        Qaux: Applied auxiliary fields; ``None`` takes every field not in ``Q``.
        dae: Keep a non-diagonal or singular mass matrix (index-1 DAE) instead of raising.

    Returns:
        Dict of ``SystemModel`` keyword arguments (``time, space, state, aux_state, parameters,
            flux, hydrostatic_pressure, nonconservative_matrix, source, mass_matrix,
            diffusion_matrix, ...``) plus the ``unclassified`` remainders.
    """
    t = model.coords[0]
    # ``space`` = the spatial coordinates the STATE fields actually depend on, in
    # coord order (horizontals first, then the — possibly σ-mapped — vertical).
    # The vertical is NOT special-cased out of the flux directions: the σ-map
    # having converted ``z → ζ`` in ``coords`` is funneled straight through, so a
    # stay-3D model carrying ``ũ(t, x, ζ)`` extracts over ``space = (x, ζ)`` (∂_ζ
    # routes as a genuine flux / diffusion direction).  A depth-reduced model
    # (SME / VAM / multilayer — the vertical integrated out, every state field in
    # ``(t, *horiz)``) yields exactly its horizontals, byte-identical to the
    # former ``model.horizontal`` path.  The fallback keeps ≥1 horizontal for a
    # degenerate all-spatially-constant state.
    spatial = list(model.coords[1:])
    used = {a for f in Q for a in getattr(f, "args", ()) if a in spatial}
    space = [c for c in spatial if c in used] or [model.coords[1]]
    n_dim = len(space)

    param_names = set(model.parameters.keys())
    gravity_param = (model.parameters.g
                     if "g" in model.parameters.keys() else None)

    # ── field coverage validation ────────────────────────────────────────
    all_fields = set()
    for eq in model._equations.values():
        all_fields |= _collect_fields(eq.expr, param_names)

    q_symbols = {_state_symbol(f) for f in Q}

    if Qaux is None:
        Qaux = []
        seen = set(q_symbols)
        for f in sorted(all_fields, key=str):
            s = _state_symbol(f)
            if s in seen:
                continue
            seen.add(s)
            Qaux.append(f)
    aux_symbols = {_state_symbol(f) for f in Qaux}

    covered = q_symbols | aux_symbols
    uncovered = sorted(
        {f for f in all_fields if _state_symbol(f) not in covered}, key=str)
    if uncovered:
        names = ", ".join(sp.sstr(f) for f in uncovered)
        raise ValueError(
            "SystemModel.from_model: the following field(s) appear in the "
            f"equations but are in neither Q nor Qaux: {names}.  Add each to "
            "Q (state) or Qaux (auxiliary).")

    # ── state / aux Symbol lists (ordered as given) ──────────────────────
    state = [_state_symbol(f) for f in Q]
    aux_state = [_state_symbol(f) for f in Qaux]
    # Per-state-field applied-Function map: carry each field's FULL coordinate
    # signature (``h(t,x)``, ``ũ(t,x,ζ)``) so per-field dimensionality survives
    # the ``_state_symbol`` name-collapse.  Consumed by
    # ``SystemModel.is_vertical_dependent`` / a dimensional split.  (Aux applied
    # forms also live in each ``aux_registry`` entry's ``"atom"``.)
    state_function_map = {s: f for s, f in zip(state, Q)}
    n_eq = len(state)
    n_state = len(state)

    # Map every field application → its state/aux Symbol-as-function, carrying
    # ONLY the spatial coords that field actually depends on (in ``space`` order)
    # — PER-FIELD dimensionality.  A depth field ``h(t,x)`` therefore stays
    # ζ-INDEPENDENT (``∂_ζ h ≡ 0``) even on a stay-3D ``(x,ζ)`` extraction, so a
    # ζ-diffusion of ``mom/h`` yields ``∂_ζ(ν/h² ∂_ζ mom)`` alone — no spurious
    # ``∂_ζ h`` coupling.  Depth-reduced models (every field on ``(t,*horiz)``,
    # ``space == horiz``) get ``(t,*space)`` for every field, byte-identical to
    # the former uniform mapping.
    def _canon(s, f):
        """State/aux Function of ``(t, *space)`` limited to the coordinates of ``f``."""
        used_sp = [c for c in space if c in getattr(f, "args", ())]
        return sp.Function(str(s), real=True)(t, *used_sp)

    state_fn = {_state_symbol(f): _canon(_state_symbol(f), f) for f in Q}
    aux_fn = {_state_symbol(f): _canon(_state_symbol(f), f) for f in Qaux}
    field_to_fn = {}
    for f in Q:
        field_to_fn[f] = state_fn[_state_symbol(f)]
    for f in Qaux:
        field_to_fn[f] = aux_fn[_state_symbol(f)]

    state_funcs = [state_fn[s] for s in state]

    # ── operator tensors ─────────────────────────────────────────────────
    F = sp.zeros(n_eq, n_dim)
    P = sp.zeros(n_eq, n_dim)
    B = sp.MutableDenseNDimArray.zeros(n_eq, n_state, n_dim)
    S = sp.zeros(n_eq, 1)
    M = sp.zeros(n_eq, n_state)
    A = sp.MutableDenseNDimArray.zeros(n_eq, n_state, n_dim, n_dim)
    A_nonzero = False

    rows = _assign_rows(model, Q, field_to_fn, state, state_fn, t)

    residuals = []
    for i in range(n_eq):
        residual = sp.expand(rows[i].xreplace(field_to_fn))
        residuals.append(residual)
        _classify_row(
            residual, i, state, state_funcs, t, space, gravity_param,
            F, P, B, S, M, A,
        )
        for d in range(n_dim):
            for j in range(n_state):
                for e in range(n_dim):
                    if A[i, j, d, e] != 0:
                        A_nonzero = True

    # ── classification completeness guard ────────────────────────────────
    # Term-conservation check: reassemble each row from the slots it was
    # routed to (``M∂_t Q + ∇·(F+P) + B∂_x Q + ∇·(A∇Q) − S``) and subtract
    # the ORIGINAL residual.  A nonzero remainder is a term the classifier
    # matched NO branch for — a SILENT DROP (the class of bug that hid the
    # VAM bed-slope ``A[v,b]`` diffusion for weeks).  For a correctly-routed
    # model every remainder is identically zero; the healthy families
    # therefore report an EMPTY list.  Computed in state-Function form (before
    # the Symbol back-substitution) so ``∂_x`` acts on coordinate-dependent
    # atoms.
    unclassified = _classification_remainder(
        residuals, state_funcs, t, space, F, P, B, S, M, A)

    # Back-substitute Symbol-functions → bare Symbols for storage.
    rev = {v: k for k, v in {**state_fn, **aux_fn}.items()}
    unclassified = [(i, r.xreplace(rev)) for i, r in unclassified]

    def _to_sym(m):
        """Substitute the Symbol-functions back by bare Symbols."""
        return m.xreplace(rev) if hasattr(m, "xreplace") else m

    F = _to_sym(F)
    P = _to_sym(P)
    S = _to_sym(S)
    M = _to_sym(M)
    B = sp.MutableDenseNDimArray(
        [[[_to_sym(sp.sympify(B[r, c, d])) for d in range(n_dim)]
          for c in range(n_state)] for r in range(n_eq)])
    A_out = None
    if A_nonzero:
        A_out = sp.MutableDenseNDimArray(
            [[[[_to_sym(sp.sympify(A[r, c, dd, ee]))
                for ee in range(n_dim)] for dd in range(n_dim)]
              for c in range(n_state)] for r in range(n_eq)])

    # ── mass-matrix contract ───────────────────────────────────────────
    # The runtime solvers integrate ``∂_t Q = RHS`` — they do NOT consume a
    # mass matrix.  The DERIVATION owns the normalization: every model
    # applies ``InvertMassMatrix()`` after its conservative CoV, so dynamic
    # rows arrive with unit ∂_t coefficient.  The extraction only CHECKS:
    # a non-unit (or non-constant / off-diagonal) mass entry raises — no
    # silent rescaling, no silently-wrong (2k+1)× dynamics.
    #
    # DAE opt-in: a genuine index-1 DAE (VAM pressure, curvilinear
    # double-moment) legitimately carries a non-diagonal / singular M(Q) that
    # the DAESolver inverts numerically per cell.  ``dae=True`` retains it
    # verbatim (cancel-normalised) and skips the diagonal contract.
    parameters = Zstruct(**{k: model.parameters[k]
                            for k in model.parameters.keys()})
    parameters._symbolic_name = "p"
    parameter_values = Zstruct(**{k: getattr(model.parameter_values, k, 0.0)
                                  for k in model.parameters.keys()})

    if dae:
        M = sp.Matrix(n_eq, n_state,
                      lambda i, j: sp.cancel(sp.sympify(M[i, j])))
    else:
        for i in range(n_eq):
            offdiag = [sp.simplify(M[i, j]) for j in range(n_state)
                       if j != i and sp.simplify(M[i, j]) != 0]
            if offdiag:
                raise ValueError(
                    f"row {i}: off-diagonal mass-matrix entries {offdiag} — "
                    "the runtime cannot integrate this; apply InvertMassMatrix "
                    "(or an explicit inversion) in the derivation.")
            m_ii = sp.cancel(sp.sympify(M[i, i])) if i < n_state else sp.S.Zero
            M[i, i] = m_ii
            if m_ii not in (sp.S.Zero, sp.S.One):
                raise ValueError(
                    f"row {i}: mass-matrix diagonal {m_ii} != 1 — the runtime "
                    "integrates ∂_t Q = RHS, so this row would evolve "
                    f"{sp.nsimplify(1/m_ii)}× too fast.  Apply "
                    "InvertMassMatrix() in the derivation (after the "
                    "conservative change of variables).")

    return dict(
        time=t, space=space, state=state, aux_state=aux_state,
        parameters=parameters, parameter_values=parameter_values,
        flux=F, hydrostatic_pressure=P,
        nonconservative_matrix=B, source=S, mass_matrix=M,
        diffusion_matrix=A_out,
        state_function_map=state_function_map,
        unclassified=unclassified,
    )


def _classification_remainder(residuals, state_funcs, t, space,
                              F, P, B, S, M, A):
    """Rebuild each row from the slots it was routed to and subtract the original residual; a
    nonzero remainder is a silently dropped term.

    Returns:
        List of ``(row, remainder)`` for nonzero rows (empty if every term is accounted for).
    """
    n_eq = len(residuals)
    n_state = len(state_funcs)
    n_dim = len(space)

    def _unwrap(expr):
        """Strip the ``HydrostaticPressure`` and ``ViscousDiffusion`` markers."""
        expr = expr.replace(
            lambda e: isinstance(e, HydrostaticPressure), lambda e: e.args[0])
        expr = expr.replace(
            lambda e: isinstance(e, ViscousDiffusion), lambda e: e.args[0])
        return expr

    out = []
    for i in range(n_eq):
        recon = sp.S.Zero
        for j in range(n_state):
            if M[i, j] != 0:
                recon += M[i, j] * sp.Derivative(state_funcs[j], t)
        for d in range(n_dim):
            if F[i, d] != 0:
                recon += sp.Derivative(F[i, d], space[d])
            if P[i, d] != 0:
                recon += sp.Derivative(P[i, d], space[d])
            for j in range(n_state):
                if B[i, j, d] != 0:
                    recon += B[i, j, d] * sp.Derivative(state_funcs[j], space[d])
                for e in range(n_dim):
                    if A[i, j, d, e] != 0:
                        recon += sp.Derivative(
                            A[i, j, d, e] * sp.Derivative(state_funcs[j],
                                                          space[e]),
                            space[d])
        recon -= S[i, 0]
        remainder = sp.expand((_unwrap(residuals[i]) - recon).doit())
        if remainder != 0:
            remainder = sp.simplify(remainder)
        if remainder != 0:
            out.append((i, remainder))
    return out


def _classify_row(residual, i, state, state_funcs, t, space, gravity_param,
                  F, P, B, S, M, A):
    """Split one row residual term by term into the ``M / F / P / B / S / A`` slots, filling them in
    place.
    """
    from zoomy_core.model.derivation.tag_extraction import (
        _split_coeff_and_derivative, _first_order_direction,
    )

    n_state = len(state)
    state_set = set(state_funcs)
    sym_of_func = {sf: state[j] for j, sf in enumerate(state_funcs)}

    for term in sp.Add.make_args(residual):
        if term == 0:
            continue

        # 0. Unresolved Galerkin integrals must NOT reach the runtime —
        # they appear when a material closure is not analytically
        # integrable (e.g. Bingham): the user must opt into numerical
        # integration explicitly.
        if term.atoms(sp.Integral):
            raise ValueError(
                f"row {i}: analytically unresolved integral in {term} — "
                "the Galerkin projection of this term has no closed form "
                "(non-polynomial material closure?). Build the model with "
                "quadrature_order=N (Gauss-Legendre numerical integration "
                "via the GaussQuadrature operation) to resolve it.")

        # 1. Time derivative → mass matrix (no sign flip — LHS).
        if any(isinstance(d, sp.Derivative) and t in d.variables
               for d in term.atoms(sp.Derivative)):
            coeff, deriv = _split_coeff_and_derivative(term)
            if deriv is not None and deriv.variables == (t,):
                inner = deriv.args[0]
                if inner in sym_of_func:
                    j = state_funcs.index(inner)
                    M[i, j] = M[i, j] + coeff
                    continue
                # ``∂_t(c · Q_j)`` — the conservative time term carries the Gram
                # factor INSIDE the derivative (``∂_t(q_k/(2k+1))``).  Pull the
                # state-free constant ``c`` out → mass-matrix entry ``coeff·c``.
                c_in, base = inner.as_independent(*state_funcs, as_Mul=True)
                if base in sym_of_func:
                    j = state_funcs.index(base)
                    M[i, j] = M[i, j] + coeff * c_in
                    continue
            # Fallback: ∂_t of a non-state function (shouldn't happen for SME).
            raise ValueError(
                f"row {i}: unhandled time-derivative term {term}")

        coeff, deriv = _split_coeff_and_derivative(term)

        # 1b. ViscousDiffusion-marked compound → ALWAYS the diffusion tensor.
        # ``package_viscous`` tags its conservative viscous flux
        # ``∂_{x_d}(ViscousDiffusion(D·∂_{x_e} Q_j))`` so the extractor routes
        # the WHOLE rank-4 A[i, j, d, e] — including the σ-metric OFF-DIAGONAL
        # cross pieces ``D·∂_x h`` / ``D·∂_x b`` (``j`` a foreign state) that
        # ``_is_self_diffusion`` would otherwise send to the flux (dropping the
        # in-plane h-column; REQ-176(4)).  Unwrap the marker and route.
        if term.has(ViscousDiffusion):
            unwrapped = term.replace(
                lambda e: isinstance(e, ViscousDiffusion), lambda e: e.args[0])
            coeff_u, deriv_u = _split_coeff_and_derivative(unwrapped)
            _route_diffusion(unwrapped, i, deriv_u, coeff_u,
                             state_funcs, space, A)
            continue

        # 2. Second-order ``∂_{x_d}( A · ∂_{x_e}(·) )`` — split by WHAT is being
        # differentiated inside:
        #
        #   (a) GENUINE viscous self-diffusion — the inner derivative carries the
        #       row's OWN conserved variable (the diffusive flux ``Fᵈ = A ∂_x u``
        #       of this row's momentum, incl. the CoV cross piece ``∂_x h`` that
        #       ``∂_x(q/h)`` expands into) → ``diffusion_matrix`` (checked before
        #       flux/NCP: the outer ``∂_{x_d}`` is first-order but its argument is
        #       itself a gradient, so it must NOT be mistyped as advective flux).
        #
        #   (b) OFF-DIAGONAL ``∂_{x_d}( A(Q) · ∂_{x_e}(field) )`` whose inner
        #       derivative is of a FOREIGN field (``field`` ≠ this row's own
        #       state var) — NOT a viscous flux.  It is a CONSERVATIVE flux that
        #       merely carries a foreign-field gradient: the canonical case is the
        #       bed-slope vertical-momentum flux ``∂_x(A·∂_x b)`` arising from the
        #       bottom KBC ``w|_bed = u·∂_x b`` (VAM r_k rows).  Route it to the
        #       FLUX column ``d`` with the inner ``∂_{x_e}(field)`` left intact —
        #       ``expose_aux_atoms`` then freezes it to a gradient-aux (``b_x``),
        #       exactly as the hand-rolled VAM treats ``b_x`` as a derivative-aux
        #       and the bed-slope term as a flux integrand.  ``diffusion_matrix``
        #       is RESERVED for the genuine ν self-diffusion of case (a); an
        #       off-diagonal A[v, b] entry is dropped at runtime (numpy raises on
        #       off-diagonal A; jax-Chorin has no diffusion path), which silently
        #       killed the bed-slope coupling (REQ-80).
        dirs2 = (_second_order_dirs(deriv, space)
                 if deriv is not None else None)
        if dirs2 is not None:
            own = state_funcs[i]
            if _is_self_diffusion(deriv, own, space):
                _route_diffusion(term, i, deriv, coeff, state_funcs, space, A)
            else:
                d_out = dirs2[0]
                F[i, d_out] = F[i, d_out] + coeff * deriv.args[0]
            continue

        # 3. No first-order spatial derivative → source (sign-flipped: S = −LHS).
        d = (_first_order_direction(deriv, space)
             if deriv is not None else None)
        if d is None:
            S[i, 0] = S[i, 0] - term
            continue
        x = space[d]

        inner = deriv.args[0]
        coeff_has_state = any(coeff.has(f) for f in state_set)

        # 4. coeff·∂_{x_d}(state).  A state-free, (t, x_d)-free coeff is an EXACT
        # divergence ``∂_{x_d}(coeff·Q_j)`` — a conservative FLUX in column d,
        # never a ``B·∂_{x_d} Q`` coupling (the mass row ``∂_{x_d} q_0`` must
        # reach the generated flux kernels: interface/open-boundary mass
        # exchange uses flux + fluctuations, mass-in-NCP transmits only the
        # jump).  Genuinely non-conservative couplings (state-dependent coeff:
        # the bed slope ``g h ∂_{x_d} b``, the SME cross-mode
        # ``q_i/h·∂_{x_d} q_j``) stay in B.
        if inner in sym_of_func:
            if not coeff_has_state and not coeff.has(x) and not coeff.has(t):
                # Bare-state conservative flux ∂_{x_d}(c·Q_j) → flux.  (A bare
                # state is never a pressure term — pressure is g·h²/2, marked
                # explicitly and handled in branch 5.)
                F[i, d] = F[i, d] + coeff * inner
                continue
            j = state_funcs.index(inner)
            B[i, j, d] = B[i, j, d] + coeff
            continue

        inner_has_state = any(inner.has(f) for f in state_set)

        # 5. ∂_{x_d}(F(state)) with state-free coeff → flux (col d), UNLESS a
        # summand is wrapped in the HydrostaticPressure marker → that summand
        # routes to hydrostatic_pressure (unwrapped).  Pressure routing is
        # MANUAL: the model marks ``g·h²/2`` (no gravity-guessing here), which
        # also splits a bundled ∂_x(g h²/2 + advection) correctly.
        if inner_has_state and not coeff_has_state:
            for piece in sp.Add.make_args(sp.expand(inner)):
                if piece.has(HydrostaticPressure):
                    unwrapped = piece.replace(
                        lambda e: isinstance(e, HydrostaticPressure),
                        lambda e: e.args[0])
                    P[i, d] = P[i, d] + coeff * unwrapped
                else:
                    F[i, d] = F[i, d] + coeff * piece
            continue

        # 6. state-dependent coeff on a ∂_{x_d} argument — the non-conservative
        # product that production keeps unfolded (the SME cross-mode couplings
        # ``q_i/h · ∂_{x_d} q_j``).  Route to ``B`` column d (or source).
        _route_nonconservative_product(
            term, i, state, state_funcs, space, d, B, S)


def _second_order_dirs(deriv, space):
    """Outer and inner spatial direction indices of a second-order spatial derivative.

    Returns:
        ``(outer, inner)`` or ``None``.
    """
    v = deriv.variables
    # Pure second derivative ``∂_{x_d} ∂_{x_e}`` (covers ``∂_{x_d}²``).
    if len(v) == 2 and v[0] in space and v[1] in space:
        return space.index(v[0]), space.index(v[1])
    # Outer ``∂_{x_d}`` whose argument carries an inner spatial ``∂_{x_e}``.
    if len(v) == 1 and v[0] in space:
        d = space.index(v[0])
        inner = deriv.args[0]
        for dd in inner.atoms(sp.Derivative):
            if len(dd.variables) == 1 and dd.variables[0] in space:
                return d, space.index(dd.variables[0])
    return None


def _is_self_diffusion(deriv, own_func, space):
    """True iff a second-order term differentiates a quantity carrying the row's own state variable
    (genuine viscosity), not a foreign field such as the bed gradient.
    """
    v = deriv.variables
    # Bare ``∂_{x_d}∂_{x_e} Q_j``: self iff the differentiated arg is the own var.
    if len(v) == 2 and v[0] in space and v[1] in space:
        return deriv.args[0].has(own_func)
    # Compound ``∂_{x_d}( A · ∂_{x_e}(arg) )``: self iff an inner spatial
    # derivative differentiates a quantity carrying the own var.
    inner = deriv.args[0]
    for dd in inner.atoms(sp.Derivative):
        if (any(var in space for var in dd.variables)
                and dd.args[0].has(own_func)):
            return True
    return False


def _route_diffusion(term, i, deriv, coeff, state_funcs, space, A):
    """Route ``coeff d_xd(D d_xe Q_j)`` into ``A[i, j, d, e]`` with the sign of the residual side
    (minus the coefficient); raises if no ``D d_x Q`` reading exists.
    """
    v = deriv.variables
    # Bare second derivative ``coeff · ∂_{x_d}∂_{x_e} Q_j``: identical to
    # ``∂_{x_d}(coeff · ∂_{x_e} Q_j)`` ONLY for a (t, spatial, state)-free coeff
    # — route that; anything else has no conservative reading here → raise
    # (silent drops in this branch hid the VAM b″ bathymetry terms for weeks).
    if len(v) == 2 and v[0] in space and v[1] in space:
        d, e = space.index(v[0]), space.index(v[1])
        q = deriv.args[0]
        if (q in state_funcs and not any(coeff.has(s) for s in space)
                and not any(coeff.has(f) for f in state_funcs)):
            j = state_funcs.index(q)
            A[i, j, d, e] = A[i, j, d, e] - coeff
            return
        raise ValueError(
            f"row {i}: cannot route bare second-derivative term {term} — "
            "rewrite it as a conservative compound ∂_x(D·∂_x Q) in the "
            "derivation (own atom, not mixed with derivative-free flux "
            "parts).")

    d = space.index(v[0])           # outer-derivative flux column
    inner = deriv.args[0]           # D · ∂_{x_e} Q_j  (the diffusive flux)
    # ``.doit()`` resolves an inner ``∂_{x_e}(q_j/h)`` (the CoV-introduced
    # conserved-variable derivative) into ``∂_{x_e} q_j/h − q_j·∂_{x_e} h/h²`` so
    # the BARE state derivatives surface and are routed; a raw ``∂_{x_e}(q_j/h)``
    # would otherwise carry a non-state ``q_j/h`` argument and the dominant
    # diagonal diffusion would be silently dropped.
    for sub in sp.Add.make_args(sp.expand(inner.doit())):
        c2, d2, e = _split_inner_x_derivative(sub, space)
        if d2 is not None and d2.args[0] in state_funcs:
            j = state_funcs.index(d2.args[0])
            A[i, j, d, e] = A[i, j, d, e] - coeff * c2
            continue
        raise ValueError(
            f"row {i}: diffusion-compound piece {sub} of ∂_x({inner}) has "
            "no ``D·∂_x Q`` reading — derivative-free flux content mixed "
            "into a ∂_x b-bearing compound is silently lost otherwise; "
            "keep bx-bearing flux parts in their OWN compound atom.")


def _split_inner_x_derivative(term, space):
    """Split a term into ``coeff * d_xe Q``.

    Returns:
        ``(coeff, derivative, e)`` or ``(None, None, None)``.
    """
    factors = term.args if isinstance(term, sp.Mul) else [term]
    deriv = None
    for f in factors:
        if (isinstance(f, sp.Derivative) and len(f.variables) == 1
                and f.variables[0] in space):
            deriv = f
            break
    if deriv is None:
        return None, None, None
    e = space.index(deriv.variables[0])
    coeff = sp.Mul(*[f for f in factors if f is not deriv])
    return coeff, deriv, e


def _route_nonconservative_product(term, i, state, state_funcs, space, d, B, S):
    """Route ``coeff(state) d_xd(F(state))`` into the nonconservative matrix ``B`` via the product
    rule, or to the source if it is not a clean coupling.
    """
    from zoomy_core.model.derivation.tag_extraction import (
        _split_coeff_and_derivative, _first_order_direction,
    )
    coeff, deriv = _split_coeff_and_derivative(term)
    if deriv is not None and _first_order_direction(deriv, space) == d:
        inner = deriv.args[0]
        if inner in state_funcs:
            j = state_funcs.index(inner)
            B[i, j, d] = B[i, j, d] + coeff
            return
        # Compound argument (``∂_{x_d}(P_1·h²)`` etc.): expand by the product
        # rule and route every bare-state derivative into B.  Leaving the
        # compound in SOURCE would surface it as an opaque LSQ aux symbol
        # (``P_1*h**2_x``) — the CFL path, the Riemann dissipation AND the
        # Chorin splitter are then blind to it (a pressure force frozen at
        # P^n that the corrector never sees).
        expanded = sp.expand(deriv.doit())
        pieces = []
        routed_all = True
        for sub in sp.Add.make_args(expanded):
            c2, d2, e = _split_inner_x_derivative(sub, space)
            if d2 is not None and d2.args[0] in state_funcs:
                pieces.append((state_funcs.index(d2.args[0]), e, coeff * c2))
            else:
                routed_all = False
                break
        if routed_all and pieces:
            for j, e, c in pieces:
                B[i, j, e] = B[i, j, e] + c
            return
    # Not a clean coupling — keep as source (sign-flipped).
    S[i, 0] = S[i, 0] - term


def _assign_rows(model, Q, field_to_fn, state, state_fn, t):
    """Order the equations so that row ``i`` is the one whose time derivative carries ``state[i]``,
    falling back to declaration order for algebraic rows.
    """
    eqs = list(model._equations.values())

    def _has_dt_of(expr, sf):
        """True iff ``expr`` has a time-derivative term of ``c * sf``, including a constant inside
        the derivative.
        """
        for der in expr.atoms(sp.Derivative):
            if der.variables == (t,):
                _, base = der.args[0].as_independent(sf, as_Mul=True)
                if base == sf:
                    return True
        return False

    rows = [None] * len(state)
    used = set()
    for i, s in enumerate(state):
        sf = state_fn[s]
        for k, eq in enumerate(eqs):
            if k in used:
                continue
            if _has_dt_of(sp.expand(eq.expr.xreplace(field_to_fn)), sf):
                rows[i] = eq.expr
                used.add(k)
                break
    remaining = [eq for k, eq in enumerate(eqs) if k not in used]
    ri = 0
    for i in range(len(state)):
        if rows[i] is None:
            rows[i] = remaining[ri].expr if ri < len(remaining) else sp.S.Zero
            ri += 1
    return rows
