"""Generic linearisation around a base state for a `SystemModel`.

Replaces every state slot `q` in the SystemModel's operator-form
residual `M·∂_t Q + ∂_x F + ∂_x P + B·∂_x Q − S` with
`q_0 + ε δq`, expands to first order in ε, and returns a new
`SystemModel` whose state is the perturbation vector `δq`.

Works for arbitrary state shapes — differential or algebraic rows
(zero mass-matrix rows pass through unchanged in form).  The base
values `q_0` may themselves be expressions in coordinates (useful
when the steady state is non-uniform, e.g. a varying bottom profile
`b(x)`).
"""
from __future__ import annotations

from typing import Dict

import sympy as sp


def linearise(sm, base_state: Dict, *, eps=None, simplify=True):
    """Insert `Q = Q_0 + ε δQ`, expand, return O(ε) SystemModel.

    Parameters
    ----------
    sm : SystemModel
        Input operator-form system.
    base_state : dict
        Maps each entry in `sm.state` to its base value (a scalar
        or an expression in coordinates).
    eps : sympy Symbol, optional
        Small parameter; created internally if None.
    simplify : bool
        Apply `sp.expand` to each linearised operator entry.

    Returns
    -------
    SystemModel
        New SystemModel whose state is `[δq_0, …, δq_{n-1}]` and
        whose operator matrices are linearised around `base_state`.
    """
    from zoomy_core.systemmodel.system_model import SystemModel

    if eps is None:
        eps = sp.Symbol("epsilon", positive=True)

    if set(base_state.keys()) != set(sm.state):
        missing = set(sm.state) - set(base_state.keys())
        extra = set(base_state.keys()) - set(sm.state)
        raise ValueError(
            f"base_state must list every state entry; "
            f"missing={missing!r}, extra={extra!r}."
        )

    # Build the perturbation state and the substitution map.
    delta_state = []
    repl = {}
    for s in sm.state:
        name = str(s)
        delta = sp.Symbol(rf"\delta {name}", real=True)
        delta_state.append(delta)
        repl[s] = base_state[s] + eps * delta

    # F, P, S are functions of Q evaluated INSIDE a derivative
    # ``∂_x F(Q)`` in the residual; their linearisation is the O(ε)
    # coefficient of ``F(Q_0 + ε δQ)`` (= ``∇F(Q_0)·δQ``).
    base_only = {s: base_state[s] for s in sm.state}

    def _lin_func(expr):
        e = sp.sympify(expr).xreplace(repl)
        try:
            lin = sp.expand(e).coeff(eps, 1)
            if lin.has(eps):
                raise ValueError("ε-dependent residue after coeff")
        except (sp.PolynomialError, ValueError, AttributeError):
            lin = sp.series(e, eps, 0, 2).removeO().coeff(eps, 1)
        if simplify:
            lin = sp.expand(lin)
        return lin

    # M, B are coefficients of ∂_t Q[j] and ∂_x Q[j] respectively.  The
    # residual contribution is e.g. ``M[i, j](Q)·∂_t Q[j]``; with
    # ``∂_t Q = ε ∂_t δQ`` the O(ε) coefficient is ``M[i, j](Q_0)·∂_t δQ[j]``.
    # So M_lin[i, j] = M[i, j] evaluated AT Q_0 (NOT a coefficient of ε).
    def _eval_at_base(expr):
        e = sp.sympify(expr).xreplace(base_only)
        if simplify:
            e = sp.expand(e)
        return e

    n_eq = sm.n_equations
    n_st = sm.n_state
    n_dim = sm.n_dim

    F_lin = sp.zeros(n_eq, n_dim)
    P_lin = sp.zeros(n_eq, n_dim)
    M_lin = sp.zeros(n_eq, n_st)
    S_lin = sp.zeros(n_eq, 1)
    B_lin = sp.MutableDenseNDimArray.zeros(n_eq, n_st, n_dim)

    for i in range(n_eq):
        for d in range(n_dim):
            F_lin[i, d] = _lin_func(sm.flux[i, d])
            P_lin[i, d] = _lin_func(sm.hydrostatic_pressure[i, d])
        S_lin[i, 0] = _lin_func(sm.source[i, 0])
        for j in range(n_st):
            M_lin[i, j] = _eval_at_base(sm.mass_matrix[i, j])
            for d in range(n_dim):
                B_lin[i, j, d] = _eval_at_base(
                    sm.nonconservative_matrix[i, j, d])

    sm_lin = SystemModel(
        time=sm.time,
        space=list(sm.space),
        state=delta_state,
        aux_state=list(sm.aux_state),
        parameters=sm.parameters,
        parameter_values=sm.parameter_values,
        flux=F_lin,
        hydrostatic_pressure=P_lin,
        nonconservative_matrix=B_lin,
        source=S_lin,
        mass_matrix=M_lin,
        equation_to_state_index=(
            list(sm.equation_to_state_index)
            if sm.equation_to_state_index is not None else None
        ),
    )
    if hasattr(sm, "equation_names"):
        sm_lin.equation_names = list(sm.equation_names)
    sm_lin.history.append({
        "name": "linearise",
        "description": f"linearised around base_state ({len(base_state)} fields)",
    })
    return sm_lin
