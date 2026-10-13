"""Symbolic inverse of the reconstruction-variables map.

Given a forward map `wb_k = f_k(state)` (one entry per state slot,
as ZArray), `invert_reconstruction` returns the closed-form
inverse `state_k = g_k(WB)`, where `WB` is a vector of fresh
`WB_<state_name>` symbols.  The inverse is auto-derived via
`sympy.solve`; topological ordering between slots (e.g. depth-first,
then momentum) is left to sympy.  Used by both
`zoomy_core.model.basemodel.Model.state_from_reconstruction`
and the corresponding SystemModel-level derived field, so the same
inverse logic is shared symbol-by-symbol whether invoked on the
model's primitive state or on the SystemModel's post-CoV state.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.misc.misc import ZArray


def reconstruction_symbols(state_syms):
    """Return the fresh `WB_<name>` symbols paired with `state_syms`.

    One `WB_<state_name>` Symbol per state slot, in slot order.  The
    inverse is expressed in terms of these — at lambdification time the
    runtime feeds them with the limiter's primitive face values.
    """
    return [sp.Symbol(f"WB_{s.name}", real=True) for s in state_syms]


def invert_reconstruction(forward, state_syms):
    """Return the symbolic inverse of `forward` as a ZArray of length
    `len(state_syms)`.

    Parameters
    ----------
    forward : ZArray | sequence
        Reconstruction-variables map — entry `k` is the symbolic
        expression for the WB primitive in slot `k`, in terms of
        `state_syms`.
    state_syms : sequence of sympy.Symbol
        State symbols, in slot order.

    Returns
    -------
    ZArray
        Inverse map — entry `k` is `state_syms[k]` expressed as a
        sympy expression in `WB_<state_name>` symbols.  For slots
        where the forward is the identity (`forward[k] == state[k]`)
        the inverse is also the identity (`WB_<state_name>`), no
        `solve` call.
    """
    n = len(state_syms)
    wb_syms = reconstruction_symbols(state_syms)

    # Identity short-circuit per slot — skip slots where the forward is
    # ``state[k]`` literally; for the remaining slots, build one
    # ``sympy.Eq`` and solve jointly for the affected state symbols.
    nontrivial_eqs = []
    nontrivial_states = []
    out = list(wb_syms)        # start with identity, override below
    for k in range(n):
        fwd_k = sp.sympify(forward[k])
        if fwd_k == state_syms[k]:
            continue
        nontrivial_eqs.append(sp.Eq(fwd_k, wb_syms[k]))
        nontrivial_states.append(state_syms[k])

    if not nontrivial_eqs:
        return ZArray(out)

    sols = sp.solve(nontrivial_eqs, nontrivial_states, dict=True)
    if not sols:
        raise RuntimeError(
            "invert_reconstruction: sympy.solve could not invert the "
            "reconstruction-variables map symbolically — override "
            "state_from_reconstruction() on the model with an explicit "
            "inverse.\n"
            f"  state = {state_syms}\n"
            f"  forward = {[sp.sympify(f) for f in forward]}\n"
            f"  WB symbols = {wb_syms}"
        )
    sol = sols[0]
    # ``sympy.solve`` leaves the identity slots' state symbols in place
    # (they weren't part of ``nontrivial_states``).  At face time those
    # values arrive via their WB symbol (forward identity ⇒ WB = state),
    # so substitute every identity-slot state-symbol with its WB-symbol
    # across the entire inverse — both the slots we solved for and the
    # slots that pass through unchanged.
    identity_subs = {state_syms[k]: wb_syms[k]
                     for k in range(n)
                     if sp.sympify(forward[k]) == state_syms[k]}
    for k, s in enumerate(state_syms):
        expr = sol[s] if s in sol else wb_syms[k]
        out[k] = sp.sympify(expr).xreplace(identity_subs)
    return ZArray(out)


__all__ = ["invert_reconstruction", "reconstruction_symbols"]
