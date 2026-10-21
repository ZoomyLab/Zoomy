"""Module `zoomy_core.transformation.vectorize`.

Shared constant-entry rank-normalization seam (REQ-84).

Vector backends batch every kernel row over cells/faces, so a row whose
symbolic entry carries no dependence on a vector symbol (state / aux /
normal / position group) lowers as a *scalar* while its siblings lower as
*arrays* — the generated `jnp.array` / `np.array` then cannot stack the
mixed-rank rows.  Wrapping each such constant entry with `zeros_like` /
`c * ones_like` of an *anchor* vector symbol forces the constant to adopt
the batch rank of its neighbours *without* touching the active rows' CSE
(the whole point — broadcasting all rows regressed SME lake-at-rest WB).

The numpy printer emits `ones_like` / `zeros_like` directly; UFL lowers
them as `0*x+1` / `0*x`; jax binds them to `jnp.ones_like` /
`jnp.zeros_like` in its lambdify module dict.
"""

import sympy as sp


def uniform_rank(arr, vector_symbols, anchor):
    """Wrap every constant (vector-symbol-free) entry of `arr` with
    `zeros_like(anchor)` / `c * ones_like(anchor)` so all rows share the
    batch rank of the vector symbols.

    Parameters
    ----------
    arr : sympy.Array
        The flat/rank-N symbolic array to normalize.
    vector_symbols : iterable of sympy.Symbol
        Symbols that broadcast over the batch dimension (state / aux /
        normal / position groups).
    anchor : sympy.Symbol or None
        A representative vector symbol (conventionally the first state
        symbol) whose batch shape the constants borrow.

    Returns
    -------
    sympy.Array
        `arr` with constant entries wrapped; active entries untouched.
        Returned unchanged if there are no vector symbols or no anchor.
    """
    vector_symbols = tuple(vector_symbols)
    if not vector_symbols or anchor is None:
        return arr

    ones_like = sp.Function("ones_like")
    zeros_like = sp.Function("zeros_like")
    flat = []
    for item in list(arr._array):
        if isinstance(item, sp.Basic) and not item.has(*vector_symbols):
            if item.is_zero:
                flat.append(zeros_like(anchor))
            else:
                flat.append(item * ones_like(anchor))
        else:
            flat.append(item)
    return sp.Array(flat, arr.shape)
