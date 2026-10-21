"""Generic singular-pencil reduction for the principal-symbol problem.

When a linearised PDE system has algebraic constraint rows (M_t row =
zero), the principal-symbol pencil `(M_x, M_t)` is *singular* —
`det(M_x − λ M_t) ≡ 0` and standard generalised-eigenvalue solvers
return garbage.

This module reduces a singular pencil to a regular one by repeatedly:

  1. Picking an algebraic row (`M_t` row identically zero).
  2. Solving its M_x-row for one field (the field with the simplest
     non-zero coefficient in that row).
  3. Substituting that field's expression into every other row of
     `M_x` and `M_t`, then dropping the row + the eliminated column.

After reduction (when no all-zero `M_t` row remains), the pencil is
regular and standard generalised-eigenvalue routines work.

The algorithm is *generic* — it works for any system with algebraic
constraints, regardless of the model.  It chooses elimination order by
"simplest coefficient first", which is deterministic but not unique;
the **finite eigenvalues are basis-invariant** under valid elimination
choices, so the result is independent of the order.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import sympy as sp


def reduce_singular_pencil(M_x: sp.Matrix, M_t: sp.Matrix,
                           fields: List,
                           M_0: Optional[sp.Matrix] = None,
                           *, verbose: bool = False
                           ) -> Tuple[sp.Matrix, sp.Matrix, List]:
    """Eliminate algebraic-constraint rows + the corresponding fields.

    Three classes of "algebraic" rows are handled:

      a) `M_t row = 0` and `M_x row != 0` — principal-symbol
         algebraic constraint at high k.  Use the M_x row to solve
         for one field and substitute into all remaining rows
         (M_x, M_t and M_0 if provided).

      b) `M_t row = 0` and `M_x row = 0` and `M_0 row != 0` —
         zeroth-order algebraic constraint (k-independent).  Use the
         M_0 row to solve.  REQUIRES `M_0` argument.

      c) All three rows zero — redundant; drop the row.

    Returns `(M_x_reduced, M_t_reduced, fields_reduced)` with no
    all-zero `M_t` rows.  If `M_0` was provided it is also
    reduced; access via the returned tuple's caller side-effect (the
    matrix is mutated in place — pass a copy if you need the original).
    """
    M_x = sp.Matrix(M_x)
    M_t = sp.Matrix(M_t)
    if M_0 is not None:
        M_0 = sp.Matrix(M_0)
    fields = list(fields)

    def _is_zero_row(M, i):
        return all(sp.simplify(M[i, j]) == 0 for j in range(M.cols))

    while True:
        zero_t_rows = [i for i in range(M_t.rows) if _is_zero_row(M_t, i)]
        if not zero_t_rows:
            break
        i = zero_t_rows[0]
        # First try M_x for principal-symbol constraint (case a).
        # Then fall back to M_0 (case b).
        coef_source_name = None
        coef_source_row = None
        for source_name, source_mat in (("M_x", M_x),
                                        ("M_0", M_0 if M_0 is not None else None)):
            if source_mat is None:
                continue
            row = [sp.simplify(source_mat[i, j]) for j in range(source_mat.cols)]
            if any(r != 0 for r in row):
                coef_source_name = source_name
                coef_source_row = row
                break
        if coef_source_row is None:
            # Case (c): redundant row.  Drop it.
            if verbose:
                print(f"  reduce_pencil: dropping fully-zero row {i}")
            M_x.row_del(i)
            M_t.row_del(i)
            if M_0 is not None:
                M_0.row_del(i)
            continue

        # Pick column with simplest non-zero coefficient.
        candidates = [(j, c) for j, c in enumerate(coef_source_row) if c != 0]
        candidates.sort(key=lambda x: len(str(x[1])))
        j_elim, coef_jj = candidates[0]
        if verbose:
            print(f"  reduce_pencil: eliminating field "
                  f"{fields[j_elim]} via row {i} (source={coef_source_name}, "
                  f"coef={coef_jj})")

        # Solve row i (in coef_source_row) for q̂_{j_elim}:
        #   q̂_{j_elim} = -Σ_{k≠j_elim} (coef_source_row[k]/coef_jj) · q̂_k
        substitute_row = [-sp.cancel(coef_source_row[k] / coef_jj)
                          for k in range(len(coef_source_row))
                          if k != j_elim]

        # Build a column-elimination map: new column kc corresponds to
        # old column k (for k != j_elim).
        def _eliminate_col(M):
            new = sp.zeros(M.rows - 1, M.cols - 1)
            for r in range(M.rows):
                if r == i:
                    continue
                r_new = r if r < i else r - 1
                kc = 0
                for k in range(M.cols):
                    if k == j_elim:
                        continue
                    new[r_new, kc] = sp.expand(
                        M[r, k] + M[r, j_elim] * substitute_row[kc]
                    )
                    kc += 1
            return new

        M_x = _eliminate_col(M_x)
        M_t = _eliminate_col(M_t)
        if M_0 is not None:
            M_0 = _eliminate_col(M_0)
        fields = fields[:j_elim] + fields[j_elim + 1:]
        if verbose:
            print(f"    pencil now {M_x.shape}")

    return M_x, M_t, fields
