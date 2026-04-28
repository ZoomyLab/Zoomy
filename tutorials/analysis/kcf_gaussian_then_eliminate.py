"""Singular-pencil reduction via Gaussian elimination on M_t.

Step 1: Gaussian-eliminate on M_t (with the same row ops applied to
        M_x and M_0) to bring M_t to row echelon form.  Rows with
        non-zero M_t are "evolution rows"; rows where M_t becomes
        zero are "algebraic rows" (constraints).

Step 2: Use the algebraic rows (now obviously zero in M_t) to
        eliminate one field per row via M_x or M_0 constraint.

This is the principled "gauss-style elimination" that exposes all
hidden algebraic constraints — including those that arise only after
combining evolution rows (like cont j>=1 in SME, which has ∂_t h via
the ω polynomial in the IBP integrand and becomes algebraic after
subtracting cont j=0).
"""
import sympy as sp


def gaussian_eliminate_on_M_t(M_x, M_t, M_0=None, *, verbose=False):
    """Apply Gaussian elimination to M_t (with same ops on M_x, M_0).

    After this, the first ``rank(M_t)`` rows have non-zero M_t (one
    per pivot column); the remaining rows have M_t = 0 (algebraic).
    """
    M_x = sp.Matrix(M_x)
    M_t = sp.Matrix(M_t)
    if M_0 is not None:
        M_0 = sp.Matrix(M_0)
    n_rows, n_cols = M_t.shape
    pivot_row = 0
    for col in range(n_cols):
        if pivot_row >= n_rows:
            break
        candidates = [r for r in range(pivot_row, n_rows)
                      if sp.simplify(M_t[r, col]) != 0]
        if not candidates:
            continue
        r_pivot = candidates[0]
        if r_pivot != pivot_row:
            M_t.row_swap(pivot_row, r_pivot)
            M_x.row_swap(pivot_row, r_pivot)
            if M_0 is not None:
                M_0.row_swap(pivot_row, r_pivot)
        for r in range(n_rows):
            if r == pivot_row:
                continue
            if sp.simplify(M_t[r, col]) == 0:
                continue
            factor = sp.cancel(M_t[r, col] / M_t[pivot_row, col])
            for c in range(n_cols):
                M_t[r, c] = sp.expand(M_t[r, c] - factor * M_t[pivot_row, c])
                M_x[r, c] = sp.expand(M_x[r, c] - factor * M_x[pivot_row, c])
                if M_0 is not None:
                    M_0[r, c] = sp.expand(M_0[r, c] - factor * M_0[pivot_row, c])
        pivot_row += 1
    if verbose:
        print(f"  M_t after Gaussian elim:")
        sp.pprint(M_t)
        print(f"  M_t rank = {pivot_row} (evolution rows: 0..{pivot_row - 1}; "
              f"algebraic rows: {pivot_row}..{n_rows - 1})")
    return M_x, M_t, M_0


def reduce_after_gaussian(M_x, M_t, fields, M_0=None, *,
                          prefer_eliminate=None, verbose=False):
    """Combined: Gaussian-eliminate M_t, then drop algebraic rows by
    solving for one field each.

    Args:
        prefer_eliminate: list of fields (in priority order) that should
            be eliminated first when there's a choice.  Useful for
            specifying "treat these as constrained, the others as
            primary state".
    """
    M_x, M_t, M_0 = gaussian_eliminate_on_M_t(M_x, M_t, M_0, verbose=verbose)
    fields = list(fields)
    if prefer_eliminate is None:
        prefer_eliminate = []

    def _field_priority(j):
        """Lower number = higher priority for elimination."""
        for k, pf in enumerate(prefer_eliminate):
            if fields[j] == pf:
                return k
        return len(prefer_eliminate) + j   # tie-break by column index

    while True:
        zero_t_rows = [i for i in range(M_t.rows)
                       if all(sp.simplify(M_t[i, j]) == 0
                              for j in range(M_t.cols))]
        if not zero_t_rows:
            break
        i = zero_t_rows[0]
        # Try M_x first, then M_0.
        for source_name, source in (("M_x", M_x), ("M_0", M_0)):
            if source is None:
                continue
            row = [sp.simplify(source[i, j]) for j in range(source.cols)]
            if any(r != 0 for r in row):
                break
        else:
            if verbose:
                print(f"  drop redundant row {i}")
            M_x.row_del(i); M_t.row_del(i)
            if M_0 is not None: M_0.row_del(i)
            continue
        # Pick pivot: prefer fields in prefer_eliminate; tie-break by
        # coefficient simplicity.
        pivots = [(j, row[j]) for j in range(len(row)) if row[j] != 0]
        pivots.sort(key=lambda x: (_field_priority(x[0]), len(str(x[1]))))
        j_elim, coef = pivots[0]
        if verbose:
            print(f"  algebraic row {i}: solve for {fields[j_elim]} "
                  f"(coef={coef}, source={source_name})")
        substitute = [-sp.cancel(row[k] / coef)
                      for k in range(len(row)) if k != j_elim]

        def _col_eliminate(M):
            new = sp.zeros(M.rows - 1, M.cols - 1)
            for r in range(M.rows):
                if r == i: continue
                r_new = r if r < i else r - 1
                kc = 0
                for k in range(M.cols):
                    if k == j_elim: continue
                    new[r_new, kc] = sp.expand(
                        M[r, k] + M[r, j_elim] * substitute[kc])
                    kc += 1
            return new

        M_x = _col_eliminate(M_x); M_t = _col_eliminate(M_t)
        if M_0 is not None: M_0 = _col_eliminate(M_0)
        fields = fields[:j_elim] + fields[j_elim + 1:]
    return M_x, M_t, fields


def regular_eigenvalues(M_x, M_t):
    lam = sp.Symbol("__kcf_lam__")
    char = sp.expand((M_x - lam * M_t).det(method="berkowitz"))
    return sp.solve(sp.Eq(char, 0), lam)


# ---- test ----
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                     "tutorials", "sme"))
    from sme_builder import build_sme_pde_system
    from zoomy_core.derivation import (
        HydrostaticFlow, PolynomialAnsatz, GalerkinProjection,
        kbc_bottom_solve_w_N,
    )
    from zoomy_core.analysis import (
        PDESystem, linearise, extract_quasilinear_pencil,
    )

    sys_std, h, u_coeffs = build_sme_pde_system(1)
    H = sp.Symbol("H", positive=True)
    g = sp.Symbol("g", positive=True)
    U_0 = sp.Symbol("U_0", real=True); U_1 = sp.Symbol("U_1", real=True)
    base = {h: H, u_coeffs[0]: U_0, u_coeffs[1]: U_1}
    sys_lin = linearise(sys_std, base)
    M_t, M_xa, _ = extract_quasilinear_pencil(sys_lin)
    eigs_std = sorted([sp.simplify(e) for e in regular_eigenvalues(M_xa[0], M_t)],
                       key=str)
    print("Reference (standard SME L=1):")
    for e in eigs_std: print(" ", e)

    # Augmented.
    print()
    print("=== AUGMENTED SME L=1 → Gaussian-then-eliminate ===")
    flow = HydrostaticFlow.with_defaults()
    ansatz = PolynomialAnsatz(t=flow.t, x=flow.x, xi=flow.xi,
                               M=1, N_w=2, N_p=-1)
    proj = GalerkinProjection(flow=flow, ansatz=ansatz, w_mode="state")
    eqs = [
        proj.project_continuity(0),
        proj.project_x_momentum(0),
        proj.project_x_momentum(1),
        proj.project_continuity(1),
        proj.project_continuity(2),
        ansatz.w_coeffs[2] - kbc_bottom_solve_w_N(ansatz, flow),
    ]
    fields_orig = [flow.h] + list(ansatz.u_coeffs) + list(ansatz.w_coeffs)
    sys_aug = PDESystem(equations=eqs, fields=fields_orig,
                        time=flow.t, space=[flow.x])
    sys_aug = sys_aug.with_substitutions({flow.b: -H})
    base_aug = {flow.h: H,
                ansatz.u_coeffs[0]: U_0, ansatz.u_coeffs[1]: U_1,
                ansatz.w_coeffs[0]: sp.Symbol("W_0", real=True),
                ansatz.w_coeffs[1]: sp.Symbol("W_1", real=True),
                ansatz.w_coeffs[2]: sp.Symbol("W_2", real=True)}
    sys_lin_aug = linearise(sys_aug, base_aug)
    M_t_aug, M_xa_aug, M_0_aug = extract_quasilinear_pencil(sys_lin_aug)
    print(f"  Original {M_t_aug.shape}, M_t rank {M_t_aug.rank()}")

    # Tell the algorithm: prefer to eliminate w_i (constrained), keep u_i.
    delta_w_fields = [f for f in sys_lin_aug.fields
                      if str(f.func).startswith("\\delta w")]
    M_x_red, M_t_red, fields_red = reduce_after_gaussian(
        M_xa_aug[0], M_t_aug, sys_lin_aug.fields, M_0=M_0_aug,
        prefer_eliminate=delta_w_fields, verbose=True,
    )
    print(f"  Reduced {M_x_red.shape}, rank(M_t) = {M_t_red.rank()}")
    print(f"  Fields: {[str(f.func) for f in fields_red]}")
    eigs = sorted([sp.simplify(e) for e in regular_eigenvalues(M_x_red, M_t_red)],
                   key=str)
    print(f"  Eigenvalues:")
    for e in eigs: print("   ", e)
    print(f"  Match standard SME L=1? {eigs == eigs_std}")
