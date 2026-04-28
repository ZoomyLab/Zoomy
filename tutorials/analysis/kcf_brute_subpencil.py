"""Brute-force regular-subpencil extraction (a poor-person's KCF).

For a singular pencil (M_x, M_t), find a subset of rows + columns
whose removal gives a regular pencil with the maximum number of
finite eigenvalues.

For small systems (n ≤ ~10) this is tractable by enumeration.
"""
from itertools import combinations
import sympy as sp


def is_regular(M_x, M_t):
    """det(M_x - λ M_t) is not identically zero?"""
    if M_x.shape != M_t.shape or M_x.rows != M_x.cols:
        return False
    lam = sp.Symbol("__kcf_lam__")
    char = sp.expand((M_x - lam * M_t).det(method="berkowitz"))
    return char != 0


def regular_finite_eigenvalues(M_x, M_t):
    """Return list of symbolic eigenvalues of det(M_x - λ M_t) = 0."""
    lam = sp.Symbol("__kcf_lam__")
    char = sp.expand((M_x - lam * M_t).det(method="berkowitz"))
    return sp.solve(sp.Eq(char, 0), lam)


def extract_regular_subpencil_brute(M_x, M_t, fields,
                                    *, max_drop_rows=None,
                                    max_drop_cols=None,
                                    verbose=False):
    """Try all (rows-to-drop, cols-to-drop) subsets; return the first
    that gives a SQUARE regular pencil with the most finite eigenvalues.

    Returns ``(M_x_sub, M_t_sub, dropped_rows, dropped_cols, fields_kept, eigenvalues)``.
    """
    n = M_x.rows
    m = M_x.cols
    if max_drop_rows is None:
        max_drop_rows = n - 1
    if max_drop_cols is None:
        max_drop_cols = m - 1

    best = (-1, None)         # (n_eigs, payload)

    # Try smallest drops first (prefer larger remaining pencil).
    for n_drop in range(0, max_drop_rows + 1):
        for m_drop in range(0, max_drop_cols + 1):
            if n - n_drop != m - m_drop:                # require square
                continue
            if n - n_drop == 0:
                continue
            for rows_drop in combinations(range(n), n_drop):
                rows_keep = [i for i in range(n) if i not in rows_drop]
                for cols_drop in combinations(range(m), m_drop):
                    cols_keep = [j for j in range(m) if j not in cols_drop]
                    M_x_sub = M_x.extract(rows_keep, cols_keep)
                    M_t_sub = M_t.extract(rows_keep, cols_keep)
                    if not is_regular(M_x_sub, M_t_sub):
                        continue
                    eigs = regular_finite_eigenvalues(M_x_sub, M_t_sub)
                    if verbose:
                        print(f"  drop rows {rows_drop}, cols {cols_drop} "
                              f"→ {len(eigs)} eigs: {eigs}")
                    if len(eigs) > best[0]:
                        best = (len(eigs),
                                (M_x_sub, M_t_sub, list(rows_drop),
                                 list(cols_drop),
                                 [fields[j] for j in cols_keep], eigs))
        if best[0] >= 1 and n_drop > 0:
            # Found at least one regular subpencil; could try larger drops
            # to find more eigenvalues, but usually smallest drop wins.
            break
    return best[1]


# ---- test on augmented SME L=1 ----
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

    # === Standard SME L=1 reference ===
    print("=== Standard SME L=1 (reference) ===")
    sys_std, h, u_coeffs = build_sme_pde_system(1)
    H = sp.Symbol("H", positive=True)
    g = sp.Symbol("g", positive=True)
    U_0 = sp.Symbol("U_0", real=True)
    U_1 = sp.Symbol("U_1", real=True)
    base = {h: H, u_coeffs[0]: U_0, u_coeffs[1]: U_1}
    sys_lin = linearise(sys_std, base)
    M_t, M_xa, _ = extract_quasilinear_pencil(sys_lin)
    eigs_std = regular_finite_eigenvalues(M_xa[0], M_t)
    for e in eigs_std:
        print(" ", sp.simplify(e))

    # === Augmented SME L=1 ===
    print()
    print("=== Augmented SME L=1 (w as state, brute KCF) ===")
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
    fields = [flow.h] + list(ansatz.u_coeffs) + list(ansatz.w_coeffs)
    sys_aug = PDESystem(equations=eqs, fields=fields,
                        time=flow.t, space=[flow.x])
    sys_aug = sys_aug.with_substitutions({flow.b: -H})

    base_aug = {flow.h: H,
                ansatz.u_coeffs[0]: U_0, ansatz.u_coeffs[1]: U_1,
                ansatz.w_coeffs[0]: sp.Symbol("W_0", real=True),
                ansatz.w_coeffs[1]: sp.Symbol("W_1", real=True),
                ansatz.w_coeffs[2]: sp.Symbol("W_2", real=True)}
    sys_lin_aug = linearise(sys_aug, base_aug)
    M_t_aug, M_xa_aug, _ = extract_quasilinear_pencil(sys_lin_aug)
    print(f"  Original pencil shape {M_t_aug.shape}, M_t rank {M_t_aug.rank()}")

    payload = extract_regular_subpencil_brute(
        M_xa_aug[0], M_t_aug, sys_lin_aug.fields,
        max_drop_rows=4, max_drop_cols=4, verbose=False,
    )
    if payload is None:
        print("  No regular subpencil found.")
    else:
        Mx_sub, Mt_sub, dropped_rows, dropped_cols, fields_kept, eigs_aug = payload
        print(f"  Found regular subpencil of shape {Mx_sub.shape}")
        print(f"  Dropped rows: {dropped_rows}  (out of {M_t_aug.rows})")
        print(f"  Dropped cols (fields): "
              f"{[str(sys_lin_aug.fields[j].func) for j in dropped_cols]}")
        print(f"  Eigenvalues:")
        for e in eigs_aug:
            print("   ", sp.simplify(e))

        # Compare to standard SME.
        match = set(sp.simplify(e) for e in eigs_aug) == set(
            sp.simplify(e) for e in eigs_std
        )
        print(f"  Match standard SME: {match}")
