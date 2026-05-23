"""Test: do the eigenvalues change if we keep w as state in SME?

Standard SME L=1: 3 fields (h, u_0, u_1), 3 equations.
Augmented SME L=1: w_0, w_1, w_2 also as state at degree N_w=2, plus
algebraic closures from continuity projections j=1, j=2 + KBC bottom.

Result: the augmented pencil is **degenerate** — det(M_x − λ M_t) ≡ 0.
The finite generalised eigenvalues do NOT match the standard SME's.

Interpretation: when ``w`` is a state polynomial fully constrained by
continuity, the augmented system's M_x and M_t share a common kernel.
The naive principal-symbol pencil cannot resolve the dynamical
eigenstructure without first eliminating w via the constraints (which
brings us back to the standard form).

Conclusion: keeping a constrained variable as state vs. substituting
its closure is NOT a free choice for spectral analysis — they give
different (and in this case degenerate) pencils.  This mirrors the
VAM situation where the paper treats ``w_2`` as a *free parameter* of
``A(U, w_2)`` rather than as a function of ``U`` via the closure.
"""
import os
import sys
import sympy as sp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sme"))
from sme_builder import build_sme_pde_system, t, x, xi, g

from zoomy_core.analysis import linearise, extract_quasilinear_pencil
from zoomy_core.derivation import (
    HydrostaticFlow,
    PolynomialAnsatz,
    GalerkinProjection,
    kbc_bottom_solve_w_N,
)

# ---- Standard SME L=1 ----
print("=" * 70)
print("STANDARD SME L=1 (w substituted out via depth-integrated continuity)")
print("=" * 70)
sys_std, h, u_coeffs = build_sme_pde_system(1)
H = sp.Symbol("H", positive=True)
U_0 = sp.Symbol("U_0", real=True)
U_1 = sp.Symbol("U_1", real=True)
base_std = {h: H, u_coeffs[0]: U_0, u_coeffs[1]: U_1}
sys_lin = linearise(sys_std, base_std)
M_t, M_xa, _ = extract_quasilinear_pencil(sys_lin)
print(f"Pencil shape: {M_t.shape}, rank(M_t) = {M_t.rank()}")
lam = sp.Symbol("lam")
char = sp.expand((M_xa[0] - lam * M_t).det(method="berkowitz"))
alpha = sp.Symbol("alpha")
print("char(lam = U_0 + alpha):")
print(" ", sp.factor(sp.expand(char.subs(lam, U_0 + alpha))))

# ---- Augmented SME L=1 with w as state at N_w=2 ----
print()
print("=" * 70)
print("AUGMENTED SME L=1 (w_0, w_1, w_2 as state; closures algebraic)")
print("=" * 70)
flow = HydrostaticFlow.with_defaults()
ansatz_aug = PolynomialAnsatz(t=flow.t, x=flow.x, xi=flow.xi,
                               M=1, N_w=2, N_p=-1)
proj_aug = GalerkinProjection(flow=flow, ansatz=ansatz_aug, w_mode="state")

eqs = [
    proj_aug.project_continuity(0),
    proj_aug.project_x_momentum(0),
    proj_aug.project_x_momentum(1),
    proj_aug.project_continuity(1),
    proj_aug.project_continuity(2),
]
# KBC bottom: w_2 = u_0·∂_x b + u_1·∂_x b - w_0 - w_1.
w_N_rhs = kbc_bottom_solve_w_N(ansatz_aug, flow)
eqs.append(ansatz_aug.w_coeffs[2] - w_N_rhs)

fields_aug = [flow.h] + list(ansatz_aug.u_coeffs) + list(ansatz_aug.w_coeffs)
print(f"#equations × #fields = {len(eqs)} × {len(fields_aug)}")

from zoomy_core.analysis import PDESystem
sys_aug = PDESystem(equations=eqs, fields=fields_aug,
                    time=flow.t, space=[flow.x])
# Flat bottom for clean comparison.
sys_aug = sys_aug.with_substitutions({flow.b: -H})

base_aug = {flow.h: H,
            ansatz_aug.u_coeffs[0]: U_0,
            ansatz_aug.u_coeffs[1]: U_1,
            ansatz_aug.w_coeffs[0]: sp.Symbol("W_0", real=True),
            ansatz_aug.w_coeffs[1]: sp.Symbol("W_1", real=True),
            ansatz_aug.w_coeffs[2]: sp.Symbol("W_2", real=True)}
sys_lin_aug = linearise(sys_aug, base_aug)
M_t_aug, M_xa_aug, _ = extract_quasilinear_pencil(sys_lin_aug)
print(f"Pencil shape: {M_t_aug.shape}, rank(M_t) = {M_t_aug.rank()}")
char_aug = sp.expand((M_xa_aug[0] - lam * M_t_aug).det(method="berkowitz"))
print("char(lam = U_0 + alpha) (full, with W_0,W_1,W_2 symbolic):")
print(" ", sp.factor(sp.expand(char_aug.subs(lam, U_0 + alpha))))
print()

# Extract finite eigenvalues numerically by sampling.
print("Numerical comparison at (H=1, U_0=0.3, U_1=0.5):")
sub_std = {H: 1.0, U_0: 0.3, U_1: 0.5, g: 9.81}
sub_aug = dict(sub_std)
sub_aug[sp.Symbol("W_0", real=True)] = 0.0
sub_aug[sp.Symbol("W_1", real=True)] = 0.0
sub_aug[sp.Symbol("W_2", real=True)] = 0.0

A_std = np.array(M_xa[0].subs(sub_std).tolist(), dtype=complex)
B_std = np.array(M_t.subs(sub_std).tolist(), dtype=complex)
from scipy.linalg import eig as scipy_eig
eigs_std = scipy_eig(A_std, B_std, left=False, right=False)
print(f"  Standard SME eigenvalues:  {sorted(eigs_std.real)}")

A_aug = np.array(M_xa_aug[0].subs(sub_aug).tolist(), dtype=complex)
B_aug = np.array(M_t_aug.subs(sub_aug).tolist(), dtype=complex)
eigs_aug = scipy_eig(A_aug, B_aug, left=False, right=False)
finite_aug = eigs_aug[np.isfinite(eigs_aug)]
print(f"  Augmented SME eigenvalues: {sorted(eigs_aug.real)}  (full)")
print(f"  Augmented SME finite:      {sorted(finite_aug.real)}")
print(f"  Expected (paper):  U_0, U_0 ± √(gh + U_1²) = "
      f"{0.3}, {0.3 - np.sqrt(9.81 + 0.25):.4f}, {0.3 + np.sqrt(9.81 + 0.25):.4f}")
