"""SME L=2: experiment with partial w-elimination (NUMERICAL).

Standard SME L=2: state = (h, u_0, u_1, u_2); w is depth-integrated
from continuity (not in state).  Eigenvalues are roots of a degree-4
polynomial — symbolically intractable but numerically simple.

Augmented SME L=2 (w as state at degree N_w ≥ 3): keep w_0..w_{N_w}
as state plus algebraic constraints (cont j=1..N_w + KBC bottom).
Various partial-elimination strategies via the ``keep_as_input``
mechanism in vam_builder, transplanted to SME by hand here.

For each variant we:
  - Build the linearised pencil at a sample base state.
  - Substitute numeric values (H=1, g=9.81, varying U_i).
  - Compute generalised eigenvalues via scipy.linalg.eig.
  - Compare to standard SME numerically.

Question: is there a partial-elimination strategy that gives the
SAME eigenvalues as standard SME — confirming the math is invariant
— or do different choices genuinely give different spectra?
"""
import os, sys, sympy as sp, numpy as np
from scipy.linalg import eig as scipy_eig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sme"))
from sme_builder import build_sme_pde_system

from zoomy_core.derivation import (
    HydrostaticFlow, PolynomialAnsatz, GalerkinProjection,
    kbc_bottom_solve_w_N,
)
from zoomy_core.analysis import (
    PDESystem, linearise, extract_quasilinear_pencil,
)


H_sym = sp.Symbol("H", positive=True)
g_sym = sp.Symbol("g", positive=True)


def numeric_eigs(M_x, M_t, sub):
    A = np.array(M_x.subs(sub).tolist(), dtype=complex)
    B = np.array(M_t.subs(sub).tolist(), dtype=complex)
    eigs = scipy_eig(A, B, left=False, right=False)
    finite = eigs[np.isfinite(eigs)]
    return sorted(np.real(finite))


def standard_sme_l2():
    sys_std, h, u_coeffs = build_sme_pde_system(2)
    U_0, U_1, U_2 = sp.symbols("U_0 U_1 U_2", real=True)
    base = {h: H_sym, u_coeffs[0]: U_0, u_coeffs[1]: U_1, u_coeffs[2]: U_2}
    sys_lin = linearise(sys_std, base)
    M_t, M_xa, _ = extract_quasilinear_pencil(sys_lin)
    return M_xa[0], M_t, (U_0, U_1, U_2)


def augmented_sme_l2(*, N_w=3, eliminated_w=(),
                     dropped_cont_j=(),
                     drop_kbc_bottom=False):
    flow = HydrostaticFlow.with_defaults()
    ansatz = PolynomialAnsatz(t=flow.t, x=flow.x, xi=flow.xi,
                               M=2, N_w=N_w, N_p=-1)
    proj = GalerkinProjection(flow=flow, ansatz=ansatz, w_mode="state")
    eqs = [proj.project_continuity(0)]
    for j in range(3):
        eqs.append(proj.project_x_momentum(j))
    for j in range(1, N_w + 1):
        if j in dropped_cont_j:
            continue
        eqs.append(proj.project_continuity(j))
    if not drop_kbc_bottom:
        eqs.append(ansatz.w_coeffs[N_w] - kbc_bottom_solve_w_N(ansatz, flow))

    fields_orig = [flow.h] + list(ansatz.u_coeffs) + list(ansatz.w_coeffs)
    input_subs = {ansatz.w_coeffs[i]: sp.Symbol(f"w_{i}_input", real=True)
                  for i in eliminated_w}
    if input_subs:
        eqs = [sp.expand(eq.xreplace(input_subs).doit()) for eq in eqs]
    fields = [f for f in fields_orig if f not in input_subs]
    sys_aug = PDESystem(equations=eqs, fields=fields,
                        time=flow.t, space=[flow.x]).with_substitutions({flow.b: -H_sym})

    U_0, U_1, U_2 = sp.symbols("U_0 U_1 U_2", real=True)
    base = {fields[0]: H_sym}
    base[ansatz.u_coeffs[0]] = U_0
    base[ansatz.u_coeffs[1]] = U_1
    base[ansatz.u_coeffs[2]] = U_2
    for w in ansatz.w_coeffs:
        if w in fields:
            base[w] = sp.Symbol(f"W_{ansatz.w_coeffs.index(w)}", real=True)
    sys_lin = linearise(sys_aug, base)
    M_t, M_xa, _ = extract_quasilinear_pencil(sys_lin)
    return M_xa[0], M_t, (U_0, U_1, U_2), sys_lin.fields


def main():
    # Standard reference.
    print("=== STANDARD SME L=2 ===")
    M_x_std, M_t_std, (U_0, U_1, U_2) = standard_sme_l2()
    print(f"  pencil shape {M_t_std.shape}")

    # Test points.
    test_states = [
        ("rest",          {U_0: 0.0, U_1: 0.0, U_2: 0.0}),
        ("U_1=0.5",       {U_0: 0.0, U_1: 0.5, U_2: 0.0}),
        ("U_1=U_2=0.5",   {U_0: 0.0, U_1: 0.5, U_2: 0.5}),
        ("U_1=2,U_2=2",   {U_0: 0.0, U_1: 2.0, U_2: 2.0}),    # near loss-of-hyp
    ]

    sub_const = {H_sym: 1.0, g_sym: 9.81}

    for label, sub_state in test_states:
        full_sub = {**sub_const, **sub_state}
        eigs_std = numeric_eigs(M_x_std, M_t_std, full_sub)
        print(f"  [{label}] standard: {[f'{e:+.4f}' for e in eigs_std]}")
    print()

    experiments = [
        ("B-aug+drop-everything",
         dict(eliminated_w=[0, 1, 2, 3], dropped_cont_j=[1, 2, 3],
              drop_kbc_bottom=True)),
        ("C-only-w_3-input",
         dict(eliminated_w=[3], dropped_cont_j=[],
              drop_kbc_bottom=True)),
        ("D-w_2,w_3-input,keep-cont-j=1",
         dict(eliminated_w=[2, 3], dropped_cont_j=[2, 3],
              drop_kbc_bottom=True)),
        ("E-keep-only-w_0",
         dict(eliminated_w=[1, 2, 3], dropped_cont_j=[1, 2, 3],
              drop_kbc_bottom=True)),
    ]

    for label, kwargs in experiments:
        print(f"=== {label}: {kwargs} ===")
        try:
            M_x, M_t, (U_0, U_1, U_2), fields = augmented_sme_l2(**kwargs)
            field_names = [str(f.func) if hasattr(f, "func") else str(f)
                           for f in fields]
            print(f"  fields: {field_names}")
            print(f"  pencil: {M_t.shape}, rank(M_t) = {M_t.rank()}")
            for state_label, sub_state in test_states:
                full = {**sub_const, **sub_state}
                # Add zero values for any W_i fields in the augmented system.
                for f in fields:
                    if hasattr(f, "func") and str(f.func).startswith("w_"):
                        sym = sp.Symbol(f"W_{str(f.func).split('_')[1]}", real=True)
                        full[sym] = 0.0
                # Add zero values for any *_input symbols.
                for sym in (M_x.free_symbols | M_t.free_symbols):
                    if "input" in str(sym):
                        full[sym] = 0.0
                try:
                    eigs = numeric_eigs(M_x, M_t, full)
                    print(f"  [{state_label}] eigs: {[f'{e:+.4f}' for e in eigs]}")
                except Exception as exc:
                    print(f"  [{state_label}] failed: {exc}")
        except Exception as exc:
            print(f"  build failed: {exc}")
        print()


if __name__ == "__main__":
    main()
