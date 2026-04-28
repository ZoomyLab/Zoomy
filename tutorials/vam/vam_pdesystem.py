"""Builder: VAM(M, N_w, N_p) as a PDESystem suitable for the DAE solver.

This is the runtime-facing version of `vam_l1_physical_z.py`: instead of
verifying references, it returns a complete `PDESystem` containing all
projected equations + KBCs + surface BC.  The DAE solver then partitions
these into evolution + algebraic via the bridge in
`tests/scripts/dae_toy/test_dae_partition_bridge.py`.

Reference: Escalante et al. 2024 (DOI 10.1007/s10915-024-02579-1
or the arXiv version).  Uses physical-z and the conservative form
of NS (caller-applied product rule + continuity), per the user's
preferred derivation pattern.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.symbolic import polynomial_integrate as poly_int
from zoomy_core.analysis import PDESystem


def build_vam_pdesystem(M: int = 1, N_w: int | None = None, N_p: int | None = None,
                        flat_bottom: bool = True):
    """Build the un-split VAM PDESystem at order (M, N_w, N_p).

    Defaults: VAM(1, 2, 2).  Returns a `PDESystem` with all projected
    equations, KBCs, and surface BC.  Pressure and velocity moments are
    state variables; continuity projections + KBCs + surface BC are
    algebraic constraints.
    """
    if N_w is None:
        N_w = M + 1
    if N_p is None:
        N_p = M + 1

    # ---- symbols ----
    t = sp.Symbol("t", real=True)
    x = sp.Symbol("x", real=True)
    z = sp.Symbol("z", real=True)
    xi = sp.Symbol("xi", real=True)
    g = sp.Symbol("g", positive=True)

    h = sp.Function("h", real=True)(t, x)
    if flat_bottom:
        b = sp.S.Zero
    else:
        b = sp.Function("b", real=True)(x)
    eta = h + b

    u_funcs = [sp.Function(f"u_{i}", real=True)(t, x) for i in range(M + 1)]
    w_funcs = [sp.Function(f"w_{i}", real=True)(t, x) for i in range(N_w + 1)]
    p_funcs = [sp.Function(f"p_{i}", real=True)(t, x) for i in range(N_p + 1)]

    # ---- shifted Legendre basis evaluated at ζ(z) = (z-b)/h ----
    zeta = (z - b) / h
    PHI = [
        sp.S.One,                                      # φ_0
        1 - 2 * zeta,                                  # φ_1
        1 - 6 * zeta + 6 * zeta ** 2,                  # φ_2
        1 - 12 * zeta + 30 * zeta ** 2 - 20 * zeta ** 3,  # φ_3
        1 - 20 * zeta + 90 * zeta ** 2 - 140 * zeta ** 3 + 70 * zeta ** 4,  # φ_4
    ]
    needed = max(M, N_w, N_p) + 1
    if needed > len(PHI):
        raise NotImplementedError(f"need φ up to degree {needed-1}; "
                                  f"add more terms to PHI")

    # ---- ansatz expressions in (t, x, z) ----
    u_ansatz = sum(u_funcs[i] * PHI[i] for i in range(M + 1))
    w_ansatz = sum(w_funcs[i] * PHI[i] for i in range(N_w + 1))
    p_ansatz = sum(p_funcs[i] * PHI[i] for i in range(N_p + 1))

    # ---- conservative-form NS ----
    cont_lhs = (sp.Derivative(u_ansatz, x).doit()
                + sp.Derivative(w_ansatz, z).doit())
    xmom_lhs = (sp.Derivative(u_ansatz, t).doit()
                + sp.Derivative(u_ansatz ** 2, x).doit()
                + sp.Derivative(u_ansatz * w_ansatz, z).doit()
                + g * sp.Derivative(eta, x).doit()
                + sp.Derivative(p_ansatz, x).doit())
    zmom_lhs = (sp.Derivative(w_ansatz, t).doit()
                + sp.Derivative(u_ansatz * w_ansatz, x).doit()
                + sp.Derivative(w_ansatz ** 2, z).doit()
                + sp.Derivative(p_ansatz, z).doit())

    # ---- Galerkin projection ----
    def project_z(integrand, basis):
        integrand_xi = sp.expand(
            (basis * integrand).xreplace({z: xi * h + b}).doit()
        )
        return poly_int(integrand_xi * h, xi, 0, 1)

    # Continuity projected against φ_0..φ_{N_w}.  Highest projection is
    # often identically zero (orthogonality); we keep it and rely on
    # the partition bridge to drop trivial rows.
    cont_js = [sp.expand(project_z(cont_lhs, PHI[j])) for j in range(N_w + 1)]
    # x-mom against φ_0..φ_M.
    xmom_js = [sp.expand(project_z(xmom_lhs, PHI[j])) for j in range(M + 1)]
    # z-mom against φ_0..φ_{N_w}.
    zmom_js = [sp.expand(project_z(zmom_lhs, PHI[j])) for j in range(N_w + 1)]

    # ---- KBCs ----
    kbc_bot = sp.expand(
        w_ansatz.xreplace({z: b}) - u_ansatz.xreplace({z: b}) * sp.Derivative(b, x).doit()
    )
    kbc_top = sp.expand(
        w_ansatz.xreplace({z: eta})
        - sp.Derivative(eta, t).doit()
        - u_ansatz.xreplace({z: eta}) * sp.Derivative(eta, x).doit()
    )

    # ---- Surface BC: p(η) = 0 ----
    surface_bc = sp.expand(p_ansatz.xreplace({z: eta}))

    # ---- Assemble equations + fields ----
    # Formulation 1 (matches Escalante's eq (4) + (5) row count and the
    # user's reading: pure-KBC algebraics for VAM(1, 2, 2), one cont_j
    # added per extra moment level beyond M = 1):
    #   - explicit mass equation as evolution (∂_t h + ∂_x(h u_0) = 0)
    #   - x-mom × (M+1) and z-mom × (N_w+1) as evolution
    #   - kbc_top *with ∂_t h substituted out via the mass equation* —
    #     becomes purely algebraic.  Equivalent to Escalante's eq (5)
    #     constraint I_2 after the ∂_t h substitution.
    #   - kbc_bot, surface_bc as algebraic
    #   - cont_j projections for j = 1, …, M − 1  (zero for M = 1)
    mass_eq = sp.Derivative(h, t) + sp.Derivative(h * u_funcs[0], x).doit()

    # kbc_top originally:   w(η) − ∂_t η − u(η)·∂_x η = 0
    #               i.e.,   w(η) − u(η)·∂_x η = ∂_t h + ∂_t b = ∂_t h     (b fixed)
    # Substitute mass eq (∂_t h = −∂_x(h u_0)) → algebraic form:
    #               w(η) − u(η)·∂_x η + ∂_x(h u_0) = 0
    kbc_top_alg = sp.expand(
        w_ansatz.xreplace({z: eta})
        - u_ansatz.xreplace({z: eta}) * sp.Derivative(eta, x).doit()
        + sp.Derivative(h * u_funcs[0], x).doit()
    )

    equations = []
    eq_names = []
    # Evolutions: 2M + 4 of them for VAM(M, M+1, M+1).
    equations.append(mass_eq); eq_names.append("mass")
    for j, e in enumerate(xmom_js):
        equations.append(e); eq_names.append(f"xmom_j{j}")
    for j, e in enumerate(zmom_js):
        equations.append(e); eq_names.append(f"zmom_j{j}")
    # Algebraic constraints: M + 2 of them for VAM(M, M+1, M+1).
    equations.append(kbc_top_alg); eq_names.append("kbc_top_alg")
    equations.append(kbc_bot); eq_names.append("kbc_bot")
    equations.append(surface_bc); eq_names.append("surface_bc")
    for j in range(1, M):
        equations.append(cont_js[j]); eq_names.append(f"cont_j{j}")

    fields = [h] + u_funcs + w_funcs + p_funcs

    sys = PDESystem(equations=equations, fields=fields,
                    time=t, space=[x], parameters={g: g})
    sys.equation_names = eq_names      # extra metadata
    return sys


if __name__ == "__main__":
    sys_vam = build_vam_pdesystem(M=1, N_w=2, N_p=2, flat_bottom=True)
    print(f"VAM(1, 2, 2):  {len(sys_vam.equations)} equations × "
          f"{len(sys_vam.fields)} fields")
    for name, eq in zip(sys_vam.equation_names, sys_vam.equations):
        print(f"  [{name:12s}] {eq}")
    print()
    print("Fields:", [f.func.__name__ for f in sys_vam.fields])
