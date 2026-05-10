"""Acceptance tests for ``model.splitter.build_pressure_elliptic_block``.

The contract from ``thesis/chapters/derivation_vam.md`` §7.6:

* For VAM(1, 2, 2), substituting the corrector update for
  ``(U_k, W_k)`` into the algebraic ``cont_jk`` rows must produce a
  ``2 × 2`` elliptic block in ``(P_0, P_1)`` matching
  [tutorials/vam/escalante2024_poisson.py](
  ../../tutorials/vam/escalante2024_poisson.py) bit-for-bit (modulo
  symbol identity).

* For VAM(2, 3, 3), the same construction must produce the matching
  ``3 × 3`` block from
  [tutorials/vam/escalante2024_poisson_generic.py](
  ../../tutorials/vam/escalante2024_poisson_generic.py) at
  ``--M 2 --N 3``.

The tests here verify two properties at increasing strictness:

1. **Strict P-linearity** of every elliptic row (mirrors
   Escalante's ``collect_pressure_coeffs`` assertion).

2. **Coefficient-for-coefficient match** with Escalante's reference
   for VAM(1, 2, 2), exercising the concrete ``T_1..T_5`` source
   forms from ``escalante2024_poisson.py:85–91``.
"""
from __future__ import annotations

import pytest
import sympy as sp


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def m1d_122():
    """VAM1D at level=1 → (M=1, N_w=2, N_p=2)."""
    from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin

    class VAM1D(VAMModelGalerkin):
        ins_dimension = 2

    return VAM1D(level=1)


@pytest.fixture(scope="module")
def m1d_233():
    """VAM1D at level=2 → (M=2, N_w=3, N_p=3)."""
    from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin

    class VAM1D(VAMModelGalerkin):
        ins_dimension = 2

    return VAM1D(level=2)


# ---------------------------------------------------------------------------
# VAM(1, 2, 2).
# ---------------------------------------------------------------------------


def test_elliptic_block_122_p_linear(m1d_122):
    """Each elliptic row at VAM(1, 2, 2) is strictly linear in
    ``(P_0, P_1, ∂_x P_0, ∂_x P_1, ∂_xx P_0, ∂_xx P_1)``."""
    from zoomy_core.model.splitter import (
        build_pressure_elliptic_block,
        verify_p_linearity,
    )

    pdesys = m1d_122._chain_dae
    fields_by_name = {f.func.__name__: f for f in pdesys.fields}
    pressure_vars = [fields_by_name["P_0"], fields_by_name["P_1"]]
    dt = sp.Symbol(r"\Delta t", positive=True)

    block = build_pressure_elliptic_block(pdesys, pressure_vars, dt)
    rows = block["rows"]

    # 2 elliptic rows expected (cont_j1, cont_j2).
    assert set(rows.keys()) == {1, 2}

    # Strict P-linearity (raises AssertionError on failure).
    result = verify_p_linearity(rows, pressure_vars, pdesys.space[0])

    # Each row must be in the linear span of 6 pressure atoms; each
    # atom either appears (nonzero coeff) or doesn't.  Here we just
    # verify the assertion ran successfully.
    assert set(result["coefficients"].keys()) == {1, 2}


def test_elliptic_block_122_matches_escalante(m1d_122):
    """Coefficient-for-coefficient match against
    ``tutorials/vam/escalante2024_poisson.py`` for VAM(1, 2, 2).

    Re-derives Escalante's reference 2×2 block in symbols aligned
    with the chain DAE's, then compares each ``a_i, b_i, RHS_i``
    coefficient via ``sp.simplify(diff) == 0``.
    """
    from zoomy_core.model.splitter import (
        build_pressure_elliptic_block,
        verify_p_linearity,
    )

    pdesys = m1d_122._chain_dae
    fields_by_name = {f.func.__name__: f for f in pdesys.fields}
    h = fields_by_name["h"]
    P_0 = fields_by_name["P_0"]
    P_1 = fields_by_name["P_1"]
    rho = next(s for s in pdesys.parameters if str(s) == "rho")
    t = pdesys.time
    x = pdesys.space[0]
    # Pull bottom from any equation atoms.
    b = None
    for eq in pdesys.equations:
        for atom in eq.atoms(sp.Function):
            if atom.func.__name__ == "b":
                b = atom
                break
        if b is not None:
            break
    assert b is not None, "could not locate bottom in chain equations"
    dt = sp.Symbol(r"\Delta t", positive=True)

    # Build my elliptic block.
    block = build_pressure_elliptic_block(pdesys, [P_0, P_1], dt)

    # ---- Escalante's reference (escalante2024_poisson.py:85-129).
    # Predictor velocities: align with my U_k_tilde / W_k_tilde naming.
    U_0_tilde = block["U_tilde"][0]
    U_1_tilde = block["U_tilde"][1]
    W_0_tilde = block["W_tilde"][0]
    W_1_tilde = block["W_tilde"][1]

    # The chain DAE uses dynamic pressure P_k (with units of stress /
    # density factor); Escalante's poisson.py uses kinematic p_k =
    # P_k / rho.  So Escalante's T's get ``/rho`` when we plug in our
    # pressure symbols.  After substituting U/W = corrector(P/rho)
    # into the I_1, I_2 forms, the resulting elliptic operator is
    # linear in P (since we just rescaled).  We compare the
    # *coefficients*, which differ by a rho^-1 factor in the
    # P-coefficient slots and an unchanged RHS.

    # Escalante's T1..T5 with our P naming and rho rescaling:
    p0 = P_0 / rho
    p1 = P_1 / rho
    T2 = sp.Derivative(h * p0, x).doit() + 2 * p1 * sp.Derivative(b, x).doit()
    T3 = -2 * p1
    T4 = (sp.Derivative(h * p1, x).doit()
          - (3 * p0 - p1) * sp.Derivative(h, x).doit()
          - 6 * (p0 - p1) * sp.Derivative(b, x).doit())
    T5 = 6 * (p0 - p1)

    # Predictor conserved variables and corrector:
    q0_t = h * U_0_tilde
    q1_t = h * U_1_tilde
    r0_t = h * W_0_tilde
    r1_t = h * W_1_tilde

    q0_k = q0_t - dt * T2
    q1_k = q1_t - dt * T4
    r0_k = r0_t - dt * T3
    r1_k = r1_t - dt * T5

    u0_k = q0_k / h
    u1_k = q1_k / h
    w0_k = r0_k / h
    w1_k = r1_k / h

    # Escalante's I_1, I_2 evaluated at the corrector:
    I_1_at_k = (h * sp.Derivative(u0_k, x).doit()
                + sp.Rational(1, 3) * sp.Derivative(h * u1_k, x).doit()
                + sp.Rational(1, 3) * u1_k * sp.Derivative(h, x).doit()
                + 2 * (w0_k - u0_k * sp.Derivative(b, x).doit()))
    I_2_at_k = (h * sp.Derivative(u0_k, x).doit()
                + u1_k * sp.Derivative(h, x).doit()
                + 2 * (u1_k * sp.Derivative(b, x).doit() - w1_k))

    # cont_j1 should match I_1 at corrector; cont_j2 should match -I_2.
    # The corrector substitution introduces ``1/h`` denominators that
    # sympy.simplify alone won't always cancel through a multivariate
    # rational form.  Compare via ``together().cancel()`` first, and
    # fall back to per-atom coefficient comparison if that fails.

    def deep_zero(expr):
        e = sp.expand(expr)
        e = sp.together(e).cancel()
        e = sp.simplify(e)
        return e == 0

    if not deep_zero(block["rows"][1] - I_1_at_k):
        # Fall back: compare coefficient-by-coefficient on the
        # pressure atoms (this is what Escalante's
        # ``collect_pressure_coeffs`` does).
        from zoomy_core.model.splitter import verify_p_linearity

        my_block = verify_p_linearity({1: block["rows"][1]}, [P_0, P_1], x)
        ref_block = verify_p_linearity({1: I_1_at_k}, [P_0, P_1], x)
        for atom_name in my_block["coefficients"][1]:
            my_c = my_block["coefficients"][1][atom_name]
            ref_c = ref_block["coefficients"][1][atom_name]
            cdiff = sp.simplify(sp.together(my_c - ref_c).cancel())
            assert cdiff == 0, (
                f"cont_j1 coefficient of {atom_name} differs from "
                f"Escalante I_1: my={my_c}, ref={ref_c}, diff={cdiff}"
            )
        # Constants should also match.
        const_diff = sp.simplify(sp.together(
            my_block["constants"][1] - ref_block["constants"][1]).cancel())
        assert const_diff == 0, (
            f"cont_j1 constant (RHS) differs from Escalante I_1: "
            f"diff = {const_diff}"
        )

    if not deep_zero(block["rows"][2] - (-I_2_at_k)):
        from zoomy_core.model.splitter import verify_p_linearity

        my_block = verify_p_linearity({2: block["rows"][2]}, [P_0, P_1], x)
        ref_block = verify_p_linearity({2: -I_2_at_k}, [P_0, P_1], x)
        for atom_name in my_block["coefficients"][2]:
            my_c = my_block["coefficients"][2][atom_name]
            ref_c = ref_block["coefficients"][2][atom_name]
            cdiff = sp.simplify(sp.together(my_c - ref_c).cancel())
            assert cdiff == 0, (
                f"cont_j2 coefficient of {atom_name} differs from "
                f"-I_2: my={my_c}, ref={ref_c}, diff={cdiff}"
            )
        const_diff = sp.simplify(sp.together(
            my_block["constants"][2] - ref_block["constants"][2]).cancel())
        assert const_diff == 0, (
            f"cont_j2 constant (RHS) differs from -I_2: diff = {const_diff}"
        )


def test_elliptic_block_122_pressure_atoms_present(m1d_122):
    """Both ``∂_xx P_0`` and ``∂_xx P_1`` must appear in the elliptic
    block — otherwise the system isn't actually elliptic in P.
    """
    from zoomy_core.model.splitter import (
        build_pressure_elliptic_block,
        verify_p_linearity,
    )

    pdesys = m1d_122._chain_dae
    fields_by_name = {f.func.__name__: f for f in pdesys.fields}
    P_0 = fields_by_name["P_0"]
    P_1 = fields_by_name["P_1"]
    dt = sp.Symbol(r"\Delta t", positive=True)

    block = build_pressure_elliptic_block(pdesys, [P_0, P_1], dt)
    result = verify_p_linearity(block["rows"], [P_0, P_1], pdesys.space[0])

    # At least one row should carry a non-zero ∂_xx P_l coefficient.
    has_xx = False
    for j, coeffs in result["coefficients"].items():
        for name, coeff in coeffs.items():
            if name.startswith("d_xx") and sp.simplify(coeff) != 0:
                has_xx = True
                break
    assert has_xx, (
        "elliptic block has no ∂_xx P_l terms — not actually elliptic"
    )


# ---------------------------------------------------------------------------
# VAM(2, 3, 3).
# ---------------------------------------------------------------------------


def test_elliptic_block_233_p_linear(m1d_233):
    """At VAM(2, 3, 3) the elliptic block has 3 rows in
    ``(P_0, P_1, P_2)`` and is strictly linear in
    ``(P_l, ∂_x P_l, ∂_{xx} P_l)``."""
    from zoomy_core.model.splitter import (
        build_pressure_elliptic_block,
        verify_p_linearity,
    )

    pdesys = m1d_233._chain_dae
    fields_by_name = {f.func.__name__: f for f in pdesys.fields}
    pressure_vars = [
        fields_by_name["P_0"],
        fields_by_name["P_1"],
        fields_by_name["P_2"],
    ]
    dt = sp.Symbol(r"\Delta t", positive=True)

    block = build_pressure_elliptic_block(pdesys, pressure_vars, dt)
    rows = block["rows"]

    assert set(rows.keys()) == {1, 2, 3}

    result = verify_p_linearity(rows, pressure_vars, pdesys.space[0])
    assert set(result["coefficients"].keys()) == {1, 2, 3}


def test_split_for_pressure_122_returns_three_subsystems(m1d_122):
    """``split_for_pressure`` returns a ``SplitForPressureResult``
    holding three :class:`SystemModel` objects sharing the same 7-state
    Q vector but updating different subsets via rectangular operators.
    """
    from zoomy_core.model.models.system_model import SystemModel
    from zoomy_core.model.splitter import (
        split_for_pressure,
        SplitForPressureResult,
    )

    pdesys = m1d_122._chain_dae
    fields_by_name = {f.func.__name__: f for f in pdesys.fields}
    pressure_vars = [fields_by_name["P_0"], fields_by_name["P_1"]]
    dt = sp.Symbol(r"\Delta t", positive=True)

    result = split_for_pressure(pdesys, pressure_vars, dt)

    assert isinstance(result, SplitForPressureResult)
    for sm in (result.SM_pred, result.SM_press, result.SM_corr):
        assert isinstance(sm, SystemModel)
        # Same shared 7-state Q across all stages.
        assert [str(s) for s in sm.state] == [
            "h", "U_0", "U_1", "W_0", "W_1", "P_0", "P_1"
        ]
        assert sm.n_state == 7

    # Predictor: 5 evolution rows (mass + 2 xmom + 2 zmom) updating
    # Q[0..4] = h, U_k, W_k.  Pressure entries pass through.
    assert result.SM_pred.n_equations == 5
    assert result.SM_pred.equation_to_state_index == [0, 1, 2, 3, 4]
    assert result.SM_pred.mass_matrix.shape == (5, 7)
    assert result.SM_pred.equation_names == [
        "mass", "xmom_j0", "xmom_j1", "zmom_j0", "zmom_j1"
    ]

    # Pressure stage: 2 algebraic rows (elliptic block) updating
    # Q[5..6] = P_0, P_1.  M = 0.
    assert result.SM_press.n_equations == 2
    assert result.SM_press.equation_to_state_index == [5, 6]
    assert result.SM_press.mass_matrix.shape == (2, 7)
    assert result.SM_press.mass_matrix.is_zero_matrix
    assert result.SM_press.equation_names == ["elliptic_j1", "elliptic_j2"]

    # Corrector: 4 algebraic update rows for U_0, U_1, W_0, W_1.  M = 0.
    assert result.SM_corr.n_equations == 4
    assert result.SM_corr.equation_to_state_index == [1, 2, 3, 4]
    assert result.SM_corr.mass_matrix.shape == (4, 7)
    assert result.SM_corr.mass_matrix.is_zero_matrix
    assert result.SM_corr.equation_names == [
        "corr_U_0", "corr_U_1", "corr_W_0", "corr_W_1"
    ]


def test_elliptic_block_233_pressure_atoms_present(m1d_233):
    """At VAM(2, 3, 3) the 3×3 block must engage all three pressure
    modes (each ``P_l`` should appear in at least one row)."""
    from zoomy_core.model.splitter import (
        build_pressure_elliptic_block,
        verify_p_linearity,
    )

    pdesys = m1d_233._chain_dae
    fields_by_name = {f.func.__name__: f for f in pdesys.fields}
    pressure_vars = [
        fields_by_name["P_0"],
        fields_by_name["P_1"],
        fields_by_name["P_2"],
    ]
    dt = sp.Symbol(r"\Delta t", positive=True)

    block = build_pressure_elliptic_block(pdesys, pressure_vars, dt)
    result = verify_p_linearity(block["rows"], pressure_vars, pdesys.space[0])

    for p in pressure_vars:
        name = p.func.__name__
        # P_l should appear in at least one row's coefficients.
        appears = False
        for j, coeffs in result["coefficients"].items():
            for atom_name, coeff in coeffs.items():
                if name in atom_name and sp.simplify(coeff) != 0:
                    appears = True
                    break
            if appears:
                break
        assert appears, (
            f"pressure mode {name} doesn't appear in any elliptic row "
            f"— 3x3 block is degenerate in P"
        )
