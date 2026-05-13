"""Tests for the chain-DAE SystemModel produced by
``SystemModel.from_model(VAMModelGalerkin(level=...))``.

At ``(M=1, N_w=2, N_p=2)`` the chain closes to 7 equations / 7 fields:

  fields    = ``h, U_0, U_1, W_0, W_1, P_0, P_1``  (W_2 closed via
              bottom KBC; P_2 closed via surface BC)
  equations = ``mass, xmom_j0, xmom_j1, zmom_j0, zmom_j1,
              cont_j1, cont_j2``  (5 evolution + 2 algebraic)

Run::
    pytest tests/integration/zoomy_core/test_vam_chain_dae.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import sympy as sp


REPO = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def m1d():
    """VAMModelGalerkin at level=1 (M=1, N_w=2, N_p=2 by default)."""
    from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
    return VAMModelGalerkin(level=1)


@pytest.fixture(scope="module")
def sm1d(m1d):
    from zoomy_core.model.models.system_model import SystemModel
    return SystemModel.from_model(m1d)


@pytest.fixture(scope="module")
def m1d_233():
    """VAMModelGalerkin at level=2 → (M=2, N_w=3, N_p=3)."""
    from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
    return VAMModelGalerkin(level=2)


@pytest.fixture(scope="module")
def sm1d_233(m1d_233):
    from zoomy_core.model.models.system_model import SystemModel
    return SystemModel.from_model(m1d_233)


def _state_by_name(sm):
    return {str(s): s for s in sm.state}


# ---------------------------------------------------------------------------
# Step 1 — chain DAE structure
# ---------------------------------------------------------------------------


def test_chain_systemmodel_n_state(sm1d):
    """7 state entries after eliminating W_2 (bottom KBC) and P_2
    (surface BC)."""
    assert sm1d.n_state == 7
    assert sm1d.n_equations == 7


def test_chain_systemmodel_state_names(sm1d):
    expected = ["h", "U_0", "U_1", "W_0", "W_1", "P_0", "P_1"]
    assert [str(s) for s in sm1d.state] == expected


def test_chain_systemmodel_equation_names(sm1d):
    expected = [
        "mass",
        "xmom_j0", "xmom_j1",
        "zmom_j0", "zmom_j1",
        "cont_j1", "cont_j2",
    ]
    assert sm1d.equation_names == expected


def test_chain_systemmodel_dae_partition(sm1d):
    """5 evolution + 2 algebraic.  Algebraic rows have zero mass-matrix
    row; evolution rows have at least one non-zero entry."""
    n = sm1d.n_equations
    M = sm1d.mass_matrix
    # Evolution rows: 0..4 (mass + 2 xmom + 2 zmom).
    for i in range(5):
        row = [M[i, j] for j in range(n)]
        assert any(r != 0 for r in row), (
            f"evolution row {i} has all-zero mass matrix"
        )
    # Algebraic rows: 5, 6 (cont_j1, cont_j2).
    for i in (5, 6):
        row = [M[i, j] for j in range(n)]
        assert all(r == 0 for r in row), (
            f"algebraic row {i} has nonzero mass matrix entries: {row}"
        )


# ---------------------------------------------------------------------------
# Step 2 — residual reconstruction matches Escalante eq (4) for j=0 rows
# ---------------------------------------------------------------------------


def test_chain_mass_residual(sm1d):
    """Reconstructed ``mass`` residual equals ``∂_t h + ∂_x(h·U_0)``."""
    residuals = sm1d.reconstruct_residuals()
    idx = sm1d.equation_names.index("mass")
    state_by_name = _state_by_name(sm1d)
    h_sym = state_by_name["h"]
    U_0_sym = state_by_name["U_0"]
    t = sm1d.time
    x = sm1d.space[0]
    h_fn = sp.Function(str(h_sym), real=True)(t, x)
    U_0_fn = sp.Function(str(U_0_sym), real=True)(t, x)
    expected = sp.Derivative(h_fn, t) + sp.Derivative(h_fn * U_0_fn, x).doit()
    diff = sp.simplify(sp.expand(residuals[idx] - expected))
    assert diff == 0, f"mass residual diverges: diff = {diff}"


def test_chain_xmom_j0_residual_matches_escalante(sm1d):
    """``xmom_j0`` residual equals Escalante eq (4) row 2 (inviscid)."""
    residuals = sm1d.reconstruct_residuals()
    idx = sm1d.equation_names.index("xmom_j0")
    name_to_sym = _state_by_name(sm1d)
    t = sm1d.time
    x = sm1d.space[0]

    def _fn(name):
        return sp.Function(name, real=True)(t, x)

    h = _fn("h")
    U_0, U_1 = _fn("U_0"), _fn("U_1")
    P_0, P_1 = _fn("P_0"), _fn("P_1")
    g = next(s for s in sm1d.parameters if str(s) == "g")
    rho = next(s for s in sm1d.parameters if str(s) == "rho")
    # Locate the bottom topography Function in residuals.
    b = None
    for r in residuals:
        for a in r.atoms(sp.Function):
            if a.func.__name__ == "b":
                b = a
                break
        if b is not None:
            break
    eta = b + h

    expected = (
        sp.Derivative(h * U_0, t).doit()
        + sp.Derivative(
            h * U_0**2
            + sp.Rational(1, 3) * h * U_1**2
            + h * P_0 / rho, x).doit()
        + g * h * sp.Derivative(eta, x).doit()
        + 2 * P_1 * sp.Derivative(b, x).doit() / rho
    )
    diff = sp.expand(residuals[idx].doit() - sp.expand(expected))
    diff = sp.simplify(diff)
    assert diff == 0, f"xmom_j0 diverges: diff = {diff}"


def test_chain_zmom_j0_residual_matches_escalante(sm1d):
    """``zmom_j0`` residual equals Escalante eq (4) row 3 (inviscid)."""
    residuals = sm1d.reconstruct_residuals()
    idx = sm1d.equation_names.index("zmom_j0")
    t = sm1d.time
    x = sm1d.space[0]

    def _fn(name):
        return sp.Function(name, real=True)(t, x)

    h = _fn("h")
    U_0, U_1 = _fn("U_0"), _fn("U_1")
    W_0, W_1 = _fn("W_0"), _fn("W_1")
    P_1 = _fn("P_1")
    rho = next(s for s in sm1d.parameters if str(s) == "rho")

    expected = (
        sp.Derivative(h * W_0, t).doit()
        + sp.Derivative(
            h * U_0 * W_0
            + sp.Rational(1, 3) * h * U_1 * W_1, x).doit()
        - 2 * P_1 / rho
    )
    diff = sp.expand(residuals[idx].doit() - sp.expand(expected))
    diff = sp.simplify(diff)
    assert diff == 0, f"zmom_j0 diverges: diff = {diff}"


def test_chain_cont_j1_residual_matches_I1(sm1d):
    """``cont_j1`` residual equals Escalante's I_1."""
    residuals = sm1d.reconstruct_residuals()
    idx = sm1d.equation_names.index("cont_j1")
    t = sm1d.time
    x = sm1d.space[0]

    def _fn(name):
        return sp.Function(name, real=True)(t, x)

    h = _fn("h")
    U_0, U_1 = _fn("U_0"), _fn("U_1")
    W_0 = _fn("W_0")
    b = None
    for r in residuals:
        for a in r.atoms(sp.Function):
            if a.func.__name__ == "b":
                b = a
                break
        if b is not None:
            break

    expected = (h * sp.Derivative(U_0, x).doit()
                + sp.Rational(1, 3) * sp.Derivative(h * U_1, x).doit()
                + sp.Rational(1, 3) * U_1 * sp.Derivative(h, x).doit()
                + 2 * (W_0 - U_0 * sp.Derivative(b, x).doit()))
    diff = sp.simplify(sp.expand(residuals[idx] - expected))
    assert diff == 0, f"cont_j1 diverges: diff = {diff}"


def test_chain_cont_j2_residual_matches_neg_I2(sm1d):
    """``cont_j2`` residual equals ``-I_2``."""
    residuals = sm1d.reconstruct_residuals()
    idx = sm1d.equation_names.index("cont_j2")
    t = sm1d.time
    x = sm1d.space[0]

    def _fn(name):
        return sp.Function(name, real=True)(t, x)

    h = _fn("h")
    U_0, U_1 = _fn("U_0"), _fn("U_1")
    W_1 = _fn("W_1")
    b = None
    for r in residuals:
        for a in r.atoms(sp.Function):
            if a.func.__name__ == "b":
                b = a
                break
        if b is not None:
            break

    I_2 = (h * sp.Derivative(U_0, x).doit()
           + U_1 * sp.Derivative(h, x).doit()
           + 2 * (U_1 * sp.Derivative(b, x).doit() - W_1))
    expected = -I_2
    diff = sp.simplify(sp.expand(residuals[idx] - expected))
    assert diff == 0, f"cont_j2 diverges: diff = {diff}"


def test_chain_p2_eliminated(sm1d):
    """``P_2`` does not appear as a state entry and does not appear in
    any operator slot."""
    state_names = [str(s) for s in sm1d.state]
    assert "P_2" not in state_names
    # No matrix entry should reference P_2.
    P_2 = sp.Symbol("P_2", real=True)
    for i in range(sm1d.n_equations):
        for j in range(sm1d.n_state):
            assert not sm1d.mass_matrix[i, j].has(P_2)
            for d in range(sm1d.n_dim):
                assert not sm1d.flux[i, d].has(P_2)
                assert not sm1d.hydrostatic_pressure[i, d].has(P_2)
                assert not sp.sympify(
                    sm1d.nonconservative_matrix[i, j, d]).has(P_2)
        assert not sm1d.source[i, 0].has(P_2)


# ---------------------------------------------------------------------------
# Step 1 (new) — constraint-modulo paper match for j ≥ 1 rows
# ---------------------------------------------------------------------------
#
# The j=1 rows are NOT pointwise equal to Escalante eq (4) — they
# carry Galerkin chain-rule cross-terms.  They ARE equal modulo
# {mass, cont_j1, cont_j2}.  Reduction recipe:
#   1. ``.doit()`` to expand ``Derivative(product, var)`` atoms.
#   2. Solve ``cont_j1=0`` for W_0; ``cont_j2=0`` for W_1; substitute.
#   3. Substitute ``∂_t h → -∂_x(h·U_0)`` from mass.


def _common_xmom_j1_setup(sm1d):
    residuals = sm1d.reconstruct_residuals()
    t = sm1d.time
    x = sm1d.space[0]

    def _fn(name):
        return sp.Function(name, real=True)(t, x)

    h, U_0, U_1 = _fn("h"), _fn("U_0"), _fn("U_1")
    W_0, W_1 = _fn("W_0"), _fn("W_1")
    P_0, P_1 = _fn("P_0"), _fn("P_1")
    rho = next(s for s in sm1d.parameters if str(s) == "rho")
    b = None
    for r in residuals:
        for a in r.atoms(sp.Function):
            if a.func.__name__ == "b":
                b = a
                break
        if b is not None:
            break

    cont_j1 = residuals[sm1d.equation_names.index("cont_j1")].doit()
    cont_j2 = residuals[sm1d.equation_names.index("cont_j2")].doit()
    W_0_sol = sp.solve(cont_j1, W_0)[0]
    W_1_sol = sp.solve(cont_j2, W_1)[0]

    return {
        "residuals": residuals, "t": t, "x": x, "h": h,
        "U_0": U_0, "U_1": U_1, "W_0": W_0, "W_1": W_1,
        "P_0": P_0, "P_1": P_1, "rho": rho, "b": b,
        "W_0_sol": W_0_sol, "W_1_sol": W_1_sol,
    }


def test_chain_xmom_j1_constraint_equivalent_to_escalante(sm1d):
    """``xmom_j1`` residual equals Escalante eq (4) row 5 (inviscid)
    modulo the ideal generated by ``{mass, cont_j1, cont_j2}``."""
    s = _common_xmom_j1_setup(sm1d)
    h, U_0, U_1 = s["h"], s["U_0"], s["U_1"]
    P_0, P_1, rho, b = s["P_0"], s["P_1"], s["rho"], s["b"]
    x = s["x"]

    chain = s["residuals"][sm1d.equation_names.index("xmom_j1")].doit()
    ref = (
        sp.Rational(1, 3) * sp.Derivative(h * U_1, s["t"]).doit()
        + sp.Rational(1, 3) * sp.Derivative(
            2 * h * U_0 * U_1 + h * P_1 / rho, x).doit()
        - sp.Rational(1, 3) * U_0 * sp.Derivative(h * U_1, x).doit()
        - (P_0 / rho - P_1 / (3 * rho)) * sp.Derivative(h, x).doit()
        - 2 * (P_0 - P_1) / rho * sp.Derivative(b, x).doit()
    )

    diff = sp.expand(chain - ref)
    diff = diff.subs({s["W_0"]: s["W_0_sol"], s["W_1"]: s["W_1_sol"]})
    diff = sp.expand(diff)
    diff = diff.subs(sp.Derivative(h, s["t"]),
                     -sp.Derivative(h * U_0, x).doit())
    reduced = sp.simplify(sp.expand(diff))
    assert reduced == 0, (
        f"xmom_j1 not equivalent to Escalante eq (4) row 5 modulo "
        f"{{mass, cont_j1, cont_j2}}; reduced diff = {reduced}"
    )


def test_chain_zmom_j1_constraint_equivalent_to_escalante(sm1d):
    """``zmom_j1`` residual equals Escalante eq (4) row 6 (inviscid)
    modulo ``{mass, cont_j1, cont_j2}`` and the closures already baked
    into the chain (``W_2`` via bottom KBC, ``P_2`` via surface BC)."""
    s = _common_xmom_j1_setup(sm1d)
    h, U_0, U_1 = s["h"], s["U_0"], s["U_1"]
    W_0, W_1 = s["W_0"], s["W_1"]
    P_0, P_1, rho, b = s["P_0"], s["P_1"], s["rho"], s["b"]
    x, t = s["x"], s["t"]

    W_2_sub = -(W_0 + W_1) + (U_0 + U_1) * sp.Derivative(b, x).doit()
    P_2_sub = P_1 - P_0
    p_b = P_0 + P_1 + P_2_sub          # = 2*P_1

    chain = s["residuals"][sm1d.equation_names.index("zmom_j1")].doit()
    ref = (
        sp.Rational(1, 3) * sp.Derivative(h * W_1, t).doit()
        + sp.Rational(1, 3) * sp.Derivative(
            h * U_0 * W_1
            + U_1 * (h * W_0 + sp.Rational(2, 5) * h * W_2_sub),
            x,
        ).doit()
        + sp.Rational(1, 3) * (sp.Rational(1, 5) * W_2_sub - W_0)
        * sp.Derivative(h * U_1, x).doit()
        + 2 * P_0 / rho - p_b / rho
    )

    diff = sp.expand(chain - ref)
    diff = diff.subs({W_0: s["W_0_sol"], W_1: s["W_1_sol"]})
    diff = sp.expand(diff)
    diff = diff.subs(sp.Derivative(h, t),
                     -sp.Derivative(h * U_0, x).doit())
    reduced = sp.simplify(sp.expand(diff))
    assert reduced == 0, (
        f"zmom_j1 not equivalent to Escalante eq (4) row 6 modulo "
        f"{{mass, cont_j1, cont_j2}}; reduced diff = {reduced}"
    )


# ---------------------------------------------------------------------------
# Form A residual fixture dump
# ---------------------------------------------------------------------------


def test_chain_form_A_residuals_match_fixture(sm1d):
    """Pretty-printed Form A residuals for VAM(1, 2, 2) match the
    committed fixture file."""
    fixture_path = (REPO / "tests" / "fixtures"
                    / "vam_122_chain_form_A.txt")
    assert fixture_path.exists(), (
        f"Missing fixture file: {fixture_path}.\n"
        "Generate it with the helper script in tests/fixtures/."
    )

    residuals = sm1d.reconstruct_residuals()
    lines = []
    for name, res in zip(sm1d.equation_names, residuals):
        lines.append(f"=== {name} ===")
        lines.append(sp.pretty(sp.expand(res), use_unicode=True))
        lines.append("")
    actual = "\n".join(lines).rstrip() + "\n"
    expected = fixture_path.read_text()
    assert actual == expected, (
        "Chain Form A residual dump differs from fixture.\n"
        "Either regenerate the fixture intentionally, or investigate "
        "what changed in the chain primitives."
    )


# ---------------------------------------------------------------------------
# VAM(2, 3, 3) — structural tests only
# ---------------------------------------------------------------------------


def test_vam_233_n_state_and_equations(sm1d_233):
    """VAM(2, 3, 3): 10 equations / 10 state entries after closures.

    Active state: ``h, U_0..U_2, W_0..W_2, P_0..P_2`` (W_3 closed via
    bot KBC; P_3 closed via surface BC).
    """
    assert sm1d_233.n_equations == 10
    assert sm1d_233.n_state == 10


def test_vam_233_state_names(sm1d_233):
    expected = {
        "h",
        "U_0", "U_1", "U_2",
        "W_0", "W_1", "W_2",
        "P_0", "P_1", "P_2",
    }
    assert {str(s) for s in sm1d_233.state} == expected


def test_vam_233_equation_names(sm1d_233):
    expected = [
        "mass",
        "xmom_j0", "xmom_j1", "xmom_j2",
        "zmom_j0", "zmom_j1", "zmom_j2",
        "cont_j1", "cont_j2", "cont_j3",
    ]
    assert sm1d_233.equation_names == expected


def test_vam_233_dae_partition(sm1d_233):
    """7 evolution + 3 algebraic at VAM(2, 3, 3)."""
    n = sm1d_233.n_equations
    M = sm1d_233.mass_matrix
    for i in range(7):
        row = [M[i, j] for j in range(n)]
        assert any(r != 0 for r in row), (
            f"evolution row {i} has all-zero mass matrix"
        )
    for i in range(7, 10):
        row = [M[i, j] for j in range(n)]
        assert all(r == 0 for r in row), (
            f"algebraic row {i} has nonzero mass matrix entries: {row}"
        )


def test_vam_233_w3_p3_eliminated(sm1d_233):
    """W_3 and P_3 are eliminated everywhere in operators."""
    state_names = [str(s) for s in sm1d_233.state]
    assert "W_3" not in state_names
    assert "P_3" not in state_names
    W_3 = sp.Symbol("W_3", real=True)
    P_3 = sp.Symbol("P_3", real=True)
    for i in range(sm1d_233.n_equations):
        for d in range(sm1d_233.n_dim):
            assert not sm1d_233.flux[i, d].has(W_3, P_3)
            assert not sm1d_233.hydrostatic_pressure[i, d].has(W_3, P_3)
        assert not sm1d_233.source[i, 0].has(W_3, P_3)
