"""Tests for ``SystemModel.change_state_variables``.

The chain DAE for VAM(1, 2, 2) lives in **primitive state**
``Q^A = (h, U_0, U_1, W_0, W_1, P_0, P_1)`` with a non-identity mass
matrix.  Applying ``change_state_variables`` with the conservative
transform ``U_k → q_{U_k}/h``, ``W_k → q_{W_k}/h`` yields **Form B**
``Q^B = (h, q_U0, q_U1, q_W0, q_W1, P_0, P_1)``.

After the change:

  * Mass matrix on ``mass`` row stays identity.
  * Mass matrix on ``xmom_j0`` / ``zmom_j0`` rows becomes
    ``[0, 1, 0, …]`` / ``[0, 0, 0, 1, …]`` — perfectly diagonal.
  * Mass matrix on ``xmom_j1`` / ``zmom_j1`` rows **does not** become
    diagonal; it carries an ``∂_t h``-column entry equal to
    ``(−q_Uk_minus_1 + q_Uk / 3) / h`` (resp. W variant).  This is
    NOT a bug — it reflects the genuine non-conservative cross-terms
    the Galerkin chain produces on the j ≥ 1 rows (see Step 1 tests:
    these rows are equivalent to Escalante eq (4) modulo the
    cont-projection constraints, not pointwise).
  * Flux ``F`` slot picks up the conservative quantities matching
    Escalante eq (4) on rows ``xmom_j0`` and ``zmom_j0``.
  * Source ``S`` slot stays diagonal as expected.

Tests below pin both: the parts that do become diagonal AND the
parts that don't.  The non-diagonal cross-terms are validated as an
explicit expected residue, so any future drift surfaces immediately.
"""
from __future__ import annotations

import pytest
import sympy as sp


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def m1d():
    """VAMModelGalerkin at level=1.

    Uses the ``escalante`` quadratic form: the post-modal-closure stages
    that eliminate ``∂_t h`` and linear-W cross-terms in the j ≥ 1
    momentum rows, yielding the diagonal Form B mass matrix
    ``diag(1, 1, 1/3, 1, 1/3, 0, 0)`` tested below.  The default
    ``cantero_chinchilla`` form keeps those cross-terms (and the
    associated off-diagonal mass-matrix entries) explicit — see
    ``test_form_B_mass_matrix_cantero_chinchilla`` for that behaviour.
    """
    from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
    return VAMModelGalerkin(level=1, quadratic_form="escalante")


@pytest.fixture(scope="function")
def sm_form_B(m1d):
    """A fresh chain-DAE SystemModel switched to Form B (conservative).

    Returned per-function (scope="function") because
    ``change_state_variables`` mutates in place.
    """
    from zoomy_core.model.models.system_model import SystemModel

    sm = SystemModel.from_model(m1d)
    h, U_0, U_1, W_0, W_1, P_0, P_1 = sm.state
    q_U0 = sp.Symbol("q_U0", real=True)
    q_U1 = sp.Symbol("q_U1", real=True)
    q_W0 = sp.Symbol("q_W0", real=True)
    q_W1 = sp.Symbol("q_W1", real=True)
    sm.change_state_variables(
        new_state=[h, q_U0, q_U1, q_W0, q_W1, P_0, P_1],
        transform={U_0: q_U0 / h, U_1: q_U1 / h,
                   W_0: q_W0 / h, W_1: q_W1 / h},
    )
    return sm, {"h": h, "q_U0": q_U0, "q_U1": q_U1,
                "q_W0": q_W0, "q_W1": q_W1, "P_0": P_0, "P_1": P_1}


# ---------------------------------------------------------------------------
# Unit test on a clean conservative system.
# ---------------------------------------------------------------------------


def test_change_state_swe_to_conservative_diagonal_mass():
    """Simple SWE-style 2-equation system in primitive state
    ``(h, U)`` has mass matrix ``[[1, 0], [U, h]]``.  Switching to
    conservative state ``(h, q_U = h*U)`` must produce identity mass
    matrix and flux/source updated consistently.
    """
    from zoomy_core.model.models.system_model import SystemModel

    t = sp.Symbol("t", real=True)
    x = sp.Symbol("x", real=True)
    h, U = sp.symbols("h U", real=True)
    g = sp.Symbol("g", positive=True)

    # Residual in operator form: M ∂_t Q + ∂_x F = 0
    # with M = [[1, 0], [U, h]], F = [h*U, h*U^2 + g*h^2/2].
    sm = SystemModel(
        time=t,
        space=[x],
        state=[h, U],
        aux_state=[],
        parameters={g: 9.81},
        flux=sp.Matrix([[h * U], [h * U**2 + g * h**2 / 2]]),
        hydrostatic_pressure=sp.zeros(2, 1),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(2, 2, 1),
        source=sp.zeros(2, 1),
        mass_matrix=sp.Matrix([[1, 0], [U, h]]),
    )

    q_U = sp.Symbol("q_U", real=True)
    sm.change_state_variables(
        new_state=[h, q_U],
        transform={U: q_U / h},
    )
    # Mass matrix should become identity.
    assert sm.mass_matrix == sp.eye(2)
    # Flux: row 0 stays h*U = q_U; row 1 becomes (h * (q_U/h)^2 +
    # g*h^2/2) = q_U^2/h + g*h^2/2.
    assert sp.simplify(sm.flux[0, 0] - q_U) == 0
    expected_f1 = q_U**2 / h + g * h**2 / 2
    assert sp.simplify(sm.flux[1, 0] - expected_f1) == 0


# ---------------------------------------------------------------------------
# VAM(1, 2, 2): the parts that DO become diagonal.
# ---------------------------------------------------------------------------


def test_change_state_122_j0_rows_mass_diagonal(sm_form_B):
    """After Form A → Form B, rows ``mass``, ``xmom_j0``, ``zmom_j0``
    have a perfectly diagonal mass matrix entry.
    """
    sm, _ = sm_form_B
    M = sm.mass_matrix
    n = sm.n_state
    # Row 0 (mass): [1, 0, 0, 0, 0, 0, 0].
    assert M[0, 0] == 1
    for j in range(1, n):
        assert M[0, j] == 0
    # Row 1 (xmom_j0): [0, 1, 0, 0, 0, 0, 0].
    assert M[1, 1] == 1
    for j in range(n):
        if j != 1:
            assert M[1, j] == 0
    # Row 3 (zmom_j0): [0, 0, 0, 1, 0, 0, 0].
    assert M[3, 3] == 1
    for j in range(n):
        if j != 3:
            assert M[3, j] == 0


def test_change_state_122_algebraic_rows_stay_zero(sm_form_B):
    """Rows ``cont_j1``, ``cont_j2`` (indices 5, 6) keep all-zero mass
    matrix after change of variables.
    """
    sm, _ = sm_form_B
    M = sm.mass_matrix
    n = sm.n_state
    for i in (5, 6):
        for j in range(n):
            assert M[i, j] == 0, f"M[{i}, {j}] = {M[i, j]} (expected 0)"


# ---------------------------------------------------------------------------
# VAM(1, 2, 2): the parts that DON'T become diagonal — the genuine
# Galerkin cross-terms.
# ---------------------------------------------------------------------------


def test_change_state_122_j1_rows_diagonal(sm_form_B):
    """Rows ``xmom_j1``, ``zmom_j1`` have a diagonal mass-matrix entry
    after Form A → Form B change of variables.

    Previously the chain produced ``(−U_0 + 2U_1/3) ∂_t h`` cross-terms
    on these rows from the Galerkin projection.  The chain now
    eliminates them via mass-equation subtraction (zero on solutions,
    cosmetic for symbolic form), so the Form B mass matrix is
    ``diag(1, 1, 1/3, 1, 1/3, 0, 0)`` — matching Escalante eq (4).
    """
    sm, sy = sm_form_B
    M = sm.mass_matrix

    # Row 2 — xmom_j1: only entry is M[2, 2] = 1/3.
    for j in range(sm.n_state):
        expected = sp.Rational(1, 3) if j == 2 else sp.S.Zero
        assert sp.simplify(M[2, j] - expected) == 0, (
            f"M[xmom_j1, {j}] = {M[2, j]}, expected {expected}"
        )

    # Row 4 — zmom_j1: only entry is M[4, 4] = 1/3.
    for j in range(sm.n_state):
        expected = sp.Rational(1, 3) if j == 4 else sp.S.Zero
        assert sp.simplify(M[4, j] - expected) == 0, (
            f"M[zmom_j1, {j}] = {M[4, j]}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# VAM(1, 2, 2): flux/source slot matches against Escalante eq (4).
# ---------------------------------------------------------------------------


def test_change_state_122_flux_xmom_j0_matches_escalante(sm_form_B):
    """Row 1 (xmom_j0) flux in Form B equals Escalante eq (4) row 2's
    flux argument: ``h·U_0² + (1/3) h·U_1² + h·P_0/ρ``
    = ``q_U0²/h + q_U1²/(3 h) + h·P_0/ρ``.
    """
    sm, sy = sm_form_B
    h, q_U0, q_U1, P_0 = sy["h"], sy["q_U0"], sy["q_U1"], sy["P_0"]
    rho = next(s for s in sm.parameters if str(s) == "rho")
    expected = q_U0**2 / h + q_U1**2 / (3 * h) + h * P_0 / rho
    assert sp.simplify(sm.flux[1, 0] - expected) == 0, (
        f"F[xmom_j0, x] = {sm.flux[1, 0]}, expected {expected}"
    )


def test_change_state_122_flux_zmom_j0_matches_escalante(sm_form_B):
    """Row 3 (zmom_j0) flux equals ``h·U_0·W_0 + (1/3) h·U_1·W_1``
    = ``q_U0·q_W0/h + q_U1·q_W1/(3 h)``.
    """
    sm, sy = sm_form_B
    h, q_U0, q_U1 = sy["h"], sy["q_U0"], sy["q_U1"]
    q_W0, q_W1 = sy["q_W0"], sy["q_W1"]
    expected = q_U0 * q_W0 / h + q_U1 * q_W1 / (3 * h)
    assert sp.simplify(sm.flux[3, 0] - expected) == 0, (
        f"F[zmom_j0, x] = {sm.flux[3, 0]}, expected {expected}"
    )


def test_change_state_122_source_zmom_j0_matches_escalante(sm_form_B):
    """Row 3 (zmom_j0) source.

    The SystemModel canonical form is
    ``M·∂_t Q + ∂_x F + ∂_x P + B·∂_x Q − S(Q) = 0``.  Escalante
    eq (4) row 3 (inviscid) has ``−2 p_1`` on the LHS; in our
    convention that becomes ``S = +2·P_1/ρ`` (the LHS term equals
    ``−S``).
    """
    sm, sy = sm_form_B
    P_1 = sy["P_1"]
    rho = next(s for s in sm.parameters if str(s) == "rho")
    expected = 2 * P_1 / rho
    assert sp.simplify(sm.source[3, 0] - expected) == 0, (
        f"S[zmom_j0] = {sm.source[3, 0]}, expected {expected}"
    )


# ---------------------------------------------------------------------------
# Round-trip: applying the forward + inverse transform must leave the
# operator surface unchanged.
# ---------------------------------------------------------------------------


def test_change_state_round_trip_preserves_operators(m1d):
    """Apply Form A → Form B → Form A; flux / NCP / source / mass
    matrix must equal the originals on every row and column."""
    from zoomy_core.model.models.system_model import SystemModel

    sm = SystemModel.from_model(m1d)
    h, U_0, U_1, W_0, W_1, P_0, P_1 = sm.state

    F_before = sp.Matrix(sm.flux)
    S_before = sp.Matrix(sm.source)
    M_before = sp.Matrix(sm.mass_matrix)
    n_eq, n_st, n_dim = sm.n_equations, sm.n_state, sm.n_dim
    B_before = sp.MutableDenseNDimArray.zeros(n_eq, n_st, n_dim)
    for i in range(n_eq):
        for j in range(n_st):
            for d in range(n_dim):
                B_before[i, j, d] = sp.sympify(
                    sm.nonconservative_matrix[i, j, d]
                )

    q_U0 = sp.Symbol("q_U0", real=True)
    q_U1 = sp.Symbol("q_U1", real=True)
    q_W0 = sp.Symbol("q_W0", real=True)
    q_W1 = sp.Symbol("q_W1", real=True)
    forward_state = [h, q_U0, q_U1, q_W0, q_W1, P_0, P_1]
    forward_transform = {U_0: q_U0 / h, U_1: q_U1 / h,
                         W_0: q_W0 / h, W_1: q_W1 / h}
    inverse_transform = {q_U0: h * U_0, q_U1: h * U_1,
                         q_W0: h * W_0, q_W1: h * W_1}

    sm.change_state_variables(new_state=forward_state,
                              transform=forward_transform)
    sm.change_state_variables(
        new_state=[h, U_0, U_1, W_0, W_1, P_0, P_1],
        transform=inverse_transform,
    )

    # Check operators.
    assert sp.simplify(sm.flux - F_before) == sp.zeros(*F_before.shape)
    assert sp.simplify(sm.source - S_before) == sp.zeros(*S_before.shape)
    assert sp.simplify(sm.mass_matrix - M_before) == sp.zeros(*M_before.shape)
    for i in range(n_eq):
        for j in range(n_st):
            for d in range(n_dim):
                d_before = sp.sympify(B_before[i, j, d])
                d_after = sp.sympify(sm.nonconservative_matrix[i, j, d])
                assert sp.simplify(d_after - d_before) == 0, (
                    f"NCP[{i}, {j}, {d}] differs after round-trip"
                )


def test_change_state_history_recorded(m1d):
    """``change_state_variables`` appends a history entry."""
    from zoomy_core.model.models.system_model import SystemModel

    sm = SystemModel.from_model(m1d)
    n_history_before = len(sm.history)
    h, U_0, U_1, W_0, W_1, P_0, P_1 = sm.state
    q_U0 = sp.Symbol("q_U0", real=True)
    q_U1 = sp.Symbol("q_U1", real=True)
    q_W0 = sp.Symbol("q_W0", real=True)
    q_W1 = sp.Symbol("q_W1", real=True)
    sm.change_state_variables(
        new_state=[h, q_U0, q_U1, q_W0, q_W1, P_0, P_1],
        transform={U_0: q_U0 / h, U_1: q_U1 / h,
                   W_0: q_W0 / h, W_1: q_W1 / h},
    )
    assert len(sm.history) == n_history_before + 1
    assert sm.history[-1]["name"] == "change_state_variables"
