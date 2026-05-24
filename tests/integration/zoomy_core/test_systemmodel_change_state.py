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
    h, U_0, U_1, W_0, W_1, b, P_0, P_1 = sm.state
    q_U0 = sp.Symbol("q_U0", real=True)
    q_U1 = sp.Symbol("q_U1", real=True)
    q_W0 = sp.Symbol("q_W0", real=True)
    q_W1 = sp.Symbol("q_W1", real=True)
    sm.change_state_variables(
        new_state=[h, q_U0, q_U1, q_W0, q_W1, b, P_0, P_1],
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
    """Rows ``cont_j1``, ``cont_j2`` (indices 6, 7 with b promoted to
    state) keep all-zero mass matrix after change of variables.
    """
    sm, _ = sm_form_B
    M = sm.mass_matrix
    n = sm.n_state
    for i in (6, 7):
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
    h, U_0, U_1, W_0, W_1, b, P_0, P_1 = sm.state

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
    forward_state = [h, q_U0, q_U1, q_W0, q_W1, b, P_0, P_1]
    forward_transform = {U_0: q_U0 / h, U_1: q_U1 / h,
                         W_0: q_W0 / h, W_1: q_W1 / h}
    inverse_transform = {q_U0: h * U_0, q_U1: h * U_1,
                         q_W0: h * W_0, q_W1: h * W_1}

    sm.change_state_variables(new_state=forward_state,
                              transform=forward_transform)
    sm.change_state_variables(
        new_state=[h, U_0, U_1, W_0, W_1, b, P_0, P_1],
        transform=inverse_transform,
    )

    # Check operators.  ``sp.simplify`` may strip ZArray type back to
    # plain NDimArray, which compares unequal to Matrix even with
    # identical content — so iterate the flat element view.
    def _flatten(nested):
        if isinstance(nested, list):
            for x in nested:
                yield from _flatten(x)
        else:
            yield nested
    for diff in (sm.flux - F_before,
                 sm.source - S_before,
                 sm.mass_matrix - M_before):
        for e in _flatten(diff.tolist()):
            assert sp.simplify(e) == 0
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
    h, U_0, U_1, W_0, W_1, b, P_0, P_1 = sm.state
    q_U0 = sp.Symbol("q_U0", real=True)
    q_U1 = sp.Symbol("q_U1", real=True)
    q_W0 = sp.Symbol("q_W0", real=True)
    q_W1 = sp.Symbol("q_W1", real=True)
    sm.change_state_variables(
        new_state=[h, q_U0, q_U1, q_W0, q_W1, b, P_0, P_1],
        transform={U_0: q_U0 / h, U_1: q_U1 / h,
                   W_0: q_W0 / h, W_1: q_W1 / h},
    )
    assert len(sm.history) == n_history_before + 1
    assert sm.history[-1]["name"] == "change_state_variables"


# ---------------------------------------------------------------------------
# remove_non_diagonal_h: push M[:, h] cross-terms into the NCP matrix B.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def m1d_cantero():
    """VAMModelGalerkin at level=1 with the default ``cantero_chinchilla``
    quadratic form — keeps the j ≥ 1 ``∂_t h`` cross-terms explicit, so
    ``remove_non_diagonal_h`` has work to do."""
    from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
    return VAMModelGalerkin(level=1, quadratic_form="cantero_chinchilla")


def _sm_cantero_conservative(m):
    """Conservative-state SystemModel with the walkthrough's modal
    rescaling ``q_U1 = h U_1 / 3``, ``q_W1 = h W_1 / 3``.  Returns
    ``(sm, syms)``."""
    from zoomy_core.model.models.system_model import SystemModel

    sm = SystemModel.from_model(m)
    h, U_0, U_1, W_0, W_1, b, P_0, P_1 = sm.state
    q_U0 = sp.Symbol("q_U0", real=True)
    q_U1 = sp.Symbol("q_U1", real=True)
    q_W0 = sp.Symbol("q_W0", real=True)
    q_W1 = sp.Symbol("q_W1", real=True)
    sm.change_state_variables(
        new_state=[h, q_U0, q_U1, q_W0, q_W1, b, P_0, P_1],
        transform={U_0: q_U0 / h, U_1: 3 * q_U1 / h,
                   W_0: q_W0 / h, W_1: 3 * q_W1 / h},
    )
    return sm, {"h": h, "q_U0": q_U0, "q_U1": q_U1,
                "q_W0": q_W0, "q_W1": q_W1}


def test_remove_non_diagonal_h_122_conservative(m1d_cantero):
    """In conservative state with modal rescaling, the j = 1 momentum
    rows carry an ``M[i, h] = (q_U_k − q_U_{k+1})/h`` cross-term.
    After ``remove_non_diagonal_h`` the ``h``-column is zero on all
    rows, the diagonal ``M[xmom_j1, q_U1] = 1`` is preserved, and the
    NCP matrix picks up exactly the predicted entry."""
    sm, sy = _sm_cantero_conservative(m1d_cantero)
    h, q_U0, q_U1 = sy["h"], sy["q_U0"], sy["q_U1"]
    q_W0, q_W1 = sy["q_W0"], sy["q_W1"]

    # Cache the NCP slab before the pass so we can measure the delta.
    n_eq, n_st, n_dim = sm.n_equations, sm.n_state, sm.n_dim
    B_before = sp.MutableDenseNDimArray(
        sm.nonconservative_matrix.tolist(),
        shape=tuple(sm.nonconservative_matrix.shape),
    )

    sm.remove_non_diagonal_h()

    # Mass matrix: h-column zero on every row except the mass row itself
    # (which keeps its `M[mass, h] = 1`).
    assert sp.simplify(sm.mass_matrix[0, 0] - 1) == 0    # mass row preserved
    for i in range(1, n_eq):
        assert sp.simplify(sm.mass_matrix[i, 0]) == 0, (
            f"M[{i}, h] = {sm.mass_matrix[i, 0]} after pass (expected 0)"
        )

    # Diagonal entries on j = 1 rows preserved (the rescaling already
    # normalised them to 1 in conservative form).
    assert sp.simplify(sm.mass_matrix[2, 2] - 1) == 0    # xmom_j1, q_U1
    assert sp.simplify(sm.mass_matrix[4, 4] - 1) == 0    # zmom_j1, q_W1

    # B-delta on j = 1 momentum rows, ∂_x q_U0 column.
    # F[mass, x] = q_U0 ⇒ ∂F/∂q_U0 = 1 ⇒ delta = -coeff·1.
    expected_xmom = (q_U0 - q_U1) / h
    expected_zmom = (q_W0 - q_W1) / h
    delta_x = sp.simplify(sm.nonconservative_matrix[2, 1, 0]
                          - B_before[2, 1, 0])
    delta_z = sp.simplify(sm.nonconservative_matrix[4, 1, 0]
                          - B_before[4, 1, 0])
    assert sp.simplify(delta_x - expected_xmom) == 0, (
        f"B[xmom_j1, q_U0, x] gained {delta_x}, expected {expected_xmom}"
    )
    assert sp.simplify(delta_z - expected_zmom) == 0, (
        f"B[zmom_j1, q_U0, x] gained {delta_z}, expected {expected_zmom}"
    )


def test_remove_non_diagonal_h_122_primitive(m1d_cantero):
    """In primitive state the j = 1 momentum rows have a
    state-dependent ``∂_t h``-column AND a state-dependent diagonal
    (``M[xmom_j1, U_1] = h/3``).  ``remove_non_diagonal_h`` clears the
    ``h``-column but **leaves the diagonal alone** — that is the input
    to ``InvertMassMatrix``, not this op's job."""
    from zoomy_core.model.models.system_model import SystemModel

    sm = SystemModel.from_model(m1d_cantero)
    h, U_0, U_1, W_0, W_1, b, P_0, P_1 = sm.state
    sm.remove_non_diagonal_h()

    # h-column cleared everywhere except the mass row itself.
    assert sp.simplify(sm.mass_matrix[0, 0] - 1) == 0
    for i in range(1, sm.n_equations):
        assert sp.simplify(sm.mass_matrix[i, 0]) == 0

    # State-dependent diagonals preserved (not normalised).
    assert sp.simplify(sm.mass_matrix[1, 1] - h) == 0       # xmom_j0
    assert sp.simplify(sm.mass_matrix[2, 2] - h / 3) == 0    # xmom_j1
    assert sp.simplify(sm.mass_matrix[3, 3] - h) == 0       # zmom_j0
    assert sp.simplify(sm.mass_matrix[4, 4] - h / 3) == 0    # zmom_j1

    # NCP gained the j = 1 cross-term in the U_0 column (primitive
    # mass flux F[mass, x] = h·U_0 ⇒ ∂F/∂U_0 = h, ∂F/∂h = U_0; the
    # j = 1 row's old M[i, h] = -U_0 + 2 U_1/3 propagates to
    # B[xmom_j1, U_0, x] = -(-U_0 + 2 U_1/3) · h = h (U_0 − 2 U_1/3)).
    expected_xmom_U0 = h * (U_0 - sp.Rational(2, 3) * U_1)
    expected_zmom_U0 = h * (W_0 - sp.Rational(2, 3) * W_1)
    assert sp.simplify(sm.nonconservative_matrix[2, 1, 0]
                       - expected_xmom_U0) == 0
    assert sp.simplify(sm.nonconservative_matrix[4, 1, 0]
                       - expected_zmom_U0) == 0


def test_remove_non_diagonal_h_residual_equivalence(m1d_cantero):
    """The substitution is exact modulo the mass equation: for every
    affected row, the change in (M ∂_t Q + ∂_x F + ∂_x P + B ∂_x Q − S)
    equals ``M_old[i, h] · (mass-residual)``."""
    from zoomy_core.model.models.system_model import SystemModel

    sm = SystemModel.from_model(m1d_cantero)
    n_eq, n_st, n_dim = sm.n_equations, sm.n_state, sm.n_dim
    t = sm.time
    space = sm.space
    state = list(sm.state)

    # Snapshot every primary before the pass.
    M_before = sp.Matrix(sm.mass_matrix.tolist())
    S_before = sp.Matrix(sm.source.tolist())
    B_before = sp.MutableDenseNDimArray(
        sm.nonconservative_matrix.tolist(),
        shape=tuple(sm.nonconservative_matrix.shape),
    )

    # Treat state entries as functions of (t, *space) so chain-rule
    # divergences materialise as honest derivative atoms.
    field = {s: sp.Function(f"_{s.name}")(t, *space) for s in state}

    def _residual_row(i, M, B, S):
        """LHS = Σ_k M[i,k] ∂_t Q_k + Σ_d ∂_d F[i,d] + Σ_d ∂_d P[i,d]
        + Σ_{k,d} B[i,k,d] ∂_d Q_k - S[i, 0], all evaluated under the
        field-substitution."""
        lhs = sp.S.Zero
        for k in range(n_st):
            lhs += M[i, k].xreplace(field) * sp.diff(field[state[k]], t)
        for d in range(n_dim):
            f_id = sm.flux[i, d].xreplace(field)
            p_id = sm.hydrostatic_pressure[i, d].xreplace(field)
            lhs += sp.diff(f_id, space[d])
            lhs += sp.diff(p_id, space[d])
        for k in range(n_st):
            for d in range(n_dim):
                lhs += (B[i, k, d].xreplace(field)
                        * sp.diff(field[state[k]], space[d]))
        lhs -= S[i, 0].xreplace(field)
        return lhs

    mass_row = 0
    mass_residual = _residual_row(mass_row, M_before, B_before, S_before)

    sm.remove_non_diagonal_h()

    M_after = sp.Matrix(sm.mass_matrix.tolist())
    S_after = sp.Matrix(sm.source.tolist())
    B_after = sp.MutableDenseNDimArray(
        sm.nonconservative_matrix.tolist(),
        shape=tuple(sm.nonconservative_matrix.shape),
    )

    for i in range(n_eq):
        if i == mass_row:
            continue
        before = _residual_row(i, M_before, B_before, S_before)
        after = _residual_row(i, M_after, B_after, S_after)
        coeff = M_before[i, 0].xreplace(field)
        delta = sp.expand(after - before)
        # The substitution must absorb exactly `-coeff · mass_residual`
        # of LHS content (because the row LHS originally contained
        # `coeff · ∂_t h` which we rewrote using ∂_t h = − rest_of_mass).
        # We compare under the field-substitution to keep all the
        # ∂_d Q_k atoms honest.
        expected = sp.expand(-coeff * mass_residual)
        diff = sp.expand(delta - expected)
        assert sp.simplify(diff) == 0, (
            f"row {i}: residual delta {delta} ≠ expected {expected}"
        )


def test_remove_non_diagonal_h_history(m1d_cantero):
    """``remove_non_diagonal_h`` appends a history entry."""
    from zoomy_core.model.models.system_model import SystemModel

    sm = SystemModel.from_model(m1d_cantero)
    n_history_before = len(sm.history)
    sm.remove_non_diagonal_h()
    assert len(sm.history) == n_history_before + 1
    assert sm.history[-1]["name"] == "remove_non_diagonal_h"


def test_remove_non_diagonal_h_idempotent(m1d_cantero):
    """Applying twice is a no-op the second time (h-column is already 0)."""
    from zoomy_core.model.models.system_model import SystemModel

    sm = SystemModel.from_model(m1d_cantero)
    sm.remove_non_diagonal_h()
    snap_M = sp.Matrix(sm.mass_matrix.tolist())
    snap_B = sp.MutableDenseNDimArray(
        sm.nonconservative_matrix.tolist(),
        shape=tuple(sm.nonconservative_matrix.shape),
    )
    snap_S = sp.Matrix(sm.source.tolist())

    sm.remove_non_diagonal_h()

    # M unchanged.
    for i in range(sm.n_equations):
        for k in range(sm.n_state):
            assert sp.simplify(sm.mass_matrix[i, k] - snap_M[i, k]) == 0
    # B unchanged.
    for i in range(sm.n_equations):
        for k in range(sm.n_state):
            for d in range(sm.n_dim):
                assert sp.simplify(
                    sm.nonconservative_matrix[i, k, d] - snap_B[i, k, d]
                ) == 0
    # S unchanged.
    for i in range(sm.n_equations):
        assert sp.simplify(sm.source[i, 0] - snap_S[i, 0]) == 0


# ---------------------------------------------------------------------------
# Full state-dependent-field propagation: BCs + IMEX twins.
#
# A CoV must rewrite *every* state-dependent field, not just the five
# primary operators.  The tests below exercise the secondary fields
# that ``change_state_variables`` extends after the primaries:
# diffusion matrices, the explicit-source twin, and the symbolic BC
# kernels.
# ---------------------------------------------------------------------------


def test_change_state_diffusion_matrix_mixed_partial():
    """Diffusion CoV under ``hu → h·u`` produces a mixed partial entry.

    SWE 1D has ``A[1, 1, 0, 0] = h·ν`` (depth-weighted eddy viscosity
    on the momentum row).  Under the CoV ``hu = h·u`` the diffusive
    flux

        F_diff[1, 0] = h·ν · ∂_x hu = h·ν · (u·∂_x h + h·∂_x u)

    decomposes into two entries on the new state ``(h, u)``:

        A[1, 0, 0, 0] = h·ν·u   (coefficient of ∂_x h)
        A[1, 1, 0, 0] = h²·ν    (coefficient of ∂_x u)
    """
    from zoomy_core.model.models.swe import SWE
    from zoomy_core.model.models.system_model import SystemModel

    m = SWE(dimension=1, manning_n=0.0, nu=0.01)
    sm = SystemModel.from_model(m)
    h, hu = sm.state
    u = sp.Symbol("u", real=True)
    sm.change_state_variables(new_state=[h, u], transform={hu: h * u})

    p = sm.parameters
    A = sm.diffusion_matrix
    nu = p.nu if hasattr(p, "nu") else sp.Symbol("nu")
    # Mass / mass row: identically zero.
    assert sp.simplify(A[0, 0, 0, 0]) == 0
    assert sp.simplify(A[0, 1, 0, 0]) == 0
    # Momentum row picks up both the h-coupling and the u-self entries.
    assert sp.simplify(A[1, 0, 0, 0] - h * nu * u) == 0
    assert sp.simplify(A[1, 1, 0, 0] - h ** 2 * nu) == 0


def test_change_state_boundary_conditions_swept_to_new_state():
    """BC ``args.variables`` Zstruct and ``definition`` no longer
    reference the old state Symbol ``hu`` after a CoV to ``(h, u)``.
    """
    from zoomy_core.model.models.swe import SWE
    from zoomy_core.model.models.system_model import SystemModel

    m = SWE(dimension=1, manning_n=0.0, nu=0.0)
    sm = SystemModel.from_model(m)
    h, hu = sm.state
    u = sp.Symbol("u", real=True)

    # Pre-CoV: args.variables names are the OLD state names.
    assert list(sm.boundary_conditions.args.variables.keys()) == ["h", "hu"]

    sm.change_state_variables(new_state=[h, u], transform={hu: h * u})

    # Post-CoV: args.variables names match the NEW state.
    assert list(sm.boundary_conditions.args.variables.keys()) == ["h", "u"]
    # The Symbols themselves are the new state Symbols.
    assert sm.boundary_conditions.args.variables.h is h
    assert sm.boundary_conditions.args.variables.u is u
    # The definition free Symbols do not include the OLD state.
    free = sm.boundary_conditions.definition.free_symbols
    assert hu not in free, (
        f"BC definition still references the OLD state Symbol hu; "
        f"free symbols are {free}"
    )


def test_change_state_aux_boundary_conditions_swept_to_new_state():
    """``aux_boundary_conditions`` follows the same propagation path
    as ``boundary_conditions`` — its args.variables Zstruct must be
    rebuilt to the new state."""
    from zoomy_core.model.models.swe import SWE
    from zoomy_core.model.models.system_model import SystemModel

    m = SWE(dimension=1, manning_n=0.0, nu=0.0)
    sm = SystemModel.from_model(m)
    if sm.aux_boundary_conditions is None:
        pytest.skip("SWE didn't expose aux_boundary_conditions; nothing to test")
    h, hu = sm.state
    u = sp.Symbol("u", real=True)
    sm.change_state_variables(new_state=[h, u], transform={hu: h * u})
    assert (list(sm.aux_boundary_conditions.args.variables.keys())
            == ["h", "u"])


def test_change_state_boundary_gradients_swept_to_new_state():
    """``boundary_gradients`` follows the same propagation path."""
    from zoomy_core.model.models.swe import SWE
    from zoomy_core.model.models.system_model import SystemModel

    m = SWE(dimension=1, manning_n=0.0, nu=0.0)
    sm = SystemModel.from_model(m)
    if sm.boundary_gradients is None:
        pytest.skip("SWE didn't expose boundary_gradients; nothing to test")
    h, hu = sm.state
    u = sp.Symbol("u", real=True)
    sm.change_state_variables(new_state=[h, u], transform={hu: h * u})
    assert (list(sm.boundary_gradients.args.variables.keys())
            == ["h", "u"])


def test_change_state_extrapolation_bc_stays_identity_on_slots():
    """The bug this pins: an extrapolation BC body is ``ZArray(Q)`` — a
    list of state Symbols treated as **index placeholders** for the
    boundary face state.  Under a CoV ``{U_0: q_U0/h, …}`` an
    ``xreplace`` would remap that placeholder into the *inverse*
    transform of the old symbol (e.g. slot 1 becomes ``q_U0/h``
    instead of ``q_U0``), so the runtime would divide inflow momentum
    by the depth at every extrapolation face and NaN ``h``
    immediately.

    The fix is to rebuild the BC kernel from the source
    :class:`BoundaryConditions` against the new state Zstruct;
    extrapolation re-emerges as ``ZArray(new_state)`` (identity on
    slot symbols).  This test asserts no ``q_*/h`` ratio leaks into
    the BC definition after a VAM(1, 2, 2) Form A → Form B CoV.
    """
    from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
    from zoomy_core.model.models.system_model import SystemModel

    m = VAMModelGalerkin(level=1, quadratic_form="escalante")
    sm = SystemModel.from_model(m)
    h, U_0, U_1, W_0, W_1, b, P_0, P_1 = sm.state
    q_U0, q_U1, q_W0, q_W1 = sp.symbols(
        "q_U0 q_U1 q_W0 q_W1", real=True)
    sm.change_state_variables(
        new_state=[h, q_U0, q_U1, q_W0, q_W1, b, P_0, P_1],
        transform={U_0: q_U0 / h, U_1: q_U1 / h,
                   W_0: q_W0 / h, W_1: q_W1 / h},
    )
    # Reduce the definition to a single sympy Tuple we can scan
    # uniformly: ZArray (default extrapolation body) is flattened to a
    # Tuple, Piecewise is wrapped in a 1-Tuple so `.has(...)` and
    # `.free_symbols` work the same way for either shape.
    defn = sm.boundary_conditions.definition
    if hasattr(defn, "tolist"):
        defn_scan = sp.Tuple(*[sp.sympify(e) for e in defn.tolist()])
    else:
        defn_scan = sp.Tuple(sp.sympify(defn))
    for q_sym in (q_U0, q_U1, q_W0, q_W1):
        atom = q_sym / h
        assert not defn_scan.has(atom), (
            f"BC definition still references the placeholder image "
            f"{atom} — the rebuild-from-source path did not run or it "
            f"fell back to xreplace.\nBody:\n{defn}"
        )

    # And the positive check: the new state Symbols ARE referenced.
    free = defn_scan.free_symbols
    for q_sym in (q_U0, q_U1, q_W0, q_W1):
        assert q_sym in free, (
            f"BC definition does not reference the new state Symbol "
            f"{q_sym}"
        )


def test_change_state_source_explicit_swept_to_new_state():
    """``source_explicit`` (default zero on most models) must still be
    re-expanded against the new state symbols, not point at the old
    ones — even when it is identically zero the substitution must run
    without raising and yield the correct shape.
    """
    from zoomy_core.model.models.swe import SWE
    from zoomy_core.model.models.system_model import SystemModel

    m = SWE(dimension=1, manning_n=0.0, nu=0.0)
    sm = SystemModel.from_model(m)
    h, hu = sm.state
    u = sp.Symbol("u", real=True)
    sm.change_state_variables(new_state=[h, u], transform={hu: h * u})
    assert sm.source_explicit is not None
    assert sm.source_explicit.shape == (2, 1)
    # Default is zero; identically zero stays zero.
    for i in range(2):
        assert sp.simplify(sm.source_explicit[i, 0]) == 0
