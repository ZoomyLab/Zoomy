"""TDD tests for VAMModelGalerkin._chain_dae and SystemModel-from-chain.

These tests drive the loop closure in two steps:

* **Step 1** — chain produces an Escalante-aligned DAE.  Tests:
  - ``test_chain_dae_attribute_exists``  : ``m._chain_dae`` returns a
    ``PDESystem`` (or compatible) at default level=1.
  - ``test_chain_dae_equation_count``    : 9 equations / 9 fields at
    (M=1, N_w=2, N_p=2).
  - ``test_chain_dae_equation_names``    : names match April's
    ``build_vam_pdesystem`` exactly.
  - ``test_chain_dae_dae_partition``     : 6 evolution + 3 algebraic.
  - ``test_chain_dae_mass_equation``     : the ``mass`` row is
    ``∂_t h + ∂_x(h · u_0)`` modulo symbol identity.
  - ``test_chain_dae_kbc_equations``     : ``kbc_top_alg`` / ``kbc_bot``
    / ``surface_bc`` match Escalante's reference shapes.

* **Step 2** — SystemModel built from chain DAE has the right
  hyperbolic + constraint structure.  Tests:
  - ``test_systemmodel_from_chain_dae_shapes`` : flux / NCP / source /
    mass_matrix have the right shape.
  - ``test_systemmodel_mass_matrix_diagonal`` : 1 on evolution rows,
    0 on algebraic rows.

Run::
    pytest tests/integration/zoomy_core/test_vam_chain_dae.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import sympy as sp


REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def m1d():
    """VAM1D at level=1 (M=1, N_w=2, N_p=2 by default)."""
    from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin

    class VAM1D(VAMModelGalerkin):
        ins_dimension = 2

    return VAM1D(level=1)


# ---------------------------------------------------------------------------
# STEP 1 — chain DAE structure
# ---------------------------------------------------------------------------

def test_chain_dae_attribute_exists(m1d):
    """``m._chain_dae`` must exist and be accessible after model
    construction — currently FAILS because we haven't added it yet."""
    assert hasattr(m1d, "_chain_dae"), (
        "VAMModelGalerkin needs a ``_chain_dae`` attribute holding the "
        "Escalante-aligned DAE (mass + projections + KBC + surface BC)."
    )
    assert m1d._chain_dae is not None


def test_chain_dae_equation_count(m1d):
    """8 equations / 8 fields after eliminating ``P_{N_p}`` via the
    non-hydrostatic surface BC."""
    pdesys = m1d._chain_dae
    assert len(pdesys.equations) == 8
    assert len(pdesys.fields) == 8


def test_chain_dae_equation_names(m1d):
    expected = [
        "mass",
        "xmom_j0", "xmom_j1",
        "zmom_j0", "zmom_j1", "zmom_j2",
        "kbc_top", "kbc_bot",
    ]
    assert m1d._chain_dae.equation_names == expected


def test_chain_dae_dae_partition(m1d):
    """6 evolution + 2 algebraic per Escalante (with ``P_2``
    eliminated via the surface BC, ``surface_bc`` is consumed and
    only the two kinematic BCs remain as algebraic rows)."""
    sys.path.insert(0, str(REPO / "thesis/notebooks/verification/dae_solver"))
    try:
        from test_dae_partition_bridge import dae_partition  # type: ignore
    finally:
        sys.path.remove(str(REPO / "thesis/notebooks/verification/dae_solver"))
    dyn, alg, _, _ = dae_partition(m1d._chain_dae)
    pdesys = m1d._chain_dae
    dyn_names = [pdesys.equation_names[i] for i in dyn]
    alg_names = [pdesys.equation_names[i] for i in alg]
    assert dyn_names == ["mass", "xmom_j0", "xmom_j1",
                         "zmom_j0", "zmom_j1", "zmom_j2"]
    assert alg_names == ["kbc_top", "kbc_bot"]


def test_chain_dae_mass_equation(m1d):
    """``mass`` row must be ``∂_t h + ∂_x(h · U_0) = 0``."""
    pdesys = m1d._chain_dae
    idx = pdesys.equation_names.index("mass")
    mass_expr = pdesys.equations[idx]
    fields_by_name = {f.func.__name__: f for f in pdesys.fields}
    h = fields_by_name["h"]
    U_0 = fields_by_name["U_0"]
    t, x = pdesys.time, pdesys.space[0]
    expected = sp.Derivative(h, t) + sp.Derivative(h * U_0, x).doit()
    diff = sp.simplify(sp.expand(mass_expr - expected))
    assert diff == 0, f"mass equation diverges: diff = {diff}"


def test_chain_dae_kbc_bot(m1d):
    """``kbc_bot`` row must be ``w|_b - u|_b · ∂_x b``.

    For shifted Legendre on [0,1]:
        w|_b = sum_k W_k · phi_k(0) = sum_k W_k     (phi_k(0) = 1)
        u|_b = sum_k U_k · phi_k(0) = sum_k U_k
    """
    pdesys = m1d._chain_dae
    idx = pdesys.equation_names.index("kbc_bot")
    kbc = pdesys.equations[idx]
    fields_by_name = {f.func.__name__: f for f in pdesys.fields}
    # b is non-evolving (parameter-like); extract from equation atoms.
    b = next(a for a in kbc.atoms(sp.Function) if a.func.__name__ == "b")
    U_funcs = [fields_by_name[f"U_{i}"] for i in range(2)]   # M+1 = 2
    W_funcs = [fields_by_name[f"W_{i}"] for i in range(3)]   # N_w+1 = 3
    x = pdesys.space[0]
    expected = (sum(W_funcs) - sum(U_funcs) * sp.Derivative(b, x).doit())
    diff = sp.simplify(sp.expand(kbc - expected))
    assert diff == 0, f"kbc_bot diverges: diff = {diff}"


def test_chain_dae_p_top_eliminated(m1d):
    """``P_{N_p}`` is consumed by the surface BC and absent from
    every equation in the DAE (eliminated via
    ``Σ_k phi_p_k(1) · P_k = 0`` solved for ``P_{N_p}``)."""
    pdesys = m1d._chain_dae
    fields_by_name = {f.func.__name__: f for f in pdesys.fields}
    # P_2 must NOT appear as a field at default (M=1, N_w=N_p=2).
    assert "P_2" not in fields_by_name, (
        f"P_2 should be eliminated; fields = {list(fields_by_name)}"
    )
    # No equation should reference P_2 either.
    for name, eq in zip(pdesys.equation_names, pdesys.equations):
        assert "P_2" not in str(eq), (
            f"equation {name!r} still references eliminated P_2"
        )


def test_chain_dae_matches_escalante_reference(m1d):
    """Bit-for-bit comparison of the chain's three j=0 evolution
    equations against the verified Escalante reference equations
    from ``thesis/notebooks/modeling/transparent_derivations/
    vam_l1_physical_z.py``.

    Compared after substituting the chain's ``U_k / W_k / P_k``
    coefficient functions with the reference's ``u_k / w_k / p_k``
    convention so symbolic identity is well-defined.
    """
    pdesys = m1d._chain_dae
    fields_by_name = {f.func.__name__: f for f in pdesys.fields}

    # Chain symbols.
    h = fields_by_name["h"]
    U_0, U_1 = fields_by_name["U_0"], fields_by_name["U_1"]
    W_0, W_1 = fields_by_name["W_0"], fields_by_name["W_1"]
    P_0, P_1 = fields_by_name["P_0"], fields_by_name["P_1"]
    t, x = pdesys.time, pdesys.space[0]
    g = next(s for s in pdesys.parameters if str(s) == "g")
    # The chain's ``b`` is a Function on (t, x) with ``∂_t b = 0``
    # already substituted; pull it from the kbc_bot equation.
    kbc_bot = pdesys.equations[pdesys.equation_names.index("kbc_bot")]
    b = next(a for a in kbc_bot.atoms(sp.Function)
             if a.func.__name__ == "b")

    # Escalante eq (4) reference equations expressed in chain
    # symbols.  Note we have to multiply Escalante's pressure terms
    # by ``rho`` because his ``p_k`` is the kinematic non-hydrostatic
    # pressure (already divided by density), whereas our ``P_k`` is
    # the dynamic pressure with units of stress.
    rho = next(s for s in pdesys.parameters if str(s) == "rho")
    eta = b + h
    ref_mass = sp.Derivative(h, t) + sp.Derivative(h * U_0, x).doit()
    ref_xmom_j0 = (
        sp.Derivative(h * U_0, t)
        + sp.Derivative(
            h * U_0**2
            + sp.Rational(1, 3) * h * U_1**2
            + h * P_0 / rho, x).doit()
        + g * h * sp.Derivative(eta, x).doit()
        + 2 * P_1 * sp.Derivative(b, x).doit() / rho
    )
    ref_zmom_j0 = (
        sp.Derivative(h * W_0, t)
        + sp.Derivative(
            h * U_0 * W_0
            + sp.Rational(1, 3) * h * U_1 * W_1, x).doit()
        - 2 * P_1 / rho
    )

    def _diff(name, ref):
        idx = pdesys.equation_names.index(name)
        chain_expr = pdesys.equations[idx]
        return sp.simplify(sp.expand(chain_expr - ref))

    assert _diff("mass", ref_mass) == 0, "mass diverges from Escalante eq (4) row 1"
    assert _diff("xmom_j0", ref_xmom_j0) == 0, (
        "xmom_j0 diverges from Escalante eq (4) row 2"
    )
    assert _diff("zmom_j0", ref_zmom_j0) == 0, (
        "zmom_j0 diverges from Escalante eq (4) row 3"
    )


# ---------------------------------------------------------------------------
# STEP 2 — SystemModel built from chain DAE
# ---------------------------------------------------------------------------

def test_systemmodel_from_chain_dae_attribute(m1d):
    """``m._chain_dae_systemmodel`` must hold a SystemModel built from
    the chain DAE — currently FAILS because we haven't wired this yet."""
    assert hasattr(m1d, "_chain_dae_systemmodel"), (
        "Need a SystemModel built from the chain DAE (mass matrix has "
        "0 rows for kbc_top_alg / kbc_bot / surface_bc)."
    )
    sm = m1d._chain_dae_systemmodel
    assert sm is not None


def test_systemmodel_mass_matrix_evolution_vs_algebraic(m1d):
    """Mass matrix rows partition the system into evolution / algebraic.

    The PDESystem is in primitive form (state = ``[h, u_0, ..., w_0,
    ..., p_0, ...]``); ``∂_t(h u_0)`` therefore expands via product
    rule to ``h ∂_t u_0 + u_0 ∂_t h``, which gives off-diagonal
    entries on the xmom rows (``M[xmom_j0, h] = u_0``,
    ``M[xmom_j0, u_0] = h``).  The contract that matters for the
    DAE solver is:

      * **algebraic rows** (``kbc_top_alg``, ``kbc_bot``,
        ``surface_bc``, indices 6..8) — the *whole row* of the mass
        matrix is zero (no time derivative present).
      * **evolution rows** (``mass``, ``xmom_j0/j1``,
        ``zmom_j0/j1/j2``, indices 0..5) — at least one entry in the
        row is non-zero.
    """
    sm = m1d._chain_dae_systemmodel
    M_mat = sm.mass_matrix
    n = M_mat.shape[0]
    assert n == 8

    # First 6 rows = evolutions (mass + xmom_j0/j1 + zmom_j0/j1/j2).
    for i in range(6):
        row = [M_mat[i, j] for j in range(n)]
        assert any(r != 0 for r in row), (
            f"evolution row {i} has all-zero mass matrix; "
            f"row contents: {row}"
        )
    # Last 2 rows = kbc_top / kbc_bot — algebraic, all-zero.
    for i in range(6, 8):
        row = [M_mat[i, j] for j in range(n)]
        assert all(r == 0 for r in row), (
            f"algebraic row {i} has nonzero mass matrix entries: {row}"
        )
