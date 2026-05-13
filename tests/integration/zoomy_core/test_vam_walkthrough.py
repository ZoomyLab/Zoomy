"""Integration tests for VAM derivation structural invariants.

These tests guard the chain-System-tree leaf structure (no notebook
execution, since the canonical ``vam_pipeline_walkthrough.py``
notebook is checked separately and ``vam_get_pde_walkthrough.py`` is
on the outdated-tutorials list awaiting migration).

Run with::

    pytest tests/integration/zoomy_core/test_vam_walkthrough.py -v
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def m1d():
    from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
    return VAMModelGalerkin(level=1)


EXPECTED_CHAIN_LEAVES = {
    # Continuity is Galerkin-projected against
    # ``φ_p_k(state.zeta)`` for k = 0..N_p (N_p + 1 leaves).  ``test_0``
    # becomes the mass evolution row; ``test_1..test_{N_p}`` are the
    # algebraic constraints (Escalante's I_1, -I_2) that determine
    # ``P_0..P_{N_p-1}``.
    ("continuity", "test_0"),
    ("continuity", "test_1"),
    ("continuity", "test_2"),
    # x-momentum projected against test_phi_u (M+1 leaves).
    ("momentum", "x", "test_0"),
    ("momentum", "x", "test_1"),
    # z-momentum projected against test_phi_w for j = 0..N_w-1
    # (W_{N_w} closed at basis level by bot KBC).
    ("momentum", "z", "test_0"),
    ("momentum", "z", "test_1"),
}


def test_chain_leaf_structure(m1d):
    """At (M=1, N_w=N_p=2) the chain System tree has exactly the seven
    leaves Escalante's projection structure expects."""
    paths = {p for p, _ in m1d._chain_system.leaves()}
    assert paths == EXPECTED_CHAIN_LEAVES, (
        f"chain leaves diverged from Escalante VAM(1,2,2) projection "
        f"structure.\n  expected: {EXPECTED_CHAIN_LEAVES}\n  got: {paths}"
    )
    assert m1d._chain_M == 1
    assert m1d._chain_N_w == 2
    assert m1d._chain_N_p == 2


def test_describe_renders_closed_chain(m1d):
    """``m1d.describe()`` should render the chain's closed primitive
    form — every integral resolved by ``EvaluateIntegrals``.  The
    presence of ``Integral(`` in the markdown means we regressed."""
    desc = m1d.describe()
    md = desc._repr_markdown_() if hasattr(desc, "_repr_markdown_") else str(desc)
    assert "Integral(" not in md, (
        "describe() leaked un-evaluated Integral atoms — chain regression"
    )
    for label in ("continuity", "momentum.x", "momentum.z"):
        assert label in md, f"missing {label!r} in describe() output"
