"""Integration tests for the VAM end-to-end walkthrough.

Lightweight TDD harness for `tutorials/vam/vam_get_pde_walkthrough.py`:

* ``test_walkthrough_runs`` — the whole script executes top-to-bottom
  with no exception, and the live solver reports a finite mass.
* ``test_describe_renders_closed_chain`` — ``m1d.describe()`` returns
  the chain's closed primitive form (every integral resolved); no
  un-evaluated ``Integral(`` atoms in the rendered markdown.
* ``test_chain_leaf_structure`` — at default level=1
  (M=1, N_w=N_p=2) the chain has exactly the eight leaves Escalante's
  projection structure expects.
* ``test_april_builder_dae_partition`` — the verified
  ``build_vam_pdesystem(1, 2, 2)`` from April 2026 still produces
  9 equations / 9 fields with a 6+3 dynamic/algebraic partition; this
  is the canonical Escalante shape we reference from the walkthrough.

Run with::

    pytest tests/integration/zoomy_core/test_vam_walkthrough.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]


def _exec_walkthrough():
    """Run the walkthrough script in a fresh namespace and return it.

    The walkthrough is a jupytext-light file with ``# +`` / ``# -``
    cell markers but otherwise valid python.  We exec it once per
    test that needs the result; the cost is dominated by the
    chain build (~1s) and the 2D dam-break (~5s).
    """
    ns: dict = {}
    src = (REPO / "tutorials/vam/vam_get_pde_walkthrough.py").read_text()
    exec(compile(src, "vam_get_pde_walkthrough.py", "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def walkthrough_ns():
    return _exec_walkthrough()


# ---------------------------------------------------------------------------
# Trivial "script runs through" — the absolute floor of TDD.
# ---------------------------------------------------------------------------

def test_walkthrough_runs(walkthrough_ns):
    """The full pipeline (chain → SystemModel → analysis → runtime →
    solver) executes top-to-bottom without raising and produces a
    finite, mass-conserving 2D dam-break."""
    diagnostics = walkthrough_ns["diagnostics"]
    assert diagnostics["all-finite Q"] is True
    assert diagnostics["all-finite p"] is True
    assert diagnostics["h_min"] > 0.5
    assert diagnostics["h_max"] < 3.0
    # Mass conserved to ~1e-7 of expected.
    assert abs(diagnostics["mass"] - diagnostics["mass expected"]) < 1e-3


# ---------------------------------------------------------------------------
# "describe is as I want it" — no un-evaluated integrals leaking.
# ---------------------------------------------------------------------------

def test_describe_renders_closed_chain(walkthrough_ns):
    """``m1d.describe()`` should render the chain's closed
    primitive-form equations — every integral resolved by
    ``EvaluateIntegrals``.  The presence of ``\\int`` or
    ``Integral(`` in the markdown means we regressed back to the
    parent's depth-integrate-then-stop system."""
    m1d = walkthrough_ns["m1d"]
    desc = m1d.describe()
    md = desc._repr_markdown_() if hasattr(desc, "_repr_markdown_") else str(desc)
    # Negative checks: no un-evaluated integrals.
    assert "Integral(" not in md, (
        "describe() leaked un-evaluated Integral atoms — chain regression"
    )
    # Positive checks: the augmented chain mentions every leaf the
    # Escalante DAE carries (projections + algebraic constraints).
    for label in ("momentum.x", "momentum.z",
                  "mass", "kbc_top_alg", "kbc_bot", "surface_bc"):
        assert label in md, f"missing {label!r} in describe() output"


# ---------------------------------------------------------------------------
# Chain leaf structure at (M=1, N_w=N_p=2).
# ---------------------------------------------------------------------------

EXPECTED_CHAIN_LEAVES = {
    # Projection leaves the chain primitives produce.
    ("momentum", "x", "test_0"),
    ("momentum", "x", "test_1"),
    ("momentum", "z", "test_0"),
    ("momentum", "z", "test_1"),
    ("momentum", "z", "test_2"),
    # Algebraic constraint leaves added at the end of the chain
    # (mass + KBC + surface BC, matching Escalante eq (4)+(5)).
    ("mass",),
    ("kbc_top_alg",),
    ("kbc_bot",),
    ("surface_bc",),
    # cont test_0 → replaced by mass; cont test_1 / test_2 dropped
    # (M=1: cont_j1..M-1 is empty; cont test_{N_w} is trivially 0).
}


def test_chain_leaf_structure(walkthrough_ns):
    m1d = walkthrough_ns["m1d"]
    paths = {p for p, _ in m1d._chain_system.leaves()}
    assert paths == EXPECTED_CHAIN_LEAVES, (
        f"chain leaves diverged from Escalante VAM(1,2,2) projection "
        f"structure.\n  expected: {EXPECTED_CHAIN_LEAVES}\n  got: {paths}"
    )
    assert m1d._chain_M == 1
    assert m1d._chain_N_w == 2
    assert m1d._chain_N_p == 2


# ---------------------------------------------------------------------------
# April-2026 builder's verified Escalante DAE partition (6 dyn + 3 alg).
# ---------------------------------------------------------------------------

def test_april_builder_dae_partition():
    """``build_vam_pdesystem(1, 2, 2)`` is the verified Escalante-aligned
    PDESystem from the April-2026 work (commits 1d86c7c1 / 8b502ac3 /
    2be6db8c).  It must keep its DAE shape: 9 equations × 9 fields,
    6 evolution rows (mass + xmom_j0..1 + zmom_j0..2) and 3 algebraic
    rows (kbc_top_alg, kbc_bot, surface_bc).

    The walkthrough references this partition as the canonical shape
    Escalante writes; if the builder ever drifts the walkthrough's
    classification table goes stale.
    """
    sys.path.insert(0, str(REPO / "tutorials/vam"))
    sys.path.insert(0, str(REPO / "thesis/notebooks/verification/dae_solver"))
    try:
        from vam_pdesystem import build_vam_pdesystem  # type: ignore
        from test_dae_partition_bridge import dae_partition  # type: ignore
    finally:
        # Don't pollute the importer.
        sys.path[:2] = []

    s = build_vam_pdesystem(M=1, N_w=2, N_p=2, flat_bottom=True)
    assert len(s.equations) == 9
    assert len(s.fields) == 9
    assert s.equation_names == [
        "mass", "xmom_j0", "xmom_j1",
        "zmom_j0", "zmom_j1", "zmom_j2",
        "kbc_top_alg", "kbc_bot", "surface_bc",
    ]
    dyn, alg, _, _ = dae_partition(s)
    dyn_names = [s.equation_names[i] for i in dyn]
    alg_names = [s.equation_names[i] for i in alg]
    assert dyn_names == ["mass", "xmom_j0", "xmom_j1",
                         "zmom_j0", "zmom_j1", "zmom_j2"]
    assert alg_names == ["kbc_top_alg", "kbc_bot", "surface_bc"]
