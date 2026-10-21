"""E4 — dry/rest SOURCE SEMANTICS: cap text vs cap MEANING (spec §1b/§2 rank 5).

A golden cannot distinguish a re-baselined (forbidden) h-floor from the
mandated momentum-only cap, nor see the Manning 0/0 at rest.  These pin the
runtime meaning on the classes that CARRY the cap (the hand-built ``SWE`` /
``MalpassetSWE`` — the derived SME(0) golden has no cap; see the flagged spec
gap in the refactor report).

Merged from: test_swe_wetdry_cap_req176.py + test_swe_manning_regularized_
req166.py + the friction-decay march from test_sme_dambreak.py.

* cap rows RAW (h-floor HARD-forbidden, user mandate) + promotion +
  Malpasset U_MAX byte-pin;
* cap bites near-dry + wet identity + lake-at-rest invariance;
* Manning Jacobian finite at rest (REQ-166, vel_eps regularizer);
* [large] friction decays uniform flow to the analytic 0.1*e^-2.5
  (the pre-fix sign grew it to ~1.12).
"""
import math

import numpy as np
import pytest
import sympy as sp

from zoomy_core.systemmodel import SystemModel
from zoomy_core.model.models.swe import SWE
from zoomy_core.model.models.malpasset import MalpassetSWE

pytestmark = [pytest.mark.model]

G = 9.81
EPS = 1e-2


def _eval_update(model):
    """Return a callable ``state_dict -> row values`` for the model's
    ``update_variables`` column, substituting g and wet_dry_eps."""
    rows = [sp.sympify(e) for e in sp.flatten(model.update_variables())]
    names = [str(s) for s in model.variables.get_list()]

    def run(**state):
        env = {"g": G, "wet_dry_eps": EPS, **state}
        out = []
        for e in rows:
            subs = {p: env[str(p)] for p in e.free_symbols}
            out.append(float(sp.N(e.subs(subs))))
        return dict(zip(names, out))

    return run


def _bound(h):
    return max(h - EPS, 0.0) * 4.0 * math.sqrt(G * max(h, 0.0))


@pytest.mark.small
@pytest.mark.gate
def test_cap_rows_raw_symbols():
    """USER MANDATE: no path may clamp/floor/truncate h (or b) — the h/b rows
    of ``update_variables`` are the RAW state symbols on EVERY state.
    ``Max(h,0)`` may appear only INSIDE the momentum bound.  The SystemModel
    promotion carries the cap, and MalpassetSWE keeps its own U_MAX=30 formula
    byte-identical (the pin that used to live on the deleted req176 file)."""
    for dim in (1, 2):
        m = SWE(dimension=dim)
        v = m.variables
        rows = list(sp.flatten(m.update_variables()))
        names = [str(s) for s in v.get_list()]
        row = dict(zip(names, rows))
        assert row["b"] is v.b        # raw symbol, not a rewritten expression
        assert row["h"] is v.h
        assert not any(sp.sympify(r).has(v.b) for k, r in row.items() if k != "b")

    # promotion: the SystemModel carries the celerity cap on the momentum rows
    sm = SystemModel.from_model(SWE(dimension=2))
    assert sm.update_variables is not None
    rows = [str(e) for e in sp.flatten(sm.update_variables)]
    assert rows[0] == "b" and rows[1] == "h"
    assert "sqrt(g)" in rows[2] and "Min(hu" in rows[2]
    assert "sqrt(g)" in rows[3] and "Min(hv" in rows[3]

    # MalpassetSWE keeps its OWN U_MAX=30 cap (no g) + zero jacobian + KP hinv
    mal = MalpassetSWE()
    uv = sp.flatten(mal.update_variables())
    assert all("g" not in [str(s) for s in sp.sympify(e).free_symbols] for e in uv)
    assert any("30.0" in str(e) for e in uv)
    assert all(e == 0 for e in sp.flatten(mal.update_variables_jacobian_wrt_variables()))
    aux = str(sp.flatten(mal.update_aux_variables())[0])
    assert "sqrt(2)" in aux and "Max(h, wet_dry_eps)" in aux


@pytest.mark.small
@pytest.mark.gate
def test_cap_bites_and_rest_invariant():
    """(a) wet identity — capped momentum == raw on wet flow, h/b bit-exact;
    (b) near-dry bound — |capped hu| <= h_wet*4*sqrt(g*h), and the cap actually
    BIT (raw momentum far outside the band) while h/b stay untouched (cap is
    NOT an h-floor); (c) lake-at-rest invariance (well-balanced)."""
    # (a) wet identity
    for dim in (1, 2):
        run = _eval_update(SWE(dimension=dim))
        st = {"b": 1.5, "h": 10.0, "hu": 3.0}
        if dim == 2:
            st["hv"] = -2.0
        out = run(**st)
        assert out["b"] == st["b"] and out["h"] == st["h"]
        assert math.isclose(out["hu"], st["hu"], rel_tol=1e-12)
        if dim == 2:
            assert math.isclose(out["hv"], st["hv"], rel_tol=1e-12)

    # (b) near-dry bound bites
    run = _eval_update(SWE(dimension=2))
    h = 2.0 * EPS
    out = run(b=0.0, h=h, hu=1e4, hv=-1e4)
    bnd = _bound(h)
    assert abs(out["hu"]) <= bnd + 1e-9 and abs(out["hv"]) <= bnd + 1e-9
    assert math.isclose(out["hu"], bnd, rel_tol=1e-9)
    assert math.isclose(out["hv"], -bnd, rel_tol=1e-9)
    assert out["h"] == h and out["b"] == 0.0     # NEVER an h-floor

    # (c) lake at rest — bit-exact no-op
    for dim in (1, 2):
        run = _eval_update(SWE(dimension=dim))
        st = {"b": 3.0, "h": 5.0, "hu": 0.0}
        if dim == 2:
            st["hv"] = 0.0
        out = run(**st)
        for k, v in st.items():
            assert out[k] == v


def _manning_jac_nonfinite_at_rest(dim):
    sm = SystemModel.from_model(
        SWE(dimension=dim, parameters={"g": 9.81, "n_m": 0.033}))
    J = sm.source_jacobian_wrt_variables
    subs = {}
    for s in sm.state:
        subs[s] = 2.0 if str(s) == "h" else 0.0            # rest: hu=hv=0
    for a in sm.aux_state:
        subs[a] = 0.5 if str(a) == "hinv" else 0.0
    for row in range(J.shape[0]):
        for col in range(J.shape[1]):
            e = sp.sympify(J[row, col])
            if e == 0:
                continue
            for p in e.free_symbols:
                nm = str(p)
                subs.setdefault(p, {"g": 9.81, "n_m": 0.033,
                                    "vel_eps": 1e-12}.get(nm, 0.0))
            val = complex(sp.N(e.subs(subs)))
            if not np.isfinite(val.real):
                return True
    return False


@pytest.mark.small
@pytest.mark.gate
@pytest.mark.parametrize("dim", [1, 2])
def test_manning_jacobian_finite_at_rest(dim):
    """REQ-166: ``hu*sqrt(hu^2+hv^2)`` has a 0/0 derivative at rest which NaNs
    the implicit/IMEX Newton on step 0 — the vel_eps regularizer keeps
    dS/dQ finite everywhere."""
    assert not _manning_jac_nonfinite_at_rest(dim)


@pytest.mark.large
def test_friction_decays_to_analytic():
    """Uniform flow h=0.2, q_0=0.1, lambda_s=0.5 => q_0(t=1) = 0.1*e^-2.5.
    The wrong (pre-fix) sign GROWS this to ~1.12 — far outside any tolerance.
    (The friction-decay slice of the deleted test_sme_dambreak.py.)"""
    from zoomy_core.model.models import SME
    from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
    from zoomy_core.model.boundary_conditions import (
        BoundaryConditions, Extrapolation)
    from zoomy_core.model.initial_conditions import Constant
    from zoomy_core.mesh import BaseMesh
    import zoomy_core.fvm.timestepping as timestepping
    from zoomy_core.fvm.solver_numpy import HyperbolicSolver
    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec

    sm = SystemModel.from_model(SME(
        closures=[Newtonian(), NavierSlip(), StressFree()], level=0,
        parameters={"lambda_s": 0.5},
        boundary_conditions=BoundaryConditions(
            [Extrapolation(tag="left"), Extrapolation(tag="right")])))
    ic = np.array([0.0, 0.2, 0.1])
    sm.initial_conditions = Constant(constants=lambda n, v=ic: v)
    sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
    mesh = BaseMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=20)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))
    solver = HyperbolicSolver(time_end=1.0,
                              compute_dt=timestepping.adaptive(CFL=0.9))
    Q, _ = solver.solve(mesh, nsm, write_output=False)
    q0 = np.asarray(Q[2, :20], dtype=float)
    assert q0.std() < 1e-12, "uniform flow stopped being uniform"
    assert abs(q0.mean() - 0.1 * np.exp(-2.5)) < 1e-3
