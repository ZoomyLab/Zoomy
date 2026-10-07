"""Gate 1a-1 (core): authoring classes -- schemes, stages, plugins, builders,
Solver.

* tableau data: Shu-Osher stage times = Butcher row sums, classical order
  conditions; ARS232 / ARS343 IMEX order conditions to roundoff;
* the assembled step / march have the D1 / D3 / D4 / D7 / D9 structure;
* the template on numpy is BITWISE equal to an independent hand-written numpy
  march (SSPRK2, D1 MOOD ladder, D4 clamp, drift-free output cadence with the
  initial and final snapshot) on a case where MOOD fires;
* D3 stage times: temporal order 1 / 2 / 3 for EULER / SSPRK2 / SSPRK3 on a
  time-dependent ODE (a stage-time drift would collapse the order), and the
  Strang split keeps per-operator clocks.
"""
import numpy as np
import pytest
import sympy as sp

from zoomy_core.solver.template import (
    ARS232, ARS343, BLOCKS, EULER, SSPRK2, SSPRK3, BlockSpec, Call,
    CompositeStage, Guard, Hyperbolic, Imex, Observer, Output, Procedure,
    Select, Setup, Solver, Stage, While, strang, lie)
from zoomy_core.solver.template.ir import iter_statements
from zoomy_core.solver.template.toy import advection_bodies, advection_state

pytestmark = pytest.mark.template_gate


# ── scheme data ─────────────────────────────────────────────────────────────


def test_tableaus():
    for sc, order in ((EULER, 1), (SSPRK2, 2), (SSPRK3, 3)):
        assert sc.check_consistency()
        A, b, c = sc.butcher()
        assert abs(b.sum() - 1) < 1e-15
        if order >= 2:
            assert abs(b @ c - 0.5) < 1e-15
        if order >= 3:
            assert abs(b @ (c * c) - 1 / 3) < 1e-15
            assert abs(b @ A @ c - 1 / 6) < 1e-15
    assert SSPRK3.c == (0.0, 1.0, 0.5)
    for tab, order in ((ARS232, 2), (ARS343, 3)):
        res = tab.order_conditions()
        assert max(res.values()) < 1e-14, res
        assert tab.order == order
        Ai = np.asarray(tab.implicit.A)
        assert np.allclose(Ai[-1], tab.implicit.b)     # stiffly accurate
    assert Imex().kind == "lie" and Imex(kind="ark").tableau is ARS232


# ── structure of the assembled template ─────────────────────────────────────


def test_step_structure_d1_d3_d7():
    s = Solver(plugins=[Output(snapshots=3)])
    march, ctx = s.template(order=2)
    step = next(st.callee for st in iter_statements(march.body)
                if isinstance(st, Call) and isinstance(st.callee, Procedure))
    body = step.body
    calls = [st for st in body if isinstance(st, Call)]
    stage_calls = [c for c in calls if c.callee_name == "explicit_stage"]
    assert len(stage_calls) == 2
    assert [c.args[4] for c in stage_calls] == ["t_stage", "t_stage"]
    t_stage = [str(st.expr) for st in body
               if getattr(st, "target", None) == "t_stage"]
    assert t_stage == ["time", "c_ssprk2_c1*dt + time"]           # D3
    aux = [c for c in calls if c.callee_name == "update_aux"]
    assert aux[0].args[-1] == "t_stage" and aux[-1].args[-1] == "t_new"  # D7
    mood = [st for st in body if isinstance(st, Select)]
    assert len(mood) == 1                                          # D1 rung 1
    inner = [st for st in mood[0].then if isinstance(st, Select)]
    assert len(inner) == 1                                         # D1 rung 2
    assert isinstance(inner[0].then[-1], Guard)                    # D1 final
    loop = next(st for st in march.body if isinstance(st, While))
    assert any(isinstance(st, Guard) and "D9" in st.note for st in loop.body)
    # order 1: no repair rungs, the guard stays
    step1 = next(st.callee for st in iter_statements(
        s.template(order=1)[0].body)
        if isinstance(st, Call) and isinstance(st.callee, Procedure))
    assert not any(isinstance(st, Select) for st in step1.body)
    assert any(isinstance(st, Guard) for st in step1.body)


# ── end to end vs an independent numpy march ────────────────────────────────


class Adv(Solver):
    name = "adv"

    def __init__(self, n=40, limiter="none", record=None, calls=None, **kw):
        super().__init__(**kw)
        self.n, self.limiter, self.record, self.calls = n, limiter, record, calls

    def setup_bodies(self, mesh, nsm, backend):
        xp = backend.xp
        Q, Qaux = advection_state(xp, self.n)
        b = advection_bodies(xp, order=2, limiter=self.limiter, record=self.record)
        if self.calls is not None:
            gu = b["gather_update"]

            def counted(*a):
                self.calls.append(1)
                return gu(*a)
            b["gather_update"] = counted
        return Setup(bodies=b, flags={"cell_ncp": False}, geom=None, Q=Q,
                     Qaux=Qaux, parameters=xp.asarray([1.0, 0.0]), dimension=1,
                     degree=1, order=2)


def reference_march(Q, Qaux, p, t_end, cfl, snapshots, n):
    """Hand-written SSPRK2 + D1 MOOD + D4 clamp + output cadence (numpy)."""
    b = advection_bodies(np, order=2, limiter="none")
    a1, c1, eps = 0.5, 1.0, 1e-12
    writes, calls = [], [0]

    def stage(Q, Qaux, t, dt, o1):
        calls[0] += 1
        bc = b["halo_bc"](Q, Qaux, p, t, None)
        QL, QR, AL, AR, g = b["reconstruct"](Q, Qaux, bc[0], bc[1], p, None, o1)
        F, Dp, Dm = b["flux_pass"](QL, QR, AL, AR, p, t, None)
        return b["gather_update"](Q, F, Dp, Dm, np.zeros_like(Q), Qaux, p, t, dt,
                                  None)

    def rk(Q, Qaux, time, dt, o1):
        Q0 = Q
        Qc, tr = stage(Q, Qaux, time, dt, o1)
        Q = Qc
        ts = c1 * dt + time
        Qaux = b["update_aux"](Q, Qaux, p, None, ts)
        Qc, tk = stage(Q, Qaux, ts, dt, o1)
        return Q0 * a1 + Qc * (1 - a1), Qaux, np.logical_or(tr, tk)

    time, it, i_snap = 0.0, 0, 0
    wi = (-time + t_end) / (snapshots - 1)
    t_ref, i_ref = time, i_snap
    writes.append((i_snap, time, Q.copy()))
    i_snap += 1
    t_last = time
    while time < t_end:
        dt = b["compute_dt"](Q, Qaux, p, None, cfl, 1, 3, np.inf)
        clamped = dt >= t_end - time
        dt = np.minimum(dt, t_end - time)
        assert np.isfinite(dt) and dt > 0
        Q0, A0 = Q, Qaux
        Q, Qaux, tr = rk(Q0, A0, time, dt, np.zeros(n, bool))
        if tr.any():
            Q, Qaux, tr = rk(Q0, A0, time, dt, tr)
            if tr.any():
                Q, Qaux, tr = rk(Q0, A0, time, dt, np.ones(n, bool))
                assert not tr.any()
        Qaux = b["update_aux"](Q, Qaux, p, None, dt + time)
        time = t_end if clamped else dt + time
        it += 1
        stamp = t_ref + wi * (-i_ref + i_snap)
        if eps * (abs(stamp) + 1) + time >= stamp:
            writes.append((i_snap, time, Q.copy()))
            i_snap += 1
            t_last = time
    if time > t_last:
        writes.append((i_snap, time, Q.copy()))
        i_snap += 1
    return Q, Qaux, time, it, i_snap, writes, calls[0]


def test_template_equals_independent_reference_with_mood():
    rec, calls = [], []
    obs = Observer(init=lambda Q, Qaux, p: np.max(Q),
                   update=lambda acc, t, dt, Q, Qaux, p: np.maximum(acc, np.max(Q)))
    s = Adv(time_end=0.25, cfl=0.9, plugins=[Output(snapshots=5), obs],
            record=rec, calls=calls)
    Q0, A0 = s.setup_simulation(None, None)
    Q, Qaux, acc = s.run_observed(Q0, A0)
    out = s.last_outputs
    Qr, Ar, tr, itr, isr, wr, cr = reference_march(
        np.asarray(Q0), np.asarray(A0), np.asarray([1.0, 0.0]), 0.25, 0.9, 5, 40)
    assert out["time"] == 0.25 == tr                          # D4: lands exactly
    assert out["iteration"] == itr == 34
    assert np.array_equal(Q, Qr) and np.array_equal(Qaux, Ar)  # bitwise
    assert len(calls) == cr == 106 > 2 * itr                  # MOOD fired
    assert out["i_snapshot"] == isr == 5
    assert [(i, t) for i, t, _ in rec] == [(i, t) for i, t, _ in wr]
    assert rec[0][1] == 0.0 and rec[-1][1] == 0.25            # initial + final
    assert all(np.array_equal(a[2], b_[2]) for a, b_ in zip(rec, wr))
    assert np.min(Q) >= 0.0                                   # admissible
    assert abs(Q.sum() - np.asarray(Q0).sum()) < 1e-13        # conservative
    assert acc == pytest.approx(1.2016289507889129, abs=0)    # observer max


def test_output_interval_mode_and_chunked_restart():
    rec = []
    s = Adv(time_end=0.2, cfl=0.9, plugins=[Output()], record=rec)
    Q0, A0 = s.setup_simulation(None, None)
    Q1, A1 = s.run_simulation(Q0, A0, write_interval=0.05, i_snapshot=0)
    i1 = s.last_outputs["i_snapshot"]
    # continue the run from time0 = 0.2 to 0.3 with the snapshot counter
    Q2, A2 = s.run_simulation(Q1, A1, time0=0.2, time_end=0.3,
                              write_interval=0.05, i_snapshot=i1)
    assert s.last_outputs["time"] == 0.3
    idx = [r[0] for r in rec]
    assert idx[:i1] == list(range(i1))
    times = [r[1] for r in rec]
    assert times[0] == 0.0 and 0.2 in times and times[-1] == 0.3


# ── D3: temporal order on a time-dependent ODE ──────────────────────────────

ODE_BLOCKS = BLOCKS.extended([
    BlockSpec("ode_stage", "Q:cell_state p:params time:real dt:real",
              "Q_cand:cell_state troubled:cell_mask"),
    BlockSpec("zero_stage", "Q:cell_state", "Q_cand:cell_state troubled:cell_mask"),
])


class OdeStage(Stage):
    name = "ode_stage_proc"

    def build(self, ctx):
        return [Call("ode_stage", ("Q", "p", "time", "dt"), ("Q_cand", "troubled"))]


class ZeroStage(Stage):
    name = "zero_stage_proc"

    def build(self, ctx):
        return [Call("zero_stage", ("Q",), ("Q_cand", "troubled"))]


def ode_bodies():
    def ode_stage(Q, p, time, dt):              # u' = cos(2 pi t) u
        return Q + dt * np.cos(2 * np.pi * time) * Q, np.zeros(Q.shape[-1], bool)

    def zero_stage(Q):
        return Q, np.zeros(Q.shape[-1], bool)

    return {
        "ode_stage": ode_stage, "zero_stage": zero_stage,
        "compute_dt": lambda Q, Qaux, p, g, cfl, d, k, dt_max: np.minimum(1.0, dt_max),
        "update_aux": lambda Q, Qaux, p, g, t: Qaux,
        "update_variables": lambda Q, Qaux, p, g, t, dt: Q,
    }


def ode_error(scheme, n, *, stages=None):
    s = Solver(scheme=scheme, stages=stages or [OdeStage()], registry=ODE_BLOCKS,
               dt_max=1.0 / n, mood="off", name="ode")
    g = s.compile(ode_bodies(), flags={"aux_from_state": False}, dimension=1,
                  degree=0, order=1)
    Q0 = np.ones((1, 3))
    Q, Qaux, t, it = g.entry(Q0, np.zeros((1, 3)), np.zeros(1), None, 0.0, 0.9)
    exact = np.exp(np.sin(2 * np.pi * 0.9) / (2 * np.pi))
    return abs(Q[0, 0] - exact), it


@pytest.mark.parametrize("scheme,order", [(EULER, 1), (SSPRK2, 2), (SSPRK3, 3)])
def test_d3_stage_times_temporal_order(scheme, order):
    e1, it1 = ode_error(scheme, 160)
    e2, it2 = ode_error(scheme, 320)
    rate = np.log2(e1 / e2)
    assert abs(it2 - 2 * it1) <= 2                    # (+1 if a roundoff tail step)
    # measured: euler 0.992, ssprk2 1.966, ssprk3 3.000
    assert rate > order - 0.1, (scheme.name, e1, e2, rate)


def test_strang_split_keeps_per_operator_clocks():
    A, Z = (SSPRK2, OdeStage()), (SSPRK2, ZeroStage())
    errs = [ode_error(strang(A, Z), n)[0] for n in (160, 320)]
    assert np.log2(errs[0] / errs[1]) > 1.85, errs
    errs = [ode_error(lie(A, Z), n)[0] for n in (160, 320)]
    assert np.log2(errs[0] / errs[1]) > 1.85, errs
    # a composite stage (A then Z inside one RK stage) is still order 2
    errs = [ode_error(SSPRK2, n, stages=[CompositeStage([OdeStage(), ZeroStage()],
                                                        refresh_aux=False)])[0]
            for n in (160, 320)]
    assert np.log2(errs[0] / errs[1]) > 1.85, errs
