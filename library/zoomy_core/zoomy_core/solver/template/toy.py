"""A toy template using EVERY statement kind -- worked example and gate fixture.

Model: 1-D scalar advection-reaction-diffusion on `n` cells of [0, 1]
(state `u` of kind `cell_state` with one row), parameters
`p = (a, k, nu, u_trouble)`:

    explicit:  u_t + a u_x = 0            (upwind, periodic)       Call
    implicit:  x + dt k x^3 = u           (cellwise Newton)        Root
    implicit:  (I - dt nu L) x = u        (L: Dirichlet-0 Laplacian) LinSolve

The march has an adaptive dt (depends on the state and on `a`), the D4 clamp
to `time_end`, a D9 `Guard` on dt, a `Select` that halves the explicit
increment where the state exceeds `u_trouble` (a MOOD-shaped redo), an
idempotent output `Callback`, a `returns` Callback, an observer-like
accumulator, a `Hook`, an `IfStatic`, a `Repeat` and a `ForEachFace`.
`stateful=True` wraps the time loop in a host window loop driven by a
STATEFUL callback (the coupling-window shape).

The bodies are written once over an array namespace `xp` (numpy or
jax.numpy).
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.solver.template.consts import ConstTable
from zoomy_core.solver.template.expr import (
    AnyTrue, IsFinite, ReduceSum, Where, ZerosLike)
from zoomy_core.solver.template.ir import (
    Assign, Call, Callback, ForEachFace, Guard, Hook, IfStatic, LinSolve,
    LinSolveOpts, Procedure, Repeat, Root, RootOpts, Select, Seq, While)
from zoomy_core.solver.template.registry import BLOCKS, BlockSpec

u, u0, r, dt, time, p = sp.symbols("u u0 r dt time p")
P = sp.IndexedBase("p")


def toy_registry():
    return BLOCKS.extended([
        BlockSpec("toy_dt", "u:cell_state p:params cfl:real", "dt:real",
                  comm="global", doc="adaptive CFL step of the toy"),
        BlockSpec("toy_rhs", "u:cell_state p:params time:real",
                  "r:cell_state troubled:cell_mask", comm="neighbor",
                  doc="upwind advection increment + a trouble mask"),
        BlockSpec("toy_lap", "x:cell_state", "y:cell_state", comm="neighbor",
                  doc="Dirichlet-0 Laplacian"),
        BlockSpec("toy_record", "time:real u:cell_state i:int", "",
                  role="callback", doc="idempotent recorder"),
        BlockSpec("toy_scale", "i:int", "fac:real", role="callback",
                  doc="deterministic host data (returns 1.0)"),
        BlockSpec("toy_window", "w:int time0:real time_end:real", "t_win:real",
                  role="callback", doc="stateful window end (counts calls)"),
    ])


def toy_constants(cfl=0.4, windows=2, n_cells=16):
    c = ConstTable()
    c.define("c_cfl", cfl, "real", "toy CFL")
    c.define("c_windows", windows, "int", "toy window count")
    c.define("c_ncell", n_cells, "int", "toy cell count (static)")
    c.define("c_root_tol", 1e-13, "real", "toy Newton tolerance")
    c.define("c_lin_tol", 1e-13, "real", "toy Krylov tolerance")
    return c


def residual_proc():
    x = sp.Symbol("x")
    return Procedure(
        "toy_reaction_residual",
        inputs="x:cell_state u:cell_state dt:real p:params",
        outputs="r:cell_state",
        body=[Assign("r", x + dt * P[1] * x**3 - u)],
        doc="R(x) = x + dt k x^3 - u (cellwise)")


def matvec_proc():
    x, lx = sp.symbols("x lx")
    return Procedure(
        "toy_diffusion_matvec",
        inputs="x:cell_state dt:real p:params",
        outputs="y:cell_state",
        body=[Call("toy_lap", ("x",), ("lx",)),
              Assign("y", x - dt * P[2] * lx)],
        doc="y = (I - dt nu L) x")


def step_proc(*, linsolve="krylov"):
    lin = LinSolveOpts(kind=linsolve, tol="c_lin_tol", restart=40, maxiter=40,
                       bandwidth=1 if linsolve == "banded" else None)
    return Procedure(
        "toy_step",
        inputs="u:cell_state p:params time:real dt:real",
        outputs="u:cell_state",
        body=[
            Assign("u0", u),
            Call("toy_rhs", ("u", "p", "time"), ("r", "troubled")),
            Assign("u", u0 + dt * r, note="explicit Euler"),
            Select(AnyTrue(sp.Symbol("troubled")),
                   then=[Assign("u", u0 + sp.Rational(1, 2) * dt * r,
                                note="redo with half the increment")],
                   note="MOOD-shaped redo"),
            Root(residual_proc(), x0="u", args=("u", "dt", "p"), out="u",
                 opts=RootOpts(tol="c_root_tol", maxiter=30, jac="ad",
                               linsolve=LinSolveOpts(kind="cellwise")),
                 stats=("newton_ok", "newton_res", "newton_it")),
            LinSolve(matvec_proc(), rhs="u", args=("dt", "p"), out="u",
                     opts=lin, stats=("lin_ok", "lin_res", "lin_it")),
            Hook("post_step", inputs=("u",), outputs=("u",), default=[]),
        ])


def march_proc(*, stateful=False, linsolve="krylov"):
    acc, acc_scale, it, t_end, t0 = sp.symbols(
        "acc acc_scale iteration time_end time0")
    s, w, fac = sp.symbols("s w fac")
    t_until = sp.Symbol("t_until")
    clamped = sp.Symbol("clamped")
    step = step_proc(linsolve=linsolve)
    inner = While(
        time < t_until,
        body=[
            Call("toy_dt", ("u", "p", "c_cfl"), ("dt",)),
            Assign("clamped", dt >= t_until - time),
            Assign("dt", sp.Min(dt, t_until - time), note="D4 clamp"),
            Guard(IsFinite(dt) & (dt > 0), "dt not admissible",
                  values=("dt", "time", "iteration"), note="D9"),
            Call(step, ("u", "p", "time", "dt"), ("u",)),
            Assign("time", Where(clamped, t_until, time + dt)),
            Assign("iteration", it + 1),
            Assign("acc", acc + acc_scale * dt * ReduceSum(u**2)),
            Callback("toy_record", ("time", "u", "iteration"),
                     kind="idempotent"),
            Callback("toy_scale", ("iteration",), ("fac",), kind="returns"),
            Assign("acc", acc * fac),
            Hook("observe", inputs=("acc", "u"), outputs=("acc",)),
        ])
    if stateful:
        loop = [
            Assign("window", sp.Integer(0)),
            While(sp.Symbol("window") < sp.Symbol("c_windows"), body=[
                Callback("toy_window", ("window", "time0", "time_end"),
                         ("t_until",), kind="stateful"),
                inner,
                Assign("window", sp.Symbol("window") + 1),
            ]),
        ]
    else:
        loop = [Assign("t_until", t_end), inner]
    body = [
        Assign("time", t0),
        Assign("iteration", sp.Integer(0)),
        Assign("acc", sp.Float(0.0)),
        Assign("t_until", t_end),
        IfStatic("double_acc", then=[Assign("acc_scale", sp.Float(2.0))],
                 otherwise=[Assign("acc_scale", sp.Float(1.0))]),
        Hook("init", inputs=("u",), outputs=("acc",), default=[]),
        Seq(loop),
        Repeat(2, index="k", body=[Assign("acc", acc + 0 * sp.Symbol("k"))]),
        Assign("w", ZerosLike(s)),
        ForEachFace("f", "c_ncell", body=[
            Assign("w[f]", 2 * sp.IndexedBase("s")[sp.Symbol("f")])]),
        Assign("acc", acc + 1e-3 * ReduceSum(w)),
    ]
    return Procedure(
        "toy_march",
        inputs="u:cell_state p:params s:cell_scalar time0:real time_end:real",
        outputs="u:cell_state time:real iteration:int acc:real",
        body=body)


def toy_bodies(xp, *, record=None, window_calls=None):
    """Bodies over `xp`.  `record` (list) receives idempotent writes,
    `window_calls` (list) the stateful window calls."""

    def toy_dt(u, p, cfl):
        dx = 1.0 / u.shape[-1]
        return cfl * dx / (xp.abs(p[0]) * (1.0 + xp.max(xp.abs(u))))

    def toy_rhs(u, p, time):
        dx = 1.0 / u.shape[-1]
        r = -p[0] * (u - xp.roll(u, 1, axis=-1)) / dx \
            + 0.1 * xp.sin(2.0 * xp.pi * time) * u
        return r, u[0] > p[3]

    def toy_lap(x):
        dx = 1.0 / x.shape[-1]
        z = xp.zeros_like(x[..., :1])
        left = xp.concatenate([z, x[..., :-1]], axis=-1)
        right = xp.concatenate([x[..., 1:], z], axis=-1)
        return (left - 2.0 * x + right) / dx**2

    def toy_record(time, u, i):
        if record is not None:
            import numpy as np
            record.append((float(time), np.asarray(u).copy(), int(i)))

    def toy_scale(i):
        import numpy as np
        return np.asarray(1.0) + 0.0 * np.asarray(i)

    def toy_window(w, time0, time_end):
        if window_calls is not None:
            window_calls.append(int(w))
        import numpy as np
        return np.asarray(float(time0) + (int(w) + 1) * (float(time_end) - float(time0))
                          / 2.0)

    return {"toy_dt": toy_dt, "toy_rhs": toy_rhs, "toy_lap": toy_lap,
            "toy_record": toy_record, "toy_scale": toy_scale,
            "toy_window": toy_window}


def toy_fills():
    """Hook fills of the toy (an observer-like plugin)."""
    acc, u = sp.symbols("acc u")
    return {"init": [("toy_init", [Assign("acc", acc + 0.0)])],
            "observe": [("toy_obs", [Assign("acc", acc + 0.0 * ReduceSum(u))])]}


def toy_inputs(xp, n=16, a=1.0, k=0.5, nu=0.01, trouble=1.15):
    import numpy as np
    x = (np.arange(n) + 0.5) / n
    u0 = 1.0 + 0.3 * np.exp(-60.0 * (x - 0.35) ** 2)
    return dict(u=xp.asarray(u0[None, :]), p=xp.asarray([a, k, nu, trouble]),
                s=xp.asarray(np.linspace(0.0, 1.0, n)))




# ── trivial bodies of the hyperbolic blocks (1-D periodic linear advection) ──


def advection_bodies(xp, *, order=2, limiter="none", record=None):
    """Bodies of the template's hyperbolic blocks for `u_t + a u_x = 0` on
    [0, 1], periodic, `p = (a, u_min)`: upwind flux (a > 0), piecewise-linear
    reconstruction at `order` 2 (`limiter` `none` = unlimited central
    slopes, which overshoot at steps, or `minmod`), admissibility
    `u >= u_min`.  `Qaux` holds du/dx.  The same physics is implemented in
    C++ by the C-walker gate."""

    def halo_bc(Q, Qaux, p, time, geom):
        return (xp.stack([Q[:, -1], Q[:, 0]], axis=-1),
                xp.stack([Qaux[:, -1], Qaux[:, 0]], axis=-1))

    def _ext(Q, bc):
        return xp.concatenate([bc[:, :1], Q, bc[:, 1:]], axis=-1)

    def reconstruct(Q, Qaux, bc_Q, bc_aux, p, geom, o1):
        dx = 1.0 / Q.shape[-1]
        e = _ext(Q, bc_Q)
        dl = (e[:, 1:-1] - e[:, :-2]) / dx
        dr = (e[:, 2:] - e[:, 1:-1]) / dx
        if order >= 2:
            if limiter == "minmod":
                g = xp.where(dl * dr > 0, xp.where(xp.abs(dl) < xp.abs(dr), dl, dr),
                             0.0)
            else:
                g = 0.5 * (dl + dr)
            g = xp.where(o1[None, :], 0.0, g)
        else:
            g = xp.zeros_like(Q)
        left = Q + 0.5 * dx * g           # value at the right face of each cell
        right = Q - 0.5 * dx * g          # value at the left face of each cell
        # periodic seam: the boundary faces see the reconstructed states of
        # the partner cells (conservative: face 0 and face n are one face)
        Q_L = xp.concatenate([left[:, -1:], left], axis=-1)
        Q_R = xp.concatenate([right, right[:, :1]], axis=-1)
        A_L = xp.concatenate([bc_aux[:, :1], Qaux], axis=-1)
        A_R = xp.concatenate([Qaux, bc_aux[:, 1:]], axis=-1)
        return Q_L, Q_R, A_L, A_R, g[:, None, :]

    def flux_pass(Q_L, Q_R, Qaux_L, Qaux_R, p, time, geom):
        a = p[0]
        F = xp.where(a > 0, a * Q_L, a * Q_R)
        Z = xp.zeros_like(F)
        return F, Z, Z

    def cell_ncp(Q, grad, Qaux, p, geom):
        return xp.zeros_like(Q)

    def gather_update(Q, F, Dp, Dm, cell_term, Qaux, p, time, dt, geom):
        dx = 1.0 / Q.shape[-1]
        Qc = Q - dt / dx * (F[:, 1:] - F[:, :-1] + Dm[:, 1:] + Dp[:, :-1]) \
            + dt * cell_term
        bad = xp.logical_or(Qc[0] < p[1],
                            xp.logical_not(xp.all(xp.isfinite(Qc), axis=0)))
        return Qc, bad

    def compute_dt(Q, Qaux, p, geom, cfl, dimension, degree_factor, dt_max):
        dx = 1.0 / Q.shape[-1]
        lam = xp.abs(p[0])
        dt = cfl * 2.0 * (0.5 * dx) / (dimension * degree_factor * lam)
        return xp.minimum(dt, dt_max)

    def update_aux(Q, Qaux, p, geom, time):
        dx = 1.0 / Q.shape[-1]
        return (xp.roll(Q, -1, axis=-1) - xp.roll(Q, 1, axis=-1)) / (2.0 * dx)

    def update_variables(Q, Qaux, p, geom, time, dt):
        return Q

    def write_fields(Q, Qaux, time, i_snapshot):
        if record is not None:
            import numpy as np
            record.append((int(i_snapshot), float(time), np.asarray(Q).copy()))

    return {"halo_bc": halo_bc, "reconstruct": reconstruct,
            "flux_pass": flux_pass, "cell_ncp": cell_ncp,
            "gather_update": gather_update, "compute_dt": compute_dt,
            "update_aux": update_aux, "update_variables": update_variables,
            "write_fields": write_fields}


def advection_state(xp, n=40):
    """A step profile (u = 1 on [0.25, 0.5), 0 elsewhere) plus a smooth bump."""
    import numpy as np
    x = (np.arange(n) + 0.5) / n
    u = np.where((x >= 0.25) & (x < 0.5), 1.0, 0.0) + 0.2 * np.sin(2 * np.pi * x) ** 2
    Q = u[None, :]
    Qaux = (np.roll(Q, -1, axis=-1) - np.roll(Q, 1, axis=-1)) * n / 2.0
    return xp.asarray(Q), xp.asarray(Qaux)


__all__ = ["toy_registry", "toy_constants", "march_proc", "step_proc",
           "residual_proc", "matvec_proc", "toy_bodies", "toy_fills",
           "toy_inputs", "advection_bodies", "advection_state"]
