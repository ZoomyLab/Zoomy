"""Plain-Python solver template: the code IS the template.

Same semantics as the IR template (SSPRK-k Shu-Osher stages with stage times,
D1 MOOD ladder, D4 clamp, D9 guard, drift-free output with initial + final
snapshot).  ``B`` is the backend primitive object (numpy / jax-fwd / jax-rev);
``k`` holds the kernels (block bodies).  No IR, no walker, no strings.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ShuOsher:
    stages: tuple            # ((a_k, c_k), ...): Q = a Q0 + (1-a) stage(Q), t_k = t + c dt


SSPRK2 = ShuOsher(((0.0, 0.0), (0.5, 1.0)))
MOOD_FAILED = ("MOOD (D1): the whole-step order-1 candidate is not admissible "
               "-- an implementation bug, order 1 is admissible under the CFL")


def explicit_stage(B, k, Q, Qaux, p, geom, t, dt, o1):
    bc_Q, bc_aux = k.halo_bc(Q, Qaux, p, t, geom)
    Q_L, Q_R, A_L, A_R, grad = k.reconstruct(Q, Qaux, bc_Q, bc_aux, p, geom, o1)
    F, Dp, Dm = k.flux_pass(Q_L, Q_R, A_L, A_R, p, t, geom)
    cell = k.cell_ncp(Q, grad, Qaux, p, geom) if k.cell_ncp else B.xp.zeros_like(Q)
    return k.gather_update(Q, F, Dp, Dm, cell, Qaux, p, t, dt, geom)


def rk(B, k, scheme, stage, Q, Qaux, p, geom, t, dt, o1):
    Q0, troubled = Q, None
    for i, (a, c) in enumerate(scheme.stages):
        t_stage = c * dt + t
        if i > 0:
            Qaux = k.update_aux(Q, Qaux, p, geom, t_stage)
        Q_cand, tr = stage(B, k, Q, Qaux, p, geom, t_stage, dt, o1)
        troubled = tr if troubled is None else B.xp.logical_or(troubled, tr)
        Q = Q_cand if a == 0.0 else Q0 * a + Q_cand * (1 - a)
    return Q, Qaux, troubled


def mood_step(B, k, scheme, stage, Q, Qaux, p, geom, t, dt):
    """D1: all cells at full order -> troubled cells at order 1 -> all at order 1."""
    xp, n = B.xp, Q.shape[-1]
    run = lambda o1: rk(B, k, scheme, stage, Q, Qaux, p, geom, t, dt, o1)

    def all_order1():
        Q3, A3, tr3 = run(xp.full((n,), True))
        (Q3,) = B.guard(xp.logical_not(B.reduce_any(tr3)), MOOD_FAILED,
                        ("time", "dt"), (t, dt), (Q3,))
        return Q3, A3

    def troubled_order1(tr):
        Q2, A2, tr2 = run(tr)
        return B.cond(B.reduce_any(tr2), all_order1, lambda: (Q2, A2), ())

    Q1, A1, tr1 = run(xp.full((n,), False))
    Q, Qaux = B.cond(B.reduce_any(tr1), troubled_order1, lambda _: (Q1, A1), (tr1,))
    Q = k.update_variables(Q, Qaux, p, geom, dt + t, dt)
    return Q, k.update_aux(Q, Qaux, p, geom, dt + t)


def march(B, k, step, Q, Qaux, p, geom, time0, time_end, *, cfl, dimension,
          degree, dt_max=float("inf"), snapshots=0, write=None, eps=1e-12):
    """The time loop.  ``step(Q, Qaux, p, geom, t, dt) -> (Q, Qaux)``."""
    xp = B.xp
    interval = (time_end - time0) / max(snapshots - 1, 1)

    def output(Q, Qaux, t, i):
        if write is not None:
            B.callback_idempotent(write, (Q, Qaux, t, i))
        return i + 1, t

    i_snap, t_last = output(Q, Qaux, time0, 0) if snapshots else (0, -xp.inf)

    def body(c):
        Q, Qaux, t, it, i_snap, t_last = c
        dt = k.compute_dt(Q, Qaux, p, geom, cfl, dimension, 2 * degree + 1, dt_max)
        clamped = dt >= time_end - t
        dt = xp.minimum(dt, time_end - t)
        (dt,) = B.guard(xp.logical_and(xp.isfinite(dt), dt > 0),
                        "dt not admissible (D9)", ("dt", "time"), (dt, t), (dt,))
        Q, Qaux = step(Q, Qaux, p, geom, t, dt)
        t = xp.where(clamped, time_end, dt + t)
        if snapshots:
            stamp = time0 + interval * i_snap
            i_snap, t_last = B.cond(eps * (xp.abs(stamp) + 1) + t >= stamp,
                                    output, lambda Q, A, t, i: (i, t_last),
                                    (Q, Qaux, t, i_snap))
        return Q, Qaux, t, it + 1, i_snap, t_last

    Q, Qaux, t, it, i_snap, t_last = B.while_loop(
        lambda c: c[2] < time_end, body, (Q, Qaux, time0, 0, i_snap, t_last))
    if snapshots:
        i_snap, t_last = B.cond(t > t_last, output,
                                lambda Q, A, t, i: (i, t_last), (Q, Qaux, t, i_snap))
    return Q, Qaux, t, it, i_snap
