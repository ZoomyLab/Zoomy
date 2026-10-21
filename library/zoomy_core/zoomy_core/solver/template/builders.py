"""The step and time-loop builders: the template written ONCE.

`build_step`       one step at a frozen dt: Shu-Osher stages unrolled with
                     stage times t + c_k dt (D3), aux refreshed between
                     stages (D7), MOOD per D1 as a `Select` that re-enters
                     the stage sequence (troubled cells at order 1, then the
                     whole step at order 1, then a fatal `Guard`), the
                     `post_step` hook (default: update_variables) and the
                     end-of-step aux update.
`build_time_loop`  the march: dt from the D4 law, clamped to land exactly
                     on `t_until` (= the traced `time_end`), D9 `Guard`
                     on dt, the step, observer / output hooks, initial and
                     final snapshot hooks.  `time0` and `time_end` are
                     traced inputs, never baked into the compiled code.

Blocks are referenced by NAME; their bodies are per backend.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.solver.template.expr import AnyTrue, IsFinite, MaskLike, Where
from zoomy_core.solver.template.ir import (
    Assign, Call, Guard, Hook, IfStatic, Procedure, Select, While)
from zoomy_core.solver.template.kinds import ports
from zoomy_core.solver.template.scheme import ShuOsher
from zoomy_core.solver.template.stage import BuildContext, Stage

Q, Qaux, Q0, Qaux0, Q_cand = sp.symbols("Q Qaux Q0 Qaux0 Q_cand")
time, dt, troubled, troubled_k = sp.symbols("time dt troubled troubled_k")

STEP_INPUTS = "Q:cell_state Qaux:cell_aux p:params geom:geom time:real dt:real"
STEP_OUTPUTS = "Q:cell_state Qaux:cell_aux"

MOOD_MODES = ("d1", "guard", "off")


def scheme_constants(scheme: ShuOsher, ctx: BuildContext, prefix=None):
    """Define the tableau coefficients as emitted constants; returns
    `[(a_sym|None, c_sym|None, beta_sym|None), ...]` (None where the
    coefficient is the identity value and the statement is simplified)."""
    prefix = prefix or f"c_{scheme.name}"
    out = []
    for k, (a, c) in enumerate(scheme.stages):
        b = scheme.beta[k]
        a_s = (ctx.constants.define(f"{prefix}_a{k}", a, "real",
                                    f"{scheme.name} Shu-Osher weight of Q^n")
               if a != 0.0 else None)
        c_s = (ctx.constants.define(f"{prefix}_c{k}", c, "real",
                                    f"{scheme.name} stage time fraction (D3)")
               if c != 0.0 else None)
        b_s = (ctx.constants.define(f"{prefix}_b{k}", b, "real",
                                    f"{scheme.name} stage dt scale")
               if b != 1.0 else None)
        out.append((a_s, c_s, b_s))
    return out


def stage_sequence(scheme: ShuOsher, stage_proc: Procedure, ctx: BuildContext,
                   o1: str = "o1") -> list:
    """The unrolled Shu-Osher stages; binds `Q`, `Qaux` and `troubled`."""
    coeffs = scheme_constants(scheme, ctx)
    body = []
    for k, (a_s, c_s, b_s) in enumerate(coeffs):
        body.append(Assign("t_stage", time if c_s is None else time + c_s * dt,
                           note=f"stage {k}: t + c_{k} dt (D3)"))
        dt_name = "dt"
        if b_s is not None:
            body.append(Assign("dt_stage", b_s * dt))
            dt_name = "dt_stage"
        if k > 0:
            body.append(Call("update_aux", ("Q", "Qaux", "p", "geom", "t_stage"),
                             ("Qaux",), note=f"stage {k}: aux refresh (D7)"))
        tr = "troubled" if k == 0 else "troubled_k"
        body.append(Call(stage_proc, ("Q", "Qaux", "p", "geom", "t_stage",
                                      dt_name, o1), ("Q_cand", tr),
                         note=f"stage {k}"))
        if k > 0:
            body.append(Assign("troubled", sp.Or(troubled, troubled_k)))
        if a_s is None:
            body.append(Assign("Q", Q_cand))
        else:
            body.append(Assign("Q", a_s * Q0 + (1 - a_s) * Q_cand,
                               note=f"Shu-Osher combination, stage {k}"))
    return body


D1_GUARD_MSG = ("MOOD (D1): the whole-step order-1 candidate is not admissible "
                "-- an implementation bug, order 1 is admissible under the CFL")


def build_step(scheme: ShuOsher, stage: Stage, ctx: BuildContext, *,
               mood: str = "d1", repair=None, name: str = "step") -> Procedure:
    """One step at a frozen `dt`.

    `mood`: `d1` (default; D1 ladder when `repair`), `guard` (no
    repair, a troubled step is fatal), `off` (no detection).  `repair`
    defaults to `stage.supports_o1` and the `recon_order >= 2` build
    option: at order 1 the order-1 rungs would recompute the same step, so
    the ladder reduces to the guard."""
    if mood not in MOOD_MODES:
        raise ValueError(f"mood must be one of {MOOD_MODES}")
    if not isinstance(scheme, ShuOsher):
        raise TypeError(f"build_step lowers ShuOsher schemes, got {scheme!r}")
    if repair is None:
        repair = stage.supports_o1 and int(ctx.options.get("recon_order", 2)) >= 2
    sp_ = stage.procedure(ctx)
    body = [Assign("Q0", Q, note="step base: Shu-Osher average and MOOD rollback"),
            Assign("Qaux0", Qaux),
            Assign("o1", MaskLike(Q, False), note="no cell forced to order 1")]
    body += stage_sequence(scheme, sp_, ctx, "o1")

    guard = Guard(sp.Not(AnyTrue(troubled)), D1_GUARD_MSG, values=("time", "dt"),
                  note="D1 final rung")
    if mood == "d1" and repair:
        rung2 = [Assign("Q", Q0), Assign("Qaux", Qaux0),
                 Assign("o1", MaskLike(Q, True))]
        rung2 += stage_sequence(scheme, sp_, ctx, "o1")
        rung2.append(guard)
        rung1 = [Assign("Q", Q0), Assign("Qaux", Qaux0),
                 Assign("o1", troubled)]
        rung1 += stage_sequence(scheme, sp_, ctx, "o1")
        rung1.append(Select(AnyTrue(troubled), then=rung2,
                            note="D1 rung 2: whole step at order 1"))
        body.append(Select(AnyTrue(troubled), then=rung1,
                           note="D1 rung 1: troubled cells at order 1"))
    elif mood in ("d1", "guard"):
        body.append(guard)

    t_new = sp.Symbol("t_new")
    body.append(Assign("t_new", time + dt))
    body.append(Hook("post_step",
                     inputs=("Q", "Qaux", "p", "geom", "t_new", "dt"),
                     outputs=("Q",),
                     default=[Call("update_variables",
                                   ("Q", "Qaux", "p", "geom", "t_new", "dt"),
                                   ("Q",))]))
    body.append(Call("update_aux", ("Q", "Qaux", "p", "geom", "t_new"), ("Qaux",),
                     note="aux of the new state at t + dt"))
    return Procedure(name, STEP_INPUTS, STEP_OUTPUTS, body,
                     doc=f"One step at a frozen dt: {scheme.name} "
                         f"({scheme.n_stages} stage(s)), MOOD={mood}.")


def build_split_step(compose, ctx: BuildContext, *, mood: str = "d1",
                     name: str = "step") -> Procedure:
    """Operator splitting (`Compose`), unrolled.

    Each part is a `(ShuOsher, Stage)` pair advanced over `frac * dt` by
    its own sub-step procedure.  Every distinct part keeps its OWN clock
    (starting at `time`, advanced by its sub-steps), so Strang
    `((A, .5), (B, 1), (A, .5))` runs A on [t, t+dt/2] and
    [t+dt/2, t+dt] and B on [t, t+dt]."""
    from zoomy_core.solver.template.scheme import Compose
    if not isinstance(compose, Compose):
        raise TypeError("build_split_step needs a Compose")
    body = []
    clocks: dict = {}
    subs: dict = {}
    for i, ((scheme, stage), frac) in enumerate(compose.parts):
        key = (id(scheme), id(stage))
        if key not in subs:
            j = len(subs)
            subs[key] = (build_step(scheme, stage, ctx, mood=mood,
                                    name=f"{name}_part{j}"), j)
        sub, j = subs[key]
        clock = f"t_part{j}"
        if key not in clocks:
            body.append(Assign(clock, time, note=f"clock of part {j}"))
            clocks[key] = clock
        f = ctx.constants.define(f"c_split{i}", frac, "real",
                                 f"dt fraction of split part {i}")
        body.append(Assign("dt_part", f * dt))
        body.append(Call(sub, ("Q", "Qaux", "p", "geom", clock, "dt_part"),
                         ("Q", "Qaux"), note=f"split part {i} ({scheme.name})"))
        body.append(Assign(clock, sp.Symbol(clock) + sp.Symbol("dt_part")))
    return Procedure(name, STEP_INPUTS, STEP_OUTPUTS, body,
                     doc=f"Split step: {len(compose.parts)} part(s).")


def define_loop_constants(ctx: BuildContext, *, cfl, dimension, degree,
                          dt_max):
    c = ctx.constants
    c.define("c_cfl", cfl, "real", "CFL safety factor (pure, never adjusted)")
    c.define("c_cfl_dimension", dimension, "int",
             "D4: horizontal mesh dimension d")
    c.define("c_cfl_degree_factor", None if degree is None else 2 * int(degree) + 1,
             "int", "D4: 2k+1 for reconstruction degree k")
    c.define("c_dt_max", None if dt_max is None else dt_max, "real",
             "dt cap (inf = none)")


LOOP_HOOKS = ("loop_init", "pre_step", "observe", "output", "loop_finalize")


def build_time_loop(step: Procedure, ctx: BuildContext, *, plugins=(),
                    name: str = "march") -> Procedure:
    """The march procedure.

    inputs:  Q, Qaux, p, geom, time0, time_end (+ plugin inputs)
    outputs: Q, Qaux, time, iteration (+ plugin outputs)"""
    t_until = sp.Symbol("t_until")
    it = sp.Symbol("iteration")
    clamped = sp.Symbol("clamped")

    def hook_outs(h, core=()):
        out = list(core)
        for pl in plugins:
            for n in pl.hook_outputs(h):
                if n not in out:
                    out.append(n)
        return tuple(out)

    body = [
        Assign("time", sp.Symbol("time0")),
        Assign("iteration", sp.Integer(0)),
        IfStatic("aux_from_state",
                 then=[Call("update_aux", ("Q", "Qaux", "p", "geom", "time"),
                            ("Qaux",), note="initial aux from the state")]),
        Hook("loop_init", inputs=("Q", "Qaux", "p", "geom", "time", "time_end"),
             outputs=hook_outs("loop_init")),
        Assign("t_until", sp.Symbol("time_end")),
        While(time < t_until, body=[
            Hook("pre_step", inputs=("Q", "Qaux", "p", "geom", "time"),
                 outputs=hook_outs("pre_step")),
            Call("compute_dt",
                 ("Q", "Qaux", "p", "geom", "c_cfl", "c_cfl_dimension",
                  "c_cfl_degree_factor", "c_dt_max"), ("dt",),
                 note="D4 local CFL law"),
            Assign("clamped", dt >= t_until - time),
            Assign("dt", sp.Min(dt, t_until - time), note="D4: clamp to t_until"),
            Guard(IsFinite(dt) & (dt > 0), "dt not admissible (D9): non-finite "
                  "or not positive", values=("dt", "time", "iteration"),
                  note="D9"),
            Call(step, ("Q", "Qaux", "p", "geom", "time", "dt"), ("Q", "Qaux")),
            Assign("time", Where(clamped, t_until, time + dt),
                   note="lands exactly on t_until when clamped"),
            Assign("iteration", it + 1),
            Hook("observe", inputs=("Q", "Qaux", "p", "time", "dt"),
                 outputs=hook_outs("observe")),
            Hook("output", inputs=("Q", "Qaux", "time", "iteration"),
                 outputs=hook_outs("output")),
        ]),
        Hook("loop_finalize", inputs=("Q", "Qaux", "time"),
             outputs=hook_outs("loop_finalize")),
    ]
    ins = list(ports("Q:cell_state Qaux:cell_aux p:params geom:geom "
                     "time0:real time_end:real"))
    outs = list(ports("Q:cell_state Qaux:cell_aux time:real iteration:int"))
    for pl in plugins:
        for pt in ports(pl.inputs()):
            if pt.name not in [x.name for x in ins]:
                ins.append(pt)
        for pt in ports(pl.outputs()):
            if pt.name not in [x.name for x in outs]:
                outs.append(pt)
    return Procedure(name, tuple(ins), tuple(outs), body,
                     doc="The march: D4 dt, D9 guard, step, observe, output.")


__all__ = ["build_step", "build_split_step", "build_time_loop", "stage_sequence",
           "scheme_constants", "define_loop_constants", "MOOD_MODES",
           "LOOP_HOOKS", "STEP_INPUTS", "STEP_OUTPUTS", "D1_GUARD_MSG"]
