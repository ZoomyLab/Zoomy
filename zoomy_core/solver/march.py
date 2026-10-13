"""`solver_march` / `solver_hyperbolic_step` — the march, expressed in the
Procedure IR and EMITTED by core.

This module is the architecture test of the solver-unification program: it
builds, from an NSM alone, the two Procedures whose lowering must reproduce
the hand-written reference march (`zoomy_jax.fvm.solver2`) exactly.  Nothing
here is jax-, C- or foam-specific; the backend supplies BODIES.

The STRUCTURE / BODY split
--------------------------
Read `solver2` and the line falls where the design says it does.

**STRUCTURE — emitted here.**  The march `While`; the step head ordering
(dt_pass -> reduce_dt -> the FATAL guard) and the fact that `dt` is FROZEN
across stages; the stage loop UNROLLED from the tableau (core DATA, Shu-Osher
form) with the stage arithmetic `Q = alpha*Q0 + (1-alpha)*Q_cand` and the
stage time `t + (0 | dt)`; the build-time branches on reconstruction order,
MOOD, the interior-NCP term, the two implicit slots, coupling and output; the
write cadence; the two honesty guards; and every named constant.

**BODY — an `ExternalProcedure`.**  The
kernel calls and the face traversals: `dt_pass`, `reduce_dt`,
`halo_bc`, `reconstruct`, `flux_pass`, `gather_update`,
`mood_resolve`, `update_variables`, `update_aux`, the interior-NCP cell
term, io, and the two guards (host-side aborts).

dt (v6 §1)
----------
`dt` is computed ONCE per step at the head and then frozen: the stage loop
receives it as a value, no stage recomputes it, and there is no lambda row
coming back out of `flux_pass`.  `solver_reduce_dt` clamps by `dt_max`
and by EITHER the remaining time to `t_end` OR the coupling window
(`dt_window`) — the choice is the `coupled` build flag, never a runtime
branch and never min-combined (D7).

Honesty guards
--------------
Both are FATAL and both are external bodies, because both must ABORT with a
diagnostic:

* `solver_assert_dt_admissible` — `dt <= 0` or non-finite.
* `solver_assert_march_progress` — the other zero-progress mode: a dt that
  collapses towards zero while staying strictly positive.  The dt guard
  cannot see it; `max_steps` can.

Neither halves dt, neither retries, neither reduces the CFL.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional

import sympy as sp

from zoomy_core.solver.constants import MarchConstants, march_constants
from zoomy_core.solver.ir import (
    Assign,
    Call,
    IfStatic,
    Procedure,
    While,
)

# ── Shu-Osher tableaus: CORE DATA (design §2 "tableau") ─────────────────────
# Q^(k) = alpha_k * Q^n + (1 - alpha_k) * (Q^(k-1) + beta_k * dt * L(Q^(k-1)))
TABLEAU_EULER = ((0.0, 1.0),)
TABLEAU_SSPRK2 = ((0.0, 1.0), (0.5, 1.0))

#: reconstruction order -> tableau.  One table, read by every backend, so the
#: stage count is never a per-backend decision.
TABLEAUS = {1: TABLEAU_EULER, 2: TABLEAU_SSPRK2}


def tableau_for(order: int):
    """Stage tableau for a reconstruction `order`; RAISES on an unmapped
    order (no silent fallback to Euler)."""
    try:
        return TABLEAUS[int(order)]
    except KeyError:
        raise KeyError(
            f"no stage tableau declared for reconstruction order {order!r} "
            f"(have {sorted(TABLEAUS)}) — add it to march.TABLEAUS; the march "
            "never guesses a stage count") from None


# ── the build flags every IfStatic in this module branches on ──────────────
#: flag -> what it selects.  ``Procedure.resolve`` RAISES on a missing key, so
#: this table is also the checklist a caller must answer.
MARCH_FLAGS = {
    "mood": "a-posteriori MOOD with the whole-step order-1 redo",
    "interior_ncp": "amendment 10: the cell-interior NCP integral (order >= 2 "
                    "with a nonzero nonconservative_matrix)",
    "implicit_source": "a_ii*dt point-implicit source solve",
    "implicit_diffusion": "a_ii*dt global implicit diffusion solve",
    "coupled": "D7: dt_window REPLACES the t_end clamp; enables the per-stage "
               "coupling face fixup",
    "write_output": "the write cadence is active",
    "dt_floor": "a dt floor is applied (DEFAULT OFF — the sanctioned march "
                "has only the FATAL guard)",
}


def flags_from_nsm(nsm, *, mood: bool = False, coupled: bool = False,
                   write_output: bool = True,
                   interior_ncp: Optional[bool] = None,
                   dt_floor: bool = False) -> dict:
    """Build-flag map derived from the NSM, with the policy knobs passed in.

    `interior_ncp` defaults to the design's own condition — order >= 2 AND a
    nonzero `nonconservative_matrix` — which is exactly the guard
    `solver2.context.build_operators` enforces.  `implicit_*` follow the
    NSM's declared slots.
    """
    order = int(getattr(getattr(nsm, "reconstruction", None), "order", 1))
    ncm = getattr(nsm, "nonconservative_matrix", None)
    has_ncm = ncm is not None and any(
        sp.sympify(e) != 0 for e in sp.flatten(ncm))
    diffusion = getattr(nsm, "diffusion", None)
    return {
        "mood": bool(mood),
        "interior_ncp": (bool(order >= 2 and has_ncm)
                         if interior_ncp is None else bool(interior_ncp)),
        "implicit_source": bool(getattr(nsm, "implicit_source", False)),
        "implicit_diffusion": bool(
            getattr(diffusion, "implicit", False) if diffusion is not None
            else False),
        "coupled": bool(coupled),
        "write_output": bool(write_output),
        "dt_floor": bool(dt_floor),
    }


# ── the emitted procedures ─────────────────────────────────────────────────

def build_should_write(consts: MarchConstants) -> Procedure:
    """The drift-free write gate (design §2 / foam's gate).

    The next stamp is RECOMPUTED as `i_snapshot * write_interval` every call
    rather than accumulated — the accumulate form is the measured double-clamp
    bug.  The tolerance is the emitted `c_write_eps`, never a literal.
    """
    time, wi = sp.symbols("time write_interval")
    i_snap = sp.Symbol("i_snapshot")
    eps = consts.symbol("c_write_eps")
    due = sp.Ge(time + eps, i_snap * wi)
    return Procedure(
        name="solver_should_write",
        args=("time", "dt", "i_snapshot", "write_interval"),
        doc=("Drift-free write gate: due = time + c_write_eps >= i_snapshot * "
             "write_interval; the stamp is recomputed, never accumulated. "
             "Writes ``do_write`` and the advanced ``i_snapshot``."),
        stmts=(
            consts.assigns("c_write_eps")
            + (
                Assign("do_write", sp.Piecewise((sp.Integer(1), due),
                                                (sp.Integer(0), True)),
                       ctype="int"),
                Assign("i_snapshot", sp.Piecewise((i_snap + 1, due),
                                                  (i_snap, True)),
                       declare=False, ctype="int"),
            )
        ),
    )


def _stage_block(k: int, alpha: float, beta: float, consts: MarchConstants):
    """One UNROLLED Shu-Osher stage.

    The tableau is core DATA and its length is known at build time, so the
    stage loop is unrolled rather than lowered to a runtime loop: the stage
    weights become emitted constants and the whole stage sequence is visible
    in the emitted text.  `dt` is read, never written — it is frozen.
    """
    time, dt = sp.symbols("time dt")
    a_k = sp.Symbol(f"c_alpha_{k}")
    b_k = sp.Symbol(f"c_beta_{k}")
    Q0, Q_cand = sp.symbols("Q0 Q_cand")

    stage_time = time if k == 0 else time + dt
    stmts = [
        # stage weights: emitted named values, not literals at the use site
        Assign(str(a_k), sp.Float(alpha)),
        Assign(str(b_k), sp.Float(beta)),
        Assign("time_stage", stage_time),
        # per-stage halo/BC synthesis — load-bearing EVERY stage (D3)
        Call("solver_halo_bc",
             args=("variables", "aux_variables", "parameters", "time_stage"),
             results=("variables", "aux_variables", "bf_values")),
        Call("solver_reconstruct",
             args=("variables", "aux_variables", "bf_values", "o1"),
             results=("Q_L", "Q_R", "lim_grad")),
    ]
    # amendment 10: the cell-interior NCP integral, WB-critical at order >= 2
    stmts.append(IfStatic(
        "interior_ncp",
        then=(Call("solver_cell_ncp",
                   args=("variables", "aux_variables", "parameters",
                         "lim_grad"),
                   results=("cell_term",)),),
        otherwise=(Call("solver_no_cell_term", args=("variables",),
                        results=("cell_term",)),),
    ))
    stmts.append(Call(
        "solver_flux_pass",
        args=("variables", "aux_variables", "parameters", "time_stage",
              "Q_L", "Q_R"),
        results=("flux_face", "d_plus", "d_minus")))
    # D8: the 7th coupling surface — STRICT no-op when coupling is inactive,
    # so it is a BUILD flag, not a runtime `if (coupled)`.
    stmts.append(IfStatic(
        "coupled",
        then=(Call("solver_coupling_face_fixup",
                   args=("flux_face", "d_plus", "d_minus", "n_faces"),
                   results=("flux_face", "d_plus", "d_minus")),),
        otherwise=()))
    # NB the first argument is the CURRENT stage state, not the step base:
    # the residual is evaluated at Q^(k-1) and added to Q^(k-1).  ``q0`` (the
    # step base) enters only in the Shu-Osher average below and in the MOOD
    # rollback — conflating the two silently turns SSP-RK2 into two Euler
    # half-steps.
    stmts.append(Call(
        "solver_gather_update",
        args=("variables", "flux_face", "d_plus", "d_minus", "dt", str(b_k),
              "cell_term", "aux_variables", "parameters", "time_stage"),
        results=("q_cand", "troubled_stage")))
    stmts.append(Call("solver_merge_troubled",
                      args=("troubled", "troubled_stage"),
                      results=("troubled",)))
    # Shu-Osher combination — the stage arithmetic IS structure
    stmts.append(Assign("Q", a_k * Q0 + (sp.Integer(1) - a_k) * Q_cand,
                        declare=False))
    return tuple(stmts)


def build_hyperbolic_step(nsm, consts: MarchConstants, *,
                          order: Optional[int] = None) -> Procedure:
    """`solver_hyperbolic_step` — one step at a FROZEN `dt`.

    `dt` is an ARGUMENT: the head computed it and no stage may recompute it
    (v6 §1).  The stage loop is unrolled from `tableau_for`; MOOD and
    the two implicit slots are build-time branches.
    """
    order = int(getattr(getattr(nsm, "reconstruction", None), "order", 1)) \
        if order is None else int(order)
    tableau = tableau_for(order)
    time, dt = sp.symbols("time dt")
    iteration = sp.Symbol("iteration")

    stmts = [
        # the stage base: kept for the RK average AND the MOOD rollback
        Call("solver_stage_base", args=("variables",), results=("q0",)),
        Call("solver_clear_troubled", args=("variables",),
             results=("troubled",)),
        # o1 = 0: the first pass runs at the built reconstruction order.  The
        # order-1 redo re-enters the SAME stage sequence with o1 = 1, which is
        # why the order-1 path needs no separate code.
        Assign("o1", sp.Integer(0), ctype="int"),
    ]
    for k, (alpha, beta) in enumerate(tableau):
        stmts.extend(_stage_block(k, alpha, beta, consts))

    # MOOD: OUTCOME semantics (amendment 5).  The strategy is the backend's;
    # the sanctioned body is the whole-step order-1 redo.
    stmts.append(IfStatic(
        "mood",
        then=(
            consts.assigns("c_mood_h_bound", "c_mood_require_finite")
            + (Call("solver_mood_resolve",
                    args=("troubled", "q0", "variables", "aux_variables",
                          "parameters", "time", "dt", "c_mood_h_bound",
                          "c_mood_require_finite"),
                    results=("variables",)),)
        ),
        otherwise=()))
    stmts.append(IfStatic(
        "implicit_source",
        then=(Call("solver_implicit_source",
                   args=("variables", "aux_variables", "parameters", "dt"),
                   results=("variables",)),),
        otherwise=()))
    stmts.append(IfStatic(
        "implicit_diffusion",
        then=(Call("solver_implicit_diffusion",
                   args=("variables", "aux_variables", "parameters", "dt"),
                   results=("variables",)),),
        otherwise=()))

    stmts.extend([
        Assign("time", time + dt, declare=False),
        Assign("iteration", iteration + 1, declare=False, ctype="int"),
        Call("solver_update_variables",
             args=("variables", "aux_variables", "parameters", "time", "dt"),
             results=("variables",)),
        Call("solver_update_aux",
             args=("variables", "aux_variables", "parameters", "time", "dt"),
             results=("aux_variables",)),
    ])

    return Procedure(
        name="solver_hyperbolic_step",
        args=("variables", "aux_variables", "parameters", "time", "iteration",
              "dt"),
        doc=(f"One step of the v6 march at a FROZEN dt ({len(tableau)} "
             f"Shu-Osher stage(s), reconstruction order {order}).  dt is an "
             "argument: the step head owns it and no stage recomputes it. "
             "MOOD / implicit source / implicit diffusion are BUILD-time "
             "branches."),
        stmts=tuple(stmts),
    )


def build_march(nsm, consts: MarchConstants) -> Procedure:
    """`solver_march` — the ONE `While` of the design.

    Per step: the head (dt_pass -> reduce_dt -> FATAL guard), the step at a
    frozen dt, the progress guard, then the write cadence.
    """
    time, t_end = sp.symbols("time t_end")

    # ``c_eps_h`` is emitted at the site that EVALUATES the eigenvalue slot,
    # not at the reduction: it is the wave-speed floor ``max(eps_h, h)`` of
    # the spectrum (REQ-82), and a backend whose eigenvalue kernel does not
    # already carry it must apply it HERE rather than restate the number.
    head = list(consts.assigns("c_eps_h"))
    head.append(
        Call("solver_dt_pass",
             args=("variables", "aux_variables", "parameters", "c_eps_h"),
             results=("lam_lo_f", "lam_hi_f")))
    # The CFL constants are emitted at the reduction so that site reads named
    # values: dt <= c_cfl * 2r / (c_cfl_dimension * c_cfl_degree_factor *
    # |lam|), capped by c_dt_max.
    head.extend(consts.assigns(
        "c_cfl", "c_cfl_dimension", "c_cfl_degree_factor", "c_dt_max"))
    head.append(IfStatic(
        "coupled",
        # D7: dt_window REPLACES the t_end clamp — never min-combined.
        then=(Call("solver_reduce_dt",
                   args=("lam_lo_f", "lam_hi_f", "inradius_f", "n_faces",
                         "c_cfl", "c_cfl_dimension", "c_cfl_degree_factor",
                         "c_dt_max", "dt_window"),
                   results=("dt",)),),
        otherwise=(Assign("t_remaining", t_end - time),
                   Call("solver_reduce_dt",
                        args=("lam_lo_f", "lam_hi_f", "inradius_f", "n_faces",
                              "c_cfl", "c_cfl_dimension",
                              "c_cfl_degree_factor", "c_dt_max",
                              "t_remaining"),
                        results=("dt",)),)))
    head.append(IfStatic(
        "dt_floor",
        then=(consts.assigns("c_dt_floor")
              + (Call("solver_apply_dt_floor", args=("dt", "c_dt_floor"),
                      results=("dt",)),)),
        otherwise=()))
    # honesty guard 1 (v6 §1): dt <= 0 or non-finite is FATAL.
    head.append(Call("solver_assert_dt_admissible",
                     args=("dt", "lam_lo_f", "lam_hi_f", "inradius_f", "time",
                           "iteration"),
                     results=("dt",)))

    body = tuple(head) + (
        Call("solver_hyperbolic_step",
             args=("variables", "aux_variables", "parameters", "time",
                   "iteration", "dt"),
             results=("variables", "aux_variables", "time", "iteration",
                      "troubled")),
        # honesty guard 2: the zero-progress mode the dt guard cannot see.
        Call("solver_assert_march_progress",
             args=("time", "iteration", "dt", "t_end"), results=()),
        IfStatic("write_output",
                 then=(Call("solver_should_write",
                            args=("time", "dt", "i_snapshot",
                                  "write_interval"),
                            results=("do_write", "i_snapshot")),
                       Call("solver_write_fields",
                            args=("variables", "aux_variables", "time",
                                  "i_snapshot", "do_write"))),
                 otherwise=()),
    )

    return Procedure(
        name="solver_march",
        args=("variables", "aux_variables", "parameters", "time", "iteration",
              "i_snapshot", "t_end", "dt_window", "write_interval",
              "inradius_f", "n_faces"),
        doc=("The v6 march.  Step head: dt_pass -> reduce_dt -> FATAL "
             "admissibility guard; then one step at that FROZEN dt; then the "
             "progress guard and the drift-free write cadence.  No dt-halving, "
             "no retry, no CFL reduction."),
        stmts=(
            While(sp.Lt(time, t_end),
                  carry=("variables", "aux_variables", "time", "iteration",
                         "i_snapshot"),
                  body=body),
        ),
    )


# ── the assembled program ──────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class MarchProgram:
    """Everything a backend needs to lower the emitted march.

    `march` / `step` / `should_write` are RESOLVED Procedures (no
    `IfStatic` survives); `constants` carries the emitted named values and
    their provenance; `flags` records the build-time decisions that were
    resolved away, so the emitted code can be explained after the fact.
    """

    march: Procedure
    step: Procedure
    should_write: Procedure
    constants: MarchConstants
    flags: dict
    tableau: tuple
    order: int

    def report(self) -> str:
        return "\n".join([
            f"order      : {self.order}",
            f"tableau    : {self.tableau}",
            "flags      : " + ", ".join(
                f"{k}={v}" for k, v in sorted(self.flags.items())),
            "constants  :",
            self.constants.report(),
        ])


def emit_march(nsm, *, cfl: float, dimension: int, degree: int = 0,
               flags: Optional[Mapping[str, bool]] = None,
               dt_floor: Optional[float] = None,
               dt_max: Any = None,
               order: Optional[int] = None,
               **flag_kwargs) -> MarchProgram:
    """Emit the march for `nsm` — the one entry point.

    `cfl` (user law 0.9) and `dimension` (the MESH dimension) are rulings
    and therefore inputs; everything else is read from the NSM.  `flags`
    overrides the derived build-flag map; `**flag_kwargs` is forwarded to
    `flags_from_nsm` (`mood=`, `coupled=`, `write_output=`).

    Returns a `MarchProgram` whose Procedures are already RESOLVED, so
    either walker accepts them directly.
    """
    consts = march_constants(nsm, cfl=cfl, dimension=dimension, degree=degree,
                             dt_floor=dt_floor, dt_max=dt_max)
    resolved_flags = flags_from_nsm(nsm, dt_floor=dt_floor is not None,
                                    **flag_kwargs)
    if flags:
        unknown = set(flags) - set(MARCH_FLAGS)
        if unknown:
            raise KeyError(
                f"unknown march build flag(s) {sorted(unknown)} — declare them "
                f"in MARCH_FLAGS (have {sorted(MARCH_FLAGS)})")
        resolved_flags.update({k: bool(v) for k, v in flags.items()})
    missing = set(MARCH_FLAGS) - set(resolved_flags)
    if missing:
        raise KeyError(
            f"march build flag(s) {sorted(missing)} unresolved — a build-time "
            "branch has no default")

    ordr = int(getattr(getattr(nsm, "reconstruction", None), "order", 1)) \
        if order is None else int(order)
    step = build_hyperbolic_step(nsm, consts, order=ordr).resolve(resolved_flags)
    march = build_march(nsm, consts).resolve(resolved_flags)
    sw = build_should_write(consts).resolve(resolved_flags)
    return MarchProgram(march=march, step=step, should_write=sw,
                        constants=consts, flags=resolved_flags,
                        tableau=tableau_for(ordr), order=ordr)
