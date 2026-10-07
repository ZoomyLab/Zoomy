"""Typed declaration tables the FIXED printer needs (scratch prototype).

Everything here is what core would have to carry for the C family to lower
the emitted march: a typed kind per name (slot OR march-local intermediate),
a binding rule (argument / workspace / mesh / local), the reconciled external
signatures, and the out-params of internal Procedures (a stand-in for a
``Procedure.results`` field core does not have).
"""
from __future__ import annotations

import re

from zoomy_core.solver.constants import _REAL_VALUED
from zoomy_core.solver.external import REQUIRED_PROCEDURES

# ── kinds ──────────────────────────────────────────────────────────────────
#: kind -> default C++ handle type (a backend printer overrides the spelling;
#: dmplex/vector backend typedefs them to std::vector<...>).
HANDLE_TYPES = {
    "cell_state": "CellState",    # n_vars x n_cells (Q, Q0, Q_cand, cell_term)
    "cell_aux": "CellAux",        # n_aux  x n_cells
    "params": "Params",           # n_params
    "face_state": "FaceState",    # n_vars x n_faces (Q_L, Q_R, Fface, Dp, Dm)
    "face_scalar": "FaceScalar",  # n_faces          (lam_lo_f, inradius_f)
    "cell_scalar": "CellScalar",  # n_cells          (cell_volumes)
    "cell_mask": "CellMask",      # n_cells int      (troubled)
    "cell_grad": "CellGrad",      # n_vars x dim x n_cells (lim_grad)
    "bc_values": "BCValues",      # boundary-face ghost states
    "real": "double",
    "int": "int",
}
ARRAY_KINDS = frozenset(k for k in HANDLE_TYPES if k not in ("real", "int"))

#: C spelling -> (kind, binding).  binding:
#:   "arg"   must arrive as a procedure parameter (march state)
#:   "ws"    march-local array, bound to preallocated workspace storage
#:   "mesh"  MeshRT quantity, bound read-only from the context
#:   "local" scalar local, declared in the procedure
TYPED = {
    "Q": ("cell_state", "arg"), "Qaux": ("cell_aux", "arg"),
    "p": ("params", "arg"),
    "Q0": ("cell_state", "ws"), "Q_cand": ("cell_state", "ws"),
    "cell_term": ("cell_state", "ws"),
    "Q_L": ("face_state", "ws"), "Q_R": ("face_state", "ws"),
    "Fface": ("face_state", "ws"), "Dp": ("face_state", "ws"),
    "Dm": ("face_state", "ws"),
    "lam_lo_f": ("face_scalar", "ws"), "lam_hi_f": ("face_scalar", "ws"),
    "lam_f": ("face_scalar", "ws"),
    "troubled": ("cell_mask", "ws"), "troubled_stage": ("cell_mask", "ws"),
    "lim_grad": ("cell_grad", "ws"), "bf_values": ("bc_values", "ws"),
    "inradius_f": ("face_scalar", "mesh"), "face_area": ("face_scalar", "mesh"),
    "cell_volumes": ("cell_scalar", "mesh"),
    "n_faces": ("int", "mesh"), "n_cells": ("int", "mesh"),
    "time": ("real", "local"), "dt": ("real", "local"),
    "t_end": ("real", "local"), "dt_window": ("real", "local"),
    "write_interval": ("real", "local"), "time_stage": ("real", "local"),
    "t_remaining": ("real", "local"), "dt_max": ("real", "local"),
    "iteration": ("int", "local"), "i_snapshot": ("int", "local"),
    "o1": ("int", "local"), "do_write": ("int", "local"),
}

_STAGE_CONST = re.compile(r"^c_(alpha|beta)_\d+$")


def kind_of(name: str) -> str:
    """Kind of a C spelling; RAISES on an undeclared name (no default)."""
    if name in TYPED:
        return TYPED[name][0]
    if _STAGE_CONST.match(name):
        return "real"
    if name.startswith("c_"):
        return "real" if name in _REAL_VALUED else "int"
    raise KeyError(f"undeclared name {name!r}: no kind in the declaration table")


def binding_of(name: str) -> str:
    if name in TYPED:
        return TYPED[name][1]
    if name.startswith("c_"):
        return "const"            # must be bound by an emitted Assign
    raise KeyError(f"undeclared name {name!r}")


# ── reconciled external signatures ─────────────────────────────────────────
# (param name, kind, mutable).  Convention: Call.args in order, then every
# Call.result that is NOT already an arg (in/out params appear ONCE, by
# mutable reference).  These are what the call sites in march.py actually
# pass; where they disagree with external.py the disagreement is reported by
# arity_report.py.
def _p(spec):
    out = []
    for tok in spec.split():
        mut = tok.endswith("!")
        name, kind = tok.rstrip("!").split(":")
        out.append((name, kind, mut))
    return tuple(out)


R, I = "real", "int"
CS, CA, PA = "cell_state", "cell_aux", "params"
FS, FSc, CM, CG, BC = "face_state", "face_scalar", "cell_mask", "cell_grad", "bc_values"

EXTERNAL_SIGS = {
    "solver_prepare_mesh": (),
    "solver_build_operators": (),
    "solver_initialize_state": (),
    "solver_adapt_mesh": (),
    "solver_coupling_init": (),
    "solver_coupling_checkpoint": (),
    "solver_coupling_finalize": (),
    "solver_coupling_read": _p("time:real dt:real"),
    "solver_coupling_write": _p("Q:cell_state time:real"),
    "solver_coupling_advance": _p("dt:real"),
    "solver_reduce_min": _p("lam_f:face_scalar n_faces:int"),
    "solver_halo_bc": _p("Q:cell_state! Qaux:cell_aux! p:params time_stage:real "
                         "bf_values:bc_values!"),
    "solver_reconstruct": _p("Q:cell_state Qaux:cell_aux bf_values:bc_values o1:int "
                             "Q_L:face_state! Q_R:face_state! lim_grad:cell_grad!"),
    "solver_cell_ncp": _p("Q:cell_state Qaux:cell_aux p:params lim_grad:cell_grad "
                          "cell_term:cell_state!"),
    "solver_no_cell_term": _p("Q:cell_state cell_term:cell_state!"),
    "solver_flux_pass": _p("Q:cell_state Qaux:cell_aux p:params time_stage:real "
                           "Q_L:face_state Q_R:face_state Fface:face_state! "
                           "Dp:face_state! Dm:face_state!"),
    "solver_coupling_face_fixup": _p("Fface:face_state! Dp:face_state! "
                                     "Dm:face_state! n_faces:int"),
    "solver_gather_update": _p("Q:cell_state Fface:face_state Dp:face_state "
                               "Dm:face_state dt:real beta:real cell_term:cell_state "
                               "Qaux:cell_aux p:params time_stage:real "
                               "Q_cand:cell_state! troubled_stage:cell_mask!"),
    "solver_merge_troubled": _p("troubled:cell_mask! troubled_stage:cell_mask"),
    "solver_stage_base": _p("Q:cell_state Q0:cell_state!"),
    "solver_clear_troubled": _p("Q:cell_state troubled:cell_mask!"),
    "solver_mood_resolve": _p("troubled:cell_mask Q0:cell_state Q:cell_state! "
                              "Qaux:cell_aux p:params time:real dt:real "
                              "c_mood_h_bound:real c_mood_require_finite:int"),
    "solver_implicit_source": _p("Q:cell_state! Qaux:cell_aux p:params dt:real"),
    "solver_implicit_diffusion": _p("Q:cell_state! Qaux:cell_aux p:params dt:real"),
    "solver_update_variables": _p("Q:cell_state! Qaux:cell_aux p:params time:real dt:real"),
    "solver_update_aux": _p("Q:cell_state Qaux:cell_aux! p:params time:real dt:real"),
    "solver_dt_pass": _p("Q:cell_state Qaux:cell_aux p:params c_eps_h:real "
                         "lam_lo_f:face_scalar! lam_hi_f:face_scalar!"),
    "solver_reduce_dt": _p("lam_lo_f:face_scalar lam_hi_f:face_scalar "
                           "inradius_f:face_scalar n_faces:int c_cfl:real "
                           "c_cfl_dimension:int c_cfl_degree_factor:int "
                           "c_dt_max:real clamp:real dt:real!"),
    "solver_apply_dt_floor": _p("dt:real! c_dt_floor:real"),
    "solver_assert_dt_admissible": _p("dt:real! lam_lo_f:face_scalar "
                                      "lam_hi_f:face_scalar inradius_f:face_scalar "
                                      "time:real iteration:int"),
    "solver_assert_march_progress": _p("time:real iteration:int dt:real t_end:real"),
    "solver_write_fields": _p("Q:cell_state Qaux:cell_aux time:real i_snapshot:int "
                              "do_write:int"),
}
assert set(EXTERNAL_SIGS) == set(REQUIRED_PROCEDURES), (
    set(EXTERNAL_SIGS) ^ set(REQUIRED_PROCEDURES))

#: Stand-in for a ``Procedure.results`` field: out-names a Procedure delivers
#: to its caller that are NOT among its args (appended as mutable params).
#: The jax backend hand-wires exactly these (march_emitted.py: should_write
#: returns env["do_write"]; step returns env["troubled"]).
PROC_RESULTS = {
    "solver_should_write": ("do_write",),
    "solver_hyperbolic_step": ("troubled",),
    "solver_march": (),
}
