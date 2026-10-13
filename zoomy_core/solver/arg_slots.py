"""Solver-level argument vocabulary — the march-scale extension of
`OPERATOR_ARG_SLOTS` / `GenericCppBase.ARG_MAPPING`.

`OPERATOR_ARG_SLOTS` (`system_model.py`) declares the argument groups of
the per-cell/per-face MODEL kernels (`flux` / `source` / …).  The solver
blocks of the unification design live one level up and speak about
march-scale objects the model kernels never see: the MeshRT connectivity, the
stored face arrays, the stage base `Q0`, the Butcher weight `a_stage`, the
MOOD `troubled` mask, the io counter `i_snapshot` and the two dt clamps
`dt_max` / `dt_window`.

This module is the SINGLE declaration of those slots — exactly as
`OPERATOR_ARG_SLOTS` is for kernels.  Every walker translates SYNTAX only:

* `SOLVER_ARG_KINDS`   slot -> storage kind (drives the C declaration)
* `SOLVER_ARG_MAPPING` slot -> C-family spelling (merged into
  `GenericCppBase.ARG_MAPPING`; the existing kernel keys are untouched)
* `SOLVER_ARG_SLOTS`   procedure name -> ordered slot tuple, for the
  blocks the design names in §2 / §v6.

Storage kinds
-------------
`scalar_real`     by-value `const T`            (dt, dt_max, a_stage, time)
`scalar_int`      by-value `const int`          (n_faces, i_snapshot)
`const_real_ptr`  read-only array `const T*`    (Q, cell_volumes, …)
`real_ptr`        written array `T*`            (Fface, Dp, Dm, lam_*, …)
`const_int_ptr`   read-only index array `const int*` (face_cells)
`int_ptr`         written index array `int*`    (troubled)
"""
from __future__ import annotations

_SR, _SI = "scalar_real", "scalar_int"
_CRP, _RP = "const_real_ptr", "real_ptr"
_CIP, _IP = "const_int_ptr", "int_ptr"


#: slot -> storage kind.  Adding a march-level object to the design means
#: adding it HERE first; an undeclared slot raises in ``Procedure``.
SOLVER_ARG_KINDS = {
    # ── the march state S = (time, iteration, i_snapshot, Q, Qaux) ──────
    "variables": _CRP,          # Q
    "aux_variables": _CRP,      # Qaux
    "parameters": _CRP,         # p
    "q0": _CRP,                 # stage base: RK average AND MOOD rollback
    "q_cand": _RP,              # candidate state written by gather_update
    "time": _SR,
    "iteration": _SI,
    "i_snapshot": _SI,
    # ── MeshRT ──────────────────────────────────────────────────────────
    "n_cells": _SI,
    "n_faces": _SI,
    "cell_volumes": _CRP,
    "face_area": _CRP,
    "face_normal": _CRP,
    "face_cells": _CIP,         # (owner, neigh) pairs
    "inradius_f": _CRP,
    # ── the STORED face arrays (design §1, the storage decision) ────────
    "flux_face": _RP,           # Fface(n,F)
    "d_plus": _RP,              # Dp(n,F)
    "d_minus": _RP,             # Dm(n,F)
    "lam_f": _RP,               # |lambda|max per face (order-1 reuse path)
    "lam_lo_f": _RP,            # v6 dt_pass: the two stored bounds
    "lam_hi_f": _RP,
    # ── stage / step control ────────────────────────────────────────────
    "dt": _SR,
    "dt_max": _SR,
    "dt_window": _SR,           # v6 D7: REPLACES the t_end clamp, never min'd
    "a_stage": _SR,             # Butcher weight of the current stage
    "troubled": _IP,            # MOOD flags, written by the gather pass
    # ── io ──────────────────────────────────────────────────────────────
    "write_interval": _SR,
    "t_end": _SR,
}

#: slot -> C-family identifier.  Kernel-shared slots keep the spelling
#: ``GenericCppBase.ARG_MAPPING`` already uses (``variables`` -> ``Q`` …) so
#: merging this table changes NO existing emitted signature.
SOLVER_ARG_MAPPING = {
    "variables": "Q",
    "aux_variables": "Qaux",
    "parameters": "p",
    "q0": "Q0",
    "q_cand": "Q_cand",
    "time": "time",
    "iteration": "iteration",
    "i_snapshot": "i_snapshot",
    "n_cells": "n_cells",
    "n_faces": "n_faces",
    "cell_volumes": "cell_volumes",
    "face_area": "face_area",
    "face_normal": "face_normal",
    "face_cells": "face_cells",
    "inradius_f": "inradius_f",
    "flux_face": "Fface",
    "d_plus": "Dp",
    "d_minus": "Dm",
    "lam_f": "lam_f",
    "lam_lo_f": "lam_lo_f",
    "lam_hi_f": "lam_hi_f",
    "dt": "dt",
    "dt_max": "dt_max",
    "dt_window": "dt_window",
    "a_stage": "a_stage",
    "troubled": "troubled",
    "write_interval": "write_interval",
    "t_end": "t_end",
}

# Kinds and spellings must cover the same slot set — a slot with a kind but no
# spelling would emit an unnamed argument.
assert set(SOLVER_ARG_KINDS) == set(SOLVER_ARG_MAPPING), (
    "SOLVER_ARG_KINDS / SOLVER_ARG_MAPPING disagree: "
    f"{set(SOLVER_ARG_KINDS) ^ set(SOLVER_ARG_MAPPING)}"
)


#: Declared signatures of the solver blocks named by the design (§2 + v6 §1).
#: Same role as ``OPERATOR_ARG_SLOTS``: the ONE place a block's argument order
#: is written down, read by every walker.
SOLVER_ARG_SLOTS = {
    "solver_dt_pass": (
        "variables", "aux_variables", "parameters", "face_cells",
        "face_normal", "n_faces", "lam_lo_f", "lam_hi_f"),
    "solver_reduce_dt": (
        "lam_lo_f", "lam_hi_f", "inradius_f", "n_faces", "dt_max", "dt_window"),
    "solver_flux_pass": (
        "variables", "aux_variables", "parameters", "face_cells", "face_area",
        "face_normal", "n_faces", "time", "flux_face", "d_plus", "d_minus"),
    "solver_gather_update": (
        "q0", "flux_face", "d_plus", "d_minus", "face_cells", "face_area",
        "cell_volumes", "n_cells", "n_faces", "dt", "a_stage", "q_cand",
        "troubled"),
    "solver_should_write": (
        "time", "dt", "i_snapshot", "write_interval"),
    "solver_proceed": (
        "time", "iteration", "t_end"),
}


#: C-SPELLING -> storage kind.  ``GenericCppBase._sm_arg_decl`` receives an
#: already-mapped identifier (``_operator_arg_keys`` translates slot -> spelling
#: before building the signature), so the declaration table must be keyed the
#: same way.  Slots that share a spelling share a kind — asserted here rather
#: than discovered in emitted C++.
SOLVER_DECL_KINDS: dict = {}
for _slot, _spelling in SOLVER_ARG_MAPPING.items():
    _kind = SOLVER_ARG_KINDS[_slot]
    if SOLVER_DECL_KINDS.setdefault(_spelling, _kind) != _kind:
        raise AssertionError(
            f"slots mapping to {_spelling!r} disagree on storage kind: "
            f"{SOLVER_DECL_KINDS[_spelling]!r} vs {_kind!r}"
        )
del _slot, _spelling, _kind


def solver_arg_kind(slot: str) -> str:
    """Storage kind of `slot`; RAISES on an undeclared slot (no default)."""
    try:
        return SOLVER_ARG_KINDS[slot]
    except KeyError:
        raise KeyError(
            f"undeclared solver argument slot {slot!r} — declare it in "
            "SOLVER_ARG_KINDS/SOLVER_ARG_MAPPING first"
        ) from None
