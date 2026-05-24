"""TDD verification script for the VAM → SystemModel bridge.

Run with the zoomy micromamba env:

    ~/micromamba/envs/zoomy/bin/python tests/scripts/zoomy_core/vam/test_vam_to_systemmodel.py
"""

from __future__ import annotations

import sys
import traceback


def _ok(msg):
    print(f"OK   {msg}", flush=True)


def _fail(msg, exc=None):
    print(f"FAIL {msg}", flush=True)
    if exc is not None:
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    sys.exit(1)


def main():
    try:
        from zoomy_core.model.models.vam import VAM
        from zoomy_core.model.models.system_model import SystemModel
        from zoomy_core.model.boundary_conditions import (
            BoundaryConditions,
            Extrapolation,
        )
    except Exception as e:
        _fail("imports", e)
    _ok("imports")

    boundary_conditions = BoundaryConditions([
        Extrapolation(tag="left"),
        Extrapolation(tag="right"),
    ])

    # ── (1) VAM constructs with parameters + BCs ──────────────────
    try:
        vam = VAM(
            N=1,
            parameters={"g": 9.81, "rho": 1.0},
            boundary_conditions=boundary_conditions,
        )
    except Exception as e:
        _fail("VAM(N=1, parameters=..., boundary_conditions=...) constructs", e)
    _ok(f"VAM(N=1) constructed; variables = {list(vam.variables.keys())}")

    from zoomy_core.model.basemodel import Model
    if not isinstance(vam, Model):
        _fail(f"VAM is not a basemodel.Model subclass; type={type(vam).__name__}")
    _ok(f"VAM inherits basemodel.Model (MRO: {[c.__name__ for c in type(vam).__mro__[:5]]})")

    # ── (2) Symbols vs values on the model itself ─────────────────
    if set(vam.parameters.keys()) != {"g", "rho"}:
        _fail(f"vam.parameters keys = {set(vam.parameters.keys())}")
    if not vam.parameters.g.is_Symbol:
        _fail(f"vam.parameters.g not a Symbol: {vam.parameters.g!r}")
    if abs(float(vam.parameter_values.g) - 9.81) > 1e-12:
        _fail(f"vam.parameter_values.g != 9.81 ({vam.parameter_values.g})")
    _ok("vam.parameters (Symbols) + vam.parameter_values (floats) split")

    # ── (3) Operator API populated via tag_extraction ─────────────
    F = vam.flux()
    S = vam.source()
    B = vam.nonconservative_matrix()
    n_eq = vam.n_variables
    expected_n_eq = 1 + 3 * (vam.N + 1)  # [h, q_0..q_N, r_0..r_N, p_0..p_N]
    if n_eq != expected_n_eq:
        _fail(f"vam.n_variables = {n_eq}, expected {expected_n_eq}")
    if F.shape != (n_eq, vam.dimension):
        _fail(f"vam.flux() shape {F.shape}, expected ({n_eq}, {vam.dimension})")
    if S.shape != (n_eq,):
        _fail(f"vam.source() shape {S.shape}, expected ({n_eq},)")
    if B.shape != (n_eq, n_eq, vam.dimension):
        _fail(f"vam.nonconservative_matrix() shape {B.shape}, expected ({n_eq}, {n_eq}, {vam.dimension})")
    _ok(f"vam.flux/source/NCP populated, shapes {F.shape} {S.shape} {B.shape}")

    # ── (4) Bridge to SystemModel ─────────────────────────────────
    try:
        sm = SystemModel.from_model(vam)
    except Exception as e:
        _fail("SystemModel.from_model(vam) raised", e)
    if set(sm.parameters.keys()) != {"g", "rho"}:
        _fail(f"sm.parameters keys = {set(sm.parameters.keys())}")
    if not sm.parameters.g.is_Symbol:
        _fail(f"sm.parameters.g not a Symbol: {sm.parameters.g!r}")
    if abs(float(sm.parameter_values.g) - 9.81) > 1e-12:
        _fail("sm.parameter_values.g != 9.81")
    if sm.flux.shape != (n_eq, vam.dimension):
        _fail(f"sm.flux shape {sm.flux.shape}")
    _ok(f"SystemModel.from_model(vam) — parameters carried, sm.flux shape {sm.flux.shape}")

    # ── (5) Strict missing-parameter ValueError ───────────────────
    try:
        VAM(
            N=1,
            parameters={"rho": 1.0},
            boundary_conditions=boundary_conditions,
        )
    except ValueError as e:
        if "g" not in str(e):
            _fail(f"ValueError raised but message does not mention 'g': {e}")
        _ok(f"missing-parameter ValueError raised: {e}")
    except Exception as e:
        _fail("Missing parameter should raise ValueError, raised %r" % type(e).__name__, e)
    else:
        _fail("Missing parameter did not raise — expected ValueError")

    print()
    print("ALL OK — VAM(N=1) → SystemModel bridge passes baseline.")


if __name__ == "__main__":
    main()
