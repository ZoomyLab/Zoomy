"""TDD verification script for the SME → SystemModel bridge.

Public surface tested (per ~/.claude/plans/you-are-the-zoomy-lucky-leaf.md):

* ``SME`` directly inherits ``basemodel.Model`` through ``SigmaReference``.
* ``SME(N=2, parameters={...}, boundary_conditions=...)`` constructs.
* ``sme.flux() / sme.source() / sme.nonconservative_matrix()`` populated
  via ``tag_extraction.collect_solver_tag`` walking ``sme.equations``.
* ``SystemModel.from_model(sme)`` returns a SystemModel with
  ``sm.parameters`` (Symbols) and ``sm.parameter_values`` (floats).
* Missing parameter value → ``ValueError`` (strict).

Run with the zoomy micromamba env:

    ~/micromamba/envs/zoomy/bin/python tests/scripts/zoomy_core/sme/test_sme_to_systemmodel.py
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
    # ── (0) imports ───────────────────────────────────────────────
    try:
        from zoomy_core.model.models.sme import SME
        from zoomy_core.model.models.system_model import SystemModel
        from zoomy_core.model.boundary_conditions import (
            BoundaryConditions,
            Extrapolation,
        )
    except Exception as e:
        _fail("import zoomy_core.model.models.sme + system_model", e)
    _ok("imports")

    boundary_conditions = BoundaryConditions([
        Extrapolation(tag="left"),
        Extrapolation(tag="right"),
    ])

    # ── (1) SME constructs with parameters + BCs ──────────────────
    try:
        sme = SME(
            N=2,
            parameters={"g": 9.81, "rho": 1.0},
            boundary_conditions=boundary_conditions,
        )
    except Exception as e:
        _fail("SME(N=2, parameters=..., boundary_conditions=...) constructs", e)
    _ok("SME(N=2, parameters=..., boundary_conditions=...) constructed")

    # SME must be a basemodel.Model subclass.
    from zoomy_core.model.basemodel import Model
    if not isinstance(sme, Model):
        _fail(f"SME is not a basemodel.Model subclass; type={type(sme).__name__}")
    _ok(f"SME inherits basemodel.Model (MRO: {[c.__name__ for c in type(sme).__mro__[:5]]})")

    # ── (2) Symbols vs values on the model itself ─────────────────
    if not hasattr(sme, "parameters"):
        _fail("sme.parameters attribute missing")
    if not hasattr(sme, "parameter_values"):
        _fail("sme.parameter_values attribute missing")
    if set(sme.parameters.keys()) != {"g", "rho"}:
        _fail(f"sme.parameters keys = {set(sme.parameters.keys())}, expected {{g, rho}}")
    if not sme.parameters.g.is_Symbol:
        _fail(f"sme.parameters.g not a Symbol: {sme.parameters.g!r}")
    if abs(float(sme.parameter_values.g) - 9.81) > 1e-12:
        _fail(f"sme.parameter_values.g != 9.81 ({sme.parameter_values.g})")
    if abs(float(sme.parameter_values.rho) - 1.0) > 1e-12:
        _fail(f"sme.parameter_values.rho != 1.0 ({sme.parameter_values.rho})")
    _ok("sme.parameters (Symbols) + sme.parameter_values (floats) split")

    # ── (3) Operator API populated via tag_extraction ─────────────
    try:
        F = sme.flux()
    except Exception as e:
        _fail("sme.flux() raised", e)
    try:
        S = sme.source()
    except Exception as e:
        _fail("sme.source() raised", e)
    try:
        B = sme.nonconservative_matrix()
    except Exception as e:
        _fail("sme.nonconservative_matrix() raised", e)
    n_eq = sme.n_variables
    if F.shape != (n_eq, sme.dimension):
        _fail(f"sme.flux() shape {F.shape}, expected ({n_eq}, {sme.dimension})")
    if S.shape != (n_eq,):
        _fail(f"sme.source() shape {S.shape}, expected ({n_eq},)")
    if B.shape != (n_eq, n_eq, sme.dimension):
        _fail(f"sme.nonconservative_matrix() shape {B.shape}, expected ({n_eq}, {n_eq}, {sme.dimension})")
    _ok(f"sme.flux/source/nonconservative_matrix populated, shapes {F.shape} {S.shape} {B.shape}")

    # ── (4) Bridge to SystemModel ─────────────────────────────────
    try:
        sm = SystemModel.from_model(sme)
    except Exception as e:
        _fail("SystemModel.from_model(sme) raised", e)
    if set(sm.parameters.keys()) != {"g", "rho"}:
        _fail(f"sm.parameters keys = {set(sm.parameters.keys())}")
    if not sm.parameters.g.is_Symbol:
        _fail(f"sm.parameters.g is not a Symbol: {sm.parameters.g!r}")
    if abs(float(sm.parameter_values.g) - 9.81) > 1e-12:
        _fail(f"sm.parameter_values.g != 9.81")
    if abs(float(sm.parameter_values.rho) - 1.0) > 1e-12:
        _fail(f"sm.parameter_values.rho != 1.0")
    if sm.flux.shape != (n_eq, sme.dimension):
        _fail(f"sm.flux shape {sm.flux.shape}")
    _ok(f"SystemModel.from_model(sme) — parameters carried, sm.flux shape {sm.flux.shape}")

    # ── (5) Strict missing-parameter ValueError ───────────────────
    try:
        SME(
            N=2,
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
    print("ALL OK — SME(N=2) → SystemModel bridge passes baseline.")


if __name__ == "__main__":
    main()
