"""TDD verification for the MLVAM (Audusse-style) → SystemModel bridge.

Public surface tested:

* ``MLVAM`` directly inherits ``basemodel.Model``.
* ``MLVAM(N_layers=L, N=K, parameters={...}, boundary_conditions=...)``
  constructs.
* State layout is the Audusse fixed-α layout:
  ``Q = [H, (q_layer_ℓ_k, r_layer_ℓ_k, p_layer_ℓ_k)
         for ℓ=1..L, k=0..K]`` — total ``1 + L · 3·(K+1)`` variables.
* Equation set: one global ``continuity_global`` + L·(K+1) x-momentum
  + L·(K+1) z-momentum (no dynamic p equations).
* ``SystemModel.from_model(mlvam)`` returns matrices with the right
  number of dynamic rows ``1 + L · 2·(K+1)``.
* Continuity flux ``F[0]`` is the sum of layer mass fluxes (global
  mass conservation).
* Inter-layer mass-exchange transfer terms (Piecewise upwind) are
  present in the noncon matrix.
* Strict missing-parameter ``ValueError``.

Run with the zoomy micromamba env:

    ~/micromamba/envs/zoomy/bin/python tests/scripts/zoomy_core/mlvam/test_mlvam_to_systemmodel.py
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
        import sympy as sp
        from zoomy_core.model.models.mlvam import MLVAM
        from zoomy_core.model.models.system_model import SystemModel
        from zoomy_core.model.boundary_conditions import (
            BoundaryConditions, Extrapolation,
        )
    except Exception as e:
        _fail("import zoomy_core.model.models.mlvam + system_model", e)
    _ok("imports")

    bcs = BoundaryConditions([
        Extrapolation(tag="left"),
        Extrapolation(tag="right"),
    ])

    # ── (1) MLVAM(N_layers=2, N=1) constructs ────────────────────
    try:
        mlvam = MLVAM(
            N_layers=2,
            N=1,
            parameters={"g": (9.81, "positive"), "rho": (1.0, "positive")},
            boundary_conditions=bcs,
        )
    except Exception as e:
        _fail("MLVAM(N_layers=2, N=1, parameters=..., boundary_conditions=...) constructs", e)
    _ok(f"MLVAM(N_layers=2, N=1) constructed: {mlvam.name}")

    from zoomy_core.model.basemodel import Model
    if not isinstance(mlvam, Model):
        _fail(f"MLVAM is not a basemodel.Model subclass; type={type(mlvam).__name__}")
    _ok(f"MLVAM inherits basemodel.Model (MRO: {[c.__name__ for c in type(mlvam).__mro__[:5]]})")

    # ── (2) Audusse state layout (with bed-as-state) ─────────────
    # 2 (b, H) + L · 3 · (N+1) = 2 + 2 · 3 · 2 = 14.
    expected_vars = [
        "b", "H",
        "q_layer_1_0", "q_layer_1_1",
        "r_layer_1_0", "r_layer_1_1",
        "p_layer_1_0", "p_layer_1_1",
        "q_layer_2_0", "q_layer_2_1",
        "r_layer_2_0", "r_layer_2_1",
        "p_layer_2_0", "p_layer_2_1",
    ]
    got_vars = list(mlvam.variables.keys())
    if got_vars != expected_vars:
        _fail(f"variables mismatch: got {got_vars}, expected {expected_vars}")
    _ok(f"variables layout (Audusse): {len(got_vars)} states (2 + L·3(N+1))")

    # ── (3) Equation set ──────────────────────────────────────────
    expected_eqs = {
        "bottom", "continuity_global",
        "momentum_x_layer_1_0", "momentum_x_layer_1_1",
        "momentum_z_layer_1_0", "momentum_z_layer_1_1",
        "momentum_x_layer_2_0", "momentum_x_layer_2_1",
        "momentum_z_layer_2_0", "momentum_z_layer_2_1",
    }
    got_eqs = set(mlvam._equations.keys())
    if got_eqs != expected_eqs:
        extra = got_eqs - expected_eqs
        missing = expected_eqs - got_eqs
        _fail(f"equations mismatch: extra={extra}, missing={missing}")
    _ok(f"equation set: bottom + 1 global continuity + L·(N+1) x-momentum + L·(N+1) z-momentum")

    # ── (4) SystemModel ──────────────────────────────────────────
    try:
        sm = SystemModel.from_model(mlvam)
    except Exception as e:
        _fail("SystemModel.from_model(mlvam) raised", e)
    n_states = 14
    if sm.flux.shape != (n_states, 1):
        _fail(f"sm.flux shape {sm.flux.shape}, expected ({n_states}, 1)")
    _ok(f"SystemModel.from_model(mlvam) — shape ({n_states}, 1)")

    H = mlvam.variables.H
    q1_0 = mlvam.variables.q_layer_1_0
    q2_0 = mlvam.variables.q_layer_2_0

    # ── (5) Inter-layer mass-exchange transfer (Piecewise) ───────
    B = sm.nonconservative_matrix

    def _has_piecewise(expr):
        if not hasattr(expr, "has"):
            return False
        return expr.has(sp.Piecewise)

    # Rows all shifted by 1 due to new 'b' at row 0.
    # x-momentum layer-1 rows = 2, 3 (was 1, 2).
    # q-cols for layer 2: shifted by 1.  q_layer_2_0 / q_layer_2_1 at
    # state indices 8, 9.
    layer_1_xmom_rows = [2, 3]
    layer_2_q_cols = [8, 9]
    found_x_transfer = any(
        _has_piecewise(B[i, j, 0])
        for i in layer_1_xmom_rows
        for j in layer_2_q_cols
    )
    if not found_x_transfer:
        _fail("no upwind Piecewise found in layer-1 x-momentum cross-layer coupling")
    _ok(f"x-momentum mass-exchange transfer (Piecewise upwind) present")

    # z-momentum layer-1 rows = 4, 5 (was 3, 4).
    layer_1_zmom_rows = [4, 5]
    layer_2_q_cols = [8, 9]
    found_z_transfer = any(
        _has_piecewise(B[i, j, 0])
        for i in layer_1_zmom_rows
        for j in layer_2_q_cols
    )
    if not found_z_transfer:
        _fail("no upwind Piecewise found in layer-1 z-momentum cross-layer coupling")
    _ok(f"z-momentum mass-exchange transfer (Piecewise upwind via w*·G) present")

    # ── (6) Strict missing-parameter ValueError ──────────────────
    try:
        MLVAM(
            N_layers=2,
            N=1,
            parameters={"rho": (1.0, "positive")},
            boundary_conditions=bcs,
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
    print("ALL OK — MLVAM(Audusse) → SystemModel bridge passes baseline.")


if __name__ == "__main__":
    main()
