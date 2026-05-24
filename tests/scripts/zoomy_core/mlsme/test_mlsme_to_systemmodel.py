"""TDD verification script for the MLSME (Audusse) → SystemModel bridge.

Public surface tested:

* ``MLSME`` directly inherits ``basemodel.Model``.
* ``MLSME(N_layers=L, N=K, parameters={...}, boundary_conditions=...)``
  constructs.
* State layout is the Audusse fixed-α layout:
  ``Q = [H, q_layer_1_0..K, q_layer_2_0..K, ..., q_layer_L_0..K]`` —
  total ``1 + L·(K+1)`` variables.
* Equation set: one global ``continuity_global`` (``∂_t H + ∂_x Q = 0``)
  + ``L·(K+1)`` per-layer momentum equations.
* ``SystemModel.from_model(mlsme)`` returns matrices of size
  ``1 + L·(K+1)`` × ...
* Inter-layer hydrostatic coupling is present in the
  ``nonconservative_matrix``.
* In the ``N=0`` limit (piecewise-constant velocity profile),
  the SUM of the L per-layer momentum equations under a uniform
  velocity assumption (``u_ℓ = U`` for all ℓ) reduces to the
  single-layer SWE:
      ∂_t Q + ∂_x(Q²/H + g·H²/2) + g·H·∂_x b = 0
* Strict missing-parameter ``ValueError``.

Run with the zoomy micromamba env:

    ~/micromamba/envs/zoomy/bin/python tests/scripts/zoomy_core/mlsme/test_mlsme_to_systemmodel.py
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
        from zoomy_core.model.models.mlsme import MLSME
        from zoomy_core.model.models.system_model import SystemModel
        from zoomy_core.model.boundary_conditions import (
            BoundaryConditions,
            Extrapolation,
        )
    except Exception as e:
        _fail("import zoomy_core.model.models.mlsme + system_model", e)
    _ok("imports")

    boundary_conditions = BoundaryConditions([
        Extrapolation(tag="left"),
        Extrapolation(tag="right"),
    ])

    # ── (1) MLSME(N_layers=2, N=2) constructs ────────────────────
    try:
        mlsme = MLSME(
            N_layers=2,
            N=2,
            parameters={"g": (9.81, "positive"), "rho": (1.0, "positive")},
            boundary_conditions=boundary_conditions,
        )
    except Exception as e:
        _fail("MLSME(N_layers=2, N=2, parameters=..., boundary_conditions=...) constructs", e)
    _ok(f"MLSME(N_layers=2, N=2) constructed: {mlsme.name}")

    # MLSME must be a basemodel.Model subclass.
    from zoomy_core.model.basemodel import Model
    if not isinstance(mlsme, Model):
        _fail(f"MLSME is not a basemodel.Model subclass; type={type(mlsme).__name__}")
    _ok(f"MLSME inherits basemodel.Model (MRO: {[c.__name__ for c in type(mlsme).__mro__[:5]]})")

    # ── (2) Audusse state layout: [H, q_ℓ_k] ─────────────────────
    expected_vars = ["H",
                     "q_layer_1_0", "q_layer_1_1", "q_layer_1_2",
                     "q_layer_2_0", "q_layer_2_1", "q_layer_2_2"]
    got_vars = list(mlsme.variables.keys())
    if got_vars != expected_vars:
        _fail(f"variables mismatch: got {got_vars}, expected {expected_vars}")
    _ok(f"variables layout (Audusse): {got_vars}")

    expected_eqs = {"continuity_global",
                    "momentum_x_layer_1_0", "momentum_x_layer_1_1", "momentum_x_layer_1_2",
                    "momentum_x_layer_2_0", "momentum_x_layer_2_1", "momentum_x_layer_2_2"}
    got_eqs = set(mlsme._equations.keys())
    if got_eqs != expected_eqs:
        extra = got_eqs - expected_eqs
        missing = expected_eqs - got_eqs
        _fail(f"equations mismatch: extra={extra}, missing={missing}")
    _ok(f"equation set: 1 global continuity + L·(N+1) momentum equations")

    # ── (3) SystemModel ──────────────────────────────────────────
    try:
        sm = SystemModel.from_model(mlsme)
    except Exception as e:
        _fail("SystemModel.from_model(mlsme) raised", e)
    n_eq = 1 + 2 * (2 + 1)  # 1 + L·(N+1) = 7
    if sm.flux.shape != (n_eq, 1):
        _fail(f"sm.flux shape {sm.flux.shape}, expected ({n_eq}, 1)")
    if sm.nonconservative_matrix.shape != (n_eq, n_eq, 1):
        _fail(f"sm.nonconservative_matrix shape {sm.nonconservative_matrix.shape}")
    _ok(f"SystemModel.from_model(mlsme) — matrices of shape ({n_eq}, 1) / ({n_eq}, {n_eq}, 1)")

    # Continuity row: F[0] = Q_total = Σ q_ℓ_0.
    H = mlsme.variables.H
    q1_0 = mlsme.variables.q_layer_1_0
    q2_0 = mlsme.variables.q_layer_2_0
    if sp.simplify(sm.flux[0, 0] - (q1_0 + q2_0)) != 0:
        _fail(f"continuity flux F[0] = {sm.flux[0, 0]}, expected q_1_0 + q_2_0")
    _ok(f"continuity flux F[0] = Σ q_ℓ_0  (global mass conservation)")

    # ── (4) Inter-layer hydrostatic coupling in B ────────────────
    B = sm.nonconservative_matrix
    # Row 1 = layer 1 momentum-0; col 0 = H.  Audusse intra-layer
    # hydrostatic shows up in B[1, 0, 0] as g·α_1·α_other·H.
    if B[1, 0, 0] == 0:
        _fail(f"B[mom_layer_1_0, H, x] = 0 — expected inter-layer hydrostatic g·α_1·H")
    if B[4, 0, 0] == 0:
        _fail(f"B[mom_layer_2_0, H, x] = 0 — expected inter-layer hydrostatic g·α_2·H")
    _ok(f"inter-layer hydrostatic coupling present: B[1,0,0]={B[1, 0, 0]}, B[4,0,0]={B[4, 0, 0]}")

    # ── (5) Upwinded mass-exchange present (Piecewise in B[ℓ, ℓ']) ─
    # The transfer terms produce Piecewise atoms in the noncon matrix
    # (rows for layer ℓ momentum, columns for OTHER layer's q_0).
    def _has_piecewise(expr):
        return expr.has(sp.Piecewise)
    if not _has_piecewise(B[1, 1, 0]) and not _has_piecewise(B[1, 4, 0]):
        _fail(f"no upwind Piecewise found in layer-1 momentum noncon coupling")
    _ok(f"upwind Piecewise transfer terms present (mass exchange enforced)")

    # ── (6) ML-SWE limit (N=0): global momentum reduces to SWE ────
    try:
        mlswe = MLSME(
            N_layers=2,
            N=0,
            parameters={"g": (9.81, "positive"), "rho": (1.0, "positive")},
            boundary_conditions=boundary_conditions,
        )
    except Exception as e:
        _fail("MLSME(N_layers=2, N=0) constructs (ML-SWE limit)", e)
    sm0 = SystemModel.from_model(mlswe)
    n_eq0 = 1 + 2 * 1  # 1 + L·(N+1) = 3
    if sm0.flux.shape != (n_eq0, 1):
        _fail(f"ML-SWE flux shape {sm0.flux.shape}, expected ({n_eq0}, 1)")

    # Sum the two layer-momentum rows under u_1 = u_2 = U
    # (equivalent to q_ℓ = α_ℓ·H·U and q_ℓ_x linearly: q_1 = q_2 = Q/2,
    # q_1_x = q_2_x = Q_x/2 where Q = q_1+q_2 and Q_x = ∂_x Q).
    H_var = mlswe.variables.H
    q1, q2 = mlswe.variables.q_layer_1_0, mlswe.variables.q_layer_2_0
    g_par = mlswe.parameters.g
    b_x_sym = sp.Symbol("b_x", real=True)
    H_x = sp.Symbol("H_x", real=True)
    q1_x = sp.Symbol("q_layer_1_0_x", real=True)
    q2_x = sp.Symbol("q_layer_2_0_x", real=True)

    def _physical_rhs(i):
        F = sm0.flux[i, 0]
        P = sm0.hydrostatic_pressure[i, 0]
        F_x = (sp.diff(F, H_var) * H_x + sp.diff(F, q1) * q1_x
               + sp.diff(F, q2) * q2_x)
        P_x = (sp.diff(P, H_var) * H_x + sp.diff(P, q1) * q1_x
               + sp.diff(P, q2) * q2_x)
        Bterm = (sm0.nonconservative_matrix[i, 0, 0] * H_x
                 + sm0.nonconservative_matrix[i, 1, 0] * q1_x
                 + sm0.nonconservative_matrix[i, 2, 0] * q2_x)
        return F_x + P_x + Bterm - sm0.source[i, 0]

    total_rhs = sp.simplify(_physical_rhs(1) + _physical_rhs(2))
    Q_tot = sp.Symbol("Q", positive=True)
    Q_x = sp.Symbol("Q_x", real=True)
    uniform = {q1: Q_tot / 2, q2: Q_tot / 2,
               q1_x: Q_x / 2, q2_x: Q_x / 2}
    total_uniform = sp.simplify(total_rhs.xreplace(uniform))
    # Expected SWE physical RHS: ∂_x(Q²/H + g·H²/2) + g·H·∂_x b
    swe_expected = sp.simplify(
        -Q_tot**2 * H_x / H_var**2 + 2 * Q_tot * Q_x / H_var
        + g_par * H_var * H_x + g_par * H_var * b_x_sym
    )
    if sp.simplify(total_uniform - swe_expected) != 0:
        _fail(f"ML-SWE limit FAIL:\n   got     : {total_uniform}\n   expected: {swe_expected}\n   diff    : {sp.simplify(total_uniform - swe_expected)}")
    _ok(f"ML-SWE limit: under uniform velocity, sum of layer momenta reduces to SWE exactly")

    # ── (7) Strict missing-parameter ValueError ──────────────────
    try:
        MLSME(
            N_layers=2,
            N=2,
            parameters={"rho": (1.0, "positive")},
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
    print("ALL OK — MLSME(Audusse) → SystemModel bridge passes baseline;")
    print("         MLSME(N=0) cleanly recovers ML-SWE under uniform velocity.")


if __name__ == "__main__":
    main()
