"""Plane-wave dispersion analysis on the chain-DAE SystemModel.

``SystemModel.from_model(VAMModelGalerkin)`` now returns the 7-state
chain DAE directly (no legacy 6-state operator API).  This test
verifies the dispersion pipeline runs end-to-end and returns the
expected dict structure.
"""
from __future__ import annotations

import pytest
import sympy as sp


@pytest.fixture(scope="module")
def vam_122_systemmodel():
    """``SystemModel.from_model(VAMModelGalerkin(level=1))`` linearised
    around the rest state."""
    from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
    from zoomy_core.systemmodel.system_model import SystemModel

    m = VAMModelGalerkin(level=1)
    sm = SystemModel.from_model(m)
    return m, sm


def _rest_base_state(sm, h0):
    """Build a rest base-state dict for VAM(1,2,2): h → h0, all
    velocity / pressure modes → 0."""
    base = {}
    for s in sm.state:
        name = str(s)
        if name == "h":
            base[s] = h0
        else:
            base[s] = sp.Integer(0)
    return base


def test_plane_wave_dispersion_returns_omega_k(vam_122_systemmodel):
    """``return_omega_k=True`` (default) yields ``omega_solutions`` and
    ``phase_velocity_solutions`` keys with non-trivial content."""
    from zoomy_core.analysis.system_model_analysis import (
        plane_wave_dispersion,
    )

    m, sm = vam_122_systemmodel
    h0 = sp.Symbol("h0", positive=True)
    base_state = _rest_base_state(sm, h0)
    ez_param = next(s for s, v in sm.parameters.items() if str(s) == "ez")

    result = plane_wave_dispersion(
        sm, base_state, axis=0, parameters={ez_param: 1},
    )

    assert "omega" in result
    assert "k" in result
    assert "omega_solutions" in result
    assert "phase_velocity_solutions" in result
    assert "dispersion_matrix" in result
    assert "dispersion_determinant" in result
    assert "eigenvalues" in result

    omega_solutions = result["omega_solutions"]
    assert isinstance(omega_solutions, list)


def test_plane_wave_dispersion_legacy_mode(vam_122_systemmodel):
    """``return_omega_k=False`` returns just the eigenvalues."""
    from zoomy_core.analysis.system_model_analysis import (
        plane_wave_dispersion,
    )

    m, sm = vam_122_systemmodel
    h0 = sp.Symbol("h0", positive=True)
    base_state = _rest_base_state(sm, h0)
    ez_param = next(s for s, v in sm.parameters.items() if str(s) == "ez")

    result = plane_wave_dispersion(
        sm, base_state, axis=0, parameters={ez_param: 1},
        return_omega_k=False,
    )
    assert "eigenvalues" in result
    assert "omega_solutions" not in result
    assert "phase_velocity_solutions" not in result


def _find_b_symbol(sm):
    """Locate the bathymetry Symbol ``b`` — now a STATE entry (with
    trivial evolution ``∂_t b = 0`` from the ``bathymetry`` row in
    :class:`VAMModelGalerkin`).  Returns ``None`` if absent."""
    for s in sm.state:
        if str(s) == "b":
            return s
    return None


def test_plane_wave_dispersion_matches_escalante_eq10(vam_122_systemmodel):
    """VAM(1,2,2) at rest with flat bottom reproduces Escalante eq (10):

        C² / (g H)  =  (1 + (kH)²/12) / (1 + 5(kH)²/12 + (kH)⁴/144)

    equivalently the dispersion polynomial

        ω²·(1 + 5(kH)²/12 + (kH)⁴/144)  =  g·H·k²·(1 + (kH)²/12).
    """
    from zoomy_core.analysis.system_model_analysis import (
        plane_wave_dispersion,
    )

    m, sm = vam_122_systemmodel
    h0 = sp.Symbol("h0", positive=True)
    base_state = _rest_base_state(sm, h0)
    b_sym = _find_b_symbol(sm)
    assert b_sym is not None, "Expected state Symbol 'b' in VAMModelGalerkin"
    # Flat bottom: zero ``b`` in the state base AND every derivative-
    # aux entry whose target is ``b`` (``b_x``, ``b_y``, …).
    base_state[b_sym] = sp.S.Zero
    for s in sm.aux_state:
        if str(s).startswith("b_"):
            base_state[s] = sp.S.Zero
    ez_param = next(s for s, v in sm.parameters.items() if str(s) == "ez")
    g_param = next(s for s in sm.parameters if str(s) == "g")
    rho_param = next(s for s in sm.parameters if str(s) == "rho")

    result = plane_wave_dispersion(
        sm, base_state, axis=0, parameters={ez_param: 1},
    )

    omega = result["omega"]
    k = result["k"]
    det = sp.simplify(result["dispersion_determinant"])

    # The 8-state dispersion matrix carries an extra ``i·omega`` factor
    # from the trivial ``bathymetry`` row (``∂_t b = 0`` has ω = 0 as
    # its only eigenvalue, contributing a cofactor of ``iω`` to the
    # determinant).  The algebraic constraint rows and chain's
    # structural constants contribute the original ``prefactor``;
    # both factors must be stripped to isolate the Escalante
    # surface-wave polynomial.
    prefactor = -sp.I * h0**2 * omega / (27 * rho_param**2)
    det_poly = sp.expand(det / (prefactor * sp.I * omega))

    # Escalante eq (10) polynomial.  det_poly = 144 × Escalante (sign
    # is irrelevant for det = 0).
    escalante = (omega**2 * (1 + 5*(k*h0)**2/12 + (k*h0)**4/144)
                 - g_param*h0*k**2 * (1 + (k*h0)**2/12))
    diff = sp.expand(det_poly - 144 * escalante)
    assert sp.simplify(diff) == 0, (
        f"Dispersion polynomial differs from Escalante eq (10): "
        f"det/(prefactor·iω) − 144·Escalante = {diff}"
    )

    # Non-trivial ω solutions + long-wave limit c² → g·H.
    omega_sols = result["omega_solutions"]
    nontrivial = [s for s in omega_sols if s != 0]
    assert len(nontrivial) == 2, (
        f"Expected 2 non-trivial ω(k) branches, got {len(nontrivial)}"
    )
    c_sq = sp.simplify(sp.limit((nontrivial[0] / k)**2, k, 0))
    assert sp.simplify(c_sq - g_param * h0) == 0, (
        f"Long-wave limit c² = {c_sq}, expected g·h0"
    )
