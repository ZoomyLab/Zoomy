"""#7 test_foam_readers — read_zoomyfoam / read_of_states / read_vof_raw on
the committed mini foam tree (NEW; 44 thesis call sites were unguarded).

The tree carries two OpenFOAM write-time dirs (``0/``, ``0.4/``) with
zoomyFoam states Q0..Q3 (SME(level=1) state ``[b, h, q_0, q_1]``) plus a
6x4 VoF block (``alpha.water`` + ``U``, both uniform- and nonuniform-format
variants). ``read_zoomyfoam`` lifts the frames through the MODEL'S OWN
symbolic ``interpolate_to_3d`` — the model is the committed SME1 fixture
model (derived through the full chain, warm derivation cache).
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from zoomy_plotting.column_plots import (
    read_of_states, read_vof_raw, read_zoomyfoam,
)

pytestmark = [pytest.mark.small, pytest.mark.postprocessing]

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
FOAM = os.path.join(FIXTURES, "foam_case")
MODEL_PY = os.path.join(FIXTURES, "sme1", "model.py")

N = 20
X = (np.arange(N) + 0.5) * 0.5


def _h(frame):
    return 1.5 - 0.05 * np.arange(N) + 0.1 * frame


def _q0(frame):
    return (0.8 + 0.2 * frame) * _h(frame)


def _q1(frame):
    return 0.2 * _h(frame) * (X / 10.0)


@pytest.fixture(scope="module")
def sme1_model():
    """The committed SME1 model — SME(level=1) through the full chain."""
    spec = importlib.util.spec_from_file_location("sme1_fixture_model", MODEL_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.model


def test_read_of_states_raw_frames():
    T, Q = read_of_states(FOAM, nq=4, n=N)
    np.testing.assert_allclose(T, [0.0, 0.4])
    assert Q.shape == (2, 4, N)
    np.testing.assert_allclose(Q[:, 0, :], 0.0)          # b: uniform-format
    for fi in range(2):
        np.testing.assert_allclose(Q[fi, 1], _h(fi), rtol=1e-12)
        np.testing.assert_allclose(Q[fi, 2], _q0(fi), rtol=1e-12)
        np.testing.assert_allclose(Q[fi, 3], _q1(fi), rtol=1e-12)


def test_read_zoomyfoam_lifts_through_model(sme1_model):
    """The canonical [b,h,u,v,w,p] columns come from the model's symbolic
    ``interpolate_to_3d`` — pins the x h / moment reconstruction family."""
    K = 8
    cf = read_zoomyfoam(FOAM, sme1_model, n_cells=N, x=X, K=K)
    assert set(cf.fields) == {"b", "h", "u", "v", "w", "p"}
    assert cf.fields["u"].shape == (2, N, K)
    np.testing.assert_allclose(cf.zeta, (np.arange(K) + 0.5) / K)

    g, rho = 9.81, 1.0
    for fi in range(2):
        h, q0, q1 = _h(fi), _q0(fi), _q1(fi)
        # b/h broadcast down the column unchanged
        np.testing.assert_allclose(cf.fields["b"][fi], 0.0)
        np.testing.assert_allclose(cf.fields["h"][fi],
                                   np.repeat(h[:, None], K, axis=1), rtol=1e-12)
        # u(zeta) = q0/h + (2 zeta - 1) q1/h: the midpoint zeta grid is
        # symmetric, so the depth MEAN is exactly q0/h (the x h family pin —
        # momenta divide by h, never a stray x h)
        u = cf.fields["u"][fi]
        np.testing.assert_allclose(u.mean(axis=1), q0 / h, rtol=1e-12)
        # the level-1 moment sets the top-bottom shear
        expected_shear = 2.0 * (q1 / h) * (K - 1) / K
        np.testing.assert_allclose(u[:, -1] - u[:, 0], expected_shear,
                                   rtol=1e-12, atol=1e-15)
        # v = 0 in the 1-horizontal model; w = 0 with zero aux gradients
        np.testing.assert_allclose(cf.fields["v"][fi], 0.0)
        np.testing.assert_allclose(cf.fields["w"][fi], 0.0, atol=1e-15)
        # hydrostatic pressure p = rho g h (1 - zeta)
        p_expected = rho * g * h[:, None] * (1.0 - cf.zeta[None, :])
        np.testing.assert_allclose(cf.fields["p"][fi], p_expected, rtol=1e-12)


def test_read_vof_raw_alpha_and_stations():
    nx, ny, lx, ly = 6, 4, 3.0, 1.0
    dy = ly / ny
    raw, cf = read_vof_raw(FOAM, nx, ny, lx, ly, stations=[0.75, 2.25])

    np.testing.assert_allclose(raw["t"], [0.0, 0.4])
    assert raw["alpha"].shape == (2, ny, nx)
    # committed block: rows 0/1 water, row 2 half-filled left half, row 3 air
    assert raw["alpha"][0, 0, 0] == 1.0 and raw["alpha"][0, 3, 0] == 0.0
    # station columns: left (col 1) sees the partial row, right (col 4) not
    np.testing.assert_allclose(cf.fields["h"][0, 0, :], 2.5 * dy)   # 0.625
    np.testing.assert_allclose(cf.fields["h"][0, 1, :], 2.0 * dy)   # 0.5
    # frame 0 exercises the UNIFORM vector branch: u = 0.3 on wet cells
    wet_u0 = cf.fields["u"][0]
    np.testing.assert_allclose(wet_u0, 0.3, atol=1e-12)
    # frame 1 (nonuniform U, shear in y): resampled profile is nondecreasing
    u1 = cf.fields["u"][1, 0, :]
    assert np.all(np.diff(u1) >= -1e-12)
    assert u1.min() >= 0.2 - 1e-12 and u1.max() <= 0.4 + 1e-12


def test_vof_raw_without_stations_returns_raw_only():
    raw, cf = read_vof_raw(FOAM, 6, 4, 3.0, 1.0)
    assert cf is None
    assert set(raw) == {"t", "alpha", "lx", "ly"}
