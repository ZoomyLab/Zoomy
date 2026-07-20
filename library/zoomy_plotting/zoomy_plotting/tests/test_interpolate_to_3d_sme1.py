"""#8 test_interpolate_to_3d_sme1 — ``interpolate_to_3d`` on the committed
SME1 fixture (NEW, per spec): pins the moment-reconstruction family incl.
the x h handling (state_from_reconstruction / project_from_3d x h bug family).

``zoomy_prepost.lift3d`` lifts the committed 1-D SME(level=1) dam-break store
through the MODEL'S OWN symbolic ``interpolate_to_3d`` (model resolved from
the committed fixture case folder — the full derivation chain, warm cache):

* line mesh -> quad extrusion, one .vtu per snapshot + .pvd,
* h column equals the store h — no stray x h,
* the depth MEAN of u(zeta) is exactly q_0/h (u_mean * h == q_0),
* u is LINEAR in zeta with top-bottom shear 2 q_1/h (level-1 Legendre),
* p is the hydrostatic column rho g h (1 - zeta).
"""
from __future__ import annotations

import os

import h5py
import meshio
import numpy as np
import pytest

from zoomy_prepost import lift3d

pytestmark = [pytest.mark.small, pytest.mark.postprocessing]

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SME1_DIR = os.path.join(FIXTURES, "sme1")
SME1_H5 = os.path.join(SME1_DIR, "simulation.h5")

NZ = 6
N = 20
G, RHO = 9.81, 1.0


@pytest.fixture(scope="module")
def lifted(tmp_path_factory):
    out = tmp_path_factory.mktemp("lift3d")
    pvd = lift3d(SME1_DIR, SME1_H5, str(out), Nz=NZ)
    return out, pvd


def _store_frame(i):
    with h5py.File(SME1_H5) as f:
        Q = f[f"fields/iteration_{i}/Q"][()]
    return Q  # rows [b, h, q_0, q_1], inner cells only


def test_lift3d_writes_extruded_series(lifted):
    out, pvd = lifted
    assert os.path.exists(pvd)
    vtus = sorted(f for f in os.listdir(out) if f.endswith(".vtu"))
    assert len(vtus) == 3                       # one per snapshot
    m = meshio.read(os.path.join(out, vtus[0]))
    types = {c.type for c in m.cells}
    assert types == {"quad"}                    # line -> quad extrusion
    n3d = sum(len(c.data) for c in m.cells)
    assert n3d == N * NZ
    assert set(m.cell_data) == {"b", "h", "u", "v", "w", "p"}


def test_lift3d_pins_moment_reconstruction(lifted):
    """The x h family + level-1 profile shape, checked on every snapshot."""
    out, _ = lifted
    zc = (np.arange(NZ) + 0.5) / NZ
    for i in range(3):
        Q = _store_frame(i)
        b, h, q0, q1 = Q[0], Q[1], Q[2], Q[3]
        m = meshio.read(os.path.join(out, f"simulation_3d_{i:04d}.vtu"))
        # layer-major layout: cell (iz, ix) at row iz*N + ix
        F = {k: np.asarray(v[0], dtype=float).reshape(NZ, N)
             for k, v in m.cell_data.items()}

        np.testing.assert_allclose(F["b"], np.broadcast_to(b, (NZ, N)),
                                   atol=1e-14)
        np.testing.assert_allclose(F["h"], np.broadcast_to(h, (NZ, N)),
                                   rtol=1e-12)
        np.testing.assert_allclose(F["v"], 0.0, atol=1e-14)

        u = F["u"]
        # depth mean over the symmetric midpoint layers == q0/h EXACTLY:
        # u_mean * h == q_0 (a dropped or doubled x h breaks this first)
        np.testing.assert_allclose(u.mean(axis=0) * h, q0,
                                   rtol=1e-10, atol=1e-12)
        # level-1 Legendre: u linear in zeta -> vanishing second differences
        d2 = np.diff(u, n=2, axis=0)
        np.testing.assert_allclose(d2, 0.0, atol=1e-10)
        # top-bottom shear = 2 (q1/h) (Nz-1)/Nz on the layer-centre grid
        np.testing.assert_allclose(u[-1] - u[0],
                                   2.0 * (q1 / h) * (NZ - 1) / NZ,
                                   rtol=1e-10, atol=1e-12)

        # hydrostatic pressure column rho g h (1 - zeta), zero at the surface
        p_expected = RHO * G * h[None, :] * (1.0 - zc[:, None])
        np.testing.assert_allclose(F["p"], p_expected, rtol=1e-10)

        # w comes from the REAL store gradients (Qaux) — finite everywhere,
        # and nonzero somewhere once the dam break is moving (i >= 1)
        assert np.isfinite(F["w"]).all()
        if i >= 1:
            assert np.abs(F["w"]).max() > 0.0


def test_lift3d_pvd_carries_store_times(lifted):
    out, pvd = lifted
    with h5py.File(SME1_H5) as f:
        times = [float(f[f"fields/iteration_{i}/time"][()]) for i in range(3)]
    txt = open(pvd).read()
    for t in times:
        assert f'timestep="{t}"' in txt
