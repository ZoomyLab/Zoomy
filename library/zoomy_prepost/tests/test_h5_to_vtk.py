"""#2 test_h5_to_vtk — zoomy HDF5 store -> .vtu series + .pvd (absorbs
test_steps; together with test_vtk_to_h5 this is the round-trip pair).

Two stores are exercised:

* the COMMITTED SME1 fixture (SME(level=1) 1-D dam break, numpy backend,
  3 snapshots, Q + Qaux) — the real solver artifact round-trips through
  ``hdf5_to_vtk(include_aux=True)`` and back via ``vtk_to_hdf5``;
* a synthetic store whose Q/Qaux carry GHOST COLUMNS (Q wider than
  n_inner_cells, like a raw solver dump) — the converter must drop them.
"""
from __future__ import annotations

import os

import h5py
import meshio
import numpy as np
import pytest

from zoomy_prepost import hdf5_to_vtk, vtk_to_hdf5

pytestmark = [pytest.mark.small, pytest.mark.postprocessing]

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
SME1_H5 = os.path.join(FIXTURES, "sme1", "simulation.h5")

#: SME(level=1) state / NSM-aux rows of the committed fixture
SME1_FIELDS = ["b", "h", "q_0", "q_1"]
SME1_AUX = ["dq1dx", "dq0dx", "dhdx", "dbdx", "hinv"]


def _load_store(path):
    with h5py.File(path) as f:
        n_inner = int(f["mesh/n_inner_cells"][()])
        frames = []
        for i in range(len(f["fields"])):
            g = f[f"fields/iteration_{i}"]
            frames.append((float(g["time"][()]),
                           g["Q"][()], g["Qaux"][()] if "Qaux" in g else None))
    return n_inner, frames


def test_sme1_store_to_vtk_named_and_back(tmp_path):
    """The committed SME1 run exports with NAMED Q + Qaux rows and the
    vtu/pvd series converts back to the identical inner-cell data."""
    n_inner, frames = _load_store(SME1_H5)
    assert n_inner == 20 and len(frames) == 3
    assert frames[0][2] is not None, "SME1 fixture must carry Qaux"

    pvd = hdf5_to_vtk(SME1_H5, str(tmp_path), basename="sim",
                      include_aux=True, field_names=SME1_FIELDS,
                      aux_field_names=SME1_AUX)
    assert os.path.exists(pvd)
    assert os.path.exists(tmp_path / "sim_0000.vtu")
    assert os.path.exists(tmp_path / "sim_0002.vtu")

    # ParaView-visible names are the state + aux symbols
    m0 = meshio.read(str(tmp_path / "sim_0000.vtu"))
    assert set(m0.cell_data) == set(SME1_FIELDS) | set(SME1_AUX)

    # round-trip: [Q; Qaux] on the inner cells survives unchanged
    rt = vtk_to_hdf5(pvd, str(tmp_path / "rt.h5"))
    with h5py.File(rt) as f:
        names = [n.decode() for n in f["fields"].attrs["names"]]
        assert set(names) == set(SME1_FIELDS) | set(SME1_AUX)
        assert len(f["fields"]) == 3
        for i, (t, Q, Qaux) in enumerate(frames):
            g = f[f"fields/iteration_{i}"]
            assert abs(float(g["time"][()]) - t) < 1e-12
            got = g["Q"][()]
            expected = np.vstack([Q[:, :n_inner], Qaux[:, :n_inner]])
            # rows may be name-ordered — compare via the name table
            order = [names.index(n) for n in SME1_FIELDS + SME1_AUX]
            np.testing.assert_allclose(got[order], expected,
                                       rtol=1e-12, atol=1e-14)


def _make_ghost_store(path, n_cells=12, n_ghost=2, n_vars=2, n_aux=1, nsnap=2):
    """A 1-D store whose Q/Qaux carry ghost columns (raw solver layout)."""
    x = np.linspace(0.0, 1.0, n_cells + 1)
    cells = np.vstack([np.arange(n_cells), np.arange(1, n_cells + 1)])
    with h5py.File(path, "w") as f:
        g = f.create_group("mesh")
        g.create_dataset("dimension", data=1)
        g.create_dataset("type", data="line")
        g.create_dataset("n_cells", data=n_cells + n_ghost)
        g.create_dataset("n_inner_cells", data=n_cells)
        g.create_dataset("vertex_coordinates", data=x[None, :])
        g.create_dataset("cell_vertices", data=cells)
        fields = f.create_group("fields")
        for i in range(nsnap):
            Q = np.arange(n_vars * (n_cells + n_ghost), dtype=float
                          ).reshape(n_vars, -1) + 1000.0 * i
            Qaux = -np.arange(n_aux * (n_cells + n_ghost), dtype=float
                              ).reshape(n_aux, -1) - 7.0 * i
            it = fields.create_group(f"iteration_{i}")
            it.create_dataset("time", data=0.25 * i, dtype=float)
            it.create_dataset("Q", data=Q)
            it.create_dataset("Qaux", data=Qaux)


def test_ghost_columns_dropped(tmp_path):
    """Q/Qaux ghost columns (beyond n_inner_cells) never reach the VTK."""
    store = str(tmp_path / "ghosts.h5")
    _make_ghost_store(store)

    pvd = hdf5_to_vtk(store, str(tmp_path), basename="g", include_aux=True,
                      field_names=["h", "hu"], aux_field_names=["hinv"])
    m0 = meshio.read(str(tmp_path / "g_0000.vtu"))
    assert set(m0.cell_data) == {"h", "hu", "hinv"}
    for name, blocks in m0.cell_data.items():
        assert np.asarray(blocks[0]).shape == (12,), name  # inner cells only

    rt = vtk_to_hdf5(pvd, str(tmp_path / "rt.h5"))
    with h5py.File(rt) as f:
        q0 = f["fields/iteration_0/Q"][()]
        assert q0.shape == (3, 12)
        # h row of snapshot 0 was arange(14)[:12]
        np.testing.assert_allclose(q0[0], np.arange(12, dtype=float))
