"""#1 test_vtk_to_h5 — committed tiny VTK series -> zoomy HDF5.

Covers (absorbing test_vtk_to_hdf5 + test_readers_hdf5 + the test_store
overlap):

* conversion of the COMMITTED 3-step triangle series reproduces the
  committed H5 twin (mesh datasets + per-iteration time/Q),
* fields resolve BY NAME — the REQ-158 pin: ``Rank`` is the FIRST CellData
  entry and must never shift positional reads,
* ``read_hdf5`` read-back: canonical axes, metadata, lazy handle, close.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from zoomy_prepost import vtk_to_hdf5
from zoomy_plotting import read_hdf5

pytestmark = [pytest.mark.small, pytest.mark.postprocessing]

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
PVD = os.path.join(FIXTURES, "vtk_series", "series.pvd")
TWIN = os.path.join(FIXTURES, "series.h5")


def test_committed_series_converts_to_twin(tmp_path):
    """vtk_to_hdf5 on the committed series reproduces the committed twin."""
    out = str(tmp_path / "conv.h5")
    vtk_to_hdf5(PVD, out)
    with h5py.File(out) as got, h5py.File(TWIN) as ref:
        assert [n.decode() for n in got["fields"].attrs["names"]] \
            == [n.decode() for n in ref["fields"].attrs["names"]] \
            == ["Rank", "b", "h"]
        for key in ("dimension", "n_cells", "n_inner_cells"):
            assert int(got[f"mesh/{key}"][()]) == int(ref[f"mesh/{key}"][()])
        np.testing.assert_allclose(got["mesh/vertex_coordinates"][()],
                                   ref["mesh/vertex_coordinates"][()])
        np.testing.assert_array_equal(got["mesh/cell_vertices"][()],
                                      ref["mesh/cell_vertices"][()])
        assert len(got["fields"]) == len(ref["fields"]) == 3
        for i in range(3):
            g, r = got[f"fields/iteration_{i}"], ref[f"fields/iteration_{i}"]
            assert float(g["time"][()]) == float(r["time"][()])
            np.testing.assert_allclose(g["Q"][()], r["Q"][()],
                                       rtol=1e-14, atol=1e-15)


def test_fields_resolve_by_name_req158(tmp_path):
    """REQ-158: ``Rank`` is stored FIRST; ``h`` must resolve by NAME, never
    positionally (a positional read would hand back the row before it)."""
    with read_hdf5(TWIN) as store:
        centers = store.vertices[store.cells].mean(axis=1)
        cx = centers[:, 0]
        # generation formulas of the committed series
        b_ref = 0.2 * cx
        h_ref0 = np.maximum(0.0, 1.0 - 0.6 * cx)
        np.testing.assert_allclose(store.get_cell(0, "Rank"), 0.0)
        np.testing.assert_allclose(store.get_cell(0, "b"), b_ref, atol=1e-12)
        np.testing.assert_allclose(store.get_cell(0, "h"), h_ref0, atol=1e-12)
        # snapshot 2 raises the water level by 0.2 — still resolved by name
        h_ref2 = np.maximum(0.0, 1.0 - 0.6 * cx + 0.2)
        np.testing.assert_allclose(store.get_cell(2, "h"), h_ref2, atol=1e-12)

    # exclude= drops the noise row entirely and re-derives the name table
    out = str(tmp_path / "no_rank.h5")
    vtk_to_hdf5(PVD, out, exclude=("Rank",))
    with h5py.File(out) as f:
        assert [n.decode() for n in f["fields"].attrs["names"]] == ["b", "h"]
    with read_hdf5(out) as store:
        np.testing.assert_allclose(store.get_cell(0, "b"), b_ref, atol=1e-12)


def test_read_hdf5_axes_metadata_lazy_close():
    """Canonical axes + metadata + lazy handle + context-manager close."""
    store = read_hdf5(TWIN)
    try:
        assert store.dim == 2
        assert store.cell_type == "triangle"
        assert store.vertices.shape == (20, 2)      # (n_vertices, dim)
        assert store.cells.shape == (24, 3)          # (n_cells, k)
        assert store.n_cells == 24
        assert store.n_snapshots == 3
        np.testing.assert_allclose(store.times, [0.0, 0.5, 1.0])
        assert list(store.field.keys()) == ["Rank", "b", "h"]
        # lazy: the handle is held, reads are per-(step, field) slices
        assert store._resource is not None
        v = store.get_cell(1, "h")
        assert v.shape == (24,)
        with pytest.raises(IndexError):
            store.get_cell(99, "h")
    finally:
        store.close()
    # context manager releases the handle
    with read_hdf5(TWIN) as store2:
        assert store2._resource is not None
        store2.get_cell(0, 0)
    assert store2._resource is None
