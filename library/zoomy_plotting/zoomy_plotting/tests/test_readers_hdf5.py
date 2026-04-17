"""Tests for the HDF5 reader — lazy loading + axis transpose."""

from __future__ import annotations

import os
import sys
import tempfile

import h5py
import numpy as np
import pytest

from zoomy_plotting import read_hdf5


def _write_fixture(path, *, dim=1, cell_type="line",
                   n_vertices=21, n_cells=20, n_vars=2, n_aux=1,
                   n_snaps=3):
    """Write a minimal zoomy_core-schema HDF5 file."""
    with h5py.File(path, "w") as f:
        g = f.create_group("mesh")
        g.attrs["format_version"] = 2
        g.create_dataset("dimension", data=dim)
        g.create_dataset("type", data=cell_type)
        g.create_dataset("n_cells", data=n_cells)
        g.create_dataset("n_inner_cells", data=n_cells)
        # zoomy_core stores vertex_coordinates as (dim, n)
        coords = np.zeros((dim, n_vertices), dtype=float)
        coords[0] = np.linspace(0.0, 1.0, n_vertices)
        g.create_dataset("vertex_coordinates", data=coords)
        # cell_vertices as (k, n_cells); for 'line' k=2
        k = {"line": 2, "triangle": 3, "quad": 4,
             "tetra": 4, "hexahedron": 8, "wedge": 6, "pyramid": 5}[cell_type]
        cv = np.zeros((k, n_cells), dtype=np.int64)
        cv[0] = np.arange(n_cells)
        cv[1] = np.arange(1, n_cells + 1) % n_vertices
        g.create_dataset("cell_vertices", data=cv)

        fg = f.create_group("fields")
        for t in range(n_snaps):
            it = fg.create_group(f"iteration_{t}")
            it.create_dataset("time", data=float(t))
            it.create_dataset(
                "Q",
                data=np.stack([
                    np.full(n_cells, float(t) + float(v) * 0.01)
                    for v in range(n_vars)
                ], axis=0),
            )
            if n_aux > 0:
                it.create_dataset(
                    "Qaux",
                    data=np.stack([
                        np.full(n_cells, 100.0 + t * 10 + a)
                        for a in range(n_aux)
                    ], axis=0),
                )


def test_hdf5_reader_axes_and_metadata(tmp_path):
    path = tmp_path / "mini_1d.h5"
    _write_fixture(str(path), dim=1, n_vertices=21, n_cells=20, n_vars=2, n_aux=1)
    store = read_hdf5(str(path))
    try:
        assert store.dim == 1
        assert store.cell_type == "line"
        assert store.vertices.shape == (21, 1)        # (n, dim) canonical
        assert store.cells.shape == (20, 2)            # (n, k) canonical
        # n_q == 2, n_aux == 1 → 3 total fields; names q0/q1/aux_0.
        assert list(store.field.keys()) == ["q0", "q1", "aux_0"]
        assert store.field.q0 == 0
        assert store.field.aux_0 == 2
        assert store.n_snapshots == 3
        np.testing.assert_array_equal(store.times, np.array([0.0, 1.0, 2.0]))
    finally:
        store.close()


def test_hdf5_reader_lazy_values(tmp_path):
    path = tmp_path / "mini_1d.h5"
    _write_fixture(str(path), dim=1, n_cells=20, n_vars=2, n_aux=1)
    store = read_hdf5(str(path))
    try:
        # Q[0] at t=0 should be all zeros (we filled with t + v*0.01 → 0.0).
        v_q0_t0 = store.get_cell(0, 0)
        assert v_q0_t0.shape == (20,)
        np.testing.assert_allclose(v_q0_t0, 0.0)
        # Q[1] at t=2 should be 2 + 0.01 = 2.01 everywhere.
        v_q1_t2 = store.get_cell(2, "q1")
        np.testing.assert_allclose(v_q1_t2, 2.01)
        # Qaux[0] at t=1 should be 110.0.
        v_aux_t1 = store.get_cell(1, "aux_0")
        np.testing.assert_allclose(v_aux_t1, 110.0)
    finally:
        store.close()


def test_hdf5_reader_bad_time_index(tmp_path):
    path = tmp_path / "mini_1d.h5"
    _write_fixture(str(path), dim=1, n_cells=5, n_vars=1, n_aux=0, n_snaps=2)
    store = read_hdf5(str(path))
    try:
        with pytest.raises(IndexError):
            store.get_cell(99, 0)
    finally:
        store.close()


def test_hdf5_reader_context_manager_closes_file(tmp_path):
    path = tmp_path / "mini_1d.h5"
    _write_fixture(str(path), dim=1, n_cells=5, n_vars=1, n_aux=0, n_snaps=1)
    with read_hdf5(str(path)) as store:
        _ = store.get_cell(0, 0)
        assert store._resource is not None
    # After __exit__, the handle should be released.
    assert store._resource is None


def test_hdf5_reader_is_lazy_memory_bound(tmp_path):
    """Opening a file with a large Q array should NOT materialize it."""
    path = tmp_path / "big.h5"
    n_cells = 50_000
    n_vars = 4
    n_snaps = 10
    _write_fixture(
        str(path), dim=1, n_vertices=n_cells + 1, n_cells=n_cells,
        n_vars=n_vars, n_aux=0, n_snaps=n_snaps,
    )

    # The file on disk is ~16 MB (10 snaps x 4 vars x 50000 floats x 8 bytes).
    assert os.path.getsize(str(path)) > 10_000_000

    store = read_hdf5(str(path))
    try:
        # The store itself must be small; no full Q array in memory.
        store_footprint = (
            sys.getsizeof(store)
            + store.vertices.nbytes
            + store.cells.nbytes
            + (store.times.nbytes if store.times is not None else 0)
        )
        # Well under the 16 MB the full timeline would consume.
        assert store_footprint < 5_000_000, (
            f"store footprint too large ({store_footprint} bytes) — the "
            f"reader is probably not lazy."
        )

        # Single field read works and returns just n_cells.
        v = store.get_cell(3, 2)
        assert v.shape == (n_cells,)
    finally:
        store.close()
