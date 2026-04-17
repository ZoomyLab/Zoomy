"""Shared pytest fixtures.

All plotting tests run with matplotlib's ``Agg`` backend so they're headless,
and ``CONFIG`` is reset between tests so mutations don't leak.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # must happen before any pyplot import

import numpy as np
import pytest

from zoomy_plotting.plot.style import reset_config
from zoomy_plotting.store import SimulationStore, Zstruct


# --------------------------------------------------------------------------
# Autouse: clean CONFIG state between tests.
# --------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_config():
    reset_config()
    yield
    reset_config()


# --------------------------------------------------------------------------
# Helper: build a store backed by an in-memory array (no reader closure
# that needs a file).
# --------------------------------------------------------------------------
def _make_store(dim, cell_type, vertices, cells, Qs, field_names, times=None):
    """Build a SimulationStore whose cell reader serves from an in-memory
    array.

    Qs : (n_snaps, n_vars, n_cells) ndarray.
    """
    Qs = np.asarray(Qs, dtype=float)
    assert Qs.ndim == 3

    def cell_reader(t, idx):
        return np.asarray(Qs[int(t), int(idx), :])

    return SimulationStore(
        dim=dim,
        cell_type=cell_type,
        vertices=np.asarray(vertices, dtype=float),
        cells=np.asarray(cells, dtype=np.int64),
        n_inner_cells=int(Qs.shape[2]),
        times=np.asarray(times) if times is not None else None,
        field=Zstruct({name: i for i, name in enumerate(field_names)}),
        _cell_reader=cell_reader,
    )


# --------------------------------------------------------------------------
# 1-D store (20 cells, 2 variables, 5 snapshots)
# --------------------------------------------------------------------------
@pytest.fixture
def store_1d():
    n = 20
    verts = np.linspace(0.0, 1.0, n + 1)[:, None]
    cells = np.column_stack([np.arange(n), np.arange(1, n + 1)])
    x = 0.5 * (verts[:-1, 0] + verts[1:, 0])
    snaps = 5
    Qs = np.empty((snaps, 2, n))
    for t in range(snaps):
        Qs[t, 0] = np.sin(2 * np.pi * (x + 0.1 * t))   # "h"
        Qs[t, 1] = np.cos(2 * np.pi * (x + 0.1 * t))   # "u"
    return _make_store(
        dim=1, cell_type="line",
        vertices=verts, cells=cells, Qs=Qs,
        field_names=["h", "u"], times=np.arange(snaps, dtype=float),
    )


# --------------------------------------------------------------------------
# 2-D quad store (5x5)
# --------------------------------------------------------------------------
@pytest.fixture
def store_2d_quad():
    nx, ny = 5, 5
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(x, y)
    verts = np.column_stack([xx.ravel(), yy.ravel()])
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            cells.append([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
    cells = np.asarray(cells, dtype=np.int64)
    centers = verts[cells].mean(axis=1)
    snaps = 3
    Qs = np.empty((snaps, 2, len(cells)))
    for t in range(snaps):
        Qs[t, 0] = centers[:, 0] + 0.1 * t
        Qs[t, 1] = centers[:, 1] + 0.1 * t
    return _make_store(
        dim=2, cell_type="quad",
        vertices=verts, cells=cells, Qs=Qs,
        field_names=["h", "u"], times=np.arange(snaps, dtype=float),
    )


# --------------------------------------------------------------------------
# 2-D triangle store (8 triangles covering the unit square)
# --------------------------------------------------------------------------
@pytest.fixture
def store_2d_tri():
    verts = np.array([
        [0.0, 0.0], [0.5, 0.0], [1.0, 0.0],
        [0.0, 0.5], [0.5, 0.5], [1.0, 0.5],
        [0.0, 1.0], [0.5, 1.0], [1.0, 1.0],
    ], dtype=float)
    cells = np.array([
        [0, 1, 4], [0, 4, 3],
        [1, 2, 5], [1, 5, 4],
        [3, 4, 7], [3, 7, 6],
        [4, 5, 8], [4, 8, 7],
    ], dtype=np.int64)
    centers = verts[cells].mean(axis=1)
    Qs = np.empty((1, 1, len(cells)))
    Qs[0, 0] = centers[:, 0] + centers[:, 1]
    return _make_store(
        dim=2, cell_type="triangle",
        vertices=verts, cells=cells, Qs=Qs, field_names=["h"],
    )


# --------------------------------------------------------------------------
# 3-D hex store (unit cube, 3x3x3 hexes)
# --------------------------------------------------------------------------
@pytest.fixture
def store_3d_hex():
    nx = ny = nz = 3
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 1.0, ny + 1)
    z = np.linspace(0.0, 1.0, nz + 1)

    # Vertex grid, ordered k slowest, then j, then i.
    verts = np.array([[xi, yi, zi] for zi in z for yi in y for xi in x], dtype=float)

    def idx(i, j, k):
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

    cells = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # Hexahedron vertex ordering: bottom CCW then top CCW.
                cells.append([
                    idx(i, j, k), idx(i + 1, j, k),
                    idx(i + 1, j + 1, k), idx(i, j + 1, k),
                    idx(i, j, k + 1), idx(i + 1, j, k + 1),
                    idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1),
                ])
    cells = np.asarray(cells, dtype=np.int64)
    centers = verts[cells].mean(axis=1)
    snaps = 2
    Qs = np.empty((snaps, 1, len(cells)))
    for t in range(snaps):
        Qs[t, 0] = np.linalg.norm(centers - 0.5, axis=1) + 0.1 * t
    return _make_store(
        dim=3, cell_type="hexahedron",
        vertices=verts, cells=cells, Qs=Qs,
        field_names=["phi"], times=np.arange(snaps, dtype=float),
    )


# --------------------------------------------------------------------------
# 3-D tetra store (single tet split cube would be complex; use a tiny
# mesh of 5 tetrahedra filling the unit cube)
# --------------------------------------------------------------------------
@pytest.fixture
def store_3d_tet():
    verts = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    # Classic 5-tet decomposition of a cube.
    cells = np.array([
        [0, 1, 2, 5],
        [0, 2, 3, 7],
        [0, 5, 7, 4],
        [2, 5, 6, 7],
        [0, 2, 5, 7],
    ], dtype=np.int64)
    centers = verts[cells].mean(axis=1)
    Qs = np.empty((1, 1, len(cells)))
    Qs[0, 0] = centers[:, 2]  # height
    return _make_store(
        dim=3, cell_type="tetra",
        vertices=verts, cells=cells, Qs=Qs, field_names=["z"],
    )
