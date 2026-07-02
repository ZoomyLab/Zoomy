"""Round-trip: synthetic VTK series -> zoomy HDF5 -> zoomy_plotting reads it.

Uses a realistically-sized triangulated grid (n_cells >> vertices/cell) so the
reader's axis-orientation heuristic is unambiguous.
"""
import os
import tempfile

import numpy as np
import meshio

from zoomy_prepost import vtk_to_hdf5


def _grid_tris(n=6):
    xs = np.linspace(0.0, 1.0, n + 1)
    ys = np.linspace(0.0, 1.0, n + 1)
    pts = np.array([[x, y, 0.0] for y in ys for x in xs], float)
    tris = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b, c, d = a + 1, a + (n + 1), a + (n + 1) + 1
            tris += [[a, b, c], [b, d, c]]
    return pts, np.array(tris)


def _make_series(d, n=6, nsnap=3):
    pts, tris = _grid_tris(n)
    n_cells = tris.shape[0]
    files = []
    for i in range(nsnap):
        # per-cell field h = cell_index + 100*snapshot, so get_cell is checkable
        h = np.arange(n_cells, dtype=float) + 100.0 * i
        m = meshio.Mesh(pts, [("triangle", tris)], cell_data={"h": [h], "hu": [h * 0.0]})
        fp = os.path.join(d, f"f_{i}.vtu")
        meshio.write(fp, m)
        files.append(fp)
    return files, n_cells


def test_vtk_series_to_hdf5_roundtrip():
    d = tempfile.mkdtemp()
    files, n_cells = _make_series(d)
    out = os.path.join(d, "conv.h5")
    vtk_to_hdf5(files, out)

    import zoomy_plotting as zp
    s = zp.read_hdf5(out)
    assert s.dim == 2, s.dim
    assert s.n_cells == n_cells, (s.n_cells, n_cells)
    assert s.n_snapshots == 3, s.n_snapshots
    # field 0 == "h": cell c at snapshot 2 -> c + 200
    got = np.asarray(s.get_cell(2, 0)).ravel()
    np.testing.assert_allclose(got, np.arange(n_cells) + 200.0)
    print(f"OK: dim={s.dim} type={s.cell_type} n_cells={s.n_cells} snaps={s.n_snapshots}")


if __name__ == "__main__":
    test_vtk_series_to_hdf5_roundtrip()
