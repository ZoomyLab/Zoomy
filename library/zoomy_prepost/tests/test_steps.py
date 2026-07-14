"""Round-trip: synthetic zoomy HDF5 store -> .vtu series + .pvd (hdf5_to_vtk)
-> back to HDF5 (vtk_to_hdf5) — Q arrays and times must survive unchanged.

The synthetic store carries ghost-cell columns (Q wider than n_inner_cells)
like a real solver output; the converter must drop them.
"""
import os
import tempfile

import numpy as np
import h5py

from zoomy_prepost import hdf5_to_vtk, vtk_to_hdf5


def _make_store(path, n_cells=40, n_ghost=2, n_vars=3, nsnap=4):
    """A 1-D line-mesh store in the zoomy layout (mesh + fields)."""
    x = np.linspace(0.0, 1.0, n_cells + 1)
    verts = x[None, :]                                    # (dim, n_verts)
    cells = np.vstack([np.arange(n_cells), np.arange(1, n_cells + 1)])  # (2, n_cells)
    times, Qs = [], []
    with h5py.File(path, "w") as f:
        g = f.create_group("mesh")
        g.create_dataset("dimension", data=1)
        g.create_dataset("type", data="line")
        g.create_dataset("n_cells", data=n_cells + n_ghost)
        g.create_dataset("n_inner_cells", data=n_cells)
        g.create_dataset("vertex_coordinates", data=verts)
        g.create_dataset("cell_vertices", data=cells)
        fields = f.create_group("fields")
        for i in range(nsnap):
            Q = np.arange(n_vars * (n_cells + n_ghost), dtype=float
                          ).reshape(n_vars, -1) + 1000.0 * i
            t = 0.1 * i
            it = fields.create_group(f"iteration_{i}")
            it.create_dataset("time", data=t, dtype=float)
            it.create_dataset("Q", data=Q)
            times.append(t)
            Qs.append(Q[:, :n_cells])
    return np.array(times), Qs


def test_hdf5_to_vtk_roundtrip():
    d = tempfile.mkdtemp()
    store = os.path.join(d, "simulation.h5")
    times, Qs = _make_store(store)

    pvd = hdf5_to_vtk(store, d, basename="simulation")
    assert os.path.exists(pvd)
    assert os.path.exists(os.path.join(d, "simulation_0000.vtu"))

    rt = vtk_to_hdf5(pvd, os.path.join(d, "roundtrip.h5"))
    with h5py.File(rt) as f:
        assert len(f["fields"]) == len(times)
        for i, (t, Q) in enumerate(zip(times, Qs)):
            grp = f[f"fields/iteration_{i}"]
            assert abs(float(grp["time"][()]) - t) < 1e-12
            np.testing.assert_allclose(grp["Q"][()], Q, rtol=1e-12, atol=1e-14)
    print(f"OK: {len(times)} snapshots round-tripped, Q shape {Qs[0].shape}")


def _make_store_with_aux(path, n_cells=30, n_ghost=2, n_vars=2, n_aux=3, nsnap=3):
    """A 1-D store carrying both Q and Qaux (ghost columns on both)."""
    x = np.linspace(0.0, 1.0, n_cells + 1)
    cells = np.vstack([np.arange(n_cells), np.arange(1, n_cells + 1)])
    times, Qs, Qauxs = [], [], []
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
            t = 0.2 * i
            it = fields.create_group(f"iteration_{i}")
            it.create_dataset("time", data=t, dtype=float)
            it.create_dataset("Q", data=Q)
            it.create_dataset("Qaux", data=Qaux)
            times.append(t)
            Qs.append(Q[:, :n_cells])
            Qauxs.append(Qaux[:, :n_cells])
    return np.array(times), Qs, Qauxs


def test_hdf5_to_vtk_qaux_roundtrip():
    """include_aux=True must append the Qaux rows (named) into the .vtu, and
    vtk_to_hdf5 must recover [Q; Qaux] unchanged."""
    import meshio

    d = tempfile.mkdtemp()
    store = os.path.join(d, "sim.h5")
    times, Qs, Qauxs = _make_store_with_aux(store)

    pvd = hdf5_to_vtk(
        store, d, basename="sim", include_aux=True,
        field_names=["h", "hu"], aux_field_names=["hinv", "dhdx", "dudx"],
    )
    assert os.path.exists(pvd)

    # the named cell_data must be exactly what a ParaView user sees
    m0 = meshio.read(os.path.join(d, "sim_0000.vtu"))
    assert set(m0.cell_data) == {"h", "hu", "hinv", "dhdx", "dudx"}

    rt = vtk_to_hdf5(pvd, os.path.join(d, "rt.h5"))
    with h5py.File(rt) as f:
        assert len(f["fields"]) == len(times)
        for i, (t, Q, Qaux) in enumerate(zip(times, Qs, Qauxs)):
            grp = f[f"fields/iteration_{i}"]
            assert abs(float(grp["time"][()]) - t) < 1e-12
            expected = np.vstack([Q, Qaux])   # cell_data is written Q then Qaux
            np.testing.assert_allclose(grp["Q"][()], expected, rtol=1e-12, atol=1e-14)
    print(f"OK: Qaux round-tripped, {len(times)} snaps, "
          f"[Q;Qaux] shape {np.vstack([Qs[0], Qauxs[0]]).shape}")


if __name__ == "__main__":
    test_hdf5_to_vtk_roundtrip()
    test_hdf5_to_vtk_qaux_roundtrip()
