# import sys
import os
import numpy as np

from attr import define
# from typing import Union

# from library.misc.custom_types import IArray, FArray, CArray
from library.mesh.mesh import Mesh


def _no_slip(Q):
    Qout = np.array(Q)
    Qout[1:] = 0.0
    return Qout


def _stress_free_surface(Q):
    Qout = np.array(Q)
    return Qout


@define(slots=True, frozen=True)
class MeshLayered:
    n_layers: int
    mesh: Mesh

    @classmethod
    def from_Mesh(cls, n_layers, mesh):
        return cls(n_layers, mesh)

    def get_n_cells_total(self):
        n_cells_loc = self.mesh.n_cells
        n_layers = self.n_layers
        return n_cells_loc * n_layers

    def get_layer(self, Q_glob, layer):
        n_cells = self.mesh.n_cells
        i_start = layer * n_cells
        i_end = i_start + n_cells
        return Q_glob[:, i_start:i_end]

    def write_layer(self, Q_glob, Q_loc, layer):
        n_cells = self.mesh.n_cells
        i_start = layer * n_cells
        i_end = i_start + n_cells
        Q_glob[:, i_start:i_end] = Q_loc
        return Q_glob

    def get_column(self, U_glob, idx):
        n_cells = self.mesh.n_cells
        n_layers = self.n_layers
        assert idx < n_cells
        Ucol = np.zeros((n_layers), dtype=float)
        for layer in range(n_layers):
            i_offset = layer * n_cells
            Ucol[layer] = U_glob[i_offset + idx]
        return Ucol

    def write_column(self, U_glob, U_loc, idx):
        n_cells = self.mesh.n_cells
        n_layers = self.n_layers
        assert idx < n_cells
        for layer in range(n_layers):
            i_offset = layer * n_cells
            U_glob[i_offset + idx] = U_loc[layer]
        return U_glob

    def integrate_column(self, Ucol):
        Qint = np.zeros_like(Ucol)
        n_layers = self.n_layers
        dz = 1.0 / n_layers
        Qint[0] = Ucol[0] * dz / 2
        for layer in range(1, n_layers):
            Qint[layer] = Qint[layer - 1] + Ucol[layer] * dz / 2
        return Qint

    def averaging_operator(self, U):
        # WARNING:  not vectorized!
        psi = np.zeros_like(U)
        n_inner_cells = self.mesh.n_inner_cells
        n_cells = self.mesh.n_cells
        for i in range(n_cells):
            psi = self.write_column(
                psi, self.integrate_column(self.get_column(U, i)), i
            )
        return psi

    def update_omega(self, Q, Qaux, Qinfo):
        i_dhu_dx = Qinfo["grad"][1][0]
        dhu_dx = Qaux[i_dhu_dx]
        h = Q[0]
        avg = self.averaging_operator(dhu_dx)
        omega = 1 / h * avg
        i_omega = Qinfo["omega"]
        Qaux[i_omega] = omega
        return Qaux

    def update_gradient(self, Q, Qaux, Qinfo):
        # TODO finish
        # do for each layer and write to Qaux
        i_dQdx = Qinfo["grad"][:][0]
        i_dQdy = Qinfo["grad"][:][1]
        lsq_gradQ = self.mesh.lsq_gradQ
        n_cells = self.mesh.n_cells
        n_layers = self.n_layers
        for layer in range(n_layers):
            Q_layer = self.get_layer(Q, layer)
            gradQ = np.einsum("...dn, kn ->k...d", lsq_gradQ, Q_layer)
            i_offset = n_cells * layer
            Qaux[i_dQdx, i_offset : i_offset + n_cells] = gradQ[:, :, 0]
            Qaux[i_dQdy, i_offset : i_offset + n_cells] = gradQ[:, :, 1]
        return Qaux

    def apply_layer_boundary_conditions(
        self, Q, bottom=_no_slip, top=_stress_free_surface
    ):
        Qbot = bottom(self.get_layer(Q, 1))
        Q = self.write_layer(Q, Qbot, 0)

        Qtop = top(self.get_layer(Q, self.n_layers - 1 - 1))
        Q = self.write_layer(Q, Qtop, self.n_layers - 1)
        return Q


def _initialize_test_fields():
    main_dir = os.getenv("SMS")
    path = os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh")
    mesh = Mesh.from_gmsh(path)
    mesh = MeshLayered.from_Mesh(3, mesh)
    n_fields = 2
    # omega + grad
    n_aux_fields = 1 + n_fields * 2
    n_cells_total = mesh.get_n_cells_total()
    Q = np.empty((n_fields, n_cells_total), dtype=float)
    Qaux = np.empty((n_aux_fields, n_cells_total), dtype=float)
    dims = 2
    offset = 0
    grad_fields = [
        list(range(offset + n_fields * d, offset + n_fields * (d + 1)))
        for d in range(dims)
    ]
    Qinfo = {"grad": grad_fields, "omega": n_fields * dims}
    n_layers = mesh.n_layers
    n_cells = mesh.mesh.n_cells

    ## write local
    for layer in range(n_layers):
        Qloc = (layer + 1) * np.ones((n_fields, n_cells), dtype=float)
        Qloc[0] = 1.0
        Qauxloc = np.ones((n_aux_fields, n_cells), dtype=float)
        mesh.write_layer(Q, Qloc, layer)
        mesh.write_layer(Qaux, Qauxloc, layer)
    return mesh, Q, Qaux, Qinfo


def test_get_n_cells_total():
    main_dir = os.getenv("SMS")
    path = os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh")
    mesh = Mesh.from_gmsh(path)
    mesh = MeshLayered.from_Mesh(3, mesh)
    n_cells = mesh.mesh.n_cells
    n_cells_total = mesh.get_n_cells_total()
    assert n_cells * 3 == n_cells_total


def test_layers():
    mesh, Q, Qaux, Qinfo = _initialize_test_fields()
    n_layers = mesh.n_layers
    for layer in range(n_layers):
        Qloc = mesh.get_layer(Q, layer)
        assert np.allclose(Qloc[0], 1)
        assert np.allclose(Qloc[1:], layer + 1)


def test_columns():
    mesh, Q, _, _ = _initialize_test_fields()
    n_cells = mesh.mesh.n_cells
    U = Q[1]
    for i_cell in range(n_cells):
        Ucol = mesh.get_column(U, i_cell)
        assert np.allclose(Ucol, np.array([1, 2, 3], dtype=float))
        Uint = mesh.integrate_column(Ucol)
        assert np.allclose(Uint, np.array([1.0 / 6, 1.0 / 2, 1.0], dtype=float))


def test_averaging_operator():
    mesh, Q, _, _ = _initialize_test_fields()
    n_cells = mesh.mesh.n_cells
    U = Q[1]
    Unew = mesh.averaging_operator(U)
    for i_cell in range(n_cells):
        Uint = mesh.get_column(Unew, i_cell)
        assert np.allclose(Uint, np.array([1.0 / 6, 1.0 / 2, 1.0], dtype=float))


def test_update_omega():
    mesh, Q, Qaux, Qinfo = _initialize_test_fields()
    Qaux = mesh.update_omega(Q, Qaux, Qinfo)
    omega = Qaux[Qinfo["omega"]]
    omega_0 = mesh.get_column(omega, 0)
    assert np.allclose(omega_0, np.array([1 / 6, 1 / 3, 1 / 2], dtype=float))


def test_update_gradients():
    mesh, Q, Qaux, Qinfo = _initialize_test_fields()
    Qaux = mesh.update_gradient(Q, Qaux, Qinfo)
    assert np.allclose(Qaux[Qinfo["grad"][1][0]], 0)
    assert np.allclose(Qaux[Qinfo["grad"][1][1]], 0)


def test_apply_boundary_conditions():
    mesh, Q, Qaux, Qinfo = _initialize_test_fields()
    Q = mesh.apply_layer_boundary_conditions(Q)
    assert np.allclose(mesh.get_layer(Q, 0)[0], 1)
    assert np.allclose(mesh.get_layer(Q, 1)[0], 1)
    assert np.allclose(mesh.get_layer(Q, 2)[0], 1)
    assert np.allclose(mesh.get_layer(Q, 0)[1], 0)
    assert np.allclose(mesh.get_layer(Q, 1)[1], 2)
    assert np.allclose(mesh.get_layer(Q, 2)[1], 2)


if __name__ == "__main__":
    test_get_n_cells_total()
    test_layers()
    test_columns()
    test_averaging_operator()
    test_update_omega()
    test_update_gradients()
    test_apply_boundary_conditions()
