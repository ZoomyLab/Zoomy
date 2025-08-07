# import sys
import os
import numpy as np
import pyprog

from attr import define
from copy import deepcopy
from time import time as gettime
# from typing import Union

# from library.misc.custom_types import IArray, FArray, CArray
from library.mesh.mesh import Mesh
from library.pysolver.solver import (
    load_runtime_model,
    apply_boundary_conditions,
    _get_semidiscrete_solution_operator,
    _get_source,
    _get_source_jac,
    _get_compute_max_abs_eigenvalue,
    Settings,
)
from library.model.model import Advection
import library.misc.io as io
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.pysolver.ode import RK1
import library.pysolver.reconstruction as recon
import library.pysolver.flux as flux
import library.pysolver.nonconservative_flux as nonconservative_flux

# import library.pysolver.ader_flux as ader_flux
import library.pysolver.timestepping as timestepping


def _no_slip(Q):
    Qout = np.array(Q)
    Qout[1:] = 0.0
    return Qout


def _stress_free_surface(Q):
    Qout = np.array(Q)
    return Qout


def _initialize_problem(model, mesh, settings=None):
    model.boundary_conditions.initialize(
        mesh.base,
        model.time,
        model.position,
        model.distance,
        model.variables,
        model.aux_variables,
        model.parameters,
        model.normal,
    )

    n_variables = model.n_variables
    n_cells = mesh.mesh.n_cells
    n_aux_variables = model.aux_variables.length()
    if settings:
        if settings.compute_gradient:
            n_aux_variables_ext = n_aux_variables + n_variables * mesh.base.dimension

    Q = np.empty((n_variables, n_cells), dtype=float)
    Qaux = np.zeros((n_aux_variables_ext, n_cells), dtype=float)

    Q = model.initial_conditions.apply(mesh.mesh.cell_centers, Q)
    Qaux = model.aux_initial_conditions.apply(mesh.mesh.cell_centers, Qaux)
    return Q, Qaux


@define(slots=True, frozen=True)
class MeshLayered:
    n_layers: int
    base: Mesh
    mesh: Mesh

    @classmethod
    def from_Mesh(cls, n_layers, base):
        mesh = Mesh.extrude_mesh(base, n_layers)
        return cls(n_layers, base, mesh)

    def _get_gradient_field_index_list(self, Q, Qaux):
        n_variables = Q.shape[0]
        dims = self.base.dimension
        n_grad_fields = n_variables * dims
        n_aux_variables = Qaux.shape[0]
        i_grad_start = n_aux_variables - n_grad_fields
        out = np.zeros((dims, n_variables), dtype=int)
        for d in range(dims):
            out[d, :] = np.array(
                list(
                    range(
                        i_grad_start + d * n_variables, i_grad_start + (d + 1) * n_variables
                    )
                )
            )
        out = out.reshape((dims, n_variables))
        return out

    def get_n_cells_total(self):
        n_cells_loc = self.base.n_cells
        n_layers = self.n_layers
        return n_cells_loc * n_layers

    def get_layer(self, Q_glob, layer):
        n_cells = self.base.n_cells
        i_start = layer * n_cells
        i_end = i_start + n_cells
        return Q_glob[:, i_start:i_end]

    def write_layer(self, Q_glob, Q_loc, layer):
        n_cells = self.base.n_cells
        i_start = layer * n_cells
        i_end = i_start + n_cells
        Q_glob[:, i_start:i_end] = Q_loc
        return Q_glob

    def get_column(self, U_glob, idx):
        n_cells = self.base.n_cells
        n_layers = self.n_layers
        assert idx < n_cells
        Ucol = np.zeros((n_layers), dtype=float)
        for layer in range(n_layers):
            i_offset = layer * n_cells
            Ucol[layer] = U_glob[i_offset + idx]
        return Ucol

    def write_column(self, U_glob, U_loc, idx):
        n_cells = self.base.n_cells
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
        n_inner_cells = self.base.n_inner_cells
        n_cells = self.base.n_cells
        for i in range(n_cells):
            psi = self.write_column(
                psi, self.integrate_column(self.get_column(U, i)), i
            )
        return psi

    def update_omega(self, Q, Qaux):
        gradQinfo = self._get_gradient_field_index_list(Q, Qaux)
        dhu_dx = Qaux[gradQinfo[0][1]]
        h = Q[0]
        avg = self.averaging_operator(dhu_dx)
        omega = 1 / h * avg
        i_omega = 0
        Qaux[i_omega] = omega
        return Qaux

    def update_gradient(self, Q, Qaux):
        # do for each layer and write to Qaux
        gradQinfo = self._get_gradient_field_index_list(Q, Qaux)
        i_dQdx = gradQinfo[0][:]
        i_dQdy = gradQinfo[1][:]
        lsq_gradQ = self.base.lsq_gradQ
        n_cells = self.base.n_cells
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
        assert self.n_layers >= 3
        Qbot = bottom(self.get_layer(Q, 1))
        Q = self.write_layer(Q, Qbot, 0)

        Qtop = top(self.get_layer(Q, self.n_layers - 1 - 1))
        Q = self.write_layer(Q, Qtop, self.n_layers - 1)
        return Q


def solver_price_c_layered(
    mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
):
    iteration = 0
    time = 0.0

    output_hdf5_path = os.path.join(settings.output.directory, f"{settings.name}.h5")

    assert model.dimension == mesh.base.dimension

    progressbar = pyprog.ProgressBar(
        "{}: ".format(settings.name),
        "\r",
        settings.time_end,
        complete_symbol="█",
        not_complete_symbol="-",
    )
    progressbar.progress_explain = ""
    progressbar.update()

    Q, Qaux = _initialize_problem(model, mesh, settings)

    parameters = model.parameter_values
    pde, bcs = load_runtime_model(model)

    for layer in range(mesh.n_layers):
        Qloc = mesh.get_layer(Q, layer)
        Qauxloc = mesh.get_layer(Qaux, layer)
        Qloc = apply_boundary_conditions(
            mesh.base, time, Qloc, Qauxloc[: model.n_aux_variables, :], parameters, bcs
        )
        Q = mesh.write_layer(Q, Qloc, layer)
    Qaux = mesh.update_gradient(Q, Qaux)
    Q = mesh.apply_layer_boundary_conditions(Q)
    Qaux = mesh.update_omega(Q, Qaux)

    i_snapshot = 0
    dt_snapshot = settings.time_end / (settings.output_snapshots - 1)
    io.init_output_directory(settings.output.directory, settings.output_clean_dir)
    mesh.write_to_hdf5(output_hdf5_path)
    i_snapshot = io.save_fields(
        output_hdf5_path, time, 0, i_snapshot, Q, Qaux, settings.output_write_all
    )

    Qnew = deepcopy(Q)

    flux_operator = _get_semidiscrete_solution_operator(
        mesh.base, pde, bcs, settings
    )
    compute_source = _get_source(mesh.base, pde, settings)
    compute_source_jac = _get_source_jac(mesh.base, pde, settings)

    compute_max_abs_eigenvalue = _get_compute_max_abs_eigenvalue(
        mesh.base, pde, settings
    )
    min_inradius = np.min(mesh.mesh.cell_inradius)

    time_start = gettime()
    while time < settings.time_end:
        #     # print(f'in loop from process {os.getpid()}')
        Q = deepcopy(Qnew)
        dt = settings.compute_dt(
            Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue
        )
        assert dt > 10 ** (-8)
        assert not np.isnan(dt) and np.isfinite(dt)

        #     # add 0.001 safty measure to avoid very small time steps
        if settings.truncate_last_time_step:
            if time + dt * 1.001 > settings.time_end:
                dt = settings.time_end - time + 10 ** (-10)

        for layer in range(mesh.n_layers):
            Qloc = mesh.get_layer(Q, layer)
            Qauxloc = mesh.get_layer(Qaux, layer)
            Qnewloc = ode_solver_flux(
                flux_operator, Qloc, Qauxloc, parameters, dt
            )

            Qnewloc = ode_solver_source(
                compute_source,
                Qnewloc,
                Qauxloc,
                parameters,
                dt,
                func_jac=compute_source_jac,
            )
            Qnewloc = apply_boundary_conditions(
                mesh.base,
                time,
                Qnewloc,
                Qauxloc[: model.n_aux_variables, :],
                parameters,
                bcs,
            )
            Qnew = mesh.write_layer(Qnew, Qnewloc, layer)
        Qaux = mesh.update_gradient(Qnew, Qaux)
        Qnew = mesh.apply_layer_boundary_conditions(Qnew)
        Qaux = mesh.update_omega(Qnew, Qaux)

        # Update solution and time
        time += dt
        iteration += 1
        print(iteration, time, dt)

        i_snapshot = io.save_fields(
            output_hdf5_path,
            time,
            (i_snapshot + 1) * dt_snapshot,
            i_snapshot,
            Qnew,
            Qaux,
            settings.output_write_all,
        )

    print(f"Runtime: {gettime() - time_start}")

    progressbar.end()
    return settings


def _initialize_test_fields():
    main_dir = os.getenv("ZOOMY_DIR")
    path = os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh")
    mesh = Mesh.from_gmsh(path)
    mesh = MeshLayered.from_Mesh(3, mesh)
    n_variables = 2
    # omega + grad
    n_aux_variables = 1 + n_variables * 2
    n_cells_total = mesh.get_n_cells_total()
    Q = np.empty((n_variables, n_cells_total), dtype=float)
    Qaux = np.empty((n_aux_variables, n_cells_total), dtype=float)
    dims = 2
    offset = 0
    grad_fields = [
        list(range(offset + n_variables * d, offset + n_variables * (d + 1)))
        for d in range(dims)
    ]
    n_layers = mesh.n_layers
    n_cells = mesh.base.n_cells

    ## write local
    for layer in range(n_layers):
        Qloc = (layer + 1) * np.ones((n_variables, n_cells), dtype=float)
        Qloc[0] = 1.0
        Qauxloc = np.ones((n_aux_variables, n_cells), dtype=float)
        mesh.write_layer(Q, Qloc, layer)
        mesh.write_layer(Qaux, Qauxloc, layer)
    return mesh, Q, Qaux


def test_get_n_cells_total():
    main_dir = os.getenv("ZOOMY_DIR")
    path = os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh")
    mesh = Mesh.from_gmsh(path)
    mesh = MeshLayered.from_Mesh(3, mesh)
    n_cells = mesh.base.n_cells
    n_cells_total = mesh.get_n_cells_total()
    assert n_cells * 3 == n_cells_total


def test_layers():
    mesh, Q, Qaux = _initialize_test_fields()
    n_layers = mesh.n_layers
    for layer in range(n_layers):
        Qloc = mesh.get_layer(Q, layer)
        assert np.allclose(Qloc[0], 1)
        assert np.allclose(Qloc[1:], layer + 1)


def test_columns():
    mesh, Q, _ = _initialize_test_fields()
    n_cells = mesh.base.n_cells
    U = Q[1]
    for i_cell in range(n_cells):
        Ucol = mesh.get_column(U, i_cell)
        assert np.allclose(Ucol, np.array([1, 2, 3], dtype=float))
        Uint = mesh.integrate_column(Ucol)
        assert np.allclose(Uint, np.array([1.0 / 6, 1.0 / 2, 1.0], dtype=float))


def test_averaging_operator():
    mesh, Q, _ = _initialize_test_fields()
    n_cells = mesh.base.n_cells
    U = Q[1]
    Unew = mesh.averaging_operator(U)
    for i_cell in range(n_cells):
        Uint = mesh.get_column(Unew, i_cell)
        assert np.allclose(Uint, np.array([1.0 / 6, 1.0 / 2, 1.0], dtype=float))


def test_update_omega():
    mesh, Q, Qaux = _initialize_test_fields()
    Qaux = mesh.update_omega(Q, Qaux)
    omega = Qaux[0]
    omega_0 = mesh.get_column(omega, 0)
    assert np.allclose(omega_0, np.array([1 / 6, 1 / 3, 1 / 2], dtype=float))


def test_update_gradients():
    mesh, Q, Qaux = _initialize_test_fields()
    Qaux = mesh.update_gradient(Q, Qaux)
    gradQinfo = mesh._get_gradient_field_index_list(Q, Qaux)
    assert np.allclose(Qaux[gradQinfo[0][1]], 0)
    assert np.allclose(Qaux[gradQinfo[1][1]], 0)


def test_apply_boundary_conditions():
    mesh, Q, _ = _initialize_test_fields()
    Q = mesh.apply_layer_boundary_conditions(Q)
    assert np.allclose(mesh.get_layer(Q, 0)[0], 1)
    assert np.allclose(mesh.get_layer(Q, 1)[0], 1)
    assert np.allclose(mesh.get_layer(Q, 2)[0], 1)
    assert np.allclose(mesh.get_layer(Q, 0)[1], 0)
    assert np.allclose(mesh.get_layer(Q, 1)[1], 2)
    assert np.allclose(mesh.get_layer(Q, 2)[1], 2)


def test_simulation():
    # mesh = Mesh.create_1d((-1, 1), 30)
    main_dir = os.getenv("ZOOMY_DIR")
    mesh = Mesh.from_gmsh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_fine.msh"),
    )
    mesh = MeshLayered.from_Mesh(3, mesh)

    bcs = BC.BoundaryConditions(
        [
            BC.Extrapolation(physical_tag="left"),
            BC.Extrapolation(physical_tag="right"),
        ]
    )

    def define_model():
        def custom_ic(x):
            Q = np.zeros(1, dtype=float)
            Q[0] = np.where(x[0] < 0, 1.0, 0.05)
            return Q

        ic = IC.UserFunction(custom_ic)

        fields = ["q"]
        model = Advection(
            dimension=2,
            boundary_conditions=bcs,
            initial_conditions=ic,
            fields=fields,
            aux_variables=1,
            settings={"friction": []},
            parameters={"p0": 1.0, "p1": 1.0},
        )
        return model

    def run_model():
        model = define_model()

        settings = Settings(
            reconstruction=recon.constant,
            num_flux=flux.Zero(),
            nc_flux=nonconservative_flux.segmentpath(3),
            compute_dt=timestepping.adaptive(CFL=0.9),
            time_end=1.0,
            output_snapshots=100,
            output_clean_dir=True,
            output_dir=f"outputs/test/layered",
            compute_gradient=True,
        )

        solver_price_c_layered(
            mesh,
            model,
            settings,
            ode_solver_source=RK1,
        )

    print("run")
    run_model()


if __name__ == "__main__":
    # test_get_n_cells_total()
    # test_layers()
    # test_columns()
    # test_averaging_operator()
    # test_update_omega()
    # test_update_gradients()
    # test_apply_boundary_conditions()
    test_simulation()
