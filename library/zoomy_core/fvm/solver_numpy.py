import os
from time import time as gettime

import numpy as np
from attr import define

from typing import Callable
from attrs import define, field

from zoomy_core.misc.logger_config import logger



import zoomy_core.fvm.flux as fvmflux
import zoomy_core.fvm.nonconservative_flux as nonconservative_flux
import zoomy_core.misc.io as io
from zoomy_core.misc.misc import Zstruct, Settings
import zoomy_core.fvm.ode as ode
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel


@define(frozen=True, slots=True, kw_only=True)            
class Solver():
    settings: Zstruct = field(factory=lambda: Settings.default())

    def __attrs_post_init__(self):
        defaults = Settings.default()
        defaults.update(self.settings)
        object.__setattr__(self, 'settings', defaults)
        

    def initialize(self, mesh, model):
        # model.boundary_conditions.initialize(
        #     mesh,
        #     model.time,
        #     model.position,
        #     model.distance,
        #     model.variables,
        #     model.aux_variables,
        #     model.parameters,
        #     model.normal,
        # )

        n_variables = model.n_variables
        n_cells = mesh.n_cells
        n_aux_variables = model.aux_variables.length()

        Q = np.empty((n_variables, n_cells), dtype=float)
        Qaux = np.empty((n_aux_variables, n_cells), dtype=float)
        return Q, Qaux
        
    def create_runtime(self, Q, Qaux, mesh, model):      
        mesh.resolve_periodic_bcs(model.boundary_conditions)
        Q, Qaux = np.asarray(Q), np.asarray(Qaux)
        parameters = np.asarray(model.parameter_values)
        runtime_model = NumpyRuntimeModel(model)        
        return Q, Qaux, parameters, mesh, runtime_model

    def get_compute_source(self, mesh, model):
        def compute_source(dt, Q, Qaux, parameters, dQ):
            dQ = model.source(
                    Q[:, :],
                    Qaux[:, :],
                    parameters,
            )
            return dQ

        return compute_source

    def get_compute_source_jacobian_wrt_variables(self, mesh, model):
        def compute_source(dt, Q, Qaux, parameters, dQ):
            dQ = model.source_jacobian_wrt_variables(
                    Q[:, : mesh.n_inner_cells],
                    Qaux[:, : mesh.n_inner_cells],
                    parameters,
                )
            return dQ

        return compute_source
    
    def get_apply_boundary_conditions(self, mesh, model):
        
        def apply_boundary_conditions(time, Q, Qaux, parameters):
            for i in range(mesh.n_boundary_faces):
                i_face = mesh.boundary_face_face_indices[i]
                i_bc_func = mesh.boundary_face_function_numbers[i]
                q_cell = Q[:, mesh.boundary_face_cells[i]]  # Shape: (Q_dim,)
                qaux_cell = Qaux[:, mesh.boundary_face_cells[i]]
                normal = mesh.face_normals[:, i_face]
                position = mesh.face_centers[i_face, :]
                position_ghost = mesh.cell_centers[:, mesh.boundary_face_ghosts[i]]
                distance = np.linalg.norm(position - position_ghost)
                q_ghost = model.boundary_conditions(i_bc_func, time, position, distance, q_cell, qaux_cell, parameters, normal)
                Q[:,  mesh.boundary_face_ghosts[i]] = q_ghost
            return Q

        return apply_boundary_conditions
    
    def update_q(self, Q, Qaux, mesh, model, parameters):
        """
        Update variables before the solve step.
        """
        # This is a placeholder implementation. Replace with actual logic as needed.
        return Q
    
    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):
        """
        Update auxiliary variables
        """
        # This is a placeholder implementation. Replace with actual logic as needed.
        return Qaux
    
    def solve(self, mesh, model):
        logger.error(
            "Solver.solve() is not implemented. Please implement this method in the derived class."
        )
        raise NotImplementedError("Solver.solve() must be implemented in derived classes.")

 
@define(frozen=True, slots=True, kw_only=True)            
class HyperbolicSolver(Solver):
    settings: Zstruct = field(factory=lambda: Settings.default())
    compute_dt: Callable = field(factory=lambda: timestepping.adaptive(CFL=0.45))
    flux: fvmflux.Flux = field(factory=lambda: fvmflux.Zero())
    nc_flux: nonconservative_flux.NonconservativeFlux = field(
        factory=lambda: nonconservative_flux.Rusanov()
    )
    time_end: float = 0.1


    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        defaults = Settings.default()
        defaults.output.update(Zstruct(snapshots=10))
        defaults.update(self.settings)
        object.__setattr__(self, 'settings', defaults)
        

    def initialize(self, mesh, model):
        Q, Qaux = super().initialize(mesh, model)
        Q = model.initial_conditions.apply(mesh.cell_centers, Q)
        Qaux = model.aux_initial_conditions.apply(mesh.cell_centers, Qaux)
        return Q, Qaux

    def get_compute_max_abs_eigenvalue(self, mesh, model):
        def compute_max_abs_eigenvalue(Q, Qaux, parameters):
            max_abs_eigenvalue = -np.inf
            i_cellA = mesh.face_cells[0]
            i_cellB = mesh.face_cells[1]
            qA = Q[:, i_cellA]
            qB = Q[:, i_cellB]
            qauxA = Qaux[:, i_cellA]
            qauxB = Qaux[:, i_cellB]
            normal = mesh.face_normals
            evA = model.eigenvalues(qA, qauxA, parameters, normal)
            evB = model.eigenvalues(qB, qauxB, parameters, normal)
            max_abs_eigenvalue = np.maximum(np.abs(evA).max(axis=0), np.abs(evB).max(axis=0))
            return max_abs_eigenvalue
        return compute_max_abs_eigenvalue

    def get_flux_operator(self, mesh, model):
        compute_num_flux = self.flux.get_flux_operator(model)
        compute_nc_flux = self.nc_flux.get_flux_operator(model)
        def flux_operator(dt, Q, Qaux, parameters, dQ):

            # Initialize dQ as zeros using jax.numpy
            dQ = np.zeros_like(dQ)

            iA = mesh.face_cells[0]
            iB = mesh.face_cells[1]

            qA = Q[:, iA]
            qB = Q[:, iB]
            qauxA = Qaux[:, iA]
            qauxB = Qaux[:, iB]
            normals = mesh.face_normals
            face_volumes = mesh.face_volumes
            cell_volumesA = mesh.cell_volumes[iA]
            cell_volumesB = mesh.cell_volumes[iB]
            svA = mesh.face_subvolumes[:, 0]
            svB = mesh.face_subvolumes[:, 1]

            Dp, Dm = compute_nc_flux(
                qA,
                qB,
                qauxA,
                qauxB,
                parameters,
                normals,
                svA,
                svB,
                face_volumes,
                dt,
            )
            flux_out = Dm * face_volumes / cell_volumesA
            flux_in = Dp * face_volumes / cell_volumesB

        
            # dQ[:, iA]-= flux_out does not guarantee correct accumulation
            # dQ[:, iB]-= flux_in
            np.add.at(dQ, (slice(None), iA), -flux_out)
            np.add.at(dQ, (slice(None), iB), -flux_in)
            return dQ
        return flux_operator

    def solve(self, mesh, model, write_output=True):
        Q, Qaux = self.initialize(mesh, model)
        
        Q, Qaux, parameters, mesh, model = self.create_runtime(Q, Qaux, mesh, model)
        
        # init once with dummy values for dt
        Qaux = self.update_qaux(Q, Qaux, Q, Qaux, mesh, model, parameters, 0.0, 1.0)

        
        if write_output:
            output_hdf5_path = os.path.join(
                self.settings.output.directory, f"{self.settings.output.filename}.h5"
            )
            save_fields = io.get_save_fields(output_hdf5_path, write_all=False)
        else:
            def save_fields(time, time_stamp, i_snapshot, Q, Qaux):
                return i_snapshot
            

        def run(Q, Qaux, parameters, model):
            iteration = 0.0
            time = 0.0

            i_snapshot = 0.0
            dt_snapshot = self.time_end / (self.settings.output.snapshots - 1)
            if write_output:
                io.init_output_directory(
                    self.settings.output.directory, self.settings.output.clean_directory
                )
                mesh.write_to_hdf5(output_hdf5_path)
                io.save_settings(self.settings)
            i_snapshot = save_fields(time, 0.0, i_snapshot, Q, Qaux)

            Qnew = Q
            Qauxnew = Qaux

            compute_max_abs_eigenvalue = self.get_compute_max_abs_eigenvalue(mesh, model)
            flux_operator = self.get_flux_operator(mesh, model)
            source_operator = self.get_compute_source(mesh, model)
            boundary_operator = self.get_apply_boundary_conditions(mesh, model)
            Qnew = boundary_operator(time, Qnew, Qaux, parameters)


            cell_inradius_face = np.minimum(mesh.cell_inradius[mesh.face_cells[0, :]], mesh.cell_inradius[mesh.face_cells[1,:]])
            cell_inradius_face = cell_inradius_face.min()

            while time < self.time_end:
                Q = Qnew
                Qaux = Qauxnew
                
                dt = self.compute_dt(
                    Q, Qaux, parameters, cell_inradius_face, compute_max_abs_eigenvalue
                )

                Q1 = ode.RK1(flux_operator, Q, Qaux, parameters, dt)
                Q2 = ode.RK1(
                    source_operator,
                    Q1,
                    Qaux,
                    parameters,
                    dt,
                )

                Q3 = boundary_operator(time, Q2, Qaux, parameters)
                
                # Update solution and time
                time += dt
                iteration += 1

                time_stamp = (i_snapshot) * dt_snapshot
                
                Qnew = self.update_q(Q3, Qaux, mesh, model, parameters)
                Qauxnew = self.update_qaux(Qnew, Qaux, Q, Qaux, mesh, model, parameters, time, dt)


                i_snapshot = save_fields(time, time_stamp, i_snapshot, Qnew, Qauxnew)

                if iteration % 10 == 0:
                    logger.info(
                        f"iteration: {int(iteration)}, time: {float(time):.6f}, "
                        f"dt: {float(dt):.6f}, next write at time: {float(time_stamp):.6f}"
                    )

            return Qnew, Qaux

        time_start = gettime()
        Qnew, Qaux = run(Q, Qaux, parameters, model)
        time = gettime() - time_start
        logger.info(f"Finished simulation with in {time:.3f} seconds")
        return Qnew, Qaux
    
