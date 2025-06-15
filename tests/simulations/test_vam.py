import os
import numpy as np
import jax
from jax import numpy as jnp
import pytest
from types import SimpleNamespace
from sympy import cos, pi
from time import time as gettime

from library.fvm.solver import Solver, Settings
from library.fvm.ode import RK1
import library.fvm.reconstruction as recon
import library.fvm.timestepping as timestepping
import library.fvm.flux as flux
import library.fvm.nonconservative_flux as nc_flux
from library.model.boundary_conditions import BoundaryCondition

from library.model.model import *
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
import library.misc.io as io
from library.mesh.mesh import compute_gradient


import library.mesh.mesh as petscMesh
import library.postprocessing.postprocessing as postprocessing
from library.mesh.mesh import convert_mesh_to_jax
import argparse



class HyperbolicSolver(Solver):
    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):
        h = Q[0]
        u0 = Q[1]/h
        grad_b = compute_gradient(Q[5], mesh)
        dbdx = grad_b[:, 0]
        u1 = Q[2]/h
        w0 = Q[3]/h
        w1 = Q[4]/h
        hw2 = -(w0 + w1) + (u0 + u1) * dbdx
        Qaux = Qaux.at[0].set(hw2)
        return Qaux

class PoissonSolver(Solver):
    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):
        grad_h = compute_gradient(Q[0], mesh)
        grad_hu0 = compute_gradient(Q[1], mesh)
        grad_hu1 = compute_gradient(Q[2], mesh)
        grad_hw0 = compute_gradient(Q[3], mesh)
        grad_hw1 = compute_gradient(Q[4], mesh)
    

        h = Q[0]
        u0 = Q[1]/h
        p0 = Q[6]/h
        p1 = Q[7]/h
        grad_u0 = compute_gradient(u0, mesh)
        grad_b = compute_gradient(Q[5], mesh)
        
        grad_p0 = compute_gradient(p0, mesh)
        lap_p0 = compute_gradient(grad_p0[:,0], mesh)
        grad_p1 = compute_gradient(p0, mesh)
        lap_p1 = compute_gradient(grad_p1[:,0], mesh)
        Qaux = Qaux.at[13].set(lap_p0[:, 0])
        Qaux = Qaux.at[14].set(lap_p1[:, 0])



        Qaux = Qaux.at[0].set((Q[0]-Qold[0])/dt)
        Qaux = Qaux.at[1].set((Q[1]-Qold[1])/dt)
        Qaux = Qaux.at[2].set((Q[2]-Qold[2])/dt)
        Qaux = Qaux.at[3].set((Q[3]-Qold[3])/dt)
        Qaux = Qaux.at[4].set((Q[4]-Qold[4])/dt)
        Qaux = Qaux.at[5].set(grad_h[:, 0])
        Qaux = Qaux.at[6].set(grad_hu0[:, 0])
        Qaux = Qaux.at[7].set(grad_hu1[:, 0])
        Qaux = Qaux.at[8].set(grad_hw0[:, 0])
        Qaux = Qaux.at[9].set(grad_hw1[:, 0])
        Qaux = Qaux.at[10].set(grad_b[:, 0])
        Qaux = Qaux.at[11].set(grad_u0[:, 0])
        
        
        u1 = Q[2]/h
        w0 = Q[3]/h
        w1 = Q[4]/h
        dbdx = grad_b[:, 0]
        hw2 = -(w0 + w1) + (u0 + u1) * dbdx
        Qaux = Qaux.at[12].set(hw2)
        return Qaux

def solve_vam(
    mesh, model1, model2, settings, ode_solver_flux=RK1, ode_solver_source=RK1
):
    solverQ = HyperbolicSolver()
    solverP = PoissonSolver()
    
    Q, Qaux = solverQ.initialize(model1, mesh)
    P, Paux = solverP.initialize(model2, mesh)

    parameters1 = model1.parameter_values
    parameters2 = model2.parameter_values

    parameters1 = jnp.asarray(parameters1)
    parameters2 = jnp.asarray(parameters2)
    
    mesh = convert_mesh_to_jax(mesh)


    pde1, bcs1 = solverQ._load_runtime_model(model1)
    pde2, bcs2 = solverP._load_runtime_model(model2)
    output_hdf5_path = os.path.join(settings.output_dir, f"{settings.name}.h5")
    save_fields = io.get_save_fields(output_hdf5_path, settings.output_write_all)

    def run(Q, Qaux, parameters1, pde1, bcs1, P, Paux, parameters2, pde2, bcs2):
        iteration = 0.0
        time = 0.0
        assert model1.dimension == mesh.dimension
        assert model2.dimension == mesh.dimension

        i_snapshot = 0.0
        dt_snapshot = settings.time_end / (settings.output_snapshots - 1)
        io.init_output_directory(settings.output_dir, settings.output_clean_dir)
        mesh.write_to_hdf5(output_hdf5_path)
        _ = save_fields(time, 0.0, i_snapshot, Q, Qaux)
        i_snapshot = save_fields(time, 0.0, i_snapshot, P, Paux)

        Q0 = Q
        Qnew = Q
        Pnew = P
        Qauxnew = Qaux
        Pauxnew = Paux

        min_inradius = jnp.min(mesh.cell_inradius)


        compute_max_abs_eigenvalue = solverQ.get_compute_max_abs_eigenvalue(
            mesh, pde1, settings
        )
        space_solution_operator = solverQ.get_space_solution_operator(
            mesh, pde1, bcs1, settings
        )
        source_operator = solverQ.get_compute_source(mesh, pde1, settings)
        boundary_operator1 = solverQ.get_apply_boundary_conditions(mesh, bcs1)
        
        source_operator2 = solverP.get_compute_source(mesh, pde2, settings)
        boundary_operator2 = solverP.get_apply_boundary_conditions(mesh, bcs2)

        time_start = gettime()

        #@jax.jit
        def time_loop(time, iteration, i_snapshot, Q, Qaux, Pnew, Paux, Qold, Qauxold, Pold, Pauxold):
            loop_val = (time, iteration, i_snapshot, Q, Qaux, Pnew, Paux, Qold, Qauxold, Pold, Pauxold)

            def loop_body(init_value):
                time, iteration, i_snapshot, Q, Qaux, P, Paux, Qold, Qauxold, Pold, Pauxold = init_value

                dt = settings.compute_dt(
                    Q, Qaux, parameters1, min_inradius, compute_max_abs_eigenvalue
                )

                Qauxnew = solverQ.update_qaux(
                    Q, Qaux, Qold, Qauxold, mesh, pde1, parameters1, time, dt
                )
                Q1 = ode_solver_flux(
                    space_solution_operator, Q, Qauxnew, parameters1, dt
                )
                Qauxnew = solverQ.update_qaux(
                    Q1, Qauxnew, Qold, Qauxold, mesh, pde1, parameters1, time, dt
                )
                Qnew = ode_solver_source(
                    source_operator,
                    Q1,
                    Qauxnew,
                    parameters1,
                    dt,
                    func_jac=solverQ.compute_source_jac,
                )

                Qnew = boundary_operator1(time, Qnew, Qauxnew, parameters1)
                Qauxnew = solverQ.update_qaux(
                    Qnew, Qauxnew, Q, Qaux, mesh, pde1, parameters1, time, dt
                )

                # Update solution and time
                time += dt
                iteration += 1
                
                time_stamp = (i_snapshot + 1) * dt_snapshot


                jax.debug.print(
                    "iteration: {iteration}, time: {time}, dt: {dt}, i_snapshot: {i_snapshot}, time_stamp: {time_stamp}",
                    iteration=iteration,
                    time=time,
                    dt=dt,
                    i_snapshot=i_snapshot,
                    time_stamp=time_stamp
                )

                
                P = P.at[:Qnew.shape[0]].set(Qnew)
                Paux = solverP.update_qaux(
                    P, Paux, Pold, Pauxold, mesh, pde2, parameters2, time, dt
                )
                
                Pnew = solverP.implicit_solve(P, Paux, Pold, Pauxold, mesh, model2, pde2, parameters2, time, dt, boundary_operator2, debug=False)
                Pnew = Pnew.at[5].set(Q0[5])
                Pauxnew = solverP.update_qaux(
                    Pnew, Paux, Pold, Pauxold, mesh, pde2, parameters2, time, dt
                )
                Qnew = Qnew.at[:6].set(Pnew[:6])
                Qauxnew = solverQ.update_qaux(
                    Qnew, Qauxnew, Qold, Qauxold, mesh, pde1, parameters1, time, dt
                )

                
                i_snapshot = save_fields(time, time_stamp, i_snapshot, Pnew, Pauxnew)


                return (time, iteration, i_snapshot, Qnew, Qauxnew, Pnew, Pauxnew, Q, Qaux, P, Paux)

            def proceed(loop_val):
                time, iteration, i_snapshot, Q, Qaux, P, Paux, Qold, Qauxold, Pold, Pauxold = loop_val
                return time < settings.time_end

            (time, iteration, i_snapshot, Q, Qaux, P, Paux, Qold, Qauxold, Pold, Pauxold) = jax.lax.while_loop(
                proceed, loop_body, loop_val
            )
            
            return P, Paux

        P, Paux = time_loop(time, iteration, i_snapshot, Qnew, Qauxnew, Pnew, Pauxnew, Q, Qaux, P, Paux)
        return P, Paux

    time_start = gettime()
    P, Paux = run(Q, Qaux, parameters1, pde1, bcs1, P, Paux, parameters2, pde2, bcs2)

    print(f"Runtime: {gettime() - time_start}")

    return P, Paux


@pytest.mark.critical
@pytest.mark.unfinished
def test_vam_1d():
    settings = Settings(
        name="VAM",
        parameters={
            "g": 9.81,
        },
        reconstruction=recon.constant,
        num_flux=flux.Zero(),
        nc_flux=nc_flux.segmentpath(),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=10.0,
        output_snapshots=100,
        output_dir="outputs/vam",
    )

    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs1 = BC.BoundaryConditions(
        [
            BC.Lambda(physical_tag='left', prescribe_fields={1: lambda t, x, dx, q, qaux, p, n: .11197}),
            BC.Extrapolation(physical_tag='right')
        ]
    )
    
    bcs2 = BC.BoundaryConditions(
        [
            BC.Lambda(physical_tag='left', prescribe_fields={1: lambda t, x, dx, q, qaux, p, n: .11197}),
            BC.Extrapolation(physical_tag='right')
        ]
    )
    
    def custom_ic1(x):
        Q = np.zeros(6, dtype=float)
        Q[1] = np.where(x[0] < 1, 0.11197, 0.)
        Q[5] = 0.2*np.exp(-x[0]**2 / 1.0**2) 
        Q[0] = np.where(x[0] < 1, 0.34, 0.015) - Q[5]
        Q[0] = np.where(Q[0] > 0.015, Q[0], 0.015)
        return Q
    
    def custom_ic2(x):
        Q = np.zeros(8, dtype=float)
        Q[1] = np.where(x[0] < 1, 0.11197, 0.)
        Q[5] = 0.2*np.exp(-x[0]**2 / 1.0**2) 
        Q[0] = np.where(x[0] < 1, 0.34, 0.015) - Q[5]
        Q[0] = np.where(Q[0] > 0.015, Q[0], 0.015)

        return Q

    ic1 = IC.UserFunction(custom_ic1)
    ic2 = IC.UserFunction(custom_ic2)
    
    
    model1 = VAMHyperbolic(
        parameters=settings.parameters,
        boundary_conditions=bcs1,
        initial_conditions=ic1,
        settings={},
    )
    
    model2 = VAMPoisson(
        parameters=settings.parameters,
        boundary_conditions=bcs2,
        initial_conditions=ic2,
        settings={},
    )

    mesh = petscMesh.Mesh.create_1d((-1.5, 1.5), 30)

    P, Paux = solve_vam(
        mesh,
        model1,
        model2,
        settings,
    )
    io.generate_vtk(os.path.join(settings.output_dir, f"{settings.name}.h5"))

if __name__ == "__main__":
    test_vam_1d()
