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

        h=Q[0]
        hu0=Q[1]
        hu1=Q[2]
        hw0=Q[3]
        hw1=Q[4]
        b=Q[5]

        p0 = Qaux[1]
        p1 = Qaux[2]
        u0 = hu0/h
        u1 = hu1/h
        w0 = hw0/h
        w1 = hw1/h

        dhdx   = compute_gradient(h, mesh)[:, 0]
        dbdx  = compute_gradient(b, mesh)[:,0]
        dhp0dx  = compute_gradient(h*p0, mesh)[:,0]
        dhp1dx  = compute_gradient(h*p1, mesh)[:,0]
        hw2 = -(w0 + w1) + (u0 + u1) * dbdx


        #aux_variables=['hw2', 'p0', 'p1', 'dbdx', 'dhdx', 'dhp0dx', 'dhp1dx'],
        Qaux = Qaux.at[0].set(hw2)
        Qaux = Qaux.at[3].set(dbdx)
        Qaux = Qaux.at[4].set(dhdx)
        Qaux = Qaux.at[5].set(dhp0dx)
        Qaux = Qaux.at[6].set(dhp1dx)
        return Qaux


    #@jax.jit
    def compute_source_pressure(self, dt, Q, Qaux, parameters, mesh, pde):
        dQ = jnp.zeros_like(Q)
        dQ = dQ.at[:, : mesh.n_inner_cells].set(
            pde.source_implicit(
                Q[:, : mesh.n_inner_cells],
                Qaux[:, : mesh.n_inner_cells],
                parameters,
            )
        )
        return Q - dt * dQ



class PoissonSolver(Solver):
    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):

        h = Qaux[0]
        p0 = Q[0]
        p1 = Q[1]

        dhp0dx = compute_gradient(p0, mesh)[:, 0]
        ddhp0dxx = compute_gradient(dp0dx, mesh)[:, 0]
        dp1dx = compute_gradient(p1, mesh)[:, 0]
        ddp1dxx = compute_gradient(dp1dx, mesh)[:, 0]

        Qaux = Qaux.at[13].set(dp0dx)
        Qaux = Qaux.at[14].set(ddp0dxx)
        Qaux = Qaux.at[15].set(dp1dx)
        Qaux = Qaux.at[16].set(ddp1dxx)
        return Qaux

class HyperbolicSolverHP(HyperbolicSolver):

    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):

        h=Q[0]
        hu0=Q[1]
        hu1=Q[2]
        hw0=Q[3]
        hw1=Q[4]
        b=Q[5]

        hp0 = Qaux[1]
        hp1 = Qaux[2]
        u0 = hu0/h
        u1 = hu1/h
        w0 = hw0/h
        w1 = hw1/h

        dhdx   = compute_gradient(h, mesh)[:, 0]
        dbdx  = compute_gradient(b, mesh)[:,0]
        dhp0dx  = compute_gradient(hp0, mesh)[:,0]
        dhp1dx  = compute_gradient(hp1, mesh)[:,0]
        hw2 = -(w0 + w1) + (u0 + u1) * dbdx


        #aux_variables=['hw2', 'p0', 'p1', 'dbdx', 'dhdx', 'dhp0dx', 'dhp1dx'],
        Qaux = Qaux.at[0].set(hw2)
        Qaux = Qaux.at[3].set(dbdx)
        Qaux = Qaux.at[4].set(dhdx)
        Qaux = Qaux.at[5].set(dhp0dx)
        Qaux = Qaux.at[6].set(dhp1dx)
        return Qaux

class PoissonSolverHP(Solver):
    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):

        hp0 = Q[0]
        hp1 = Q[1]

        dhp0dx = compute_gradient(hp0, mesh)[:, 0]
        ddhp0dxx = compute_gradient(dhp0dx, mesh)[:, 0]
        ddhp0dxx = compute_gradient(dhp0dx, mesh)[:, 0]
        dhp1dx = compute_gradient(hp1, mesh)[:, 0]
        ddhp1dxx = compute_gradient(dhp1dx, mesh)[:, 0]

        #d3hp0dx3 = compute_gradient(ddhp0dxx, mesh)[:, 0]
        #d4hp0dx4 = compute_gradient(d3hp0dx3, mesh)[:, 0]
        #d3hp1dx3 = compute_gradient(ddhp1dxx, mesh)[:, 0]
        #d4hp1dx4 = compute_gradient(d3hp1dx3, mesh)[:, 0]

        Qaux = Qaux.at[13].set(dhp0dx)
        Qaux = Qaux.at[14].set(ddhp0dxx)
        Qaux = Qaux.at[15].set(dhp1dx)
        Qaux = Qaux.at[16].set(ddhp1dxx)
        #Qaux = Qaux.at[17].set(d4hp0dx4)
        #Qaux = Qaux.at[18].set(d4hp1dx4)
        return Qaux


class PoissonSolverFull(Solver):
    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):

        h = Q[0]
        hu0 = Q[1]
        hu1 = Q[2]
        hw0 = Q[3]
        hw1 = Q[4]
        b = Q[5]
        u0 = hu0/h
        u1 = hu1/h
        w0 = hw0/h
        w1 = hw1/h
        hp0 = Q[6]
        hp1 = Q[7]
        p0 = hp0/h
        p1 = hp1/h
        _h = Qold[0]
        _hu0 = Qold[1]
        _hu1 = Qold[2]
        _hw0 = Qold[3]
        _hw1 = Qold[4]
        _b = Qold[5]

        dbdx = compute_gradient(b, mesh)[:, 0]
        ddbdxx = compute_gradient(dbdx, mesh)[:, 0]
        dhdx = compute_gradient(h, mesh)[:,0]
        ddhdxx = compute_gradient(dhdx, mesh)[:,0]
        du0dx = compute_gradient(u0, mesh)[:, 0]
        du1dx = compute_gradient(u1, mesh)[:, 0]
        dp0dx = compute_gradient(p0, mesh)[:, 0]
        ddp0dxx = compute_gradient(dp0dx, mesh)[:, 0]
        dp1dx = compute_gradient(p1, mesh)[:, 0]
        ddp1dxx = compute_gradient(dp1dx, mesh)[:, 0]
        hw2 = -(w0 + w1) + (u0 + u1) * dbdx

        dhu0dx = compute_gradient(hu0, mesh)[:, 0]
        dhu1dx = compute_gradient(hu1, mesh)[:, 0]
        dhw0dx = compute_gradient(hw0, mesh)[:, 0]
        dhw1dx = compute_gradient(hw1, mesh)[:, 0]
        dhp0dx = compute_gradient(hp0, mesh)[:, 0]
        dhp1dx = compute_gradient(hp1, mesh)[:, 0]

        dhdt = (h-_h)/dt
        dhu0dt = (hu0-_hu0)/dt
        dhu1dt = (hu1-_hu0)/dt
        dhw0dt = (hw0-_hw0)/dt
        dhw1dt = (hw1-_hw1)/dt
        dbdt = (b-_b)/dt

        Qaux = Qaux.at[0].set(dhdt)
        Qaux = Qaux.at[1].set(dhu0dt)
        Qaux = Qaux.at[2].set(dhu1dt)
        Qaux = Qaux.at[3].set(dhw0dt)
        Qaux = Qaux.at[4].set(dhw1dt)
        Qaux = Qaux.at[5].set(dbdt)
        Qaux = Qaux.at[6].set(dhdx)
        Qaux = Qaux.at[7].set(dhu0dx)
        Qaux = Qaux.at[8].set(dhu1dx)
        Qaux = Qaux.at[9].set(dhw0dx)
        Qaux = Qaux.at[10].set(dhw1dx)
        Qaux = Qaux.at[11].set(dhp0dx)
        Qaux = Qaux.at[12].set(dhp1dx)
        Qaux = Qaux.at[13].set(dbdx)
        Qaux = Qaux.at[14].set(hw2)
        Qaux = Qaux.at[15].set(ddp0dxx)
        Qaux = Qaux.at[16].set(ddp1dxx)
        Qaux = Qaux.at[17].set(du0dx)
        Qaux = Qaux.at[18].set(du1dx)
        return Qaux

def solve_vam(
    mesh, model1, model2, settings, ode_solver_flux=RK1, ode_solver_source=RK1
):
    solverQ = HyperbolicSolverHP()
    solverP = PoissonSolverHP()
    
    Q, Qaux = solverQ.initialize(model1, mesh)
    P, Paux = solverP.initialize(model2, mesh)

    parameters1 = model1.parameter_values
    parameters2 = model2.parameter_values

    parameters1 = jnp.asarray(parameters1)
    parameters2 = jnp.asarray(parameters2)
    
    mesh = convert_mesh_to_jax(mesh)


    pde1, bcs1 = solverQ.transform_in_place(model1)
    pde2, bcs2 = solverP.transform_in_place(model2)
    output_hdf5_path = os.path.join(settings.output.directory, f"{settings.name}.h5")
    save_fields = io.get_save_fields(output_hdf5_path, settings.output_write_all)

    def run(Q, Qaux, parameters1, pde1, bcs1, P, Paux, parameters2, pde2, bcs2):
        iteration = 0.0
        time = 0.0
        assert model1.dimension == mesh.dimension
        assert model2.dimension == mesh.dimension

        i_snapshot = 0.0
        dt_snapshot = settings.time_end / (settings.output_snapshots - 1)
        io.init_output_directory(settings.output.directory, settings.output_clean_dir)
        mesh.write_to_hdf5(output_hdf5_path)
        _ = save_fields(time, 0.0, i_snapshot, Q, Qaux)
        i_snapshot = save_fields(time, 0.0, i_snapshot, Q, Qaux)

        Q0 = Q
        Qnew = Q
        Pnew = P
        Qauxnew = Qaux
        Pauxnew = Paux

        min_inradius = jnp.min(mesh.cell_inradius)


        compute_max_abs_eigenvalue = solverQ.get_compute_max_abs_eigenvalue(
            mesh, pde1, settings
        )
        flux_operator = solverQ.get_flux_operator(
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

                #############################################################
                #####################VELOCITY PREDICTOR######################
                #############################################################
                Qauxnew = solverQ.update_qaux(
                    Q, Qaux, Qold, Qauxold, mesh, pde1, parameters1, time, dt
                )
                Q1 = ode_solver_flux(
                    flux_operator, Q, Qauxnew, parameters1, dt
                )

                Q1 = Q1.at[5].set(Q0[5])

                Qauxnew = solverQ.update_qaux(
                    Q1, Qauxnew, Qold, Qauxold, mesh, pde1, parameters1, time, dt
                )
                #Qnew = ode_solver_source(
                #    source_operator,
                #    Q1,
                #    Qauxnew,
                #    parameters1,
                #    dt,
                #    func_jac=solverQ.compute_source_jac,
                #)

                #Qauxnew = solverQ.update_qaux(
                #    Qnew, Qauxnew, Q, Qaux, mesh, pde1, parameters1, time, dt
                #)
                Qnew = Q1
                Qnew = boundary_operator1(time, Qnew, Qauxnew, parameters1)


                #############################################################
                #########################PRESSURE############################
                #############################################################

                h = Q[0]
                u0 = Q[1]/h
                u1 = Q[2]/h
                w0 = Q[3]/h
                w1 = Q[4]/h
                b = Q[5]
                dbdx = compute_gradient(b, mesh)[:, 0]
                ddbdxx = compute_gradient(dbdx, mesh)[:, 0]
                dhdx = compute_gradient(h, mesh)[:,0]
                ddhdxx = compute_gradient(dhdx, mesh)[:,0]
                du0dx = compute_gradient(u0, mesh)[:, 0]
                du1dx = compute_gradient(u1, mesh)[:, 0]
                hw2 = -(w0 + w1) + (u0 + u1) * dbdx

                Paux = Paux.at[6].set(hw2)
                Paux = Paux.at[7].set(dbdx)
                Paux = Paux.at[8].set(ddbdxx)
                Paux = Paux.at[9].set(dhdx)
                Paux = Paux.at[10].set(ddhdxx)
                Paux = Paux.at[11].set(du0dx)
                Paux = Paux.at[12].set(du1dx)
                Paux = Paux.at[17].set(dt)


                Paux = Paux.at[:Qnew.shape[0]].set(Qnew)
                Paux = solverP.update_qaux(
                    P, Paux, Pold, Pauxold, mesh, pde2, parameters2, time, dt
                )

                def residual(P):

                    p = jnp.zeros_like(P)
                    p = p.at[:, 1:-1].set(P[:, :mesh.n_inner_cells])
                    p = p.at[0].set(p[1])
                    p = p.at[-1].set(p[-2])


                    dx = min_inradius*2
                    dpdx = jnp.zeros_like(P)
                    dpdx = dpdx.at[:, 1:-1].set((p[:, 0:-2] - 2 * p[:, 1:-1] + p[:, 2:])/2/dx)
                    dpdx = dpdx.at[0].set(dpdx[1])
                    dpdx = dpdx.at[-1].set(dpdx[-2])
                    ddpdxx = jnp.zeros_like(P)
                    ddpdxx = ddpdxx.at[:, 1:-1].set((dpdx[:, 0:-2] - 2 * dpdx[:, 1:-1] + dpdx[:, 2:])/2/dx)
                    ddpdxx = ddpdxx.at[0].set(ddpdxx[1])
                    ddpdxx = ddpdxx.at[-1].set(ddpdxx[-2])

                    paux = Paux
                    
                    paux = paux.at[13].set(dpdx[0])
                    paux = paux.at[14].set(ddpdxx[0])
                    paux = paux.at[15].set(dpdx[1])
                    paux = paux.at[16].set(ddpdxx[1])

                    res = pde2.source_implicit(p, paux, parameters2)
                    res = res.at[:, mesh.n_inner_cells:].set(0.)
                    return res


                Pnew = solverP.implicit_solve(P, Paux, Pold, Pauxold, mesh, model2, pde2, parameters2, time, dt, boundary_operator2, debug=[True, False], user_residual=None)

                #############################################################
                #########################CORRECTOR###########################
                #############################################################
                Qauxnew = Qauxnew.at[1].set(Pnew[0])
                Qauxnew = Qauxnew.at[2].set(Pnew[1])
                ##Qauxnew = Qauxnew.at[2].set(0.)
                Qauxnew = solverQ.update_qaux(
                    Qnew, Qauxnew, Qold, Qauxold, mesh, pde1, parameters1, time, dt
                )

                Qnew = solverQ.compute_source_pressure(dt, Qnew, Qauxnew, parameters1, mesh, pde1)
                Qauxnew = solverQ.update_qaux(
                    Qnew, Qauxnew, Qold, Qauxold, mesh, pde1, parameters1, time, dt
                )
                Qnew = boundary_operator1(time, Qnew, Qauxnew, parameters1)

                
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
                i_snapshot = save_fields(time, time_stamp, i_snapshot, Qnew, Qauxnew)


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

def solve_vam_full(
    mesh, model1, model2, settings, ode_solver_flux=RK1, ode_solver_source=RK1
):
    solverQ = HyperbolicSolver()
    solverP = PoissonSolverFull()
    
    Q, Qaux = solverQ.initialize(model1, mesh)
    P, Paux = solverP.initialize(model2, mesh)

    parameters1 = model1.parameter_values
    parameters2 = model2.parameter_values

    parameters1 = jnp.asarray(parameters1)
    parameters2 = jnp.asarray(parameters2)
    
    mesh = convert_mesh_to_jax(mesh)


    pde1, bcs1 = solverQ.transform_in_place(model1)
    pde2, bcs2 = solverP.transform_in_place(model2)
    output_hdf5_path = os.path.join(settings.output.directory, f"{settings.name}.h5")
    save_fields = io.get_save_fields(output_hdf5_path, settings.output_write_all)

    def run(Q, Qaux, parameters1, pde1, bcs1, P, Paux, parameters2, pde2, bcs2):
        iteration = 0.0
        time = 0.0
        assert model1.dimension == mesh.dimension
        assert model2.dimension == mesh.dimension

        i_snapshot = 0.0
        dt_snapshot = settings.time_end / (settings.output_snapshots - 1)
        io.init_output_directory(settings.output.directory, settings.output_clean_dir)
        mesh.write_to_hdf5(output_hdf5_path)
        _ = save_fields(time, 0.0, i_snapshot, Q, Qaux)
        i_snapshot = save_fields(time, 0.0, i_snapshot, Q, Qaux)

        Q0 = Q
        Qnew = Q
        Pnew = P
        Qauxnew = Qaux
        Pauxnew = Paux

        min_inradius = jnp.min(mesh.cell_inradius)


        compute_max_abs_eigenvalue = solverQ.get_compute_max_abs_eigenvalue(
            mesh, pde1, settings
        )
        flux_operator = solverQ.get_flux_operator(
            mesh, pde1, bcs1, settings
        )
        boundary_operator1 = solverQ.get_apply_boundary_conditions(mesh, bcs1)
        
        boundary_operator2 = solverP.get_apply_boundary_conditions(mesh, bcs2)


        #@jax.jit
        def time_loop(time, iteration, i_snapshot, Q, Qaux, Pnew, Paux, Qold, Qauxold, Pold, Pauxold):
            loop_val = (time, iteration, i_snapshot, Q, Qaux, Pnew, Paux, Qold, Qauxold, Pold, Pauxold)

            def loop_body(init_value):
                time, iteration, i_snapshot, Q, Qaux, P, Paux, Qold, Qauxold, Pold, Pauxold = init_value

                dt = settings.compute_dt(
                    Q, Qaux, parameters1, min_inradius, compute_max_abs_eigenvalue
                )

                #############################################################
                #####################VELOCITY PREDICTOR######################
                #############################################################
                Qauxnew = solverQ.update_qaux(
                    Q, Qaux, Qold, Qauxold, mesh, pde1, parameters1, time, dt
                )
                Q1 = ode_solver_flux(
                    flux_operator, Q, Qauxnew, parameters1, dt
                )

                Q1 = Q1.at[5].set(Q0[5])

                Qauxnew = solverQ.update_qaux(
                    Q1, Qauxnew, Qold, Qauxold, mesh, pde1, parameters1, time, dt
                )
                #Qnew = ode_solver_source(
                #    source_operator,
                #    Q1,
                #    Qauxnew,
                #    parameters1,
                #    dt,
                #    func_jac=solverQ.compute_source_jac,
                #)

                #Qauxnew = solverQ.update_qaux(
                #    Qnew, Qauxnew, Q, Qaux, mesh, pde1, parameters1, time, dt
                #)
                Qnew = Q1
                Qnew = boundary_operator1(time, Qnew, Qauxnew, parameters1)


                #############################################################
                #########################PRESSURE############################
                #############################################################


                #P = P.at[:Qnew.shape[0]].set(Qnew)
                #Paux = solverP.update_qaux(
                #    P, Paux, Pold, Pauxold, mesh, pde2, parameters2, time, dt
                #)
                #Pnew = solverP.implicit_solve(P, Paux, Pold, Pauxold, mesh, model2, pde2, parameters2, time, dt, boundary_operator2, debug=[True, False])

                ##############################################################
                ##########################CORRECTOR###########################
                ##############################################################
                #Qnew = Pnew[:Qnew.shape[0]]
                #Qauxnew = solverQ.update_qaux(
                #    Qnew, Qaux, Qold, Qauxold, mesh, pde1, parameters1, time, dt
                #)

                #Qnew = boundary_operator1(time, Qnew, Qauxnew, parameters1)

                
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
                i_snapshot = save_fields(time, time_stamp, i_snapshot, Qnew, Qauxnew)


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
        compute_dt=timestepping.adaptive(CFL=0.1),
        #compute_dt=timestepping.constant(dt=0.001),
        #time_end=30.07184630730286572,
        #time_end=0.55,
        #time_end=0.013077056519679052,
        #time_end=0.01,
        time_end=0.52,
        output_snapshots=100,
        output_dir="outputs/vam",
    )

    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs1 = BC.BoundaryConditions(
        [
            BC.Lambda(physical_tag='left', prescribe_fields={
                1: lambda t, x, dx, q, qaux, p, n: .11197,
                #2: lambda t, x, dx, q, qaux, p, n: 0.,
                #3: lambda t, x, dx, q, qaux, p, n: 0.,
                #4: lambda t, x, dx, q, qaux, p, n: 0.
            }),
            BC.Extrapolation(physical_tag='right')
            #BC.Lambda(physical_tag='right', prescribe_fields={
            #    3: lambda t, x, dx, q, qaux, p, n: 0,
            #    4: lambda t, x, dx, q, qaux, p, n: 0,
            #}),
        ]
    )
    
    bcs2 = BC.BoundaryConditions(
        [
            BC.Extrapolation(physical_tag='left'),
            #BC.Lambda(physical_tag='left', prescribe_fields={
            #    0: lambda t, x, dx, q, qaux, p, n: 0.,
            #    1: lambda t, x, dx, q, qaux, p, n: 0.
            #}),
            #BC.Lambda(physical_tag='right', prescribe_fields={
            #    0: lambda t, x, dx, q, qaux, p, n: 0.,
            #    1: lambda t, x, dx, q, qaux, p, n: 0.
            #}),
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
        Q = np.zeros(2, dtype=float)
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

    mesh = petscMesh.Mesh.create_1d((-1.5, 15), 100)

    Q, Qaux = solve_vam(
        mesh,
        model1,
        model2,
        settings,
    )
    io.generate_vtk(os.path.join(settings.output.directory, f"{settings.name}.h5"))

def test_vam_1d_full():
    settings = Settings(
        name="VAM",
        parameters={
            "g": 9.81,
        },
        reconstruction=recon.constant,
        num_flux=flux.Zero(),
        nc_flux=nc_flux.segmentpath(),
        compute_dt=timestepping.adaptive(CFL=0.45),
        time_end=2.,
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
            BC.Lambda(physical_tag='left', prescribe_fields={
                0: lambda t, x, dx, q, qaux, p, n: 0.,
                1: lambda t, x, dx, q, qaux, p, n: 0.
            }),
            BC.Lambda(physical_tag='right', prescribe_fields={
                0: lambda t, x, dx, q, qaux, p, n: 0.,
                1: lambda t, x, dx, q, qaux, p, n: 0.
            }),
            #BC.Extrapolation(physical_tag='left'),
            #BC.Extrapolation(physical_tag='right')
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
        return Q

    ic1 = IC.UserFunction(custom_ic1)
    ic2 = IC.UserFunction(custom_ic2)
    
    
    model1 = VAMHyperbolic(
        parameters=settings.parameters,
        boundary_conditions=bcs1,
        initial_conditions=ic1,
        settings={},
    )
    
    model2 = VAMPoissonFull(
        parameters=settings.parameters,
        boundary_conditions=bcs2,
        initial_conditions=ic2,
        settings={},
    )

    mesh = petscMesh.Mesh.create_1d((-1.5, 1.5), 30)

    P, Paux = solve_vam_full(
        mesh,
        model1,
        model2,
        settings,
    )
    io.generate_vtk(os.path.join(settings.output.directory, f"{settings.name}.h5"))

@pytest.mark.critical
@pytest.mark.unfinished
def test_vam_1d_fullyimplicit():
    settings = Settings(
        name="VAM",
        parameters={
            "g": 9.81,
        },
        reconstruction=recon.constant,
        num_flux=flux.Zero(),
        nc_flux=nc_flux.segmentpath(),
        compute_dt=timestepping.adaptive(CFL=0.1),
        #compute_dt=timestepping.constant(dt=0.001),
        #time_end=30.07184630730286572,
        #time_end=0.55,
        #time_end=0.013077056519679052,
        #time_end=0.01,
        output_snapshots=100,
        output_dir="outputs/vam",
    )

    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs = BC.BoundaryConditions(
        [
            BC.Lambda(physical_tag='left', prescribe_fields={
                1: lambda t, x, dx, q, qaux, p, n: .11197,
                #2: lambda t, x, dx, q, qaux, p, n: 0.,
                #3: lambda t, x, dx, q, qaux, p, n: 0.,
                #4: lambda t, x, dx, q, qaux, p, n: 0.
            }),
            BC.Extrapolation(physical_tag='right')
            #BC.Lambda(physical_tag='right', prescribe_fields={
            #    3: lambda t, x, dx, q, qaux, p, n: 0,
            #    4: lambda t, x, dx, q, qaux, p, n: 0,
            #}),
        ]
    )
    
    
    def custom_ic(x):
        Q = np.zeros(8, dtype=float)
        Q[1] = np.where(x[0] < 1, 0.11197, 0.)
        Q[5] = 0.2*np.exp(-x[0]**2 / 1.0**2) 
        Q[0] = np.where(x[0] < 1, 0.34, 0.015) - Q[5]
        Q[0] = np.where(Q[0] > 0.015, Q[0], 0.015)
        return Q
    
    ic = IC.UserFunction(custom_ic)
    
    
    model = VAMFullImplicit(
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={},
    )
    
    mesh = petscMesh.Mesh.create_1d((-1.5, 15), 100)

    Q, Qaux = solve_vam_fullyimplicit(
        mesh,
        model,
        settings,
    )
    io.generate_vtk(os.path.join(settings.output.directory, f"{settings.name}.h5"))

class SolverFullImplicit(Solver):

    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):

        h=Q[0]
        hu0=Q[1]
        hu1=Q[2]
        hw0=Q[3]
        hw1=Q[4]
        b=Q[5]
        hp0=Q[6]
        hp1=Q[7]

        _h=Qold[0]
        _hu0=Qold[1]
        _hu1=Qold[2]
        _hw0=Qold[3]
        _hw1=Qold[4]
        _b=Qold[5]

        u0 = hu0/h
        u1 = hu1/h
        w0 = hw0/h
        w1 = hw1/h

        dhdx   = compute_gradient(h, mesh)[:, 0]
        dbdx  = compute_gradient(b, mesh)[:,0]
        dhu0dx  = compute_gradient(hu0, mesh)[:,0]
        dhu1dx  = compute_gradient(hu1, mesh)[:,0]
        dhw0dx  = compute_gradient(hw0, mesh)[:,0]
        dhw1dx  = compute_gradient(hw1, mesh)[:,0]
        dhp0dx  = compute_gradient(hp0, mesh)[:,0]
        dhp0dx  = compute_gradient(hp0, mesh)[:,0]
        dhp1dx  = compute_gradient(hp1, mesh)[:,0]
        ddhp0dxx  = compute_gradient(dhp0dx, mesh)[:,0]
        ddhp1dxx  = compute_gradient(dhp1dx, mesh)[:,0]
        du0dx  = compute_gradient(u0, mesh)[:,0]
        du1dx  = compute_gradient(u1, mesh)[:,0]
        hw2 = -(w0 + w1) + (u0 + u1) * dbdx

        dhdt = (h-_h)/dt
        dbdt = (b-_b)/dt
        dhu0dt = (hu0-_hu0)/dt
        dhu1dt = (hu1-_hu1)/dt
        dhw0dt = (hw0-_hw0)/dt
        dhw1dt = (hw1-_hw1)/dt

        #aux_variables=[ 'hw2', 'dhdt', 'dhu0dt', 'dhu1dt', 'dhw0dt', 'dhw1dt', 'dhdx', 'dhu0dx', 'dhu1dx', 'dhw0dx', 'dhw1dx', 'dhp0dx', 'dhp1dx', 'dbdx','ddhp0dxx', 'ddhp1dxx', 'du0dx', 'du1dx'],
        Qaux = Qaux.at[0].set(hw2)
        Qaux = Qaux.at[1].set(dhdt)
        Qaux = Qaux.at[2].set(dhu0dt)
        Qaux = Qaux.at[3].set(dhu1dt)
        Qaux = Qaux.at[4].set(dhw0dt)
        Qaux = Qaux.at[5].set(dhw1dt)
        Qaux = Qaux.at[6].set(dhdx)
        Qaux = Qaux.at[7].set(dhu0dx)
        Qaux = Qaux.at[8].set(dhu1dx)
        Qaux = Qaux.at[9].set(dhw0dx)
        Qaux = Qaux.at[10].set(dhw1dx)
        Qaux = Qaux.at[11].set(dhp0dx)
        Qaux = Qaux.at[12].set(dhp1dx)
        Qaux = Qaux.at[13].set(dbdx)
        Qaux = Qaux.at[14].set(ddhp0dxx)
        Qaux = Qaux.at[15].set(ddhp1dxx)
        Qaux = Qaux.at[16].set(du0dx)
        Qaux = Qaux.at[17].set(du1dx)
        return Qaux

def solve_vam_fullyimplicit(
    mesh, model, settings
):
    solver = SolverFullImplicit()
    
    Q, Qaux = solver.initialize(model, mesh)

    parameters = model.parameter_values

    parameters = jnp.asarray(parameters)
    
    mesh = convert_mesh_to_jax(mesh)

    pde, bcs = solver.transform_in_place(model)

    output_hdf5_path = os.path.join(settings.output.directory, f"{settings.name}.h5")
    save_fields = io.get_save_fields(output_hdf5_path, settings.output_write_all)

    def run(Q, Qaux, parameters, pde, bcs):
        iteration = 0.0
        time = 0.0
        assert model.dimension == mesh.dimension

        i_snapshot = 0.0
        dt_snapshot = settings.time_end / (settings.output_snapshots - 1)
        io.init_output_directory(settings.output.directory, settings.output_clean_dir)
        mesh.write_to_hdf5(output_hdf5_path)
        _ = save_fields(time, 0.0, i_snapshot, Q, Qaux)
        i_snapshot = save_fields(time, 0.0, i_snapshot, Q, Qaux)

        Q0 = Q
        Qnew = Q
        Qauxnew = Qaux

        min_inradius = jnp.min(mesh.cell_inradius)

        compute_max_abs_eigenvalue = solver.get_compute_max_abs_eigenvalue(
            mesh, pde, settings
        )
        flux_operator = solver.get_flux_operator(
            mesh, pde, bcs, settings
        )

        dt = 0.1

        flux_operator = solver.get_flux_operator(
            mesh, pde, bcs, settings
        )
        boundary_operator = solver.get_apply_boundary_conditions(mesh, bcs)
        
        time_start = gettime()

        #@jax.jit
        def time_loop(time, iteration, i_snapshot, Q, Qaux, Qold, Qauxold):
            loop_val = (time, iteration, i_snapshot, Q, Qaux, Qold, Qauxold)

            def loop_body(init_value):
                time, iteration, i_snapshot, Q, Qaux, Qold, Qauxold= init_value


                dt = settings.compute_dt(
                    Q, Qaux, parameters, min_inradius, compute_max_abs_eigenvalue
                )

                def residual(Q):
                    q = boundary_operator(time, Q, Qaux, parameters)
                    qaux = solver.update_qaux(q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt)
                    qnew = RK1(
                        flux_operator, q, qaux, parameters, dt
                    )

                    qnew = qnew.at[5].set(Q0[5])
                    res = pde.source_implicit(qnew, qaux, parameters)
                    res = res.at[:, mesh.n_inner_cells:].set(0.)
                    return res

                Qnew = solver.implicit_solve(Q, Qaux, Qold, Qauxold, mesh, model, pde, parameters, time, dt, boundary_operator, debug=[True, False], user_residual=residual)

                Qauxnew = solver.update_qaux(
                    Qnew, Qaux, Qold, Qauxold, mesh, pde, parameters, time, dt
                )

                Qnew = boundary_operator(time, Qnew, Qauxnew, parameters)
                
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
                i_snapshot = save_fields(time, time_stamp, i_snapshot, Qnew, Qauxnew)


                return (time, iteration, i_snapshot, Qnew, Qauxnew, Q, Qaux)

            def proceed(loop_val):
                time, iteration, i_snapshot, Q, Qaux, Qold, Qauxold = loop_val
                return time < settings.time_end

            (time, iteration, i_snapshot, Q, Qaux, Qold, Qauxold ) = jax.lax.while_loop(
                proceed, loop_body, loop_val
            )
            
            return Q, Qaux

        Q, Qaux = time_loop(time, iteration, i_snapshot, Qnew, Qauxnew, Q, Qaux )
        return Q, Qaux

    time_start = gettime()
    Q, Qaux = run(Q, Qaux, parameters, pde, bcs)

    print(f"Runtime: {gettime() - time_start}")

    return Q, Qaux


if __name__ == "__main__":
    #test_vam_1d()
    test_vam_1d()
    #test_vam_1d_fullyimplicit()
    #test_vam_1d_full()
