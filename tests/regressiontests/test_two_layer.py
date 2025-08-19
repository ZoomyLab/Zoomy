import os
import numpy as np
import jax
from jax import numpy as jnp
import pytest
from time import time as gettime

from library.fvm.solver import Solver, Settings
from library.fvm.ode import RK1
import library.fvm.reconstruction as recon
import library.fvm.timestepping as timestepping
import library.fvm.flux as flux
import library.fvm.nonconservative_flux as nc_flux



import library.mesh.mesh as petscMesh
from library.mesh.mesh import convert_mesh_to_jax

import numpy as np
from sympy import Matrix, sqrt


from library.model.model import Model
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
import library.misc.io as io
from library.mesh.mesh import compute_derivatives
from library.model import *
from library.model.models.base import (
    register_sympy_attribute,
)
from library.model.models.base import Model
import library.model.initial_conditions as IC
from library.model.models.basisfunctions import *
from library.model.models.basismatrices import *


class VAMHyperbolic(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=6,
        aux_variables=['hw2', 'hp0', 'hp1', 'dbdx', 'dhdx', 'dhp0dx', 'dhp1dx'],
        parameters={},
        _default_parameters={"g": 9.81},
        settings={},
        settings_default={},
    ):
        self.variables = register_sympy_attribute(fields, "q")
        self.n_variables = self.variables.length()
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_variables=aux_variables,
            parameters=parameters,
            _default_parameters=_default_parameters,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings={**settings_default, **settings},
        )
        
    def flux(self):
        fx = Matrix([0 for i in range(self.n_variables)])
        hw2 = self.aux_variables.hw2
        h = self.variables[0]
        hu0 = self.variables[1]
        hu1 = self.variables[2]
        hw0 = self.variables[3]
        hw1 = self.variables[4]
        param = self.parameters

        u0 = hu0 / h
        u1 = hu1 / h
        w0 = hw0 / h
        w1 = hw1 / h
        
        fx[0] = hu0
        fx[1] = hu0 * u0 + 1/3 * hu1 * u1 * u1
        fx[2] = hu0 * w0 + 1/3 * hu1 * w1
        fx[3] = 2*hu0 * u1
        fx[4] = hu0 * w1 + u1 * (hw0 + 2/5*hw2)
        
        return [fx]

    def nonconservative_matrix(self):
        nc = Matrix([[0 for i in range(self.n_variables)] for j in range(self.n_variables)])

        hw2 = self.aux_variables.hw2
        h = self.variables[0]
        hu0 = self.variables[1]
        hu1 = self.variables[2]
        hw0 = self.variables[3]
        hw1 = self.variables[4]
        param = self.parameters

        u0 = hu0 / h
        u1 = hu1 / h
        w0 = hw0 / h
        w1 = hw1 / h
        w2 = hw2 / h

        nc[1, 0] = param.g * h
        nc[3, 2] = u0
        nc[4, 2] = - 1/5 * w2 + w0
        nc[1, 5] = param.g * h
        return [-nc]
    
    def eigenvalues(self):
        ev = Matrix([0 for i in range(self.n_variables)])
        h = self.variables[0]
        hu0 = self.variables[1]
        hu1 = self.variables[2]
        param = self.parameters

        u0 = hu0 / h
        u1 = hu1 / h

        ev[0] = u0
        ev[1] = u0 + 1/sqrt(3) * u1
        ev[2] = u0 - 1/sqrt(3) * u1
        ev[3] = u0 + sqrt(param.g * h + u1**2)
        ev[4] = u0 - sqrt(param.g * h + u1**2)
        ev[5] = 0
        
        return ev

    def source_implicit(self):
        R = Matrix([0 for i in range(self.n_variables)])
        hw2 = self.aux_variables.hw2
        h = self.variables[0]
        hu0 = self.variables[1]
        hu1 = self.variables[2]
        hw0 = self.variables[3]
        hw1 = self.variables[4]
        b = self.variables[5]
        param = self.parameters

        u0 = hu0 / h
        u1 = hu1 / h
        w0 = hw0 / h
        w1 = hw1 / h
        w2 = hw2 /h  


        p0 = self.aux_variables.hp0/h
        p1 = self.aux_variables.hp1/h
        dbdx = self.aux_variables.dbdx
        dhdx = self.aux_variables.dhdx
        dhp0dx = self.aux_variables.dhp0dx
        dhp1dx = self.aux_variables.dhp1dx

        R[0] = 0.
        R[1] = dhp0dx + 2 * p1 * dbdx 
        R[2] = -2*p1
        R[3] = dhp1dx - (3*p0 - p1)*dhdx  -6*(p0-p1)*dbdx
        R[4] = 6*(p0-p1)
        R[5] = 0.
        return R
        

class VAMPoisson(Model):
    def __init__(
        self,
        boundary_conditions,
        initial_conditions,
        dimension=1,
        fields=['hp0', 'hp1'],
        aux_variables=['h', 'hu0', 'hu1', 'hw0', 'hw1' ,'b', 'hw2', 'dbdx', 'ddbdxx', 'dhdx', 'ddhdxx', 'du0dx', 'du1dx', 'dhp0dx', 'ddhp0dxx', 'dhp1dx', 'ddhp1dxx', 'dt', 'd4hp0dx4', 'd4hp1dx4'],
        parameters={},
        _default_parameters={"g": 9.81},
        settings={},
        settings_default={},
    ):
        self.variables = register_sympy_attribute(fields, "q")
        self.n_variables = self.variables.length()
        super().__init__(
            dimension=dimension,
            fields=fields,
            aux_variables=aux_variables,
            parameters=parameters,
            _default_parameters=_default_parameters,
            boundary_conditions=boundary_conditions,
            initial_conditions=initial_conditions,
            settings={**settings_default, **settings},
        )

    def source_implicit(self):
        R = Matrix([0 for i in range(self.n_variables)])

        h = self.aux_variables.h
        #p0 = self.variables.p0/h
        #p1 = self.variables.p1/h
        dt = self.aux_variables.dt

        dbdx   = self.aux_variables.dbdx
        ddbdxx = self.aux_variables.ddbdxx
        dhdx   = self.aux_variables.dhdx
        ddhdxx = self.aux_variables.ddhdxx
        #dp0dx   = self.aux_variables.dp0dx
        #ddp0dxx = self.aux_variables.ddp0dxx
        #dp1dx   = self.aux_variables.dp1dx
        #ddp1dxx   = self.aux_variables.ddp1dxx
        du0dx = self.aux_variables.du0dx
        du1dx = self.aux_variables.du1dx

        u0 = self.aux_variables.hu0/h
        u1 = self.aux_variables.hu1/h
        w0 = self.aux_variables.hw0/h
        w1 = self.aux_variables.hw1/h
        
        hp0 = self.variables.hp0
        hp1 = self.variables.hp1
        dhp0dx   = self.aux_variables.dhp0dx
        ddhp0dxx = self.aux_variables.ddhp0dxx
        dhp1dx   = self.aux_variables.dhp1dx
        ddhp1dxx   = self.aux_variables.ddhp1dxx
        d4hp0dx4   = self.aux_variables.d4hp0dx4
        d4hp1dx4   = self.aux_variables.d4hp1dx4


        

        delta = 0.
        #I1 = 0.666666666666667*dt*dp0dx - 2*(-dt*(h*ddp0dxx + p0*dhdx + 2*p1*dbdx) + h*dp1dx)*dbdx/h + 2*(-dt*(-(3*p0 - p1)*dhdx - (6*p0 - 6*p1)*dbdx + h*dp0dx + p1*dhdx) + h*u1)/h + 0.333333333333333*(2*dt*p1 + h*u0)*dhdx/h + (-(-dt*(h*ddp0dxx + p0*dhdx + 2*p1*dbdx) + h*dp1dx)*dhdx/h**2 + (-dt*(h*du1dx + p0*ddhdxx + 2*p1*ddbdxx + 2*dbdx*dp0dx + 2*dhdx*ddp0dxx) + h*dhdx + dp1dx*dhdx)/h)*h + 0.333333333333333*h*du0dx + 0.333333333333333*u0*dhdx + delta * ddp0dxx
        #I2 = -2*(-dt*(6*p0 - 6*p1) + h*w0)/h + 2*(2*dt*p1 + h*u0)*dbdx/h + (2*dt*p1 + h*u0)*dhdx/h + (-(-dt*(h*ddp0dxx + p0*dhdx + 2*p1*dbdx) + h*dp1dx)*dhdx/h**2 + (-dt*(h*du1dx + p0*ddhdxx + 2*p1*ddbdxx + 2*dbdx*dp0dx + 2*dhdx*ddp0dxx) + h*dhdx + dp1dx*dhdx)/h)*h + delta * ddp1dxx
        I1 = 0.666666666666667*dt*dhp1dx/h - 0.666666666666667*dt*hp1*dhdx/h**2 - 2*(-dt*(dhp0dx + 2*hp1*dbdx/h) + h*u0)*dbdx/h + 2*(-dt*(-(3*hp0/h - hp1/h)*dhdx - (6*hp0/h - 6*hp1/h)*dbdx + dhp1dx) + h*w0)/h + 0.333333333333333*(2*dt*hp1/h + h*u1)*dhdx/h + (-(-dt*(dhp0dx + 2*hp1*dbdx/h) + h*u0)*dhdx/h**2 + (-dt*(ddhp0dxx + 2*hp1*ddbdxx/h + 2*dbdx*dhp1dx/h - 2*hp1*dbdx*dhdx/h**2) + h*du0dx + u0*dhdx)/h)*h + 0.333333333333333*h*du1dx + 0.333333333333333*u1*dhdx + delta *ddhp0dxx
        I2 = -2*(-dt*(6*hp0/h - 6*hp1/h) + h*w1)/h + 2*(2*dt*hp1/h + h*u1)*dbdx/h + (2*dt*hp1/h + h*u1)*dhdx/h + (-(-dt*(dhp0dx + 2*hp1*dbdx/h) + h*u0)*dhdx/h**2 + (-dt*(ddhp0dxx + 2*hp1*ddbdxx/h + 2*dbdx*dhp1dx/h - 2*hp1*dbdx*dhdx/h**2) + h*du0dx + u0*dhdx)/h)*h + delta *ddhp1dxx
        R[0] = I1 +I2
        R[1] = -I2 + I1

        return R
    
    def eigenvalues(self):
        ev = Matrix([0 for i in range(self.n_variables)])
        return ev


class HyperbolicSolverHP(Solver):

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

        dhdx   = compute_derivatives(h, mesh, derivatives_multi_index=([[1]]))[:, 0]
        dbdx  = compute_derivatives(b, mesh, derivatives_multi_index=([[1]]))[:,0]
        dhp0dx  = compute_derivatives(hp0, mesh, derivatives_multi_index=([[1]]))[:,0]
        dhp1dx  = compute_derivatives(hp1, mesh, derivatives_multi_index=([[1]]))[:,0]
        hw2 = -(w0 + w1) + (u0 + u1) * dbdx
        # Qaux_variables=['hw2', 'hp0', 'hp1', 'dbdx', 'dhdx', 'dhp0dx', 'dhp1dx'],
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

class PoissonSolverHP(Solver):
    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):

        hp0 = Q[0]
        hp1 = Q[1]
        
        #        aux_variables=['h', 'hu0', 'hu1', 'hw0', 'hw1' ,'b', 'hw2', 'dbdx', 'ddbdxx', 'dhdx', 'ddhdxx', 'du0dx', 'du1dx', 'dhp0dx', 'ddhp0dxx', 'dhp1dx', 'ddhp1dxx', 'dt', 'd4hp0dx4', 'd4hp1dx4'],



        dhp0dx = compute_derivatives(hp0, mesh, derivatives_multi_index=([[1]]))[:, 0]
        ddhp0dxx = compute_derivatives(hp0, mesh, derivatives_multi_index=([[2]]))[:, 0]
        dhp1dx = compute_derivatives(hp1, mesh, derivatives_multi_index=([[1]]))[:, 0]
        ddhp1dxx = compute_derivatives(hp1, mesh, derivatives_multi_index=([[2]]))[:, 0]

        Qaux = Qaux.at[13].set(dhp0dx)
        Qaux = Qaux.at[14].set(ddhp0dxx)
        Qaux = Qaux.at[15].set(dhp1dx)
        Qaux = Qaux.at[16].set(ddhp1dxx)

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


    pde1, bcs1 = solverQ.to_jax(model1)
    pde2, bcs2 = solverP.to_jax(model2)
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

                # Paux_variables=['h', 'hu0', 'hu1', 'hw0', 'hw1' ,'b', 'hw2', 'dbdx', 'ddbdxx', 'dhdx', 'ddhdxx', 'du0dx', 'du1dx', 'dhp0dx', 'ddhp0dxx', 'dhp1dx', 'ddhp1dxx', 'dt', 'd4hp0dx4', 'd4hp1dx4'],

                h = Q[0]
                u0 = Q[1]/h
                u1 = Q[2]/h
                w0 = Q[3]/h
                w1 = Q[4]/h
                b = Q[5]
                dbdx = compute_derivatives(b, mesh, derivatives_multi_index=([[1]]))[:, 0]
                ddbdxx = compute_derivatives(b, mesh, derivatives_multi_index=([[2]]))[:, 0]
                dhdx = compute_derivatives(h, mesh, derivatives_multi_index=([[1]]))[:,0]
                ddhdxx = compute_derivatives(h, mesh, derivatives_multi_index=([[2]]))[:,0]
                du0dx = compute_derivatives(u0, mesh, derivatives_multi_index=([[1]]))[:, 0]
                du1dx = compute_derivatives(u1, mesh, derivatives_multi_index=([[1]]))[:, 0]
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

                   
                    dp0dx = compute_derivatives(P[0], mesh, derivatives_multi_index=([[1]]))[:, 0]
                    ddp0dxx = compute_derivatives(P[0], mesh, derivatives_multi_index=([[2]]))[:, 0]
                    dp1dx = compute_derivatives(P[1], mesh, derivatives_multi_index=([[1]]))[:, 0]
                    ddp1dxx = compute_derivatives(P[1], mesh, derivatives_multi_index=([[2]]))[:, 0]


                    paux = Paux
                    paux = paux.at[13].set(dp0dx)
                    paux = paux.at[14].set(ddp0dxx)
                    paux = paux.at[15].set(dp1dx)
                    paux = paux.at[16].set(ddp1dxx)

                    res = pde2.source_implicit(P, paux, parameters2)
                    res = res.at[:, mesh.n_inner_cells:].set(0.)
                    return res


                Pnew = solverP.implicit_solve(P, Paux, Pold, Pauxold, mesh, model2, pde2, parameters2, time, dt, boundary_operator2, debug=[True, False], user_residual=residual)

                ############################################################
                ########################CORRECTOR###########################
                ############################################################
                Qauxnew = Qauxnew.at[1].set(Pnew[0])
                Qauxnew = Qauxnew.at[2].set(Pnew[1])
                #Qauxnew = Qauxnew.at[2].set(0.)
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
        time_end=0.1,
        output_snapshots=100,
        output_dir="outputs/vam",
    )

    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs1 = BC.BoundaryConditions(
        [
            # BC.Lambda(physical_tag='left', prescribe_fields={
            #     1: lambda t, x, dx, q, qaux, p, n: .11197,
            #     #2: lambda t, x, dx, q, qaux, p, n: 0.,
            #     #3: lambda t, x, dx, q, qaux, p, n: 0.,
            #     #4: lambda t, x, dx, q, qaux, p, n: 0.
            # }),
            BC.Extrapolation(physical_tag='left'),
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
            # BC.Lambda(physical_tag='left', prescribe_fields={
            #    0: lambda t, x, dx, q, qaux, p, n: 0.,
            #    1: lambda t, x, dx, q, qaux, p, n: 0.
            # }),
            # BC.Lambda(physical_tag='right', prescribe_fields={
            #    0: lambda t, x, dx, q, qaux, p, n: 0.,
            #    1: lambda t, x, dx, q, qaux, p, n: 0.
            # }),
            BC.Extrapolation(physical_tag='right')
        ]
    )
    
    def custom_ic1(x):
        Q = np.zeros(6, dtype=float)
        Q[1] = np.where(x[0]-5 < 1, 0.0, 0.)
        Q[5] = 0.00*np.exp(-(x[0]-5)**2 / 1.0**2) 
        Q[0] = np.where(x[0] < 100, 0.34, 0.015) - Q[5]
        Q[0] = np.where(Q[0] > 0.015, Q[0], 0.015)
        Q[0] = np.where(x[0]**2 < 0.5, 0.2, 0.1)
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

    mesh = petscMesh.Mesh.create_1d((-5, 5), 100)

    Q, Qaux = solve_vam(
        mesh,
        model1,
        model2,
        settings,
    )
    io.generate_vtk(os.path.join(settings.output.directory, f"{settings.name}.h5"))


if __name__ == "__main__":
    test_vam_1d()

