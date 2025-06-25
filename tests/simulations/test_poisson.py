import os
import numpy as np
import jax
from jax import numpy as jnp
import pytest
from types import SimpleNamespace
from sympy import cos, pi
import sympy
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
from library.mesh.mesh import compute_derivatives


import library.mesh.mesh as petscMesh
import library.postprocessing.postprocessing as postprocessing
from library.mesh.mesh import convert_mesh_to_jax
import argparse



class PoissonSolver(Solver):
    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):

        T = Q[0]
        Told = Qold[0]
        dTdx = compute_derivatives(T, mesh, [[1,]])[:, 0]
        ddTdxx_alt = compute_derivatives(dTdx, mesh, [[1,]])[:, 0]

        ddTdxx = 1.0 * compute_derivatives(T, mesh, [[2,]])[:, 0] + 0.0 * ddTdxx_alt
        # jax.debug.print("ddTdxx: {ddTdxx}", ddTdxx=ddTdxx)
        dTdt = (T-Told)/dt
        rho = 933.31 + 0.037978 * T - 3.6274 * 10**(-4) * T**2
        lam = 619.2/T + 58646/T**3 + 3.237*10**(-3)*T - 1.382*10**(-5)*T**2
        Tref = 273.16
        x = T/Tref
        c1 = 1.8343 * 10**5
        c2 = 1.6357 * 10**8
        c3 = 3.5519 * 10**9
        c4 = 1.667 * 10**2
        c5 = 6.465 * 10**4
        c6 = 1.6935 * 10**6
        Cp = x**3 * (c1 + c2*x**2 + c3*x**6)/(1 + c4*x**2 + c5*x**8)
        kappa = rho / lam / Cp
        kappa = 1



        Qaux = Qaux.at[0].set(dTdt)
        Qaux = Qaux.at[1].set(0.)
        Qaux = Qaux.at[2].set(ddTdxx)
        Qaux = Qaux.at[3].set(kappa)
        return Qaux


def solve(
    mesh, model, settings
):
    solver = PoissonSolver()
    
    Q, Qaux = solver.initialize(model, mesh)

    parameters = model.parameter_values

    parameters = jnp.asarray(parameters)
    
    mesh = convert_mesh_to_jax(mesh)


    pde, bcs = solver._load_runtime_model(model)

    output_hdf5_path = os.path.join(settings.output_dir, f"{settings.name}.h5")
    save_fields = io.get_save_fields(output_hdf5_path, settings.output_write_all)

    def run(Q, Qaux, parameters, pde, bcs):
        iteration = 0.0
        time = 0.0
        assert model.dimension == mesh.dimension

        i_snapshot = 0.0
        dt_snapshot = settings.time_end / (settings.output_snapshots - 1)
        io.init_output_directory(settings.output_dir, settings.output_clean_dir)
        mesh.write_to_hdf5(output_hdf5_path)
        _ = save_fields(time, 0.0, i_snapshot, Q, Qaux)
        i_snapshot = save_fields(time, 0.0, i_snapshot, Q, Qaux)

        Q0 = Q
        Qnew = Q
        Qauxnew = Qaux

        min_inradius = jnp.min(mesh.cell_inradius)

        dt = 0.1

        space_solution_operator = solver.get_space_solution_operator(
            mesh, pde, bcs, settings
        )
        boundary_operator = solver.get_apply_boundary_conditions(mesh, bcs)
        
        time_start = gettime()

        #@jax.jit
        def time_loop(time, iteration, i_snapshot, Q, Qaux, Qold, Qauxold):
            loop_val = (time, iteration, i_snapshot, Q, Qaux, Qold, Qauxold)

            def loop_body(init_value):
                time, iteration, i_snapshot, Q, Qaux, Qold, Qauxold= init_value


                Qauxnew = solver.update_qaux(
                    Q, Qaux, Qold, Qauxold, mesh, pde, parameters, time, dt
                )
                
                def residual(Q):
                    q = jnp.zeros_like(Q)
                    q = q.at[0, 1:-1].set(Q[0, :mesh.n_inner_cells])
                    q = q.at[0, 0].set(Q[0, mesh.n_inner_cells])
                    q = q.at[0, -1].set(Q[0, mesh.n_inner_cells +1])
                    
                    lap_q = jnp.zeros_like(q)
                    # lap_q = lap_q.at[0, 1:-1].set((q[0, 2:] - 2 * q[0, 1:-1] + q[0, :-2]) / (mesh.cell_inradius[0]*2)**2)
                    lap_q = compute_derivatives(Q[0], mesh, [[2,]])[:,0]
                    res = jnp.zeros_like(Q)
                    res = res.at[0, :mesh.n_inner_cells].set(-lap_q[:mesh.n_inner_cells])
                    return res
                    

                Qnew = boundary_operator(time, Q, Qauxnew, parameters)
                Qnew = solver.implicit_solve(Q, Qaux, Qold, Qauxold, mesh, model, pde, parameters, time, dt, boundary_operator, debug=[True, False], user_residual=residual)



                Qnew = boundary_operator(time, Qnew, Qauxnew, parameters)
                Qauxnew = solver.update_qaux(
                    Qnew, Qauxnew, Qold, Qauxold, mesh, pde, parameters, time, dt
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

@pytest.mark.critical
@pytest.mark.unfinished
def test_poisson():
    settings = Settings(
        name="Poisson",
        parameters={"alpha": 1.},
        reconstruction=recon.constant,
        num_flux=flux.Zero(),
        nc_flux=nc_flux.segmentpath(),
        compute_dt=timestepping.adaptive(CFL=0.4),
        time_end=100.2,
        output_snapshots=100,
        output_dir="outputs/poisson",
    )

    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs = BC.BoundaryConditions( [
            BC.Lambda(physical_tag='left', prescribe_fields={0: lambda t, x, dx, q, qaux, p, n: 273.16}),
            BC.Lambda(physical_tag='right', prescribe_fields={0: lambda t, x, dx, q, qaux, p, n: 350.}),
            #BC.Lambda(physical_tag='left', prescribe_fields={0: lambda t, x, dx, q, qaux, p, n: sympy.sin(3.14 * t)+2.}),
            #BC.Extrapolation(physical_tag='right')
            #BC.Periodic(physical_tag='left', periodic_to_physical_tag='right'),
            #BC.Periodic(physical_tag='right', periodic_to_physical_tag='left')
        ]
    )
    
    def custom_ic(x):
        Q = np.zeros(1, dtype=float)
        #Q[0] = 0.2*np.exp(-x[0]**2 / 1.0**2) 
        Q[0] = 273.16
        return Q
    
    ic = IC.UserFunction(custom_ic)
    
    
    model = Poisson(
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={},
    )

    mesh = petscMesh.Mesh.create_1d((-1.5, 1.5), 20, lsq_degree = 2)

    Q, Qaux = solve(
        mesh,
        model,
        settings,
    )
    io.generate_vtk(os.path.join(settings.output_dir, f"{settings.name}.h5"))

if __name__ == "__main__":
    test_poisson()
