import os
import numpy as np
import jax
from jax import numpy as jnp
import pytest
from types import SimpleNamespace
from sympy import cos, pi
import sympy
from time import time as gettime

from zoomy_core.fvm.solver_jax import Solver, Settings
from zoomy_core.fvm.ode import RK1
import zoomy_core.fvm.reconstruction as recon
import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.fvm.flux as flux
import zoomy_core.fvm.nonconservative_flux as nc_flux
from zoomy_core.model.boundary_conditions import BoundaryCondition

from zoomy_core.model.model import *
import zoomy_core.model.initial_conditions as IC
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.misc.io as io
from zoomy_core.mesh.mesh import compute_gradient


import zoomy_core.mesh.mesh as petscMesh
import postprocessing.postprocessing as postprocessing
from zoomy_core.mesh.mesh import convert_mesh_to_jax
import argparse



class GNSolver(Solver):
    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):

        dq0dt = (Q[0] - Qold[0]) / dt
        dq1dt = (Q[1] - Qold[1]) / dt
        dq1dtdx = compute_gradient(dq1dt, mesh)[:,0]
        dq1dtdxx = compute_gradient(dq1dtdx, mesh)[:,0]
        D1 = dq1dt - 1/3 * dq1dtdxx        

        Qaux = Qaux.at[0].set(dq0dt)
        Qaux = Qaux.at[1].set(D1)

        return Qaux


def solve(
    mesh, model, settings
):
    solver = GNSolver()
    
    Q, Qaux = solver.initialize(model, mesh)

    parameters = model.parameter_values

    parameters = jnp.asarray(parameters)
    
    mesh = convert_mesh_to_jax(mesh)


    pde, bcs = solver.to_jax(model)

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


                Qauxnew = solver.update_qaux(
                    Q, Qaux, Qold, Qauxold, mesh, pde, parameters, time, dt
                )

                Qnew = boundary_operator(time, Q, Qauxnew, parameters)
                
                def residual(Q):
                    dF = jnp.zeros_like(Q)
                    dF = flux_operator(dt, Qnew, Qauxnew, parameters, dF)
                    dtF = dt * dF
                    qaux = solver.update_qaux(Q, Qauxnew, Qnew, Qauxnew, mesh, model, parameters, time, dt)
                    q = boundary_operator(time, Qnew, qaux, parameters)
                    qaux = qaux.at[2:].set(dtF)
                    res = pde.source_implicit(q, qaux, parameters)
                    res = res.at[:, mesh.n_inner_cells:].set(0.0)
                    return res
                
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
        name="GN",
        parameters={},
        reconstruction=recon.constant,
        num_flux=flux.Zero(),
        nc_flux=nc_flux.segmentpath(),
        compute_dt=timestepping.adaptive(CFL=0.4),
        time_end=10.2,
        output_snapshots=100,
        output_dir="outputs/GN",
    )

    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs = BC.BoundaryConditions([
            # BC.Lambda(tag='left', prescribe_fields={0: lambda t, x, dx, q, qaux, p, n: 273.16}),
            # BC.Lambda(tag='right', prescribe_fields={0: lambda t, x, dx, q, qaux, p, n: 350.}),
            #BC.Lambda(tag='left', prescribe_fields={0: lambda t, x, dx, q, qaux, p, n: sympy.sin(3.14 * t)+2.}),
            BC.Extrapolation(tag='left'),
            BC.Extrapolation(tag='right')
            # BC.Periodic(tag='left', periodic_to_physical_tag='right'),
            # BC.Periodic(tag='right', periodic_to_physical_tag='left')
        ]
    )
    
    def custom_ic(x):
        Q = np.zeros(1, dtype=float)
        Q[0] = 0.2*np.exp(-x[0]**2 / 1.0**2) 
        # Q[0] = 273.16
        return Q
    
    ic = IC.UserFunction(custom_ic)
    
    
    model = GN(
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={},
    )

    mesh = petscMesh.Mesh.create_1d((-10.5, 10.5), 300)

    Q, Qaux = solve(
        mesh,
        model,
        settings,
    )
    io.generate_vtk(os.path.join(settings.output.directory, f"{settings.name}.h5"))

if __name__ == "__main__":
    test_poisson()
