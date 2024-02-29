import numpy as np
import pytest
from types import SimpleNamespace

from library.pysolver.solver import *
from library.model.model import *
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.pysolver.ode import RK1
import library.misc.io as io
from library.pysolver.reconstruction import GradientMesh
import library.postprocessing.postprocessing as postprocessing
import argparse


@pytest.mark.critical
@pytest.mark.unfinished
def test_ssf_2d():
    main_dir = os.getenv("SMS")
    settings = Settings(
        name="ShearShallowFlow2d",
        parameters={"g": 9.81},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        #compute_dt=timestepping.adaptive(CFL=0.15),
        compute_dt=timestepping.constant(dt=0.01),
        time_end=2.00,
        output_snapshots=100,
        output_clean_dir=True,
        output_dir=os.path.join(main_dir, "outputs/output_ssf"),
        callbacks=[]
    )

    velocity = 36.*1000./3600.
    height = 0.5
    inflow_dict = {i: 0.0 for i in range(0, 6)}
    inflow_dict[0] = height
    inflow_dict[1] = velocity * height
    outflow_dict = {}
    # outflow_dict = {0: 0.1}

    bcs = BC.BoundaryConditions(
        [
            # BC.Wall(physical_tag="hole"),
            # BC.Wall(physical_tag="top"),
            # BC.Wall(physical_tag="bottom"),
            # BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
            # BC.InflowOutflow(physical_tag="right", prescribe_fields=outflow_dict),
            # BC.Wall(physical_tag="pillar"),
            BC.Wall(physical_tag="top"),
            BC.Wall(physical_tag="bottom"),
            BC.InflowOutflow(physical_tag="inflow", prescribe_fields=inflow_dict),
            BC.InflowOutflow(physical_tag="outflow", prescribe_fields=outflow_dict),
        ]
    )

    def ic_func(x):
        Q = np.zeros(6, dtype=float)
        Q[0] = height
        Q[1] = 0.
        Q[2] = 0.
        return Q

    ic = IC.UserFunction(ic_func)

    model = ShearShallowFlow2d(
        dimension=2,
        fields=6,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        # settings={"friction": ["chezy", "newtonian"]},
        # settings={"friction": ["newtonian"]},
        settings={},
    )
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_gmsh(
    #     os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_fine.msh"),
    #     # os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_finest.msh"),
        # os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_coarse.msh"),
        # os.path.join(main_dir, 'meshes/channel_openfoam/mesh_coarse_2d.msh'),
        os.path.join(main_dir, 'meshes/simple_openfoam/mesh_2d_mid.msh'),
    #     # os.path.join(main_dir, "meshes/channel_2d_hole_sym/mesh_mid.msh"),
        "triangle",
     )
    # mesh = Mesh.from_hdf5( os.path.join(os.path.join(main_dir, settings.output_dir), "mesh.hdf5"))

    fvm_c_unsteady_semidiscete(
        mesh,
        model,
        settings,
        ode_solver_flux="RK1",
        ode_solver_source="RK1",
        rebuild_model=True,
        rebuild_mesh=True,
        rebuild_c=True,
    )

    io.generate_vtk(settings.output_dir)



if __name__ == "__main__":
    test_ssf_2d()