import numpy as np
import pytest
from types import SimpleNamespace

from library.fvm.solver import Solver, Settings
from library.model.model import *
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.pysolver.ode import RK1
import library.misc.io as io
# from library.pysolver.reconstruction import GradientMesh
import library.pysolver.reconstruction as recon
import library.pysolver.timestepping as timestepping
import library.pysolver.flux as flux
import library.mesh.mesh as petscMesh
import library.postprocessing.postprocessing as postprocessing
from library.mesh.mesh import convert_mesh_to_jax
import argparse

@pytest.mark.critical
@pytest.mark.unfinished
def test_smm_1d():
    level = 1
    settings = Settings(
        name="ShallowMoments",
        parameters={"g": 9.81, "C": 1.0, "nu": 0.000001, "lamda": 7, "rho":1, "eta":1, "c_slipmod": 1/70.},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        compute_dt=timestepping.constant(dt = 0.01),
        time_end=1.0,
        output_snapshots=100,
        output_dir = 'outputs/output_test'
    )

    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs = BC.BoundaryConditions(
        [
            #BC.Wall(physical_tag=tag, momentum_field_indices=[[i] for i in range(1, level+1)])
            BC.Extrapolation(physical_tag=tag)
            for (tag, tag_periodic_to) in zip(bc_tags, bc_tags_periodic_to)
        ]
    )
    ic = IC.RP(
        high=lambda n_field: np.array([0.2, 0.0] + [0.0 for l in range(level)]),
        low=lambda n_field: np.array([0.1, 0.0] + [0.0 for l in range(level)]),
    )
    model = ShallowMoments(
        fields=2 + level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"eigenvalue_mode": "symbolic", "friction": ["newtonian", "slip_mod"]},
    )

    mesh = petscMesh.Mesh.create_1d((-1, 30), 100)

    solver = Solver()
    solver.jax_fvm_unsteady_semidiscrete(
        mesh, model, settings
    )
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))


if __name__ == "__main__":


    test_smm_1d()
