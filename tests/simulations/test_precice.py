import numpy as np
import pytest
from types import SimpleNamespace

from library.pysolver.solver import *
from library.model.model import *
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.pysolver.ode import RK1
import library.misc.io as io
# from library.pysolver.reconstruction import GradientMesh
import library.mesh.mesh as petscMesh
import library.postprocessing.postprocessing as postprocessing
import argparse

@pytest.mark.critical
@pytest.mark.unfinished
def test_smm_1d():
    level = 4
    settings = Settings(
        name="ShallowMoments",
        parameters={"g": 9.81, "C": 30.0, "nu": 0.000001, "eta": 1.},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        nc_flux=nonconservative_flux.segmentpath(1),
        compute_dt=timestepping.adaptive(CFL=0.9),
        time_end=1.0,
        output_snapshots=100,
        output_dir = 'outputs/output_smm_1d'
    )

    bc_tags = ["left", "right"]
    bc_tags_periodic_to = ["right", "left"]

    bcs = BC.BoundaryConditions(
        [
            BC.Extrapolation(physical_tag="left"),
            BC.Extrapolation(physical_tag="right"),
        ]
    )
    ic = IC.RP(
        high=lambda n_field: np.array([0.02, 0.0] + [0.0 for l in range(level)]),
        low=lambda n_field: np.array([0.02, 0.0] + [0.0 for l in range(level)]),
    )
    model = ShallowMoments(
        fields=2 + level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"eigenvalue_mode": "symbolic", "friction": []},
    )

    mesh = petscMesh.Mesh.create_1d((0.5, 5), 300)

    precice_fvm(
        mesh, model, settings
    )
    #io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))

def test_generate_output():
    level = 0
    settings = Settings(
        name="ShallowMoments",
        parameters={"g": 9.81, "C": 30.0, "nu": 0.000001, "eta": 1.},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        nc_flux=nonconservative_flux.segmentpath(1),
        compute_dt=timestepping.adaptive(CFL=0.9),
        time_end=1.0,
        output_snapshots=100,
        output_dir = 'outputs/output_smm_1d'
    )

    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))


if __name__ == "__main__":

    test_smm_1d()
    test_generate_output()
