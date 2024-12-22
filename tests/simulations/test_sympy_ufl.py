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
main_dir = os.getenv("SMPYTHON")


@pytest.mark.critical
def test_sympy_to_ufl():
    level = 0
    settings = Settings(
        name="ShallowMoments",
        parameters={"g": 9.81},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        nc_flux=nonconservative_flux.segmentpath(1),
        compute_dt=timestepping.adaptive(CFL=.9),
        time_end=1.,
        output_snapshots=100,
        output_clean_dir=True,
        output_dir="outputs/test_sympy_ufl",
    )

    bcs = BC.BoundaryConditions(
        [
            BC.Extrapolation(physical_tag="left"),
            BC.Extrapolation(physical_tag="right"),
        ]
    )


    def ic_func(x):
        Q = np.zeros(2, dtype=float)
        Q[0] = np.where(x[0]<0.0, 2., 1.)
        return Q
    ic = IC.UserFunction(ic_func)

    level = 0
    model = ShallowMoments(
        dimension=1,
        fields=2 + level,
        aux_fields=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": []},
        basis=Basis(basis=Legendre_shifted(order=level)),
    )
    
    main_dir = os.getenv("SMS")
    mesh = petscMesh.Mesh.create_1d((-1, 1), 100)

    jax_fvm_unsteady_semidiscrete(
        mesh, model, settings, ode_solver_flux=RK1, ode_solver_source=RK1
    )
    io.generate_vtk(os.path.join(settings.output_dir, f'{settings.name}.h5'))

if __name__ == "__main__":
    test_sympy_to_ufl()
