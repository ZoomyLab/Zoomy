import os
from library.model.models import *
from library.python.fvm.solver_jax import *
import library.python.fvm.flux
import library.python.fvm.reconstruction
import library.python.fvm.nonconservative_flux
import library.python.fvm.timestepping
import library.python.mesh.mesh as petscMesh

import library.python.misc.io


def run(model_class, ic, bcs, mesh, solver, settings, **param):
    level = param['level']
    model = model_class(
        dimension=2,
        fields=3 + 2 * level,
        aux_variables=0,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings=settings,
    )

    solver(
        mesh, model, settings)

    io.generate_vtk(os.path.join(settings.output.directory, f'{settings.name}.h5'))
