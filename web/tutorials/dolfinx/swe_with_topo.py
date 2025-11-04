# ---
# title: "Dolfinx Demo"
# author: Ingo Steldermann
# date: 08/19/2025
# format:
#   html:
#     code-fold: false
#     code-tools: true
#     css: ../notebook.css
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: zoomy
#     language: python
#     name: python3
# ---

# ## Imports

# +
# | code-fold: true
# | code-summary: "Load packages"
# | output: false

import os
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy
from mpi4py import MPI
from dolfinx.io import gmshio
# import gmsh
# import dolfinx
from dolfinx import fem
import basix
import tqdm
from petsc4py import PETSc
import ufl
# import pyvista
from  dolfinx.fem import petsc
# import sys
# from dolfinx import mesh
from ufl import (
    TestFunction,
    TrialFunction,
    dx,
    inner,
)
import dolfinx
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx import mesh as dolfinx_mesh

import numpy.typing as npt


from library.zoomy_core.fvm.solver_numpy import Settings
from library.zoomy_core.model.models.shallow_water_topo import ShallowWaterEquationsWithTopo
from library.zoomy_core.model.models.shallow_water import ShallowWaterEquations
from library.zoomy_core.mesh.mesh import Mesh
import library.zoomy_core.model.initial_conditions as IC
import library.zoomy_core.model.boundary_conditions as BC
from library.zoomy_core.misc.misc import Zstruct
import library.zoomy_core.transformation.to_ufl as trafo
import library.dg.dolfinx_solver as dg


# + [markdown] vscode={"languageId": "raw"}
# # Transformation to UFL Code (Medium)
# -

# ### Map from Sympy to UFL

# +
velocity_fields = [[2,3]]
bcs = BC.BoundaryConditions(
    [
        # BC.Extrapolation(physical_tag="top"),
        # BC.Extrapolation(physical_tag="bottom"),
        # BC.Extrapolation(physical_tag="left"),
        # BC.Extrapolation(physical_tag="right"),
        BC.Wall(physical_tag="top", momentum_field_indices=velocity_fields),
        BC.Wall(physical_tag="bottom", momentum_field_indices=velocity_fields),
        BC.Wall(physical_tag="left", momentum_field_indices=velocity_fields),
        BC.Wall(physical_tag="right", momentum_field_indices=velocity_fields),
    ]
)

 ### Initial condition
def ic_q(x):
    R = 0.15
    r = np.sqrt((x[0] - 0.7)**2 + (x[1] - 0.7)**2)
    b = 0.1*np.sqrt((x[0] - 3.)**2 + (x[1] - 3.)**2)
    return np.array([b, np.where(r <= R, 1., 0.9), 0.*x[0], 0.*x[0]])

ic = IC.UserFunction(ic_q)

model = ShallowWaterEquationsWithTopo(
    dimension=2,
    aux_variables=1,
    boundary_conditions=bcs,
    initial_conditions=ic,
)


settings = Settings(name="Dolfinx", output=Zstruct(directory="outputs/dolfinx", filename='dg'))

# -

solver = dg.DolfinxHyperbolicSolver(settings=settings, time_end = 0.3,  IdentityMatrix = ufl.as_tensor([[1, 0, 0, 0],
                                                                                                        [0, 1, 0, 0],
                                                                                                        [0, 0, 1, 0],
                                                                                                        [0, 0, 0, 1]]))

main_dir = os.getenv('ZOOMY_DIR')
path_to_mesh = os.path.join(main_dir, "meshes", "channel_quad_2d", "mesh_coarse.msh")
solver.solve(path_to_mesh, model)


