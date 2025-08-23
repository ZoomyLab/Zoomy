# ---
# title: "Transformation to UFL Code"
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
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: zoomy
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Imports

# %%
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

from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, geometry
import ufl
import numpy as np
import basix
import basix.quadrature as bxquad


from library.fvm.solver import Settings
from library.model.models.shallow_water import ShallowWaterEquations
from library.mesh.mesh import Mesh
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.misc.misc import Zstruct
import library.transformation.to_ufl as trafo



# %%
main_dir = os.getenv("ZOOMY_DIR")
path_to_mesh = os.path.join(main_dir, "meshes", "channel_quad_2d", "mesh_coarse.msh")


# %% [markdown] vscode={"languageId": "raw"}
# # Transformation to UFL Code (Medium)

# %% [markdown]
# ### Map from Sympy to UFL

# %%
bcs = BC.BoundaryConditions(
    [
        # BC.Extrapolation(physical_tag="top"),
        # BC.Extrapolation(physical_tag="bottom"),
        # BC.Extrapolation(physical_tag="left"),
        # BC.Extrapolation(physical_tag="right"),
        BC.Wall(physical_tag="top"),
        BC.Wall(physical_tag="bottom"),
        BC.Wall(physical_tag="left"),
        BC.Wall(physical_tag="right"),
    ]
)

def custom_ic(x):
    Q = np.zeros(3, dtype=float)
    Q[0] = np.where(x[0] < 5., 0.005, 0.001)
    return Q

ic = IC.UserFunction(custom_ic)

model = ShallowWaterEquations(
    dimension=2,
    boundary_conditions=bcs,
    initial_conditions=ic,
)


# %%
output_dir = 'outputs/ufl'


 # %%
 ### Initial condition
def ic_q(x):
    R = 0.15
    r = np.sqrt((x[0] - 0.7)**2 + (x[1] - 0.7)**2)
    b = 0.1*np.sqrt((x[0] - 3.)**2 + (x[1] - 3.)**2)
    return np.array([np.where(r <= R, 1., 0.9), 0.*x[0], 0.*x[0]])



### Simulation
q_sol = solve_time_loop(name="sim0",path_to_mesh=path_to_mesh, model=model, weak_form_function=weak_form_swe, initial_condition=ic_q, end_time=1.0, output_path=os.path.join(output_dir, 'sim'), CFL=0.2)


# %%
