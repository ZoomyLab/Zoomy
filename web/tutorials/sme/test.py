# ---
# title: "Simple"
# author: Ingo Steldermann
# date: 07/10/2025
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

# + [markdown] vscode={"languageId": "raw"}
# # Shallow Moment Tutorial (Simple)

# + [markdown] vscode={"languageId": "raw"}
# ## Imports

# +
# | code-fold: true
# | code-summary: "Load packages"
# | output: false

import os
import numpy as np
import jax
from jax import numpy as jnp
import pytest
from types import SimpleNamespace
from sympy import cos, pi, Piecewise
import sympy as sp

from library.zoomy_core.fvm.solver_jax import HyperbolicSolver, Settings
from library.zoomy_core.fvm.ode import RK1
import library.zoomy_core.fvm.reconstruction as recon
import library.zoomy_core.fvm.timestepping as timestepping
import library.zoomy_core.fvm.flux as flux
import library.zoomy_core.fvm.nonconservative_flux as nc_flux
from library.zoomy_core.model.boundary_conditions import BoundaryCondition
from library.zoomy_core.model.models.basisfunctions import Basisfunction, Legendre_shifted
from library.zoomy_core.model.models.basismatrices import Basismatrices
from library.zoomy_core.misc.misc import Zstruct

from library.zoomy_core.model.models.shallow_moments import ShallowMoments2d, ShallowMoments
import library.zoomy_core.model.initial_conditions as IC
import library.zoomy_core.model.boundary_conditions as BC
import library.zoomy_core.misc.io as io
from library.zoomy_core.mesh.mesh import compute_derivatives
from tests.pdesoft import plots_paper
import library.postprocessing.visualization as visu


import library.zoomy_core.mesh.mesh as petscMesh
import library.postprocessing.postprocessing as postprocessing
from library.zoomy_core.mesh.mesh import convert_mesh_to_jax
import argparse
# -

# ## Model

level = 1
offset = 1+level
n_fields = 3 + 2 * level
settings = Settings(
    name="SME",
    output=Zstruct(
        directory=f"outputs/test_{level}", filename="SME", snapshots=100
    ),
)

# +
f_t = lambda t: sp.sin(t / 2 * 3.14 * 2) * 0.2 + 1.
inflow_dict = { 
    0: lambda t, x, dx, q, qaux, p, n: f_t(t),
    1: lambda t, x, dx, q, qaux, p, n: sp.sqrt(sp.Abs(f_t(t)-1) * 9.81) * q[0],
    2: lambda t, x, dx, q, qaux, p, n: 0.1 * q[0]

                }
#inflow_dict.update({
#    1+i: lambda t, x, dx, q, qaux, p, n: 0 for i in range(level)
#})

bcs = BC.BoundaryConditions(
    [
        BC.Lambda(physical_tag="left", prescribe_fields=inflow_dict),
        BC.Extrapolation(physical_tag="right"),
    ]
)

def custom_ic(x):
    Q = np.zeros(2 + level, dtype=float)
    Q[0] = 1.
    return Q

ic = IC.UserFunction(custom_ic)

model = ShallowMoments(
    level=level,
    boundary_conditions=bcs,
    initial_conditions=ic,
)

print(model.flux()[0])
print(model.nonconservative_matrix()[0])
print(model.quasilinear_matrix()[0])
print(model.source())
print(model.eigenvalues())

main_dir = os.getenv("ZOOMY_DIR")
mesh = petscMesh.Mesh.create_1d([0, 100], 500)

mesh = convert_mesh_to_jax(mesh)

class SMESolver(HyperbolicSolver):
    def update_qaux(self, Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt):
        offset = level+1
        for i in range(level+1):
            dalphadx = compute_derivatives(Q[1+i]/Q[0], mesh, derivatives_multi_index=[[1, 0]])[:,0]
            Qaux = Qaux.at[1+i].set(dalphadx)
        return Qaux

solver = SMESolver(settings=settings, time_end=100)
# -

# ## Solve

Qnew, Qaux = solver.solve(mesh, model)

# ## Visualization

io.generate_vtk(os.path.join(settings.output.directory, f"{settings.output.filename}.h5"))
# postprocessing.vtk_project_2d_to_3d(model, settings, Nz=20, filename='out_3d')

# +
# visu.pyvista_3d(settings.output.directory, scale=1.0)
# -


