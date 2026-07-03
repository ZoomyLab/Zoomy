# ---
# jupyter:
#   jupytext:
#     text_representation:
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Malpasset dam-break — Firedrake DG0
#
# Self-contained shallow-water run. The model is written directly as a
# `SystemModel`: every operator (flux, hydrostatic pressure, non-conservative
# bed-slope matrix, friction, viscous stress, eigenvalues) is given by hand, so
# there is no model-tagging and no closures. Boundary conditions are part of the
# model. The only downstream step is lifting the `SystemModel` to a
# `NumericalSystemModel` and solving.

# %% [markdown]
# ## Imports

# %%
import os
import numpy as np
import sympy as sp
from sympy import Matrix, Max, Min, sqrt, Rational, zeros, eye
import firedrake as fd
import meshio

import zoomy_core.misc.misc as misc
from zoomy_core.misc.misc import Zstruct
import zoomy_core.model.boundary_conditions as bc
import zoomy_core.model.aux_boundary_conditions as aux_bc
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeHLL
from zoomy_core.fvm.solver_numpy import Settings
from zoomy_firedrake.firedrake_solver import FiredrakeHyperbolicSolver

# %% [markdown]
# ## Mesh

# %%
mesh = os.path.join(misc.get_main_directory(), "data", "malpasset",
                    "geo_malpasset-small.msh")

# %% [markdown]
# ## Shallow-water model
#
# State `(b, h, hu, hv)`, parameters `g, n, nu, eps, u_max`. `hinv` is the
# Kurganov-Petrova desingularised `1/h`. The bed friction is classical Manning;
# the viscous term is the full deviatoric stress `div(nu h (grad u + grad u^T))`
# including the normal stresses. Friction is applied through the explicit slot.

# %%
class ShallowWater(SystemModel):
    def __init__(self, g=9.81, n=0.033, nu=1.0, eps=1e-2, u_max=30.0):
        t = sp.Symbol("t", real=True)
        x, y = sp.symbols("x y", real=True)
        n0, n1 = sp.symbols("n0 n1", real=True)
        distance = sp.Symbol("distance", real=True)
        position = Zstruct(X0=sp.Symbol("X0"), X1=sp.Symbol("X1"),
                           X2=sp.Symbol("X2"))
        position._symbolic_name = "X"

        b, h, hu, hv = sp.symbols("b h hu hv", real=True)
        state = [b, h, hu, hv]
        variables = Zstruct(b=b, h=h, hu=hu, hv=hv)
        variables._symbolic_name = "Q"
        aux = Zstruct()
        aux._symbolic_name = "Qaux"

        g_, n_, nu_, eps_, u_max_ = sp.symbols("g n nu eps u_max", positive=True)
        parameters = Zstruct(g=g_, n=n_, nu=nu_, eps=eps_, u_max=u_max_)
        parameters._symbolic_name = "p"
        parameter_values = Zstruct(g=g, n=n, nu=nu, eps=eps, u_max=u_max)
        normal = Zstruct(n0=n0, n1=n1)
        normal._symbolic_name = "n"

        hinv = sqrt(2) * h / sqrt(h ** 4 + Max(h, eps_) ** 4)
        u, w = hu * hinv, hv * hinv
        speed = sqrt(u * u + w * w)

        flux = zeros(4, 2)
        flux[1, 0], flux[1, 1] = hu, hv
        flux[2, 0], flux[2, 1] = hu * u, hu * w
        flux[3, 0], flux[3, 1] = hv * u, hv * w

        pressure = zeros(4, 2)
        pressure[2, 0] = g_ * h ** 2 / 2
        pressure[3, 1] = g_ * h ** 2 / 2

        bed_slope = sp.MutableDenseNDimArray.zeros(4, 4, 2)
        bed_slope[2, 0, 0] = g_ * h
        bed_slope[3, 0, 1] = g_ * h

        friction = -g_ * n_ ** 2 * speed / Max(h, eps_) ** Rational(1, 3)
        friction_source = Matrix([0, 0, friction * u, friction * w])

        stress = sp.MutableDenseNDimArray.zeros(4, 4, 2, 2)
        vel = {2: u, 3: w}

        def viscous(i, m, d, e, factor):
            stress[i, m, d, e] += factor * nu_
            stress[i, 1, d, e] += -factor * nu_ * vel[m]

        viscous(2, 2, 0, 0, 2); viscous(2, 2, 1, 1, 1); viscous(2, 3, 1, 0, 1)
        viscous(3, 3, 0, 0, 1); viscous(3, 2, 0, 1, 1); viscous(3, 3, 1, 1, 2)

        normal_velocity = u * n0 + w * n1
        wave = sqrt(g_ * Max(h, eps_))
        dry = sp.Function("conditional")
        eigenvalues = Matrix([
            dry(h > eps_, e, sp.S.Zero)
            for e in (sp.S.Zero, normal_velocity,
                      normal_velocity - wave, normal_velocity + wave)])

        cap = Max(h - eps_, sp.S.Zero) * u_max_
        clamp = lambda q: Max(-cap, Min(q, cap))
        update_variables = Matrix([b, h, clamp(hu), clamp(hv)])

        reconstruction = Matrix([b, b + h, hu, hv])

        walls = bc.BoundaryConditions([
            bc.Wall(tag="wall", momentum_field_indices=[[2, 3]],
                    permeability=0.0, wall_slip=1.0)])
        wall_kernel = walls.get_boundary_condition_function(
            t, position, distance, variables, aux, parameters, normal,
            function_name="boundary_conditions")
        gradient_kernel = walls.get_boundary_gradient_function(
            t, position, distance, variables, aux, parameters, normal,
            function_name="boundary_gradients")
        aux_walls = bc.BoundaryConditions([aux_bc.Extrapolation(tag="wall")])
        aux_kernel = aux_walls.get_boundary_condition_function(
            t, position, distance, variables, aux, parameters, normal,
            function_name="aux_boundary_conditions")

        super().__init__(
            time=t, space=[x, y], state=state, aux_state=[],
            parameters=parameters, parameter_values=parameter_values,
            normal=normal,
            flux=flux,
            hydrostatic_pressure=pressure,
            nonconservative_matrix=bed_slope,
            source=zeros(4, 1),
            source_explicit=friction_source,
            diffusion_matrix_explicit=stress,
            mass_matrix=eye(4),
            eigenvalues=eigenvalues,
            update_variables=update_variables,
            reconstruction_variables=reconstruction,
            boundary_conditions=wall_kernel,
            boundary_gradients=gradient_kernel,
            aux_boundary_conditions=aux_kernel)
        self._boundary_tags = walls._boundary_tags
        self.expose_aux_atoms()


model = ShallowWater()

# %% [markdown]
# ## Numerical system model

# %%
nsm = NumericalSystemModel.from_system_model(
    model, riemann=PositiveNonconservativeHLL)

# %% [markdown]
# ## Reservoir initial condition + solve
#
# The free surface `eta = b + h` is projected cell-wise (so `h = max(0, eta - b)`
# stays exactly at rest on wet/dry shorelines); the reservoir/sea depths and
# velocities come from the mesh point-data.

# %%
def reservoir(Q, m):
    grid = Q.function_space().mesh()
    data = meshio.read(mesh)
    dim = grid.geometric_dimension
    coords = np.round(grid.coordinates.dat.data_ro[:, :dim], 12)
    points = np.round(data.points[:, :dim], 12)
    lookup = {tuple(c): i for i, c in enumerate(points)}
    order = np.array([lookup[tuple(c)] for c in coords], dtype=np.int64)
    cg = fd.FunctionSpace(grid, "CG", 1)
    dg = fd.FunctionSpace(grid, "DG", 0)

    def cell_mean(values):
        field = fd.Function(cg)
        field.dat.data[:] = values
        return np.asarray(fd.Function(dg).project(field).dat.data)

    bed = data.point_data["B"][order]
    depth = data.point_data["H"][order]
    b0 = cell_mean(bed)
    eta0 = cell_mean(bed + depth)
    h0 = np.maximum(eta0 - b0, 0.0)
    wet = h0 > 0.0
    Q.dat.data[:, 0] = b0
    Q.dat.data[:, 1] = h0
    momentum_x = data.point_data["H"] * data.point_data["U"]
    momentum_y = data.point_data["H"] * data.point_data["V"]
    Q.dat.data[:, 2] = np.where(wet, cell_mean(momentum_x[order]), 0.0)
    Q.dat.data[:, 3] = np.where(wet, cell_mean(momentum_y[order]), 0.0)


solver = FiredrakeHyperbolicSolver(
    settings=Settings(name="malpasset-firedrake", output=Zstruct(
        directory="outputs/malpasset_firedrake", snapshots=40,
        filename="dg", clean_directory=True)),
    time_end=2000.0, CFL=0.5, dg_degree=0, limiter="none",
    riemann_solver_cls=PositiveNonconservativeHLL,
    initial_condition_overwrite=reservoir)
solver.setup_simulation(mesh, nsm)
solver.run_simulation()
