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
class SystemModelSpec(SystemModel):
    """Author a SystemModel by writing each operator as a method.

    Declare `variables`, `aux_variables`, `parameters` (name -> default), then
    define any of the operators below as a method returning its symbolic tensor.
    State and parameter symbols are exposed as attributes (`self.h`, `self.g`,
    ...); boundary conditions come from a `boundary_conditions()` method.
    """

    variables = ()
    aux_variables = ()
    parameters = {}
    _operators = ("flux", "hydrostatic_pressure", "nonconservative_matrix",
                  "source", "source_explicit", "diffusion_matrix_explicit",
                  "eigenvalues", "update_variables", "update_aux_variables",
                  "reconstruction_variables")

    def __init__(self, **parameter_overrides):
        self.t = sp.Symbol("t", real=True)
        self.x, self.y = sp.symbols("x y", real=True)
        self._distance = sp.Symbol("distance", real=True)
        self._position = Zstruct(X0=sp.Symbol("X0"), X1=sp.Symbol("X1"),
                                 X2=sp.Symbol("X2"))
        self._position._symbolic_name = "X"
        self.n0, self.n1 = sp.symbols("n0 n1", real=True)
        self._normal = Zstruct(n0=self.n0, n1=self.n1)
        self._normal._symbolic_name = "n"

        state = [sp.Symbol(k, real=True) for k in self.variables]
        aux = [sp.Symbol(k, real=True) for k in self.aux_variables]
        for k, s in zip(tuple(self.variables) + tuple(self.aux_variables),
                        state + aux):
            setattr(self, k, s)
        q = Zstruct(**dict(zip(self.variables, state))); q._symbolic_name = "Q"
        qaux = Zstruct(**dict(zip(self.aux_variables, aux)))
        qaux._symbolic_name = "Qaux"

        values = {**self.parameters, **parameter_overrides}
        params = Zstruct(**{k: sp.Symbol(k, positive=True) for k in values})
        params._symbolic_name = "p"
        for k in values:
            setattr(self, k, getattr(params, k))

        neq = len(state)
        fields = dict(
            time=self.t, space=[self.x, self.y], state=state, aux_state=aux,
            parameters=params, parameter_values=Zstruct(**values),
            normal=self._normal, mass_matrix=eye(neq), source=zeros(neq, 1),
            hydrostatic_pressure=zeros(neq, 2),
            nonconservative_matrix=sp.MutableDenseNDimArray.zeros(neq, neq, 2))
        cls = type(self)
        for name in self._operators:
            method = getattr(cls, name, None)
            if callable(method):
                fields[name] = method(self)

        if callable(getattr(cls, "boundary_conditions", None)):
            walls = bc.BoundaryConditions(self.boundary_conditions())
            args = (self.t, self._position, self._distance, q, qaux, params,
                    self._normal)
            fields["boundary_conditions"] = walls.get_boundary_condition_function(
                *args, function_name="boundary_conditions")
            fields["boundary_gradients"] = walls.get_boundary_gradient_function(
                *args, function_name="boundary_gradients")
            aux_walls = bc.BoundaryConditions(
                [aux_bc.Extrapolation(tag=w.tag)
                 for w in walls.boundary_conditions_list])
            fields["aux_boundary_conditions"] = aux_walls.get_boundary_condition_function(
                *args, function_name="aux_boundary_conditions")
            self._boundary_tags = walls._boundary_tags

        super().__init__(**fields)
        self.expose_aux_atoms()


# %%
class ShallowWater(SystemModelSpec):
    variables = ("b", "h", "hu", "hv")
    parameters = dict(g=9.81, n=0.033, nu=1.0, eps=1e-2, u_max=30.0)

    @property
    def hinv(self):
        return sqrt(2) * self.h / sqrt(self.h ** 4 + Max(self.h, self.eps) ** 4)

    @property
    def u(self):
        return self.hu * self.hinv

    @property
    def w(self):
        return self.hv * self.hinv

    @property
    def speed(self):
        return sqrt(self.u ** 2 + self.w ** 2)

    def flux(self):
        f = zeros(4, 2)
        f[1, 0], f[1, 1] = self.hu, self.hv
        f[2, 0], f[2, 1] = self.hu * self.u, self.hu * self.w
        f[3, 0], f[3, 1] = self.hv * self.u, self.hv * self.w
        return f

    def hydrostatic_pressure(self):
        p = zeros(4, 2)
        p[2, 0] = self.g * self.h ** 2 / 2
        p[3, 1] = self.g * self.h ** 2 / 2
        return p

    def nonconservative_matrix(self):
        bed = sp.MutableDenseNDimArray.zeros(4, 4, 2)
        bed[2, 0, 0] = self.g * self.h
        bed[3, 0, 1] = self.g * self.h
        return bed

    def source_explicit(self):
        rate = -self.g * self.n ** 2 * self.speed / Max(self.h, self.eps) ** Rational(1, 3)
        return Matrix([0, 0, rate * self.u, rate * self.w])

    def diffusion_matrix_explicit(self):
        # full deviatoric stress div(nu h (grad u + grad u^T)) — normal stresses
        # tau_xx = 2 nu du/dx, tau_yy = 2 nu dv/dy plus the shear/transpose terms.
        a = sp.MutableDenseNDimArray.zeros(4, 4, 2, 2)
        velocity = {2: self.u, 3: self.w}

        def stress(i, m, d, e, factor):
            a[i, m, d, e] += factor * self.nu
            a[i, 1, d, e] += -factor * self.nu * velocity[m]

        stress(2, 2, 0, 0, 2); stress(2, 2, 1, 1, 1); stress(2, 3, 1, 0, 1)
        stress(3, 3, 0, 0, 1); stress(3, 2, 0, 1, 1); stress(3, 3, 1, 1, 2)
        return a

    def eigenvalues(self):
        normal_velocity = self.u * self.n0 + self.w * self.n1
        wave = sqrt(self.g * Max(self.h, self.eps))
        dry = sp.Function("conditional")
        return Matrix([
            dry(self.h > self.eps, e, sp.S.Zero)
            for e in (sp.S.Zero, normal_velocity,
                      normal_velocity - wave, normal_velocity + wave)])

    def update_variables(self):
        cap = Max(self.h - self.eps, sp.S.Zero) * self.u_max
        clamp = lambda q: Max(-cap, Min(q, cap))
        return Matrix([self.b, self.h, clamp(self.hu), clamp(self.hv)])

    def reconstruction_variables(self):
        return Matrix([self.b, self.b + self.h, self.hu, self.hv])

    def boundary_conditions(self):
        return [bc.Wall(tag="wall", momentum_field_indices=[[2, 3]],
                        permeability=0.0, wall_slip=1.0)]


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
