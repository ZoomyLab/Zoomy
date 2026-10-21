import numpy as np
import sympy as sp
from sympy import Matrix

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.misc.misc import ZArray
from zoomy_core.model.derivative_workflow import DerivativeSpec, StructuredDerivativeModel


class ClassicalGNTopo1D(StructuredDerivativeModel):
    """GN-like primitive model with explicit bottom topography variable b in Q=[b,h,u]."""

    dimension = 1
    variables = ["b", "h", "u"]
    user_aux_variables = ["hinv"]
    parameters = {
        "g": (9.81, "positive"),
        "eps": (1e-8, "positive"),
    }
    auto_requested_derivatives = False
    numerics_scaled_q_indices = []

    def requested_derivatives(self):
        return [
            DerivativeSpec(field="u", axes=("t", "x", "x")),
            DerivativeSpec(field="u", axes=("x",)),
            DerivativeSpec(field="u", axes=("x", "x")),
            DerivativeSpec(field="u", axes=("x", "x", "x")),
        ]

    def flux(self):
        h = self.Q.h
        u = self.Q.u
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[0, 0] = 0
        F[1, 0] = h * u
        F[2, 0] = 0.5 * u * u
        return ZArray(F)

    def hydrostatic_pressure(self):
        h = self.Q.h
        g = self.params.g
        P = Matrix.zeros(self.n_variables, self.dimension)
        P[2, 0] = g * h
        return ZArray(P)

    def nonconservative_matrix(self):
        g = self.params.g
        A = ZArray.zeros(self.n_variables, self.n_variables, self.dimension)
        A[2, 0, 0] = g
        return A

    def source(self):
        h = self.Q.h
        u = self.Q.u
        u_txx = self.D.dtxx(self.Q.u)
        u_x = self.D.dx(self.Q.u)
        u_xx = self.D.dxx(self.Q.u)
        u_xxx = self.D.diff(self.Q.u, ("x", "x", "x"))
        S = ZArray.zeros(self.n_variables)
        S[2] = (h * h * (1.0 / 3.0)) * (u_txx + u * u_xxx - u_x * u_xx)
        return S


def make_topography(x: float, mode: str = "sine") -> float:
    if mode == "sine":
        return 0.08 * np.sin(2.0 * np.pi * x / 10.0)
    if mode == "bump":
        # smooth Gaussian-like bump centered at x=5
        return 0.12 * np.exp(-((x - 5.0) ** 2) / (2.0 * 0.9**2))
    return 0.0


def make_model(topo_mode: str = "sine"):
    bcs = BC.BoundaryConditions([
        BC.Extrapolation(tag="left"),
        BC.Extrapolation(tag="right"),
    ])

    def ic_q(x):
        X = float(x[0])
        b = make_topography(X, topo_mode)
        eta = 1.0 + 0.03 * np.exp(-((X - 2.5) ** 2) / (2.0 * 0.5**2))
        h = max(eta - b, 1e-8)
        q = np.zeros(3, dtype=float)
        q[0] = b
        q[1] = h
        q[2] = 0.02 * np.cos(2.0 * np.pi * X / 10.0)
        return q

    def ic_aux(x):
        X = float(x[0])
        b = make_topography(X, topo_mode)
        eta = 1.0 + 0.03 * np.exp(-((X - 2.5) ** 2) / (2.0 * 0.5**2))
        h = max(eta - b, 1e-8)
        return np.array([1.0 / h], dtype=float)

    return ClassicalGNTopo1D(
        boundary_conditions=bcs,
        initial_conditions=IC.UserFunction(ic_q),
        aux_initial_conditions=IC.UserFunction(ic_aux),
    )
