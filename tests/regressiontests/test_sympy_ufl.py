import numpy as np
import pytest
from types import SimpleNamespace

from firedrake import *
from copy import deepcopy

from mpi4py import MPI
from firedrake.pyplot import FunctionPlotter, tripcolor
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ufl import max_value

from library.pysolver.solver import *
from library.model.model import *
import library.model.initial_conditions as IC
import library.model.boundary_conditions as BC
from library.pysolver.ode import RK1
import library.python.misc.io as io

# from library.pysolver.reconstruction import GradientMesh
import library.python.mesh.mesh as petscMesh
import library.postprocessing.postprocessing as postprocessing
import argparse

main_dir = os.getenv("ZOOMY_DIR")


@pytest.mark.critical
def get_model():
    level = 2
    settings = Settings(
        name="ShallowMoments",
        parameters={"g": 9.81},
        reconstruction=recon.constant,
        num_flux=flux.LLF(),
        nc_flux=nonconservative_flux.segmentpath(1),
        compute_dt=timestepping.adaptive(CFL=0.9),
        time_end=1.0,
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
        Q = np.zeros(4, dtype=float)
        Q[0] = np.where(x[0] < 0.0, 2.0, 1.0)
        return Q

    ic = IC.UserFunction(ic_func)

    model = ShallowMoments(
        dimension=1,
        fields=2 + level,
        aux_variables=1,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": []},
        basis=Basis(basis=Legendre_shifted(order=level)),
    )
    return model


def model_to_firedrake(model):
    n_scalar = 4
    n_aux_scalar = 1
    N = 100
    mesh = UnitIntervalMesh(N)

    # Vl = FunctionSpace(mesh, "DG", 1)
    V = FunctionSpace(mesh, "DG", 0)
    Vout = FunctionSpace(mesh, "DG", 0)
    W = VectorFunctionSpace(mesh, "DG", 0)
    # Wl = VectorFunctionSpace(mesh, "DG", 0)
    Wout = VectorFunctionSpace(mesh, "DG", 0)

    MFS = MixedFunctionSpace([V] * n_scalar)
    MFS_aux = MixedFunctionSpace([V] * n_aux_scalar)

    x = SpatialCoordinate(mesh)

    IC = Function(MFS, name="IC")
    for i in range(n_scalar):
        h0 = Function(MFS.sub(i)).interpolate(1.0)
        IC.sub(i).assign(h0)

    Q = Function(MFS, name="Q")
    Q_ = Function(MFS, name="Qold")
    Q.assign(IC)
    Q_.assign(IC)

    n = FacetNormal(mesh)

    # var = str(model.variables.get_list())
    variables = [str(v) for v in model.variables.get_list()]
    aux_variables = [str(v) for v in model.aux_variables.get_list()]
    parameters = [str(v) for v in model.parameters.get_list()]
    parameter_values = [str(v) for v in model.parameter_values]

    symflux = model.sympy_flux

    def translate_at_facet(repr, at_facet):
        for j in range(len(variables)):
            repr = repr.replace(variables[j], f"Q.sub({j})({at_facet})")
        for j in range(len(aux_variables)):
            repr = repr.replace(aux_variables[j], f"Qaux.sub({j})({at_facet})")
        for j in range(len(parameters)):
            repr = repr.replace(parameters[j], parameter_values[j])
        for j in range(model.dimension):
            repr = repr.replace(f"n{j}", f"n[{j}]({at_facet})")
        return repr

    def translate_overwrite_normals(repr, overwrite_normals):
        for j in range(len(variables)):
            repr = repr.replace(variables[j], f"Q.sub({j})")
        for j in range(len(aux_variables)):
            repr = repr.replace(aux_variables[j], f"Qaux.sub({j})")
        for j in range(len(parameters)):
            repr = repr.replace(parameters[j], parameter_values[j])
        for j in range(model.dimension):
            repr = repr.replace(f"n{j}", f"{overwrite_normals[j]}")
        return repr

    def translate_default(repr):
        for j in range(len(variables)):
            repr = repr.replace(variables[j], f"Q.sub({j})")
        for j in range(len(aux_variables)):
            repr = repr.replace(aux_variables[j], f"Qaux.sub({j})")
        for j in range(len(parameters)):
            repr = repr.replace(parameters[j], parameter_values[j])
        for j in range(model.dimension):
            repr = repr.replace(f"n{j}", f"n[{j}]")
        return repr

    def translate(sympy_expr, overwrite_normals=None, at_facet=None):
        repr = str(sympy_expr)

        if overwrite_normals is not None:
            repr = translate_overwrite_normals(repr, overwrite_normals)
        elif at_facet is not None:
            repr = translate_at_facet(repr, at_facet)
        else:
            repr = translate_default(repr)
        return repr

    flux_q = []
    dim = 0
    for i in range(n_scalar):
        flux_q.append(eval(translate(model.sympy_flux[dim][i])))
    ev_center = []
    for i in range(n_scalar):
        ev_center.append(
            eval(translate(model.sympy_eigenvalues[i], overwrite_normals=[1.0]))
        )
        ev_center.append(
            eval(translate(model.sympy_eigenvalues[i], overwrite_normals=[-1.0]))
        )
    ev_face = []
    for i in range(n_scalar):
        ev_face.append(eval(translate(model.sympy_eigenvalues[i], at_facet="'+'")))
        ev_face.append(eval(translate(model.sympy_eigenvalues[i], at_facet="'-'")))

    max_eigenvalue = ev_face[0]
    for i in range(1, len(ev_face)):
        max_eigenvalue = max_value(max_eigenvalue, ev_face[i])
    ev_max_n = max_eigenvalue
    for i in range(1, len(ev_center)):
        max_eigenvalue = max_value(max_eigenvalue, ev_center[i])
    # TODO rename
    # ev_c = Function(V).interpolate(max_eigenvalue)

    MFS_test = TestFunctions(MFS)
    MFS_trail = TestFunctions(MFS)

    # v_, w_ = TrialFunctions(VW)

    a = 0
    for i in range(n_scalar):
        a += inner(MFS_test[i], MFS_trail[i]) * dx

    T = 3.0
    CFL = 1.0 / (2 + 2 + 1)
    incircle = 1.0 / N

    g = 9.81
    I = Identity(1)
    # ev_n = lambda q, n: abs((dot(q.sub(1), n)/q.sub(0))) + sqrt(g * q.sub(0))

    f_q = lambda q: [
        q.sub(1),
        outer(q.sub(1), q.sub(1)) / q.sub(0) + 0.5 * g * q.sub(0) ** 2 * I,
    ]

    # q = [v, w]

    p = "+"
    m = "-"

    i = 0
    # scalar
    F_Q = 0
    # for i in range(n_scalar):
    #     F_Q += ((q[i](p) - q[i](m)) * (
    #             dot(0.5 * (f_q(Q)[i](p) + f_q(Q)[i](m)), n(m))
    #             - 0.5 * (0.5*(ev_n(Q, n)(p) + ev_n(Q, n)(m))) * (Q.sub(i)(p) - Q.sub(i)(m))
    #             )) * dS

    # # vector
    # i = 1
    # for i in range(n_scalar, n_scalar + n_vector):
    #     F_Q += dot((q[i](p) - q[i](m)),(
    #         dot((0.5 * (f_q(Q)[i](p) + f_q(Q)[i](m))), n(m))
    #         - 0.5 * (0.5 * (ev_n(Q, n)(p) + ev_n(Q, n)(m)) * (Q.sub(i)(p) - Q.sub(i)(m)))
    #         )) * dS

    #     for i in range(n_scalar):
    #     F_Q += ((q[i](p) - q[i](m)) * (
    #             dot(0.5 * (f_q(Q)[i](p) + f_q(Q)[i](m)), n(m))
    #             - 0.5 * (0.5*(ev_n(Q, n)(p) + ev_n(Q, n)(m))) * (Q.sub(i)(p) - Q.sub(i)(m))
    #             )) * dS

    # vector
    i = 1
    for i in range(n_scalar):
        F_Q += (
            (MFS_test[i](p) - MFS_test[i](m))
            * (
                (0.5 * (flux_q[i](p) + flux_q[i](m))) * n[0]("-")
                - 0.5 * (Q.sub(i)(p) - Q.sub(i)(m))
            )
            * dS
        )

    t = 0.0
    print("t=", t)
    step = 0
    output_freq = 1

    ev_max = 1.0
    dt = 0.00001
    L = dt * (F_Q)

    # params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
    params = {"ksp_type": "cg"}
    dQ = Function(MFS)
    prob = LinearVariationalProblem(a, L, dQ)
    solv = LinearVariationalSolver(prob, solver_default_parameters=params)

    T = 3 * dt

    while t < T - 0.5 * dt:
        solv.solve()
        Q.assign(Q + dQ)
        step += 1
        t += dt
        print(t)


if __name__ == "__main__":
    model = get_model()
    model_to_firedrake(model)
