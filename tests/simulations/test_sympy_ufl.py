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
main_dir = os.getenv("SMS")


@pytest.mark.critical
def get_model():
    level = 2
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
        Q = np.zeros(4, dtype=float)
        Q[0] = np.where(x[0]<0.0, 2., 1.)
        return Q
    ic = IC.UserFunction(ic_func)

    model = ShallowMoments(
        dimension=1,
        fields=2 + level,
        aux_fields=1,
        parameters=settings.parameters,
        boundary_conditions=bcs,
        initial_conditions=ic,
        settings={"friction": []},
        basis=Basis(basis=Legendre_shifted(order=level)),
    )
    return model

def model_to_firedrake(model):
    
    
    n_scalar = 1
    n_vector = 1
    N = 100
    mesh = UnitIntervalMesh(N)

    # Vl = FunctionSpace(mesh, "DG", 1)
    V = FunctionSpace(mesh, "DG", 0)
    Vout = FunctionSpace(mesh, "DG", 0)
    W = VectorFunctionSpace(mesh, "DG", 0)
    # Wl = VectorFunctionSpace(mesh, "DG", 0)
    Wout = VectorFunctionSpace(mesh, "DG", 0)

    VW = V * W
    WV = W * V


    x = SpatialCoordinate(mesh)


    IC = Function(VW, name="IC")
    h0 = Function(VW.sub(0)).interpolate(1.)
    hu0 = Function(VW.sub(1)).interpolate(as_vector([1.0]))
    IC.sub(0).assign(h0)
    IC.sub(1).assign(hu0)

    Q = Function(VW, name="Q")
    Q_= Function(VW, name="Qold")
    Q.assign(IC)
    Q_.assign(IC)

    h, hu = split(Q)
    h_, hu_ = split(Q_)

    v, w = TestFunctions(VW)
    v_, w_ = TrialFunctions(VW)

    a = inner(w, w_) * dx + inner(v, v_) * dx

    T = 3.0
    CFL = 1.0 / (2 + 2 + 1)
    incircle = 1.0 / N

    g = 9.81
    I = Identity(1)
    ev_n = lambda q, n: abs((dot(q.sub(1), n)/q.sub(0))) + sqrt(g * q.sub(0))

    n = FacetNormal(mesh)

    f_q = lambda q: [q.sub(1), outer(q.sub(1), q.sub(1)) / q.sub(0) + 0.5 * g * q.sub(0)**2 * I]

    q = [v, w]

    p = "+"
    m = "-"

    i = 0 
    # scalar
    F_Q = 0
    for i in range(n_scalar):
        F_Q += ((q[i](p) - q[i](m)) * (
                dot(0.5 * (f_q(Q)[i](p) + f_q(Q)[i](m)), n(m)) 
                - 0.5 * (0.5*(ev_n(Q, n)(p) + ev_n(Q, n)(m))) * (Q.sub(i)(p) - Q.sub(i)(m))
                )) * dS
        
    # vector
    i = 1
    for i in range(n_scalar, n_scalar + n_vector):
        F_Q += dot((q[i](p) - q[i](m)),(
            dot((0.5 * (f_q(Q)[i](p) + f_q(Q)[i](m))), n(m)) 
            - 0.5 * (0.5 * (ev_n(Q, n)(p) + ev_n(Q, n)(m)) * (Q.sub(i)(p) - Q.sub(i)(m)))
            )) * dS

    t = 0.0
    print("t=", t)
    step = 0
    output_freq = 1

    ev_max = 1.
    dt = 0.00001
    L = dt * (F_Q)

    # params = {'ksp_type': 'preonly', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
    params = {"ksp_type": "cg"}
    dQ = Function(VW)
    prob = LinearVariationalProblem(a, L, dQ)
    solv = LinearVariationalSolver(prob, solver_parameters=params)

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
