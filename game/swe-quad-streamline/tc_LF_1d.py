import jax.numpy as np
import jax
import matplotlib.pyplot as plt

from functools import partial

import swe._1d.model as model
import swe._1d.bc as bc
import swe._1d.initial_conditions as ic
import swe._1d.reconstruction as reconstruction
import swe._1d.numerics as numerics
import swe._1d.solver as solver

    

# dimension of the problem
n_dim = 1

# discretization
# we assume 2 ghost points
n_elements = 100 + 2
xL = -10.
xR = 10.
dx = (xR-xL)/n_elements
X = np.linspace(xL-dx/2, xR+dx/2, n_elements)

# time stepping
dtmin = 0.0001
dtmax = 0.5
dt = dx / 10.
CFL = 0.9
t_end = 300.0

# output
dt_snapshot = 1.0

# model
n_dof = 2

# parameters
wet_tol = 0.000001
g = 9.81
nu = 0.1


# bottom topography
# bottom_gradient = np.gradient(ic.bottom_constant(X), dx)[1:-1]
bottom = ic.bottom_basins(X)
bottom_gradient = np.gradient(bottom, dx)[1:-1]


def apply_boundary_conditions(Q):
    return bc.periodic(Q)


def apply_initial_conditions(x):
    # return ic.rarefaction_shock(x)
    # return ic.rarefaction_rarefaction(x)
    # return ic.shock_shock(x)
    output = ic.dam_break(x)
    h = output[0]
    output = output.at[0].set(h - bottom)
    return np.where(output[0] > 0, output, np.zeros_like(output))



def compute_reconstruction(Q):
    return reconstruction.constant(Q)


def compute_flux(Q):
    return model.flux(Q, g)


def compute_source(Q):
    return model.source_chezy_friction(Q, nu) + model.source_bottom_topograhy(Q, bottom_gradient, g)


def compute_numerical_flux(Ql, Qr, Fl, Fr, max_speed, dt, dx):
    return numerics.lax_friedrichs(Ql, Qr, Fl, Fr, max_speed)

def compute_max_abs_eigenvalue(Q):
    return model.max_abs_eigenvalue(Q, g)

settings = {"n_dim": n_dim, "n_elements":n_elements, "xL": xL, "xR": xR, "dx": dx, "X": X, "dtmin": dtmin, "dtmax": dtmax, "dt": dt, "CFL":CFL, "t_end": t_end, "dt_snapshot": dt_snapshot, "n_dof": n_dof, "wet_tol": wet_tol, "g": g, "nu": nu, "apply_initial_conditions": apply_initial_conditions, "apply_boundary_conditions": apply_boundary_conditions, "compute_reconstruction":compute_reconstruction, "compute_source": compute_source, "compute_numerical_flux":compute_numerical_flux, "compute_max_abs_eigenvalue": compute_max_abs_eigenvalue}

def apply_test(x):
    return np.where(x > 3, 1, 0)

class Model():
    def __init__(self):
        self.n_elements = 10
        self.ic = apply_test

@partial(jax.jit, static_argnums=(0,))
def test_func(s):
    N = s.n_elements
    x = np.linspace(1, N, N)
    return s.ic(x)

@jax.jit
def test_diff(x):
    return 2 * x


if __name__ == '__main__':

    m = Model()
    x = np.linspace(1,10,10) 
    Q = test_func(m)

    X = [1.,2.,3.]
    print(jax.jvp(test_diff, X, X))

    print(Q)
    # Q, T = solver.simulate()
    # fig, ax = plt.subplots(2)
    # h0 = Q[0][0]
    # u0 = np.where(h0 > 0, Q[0][1] / h0, 0)
    # h = Q[-1][0]
    # u = np.where(h > 0, Q[-1][1] / h, 0)
    # ax[0].plot(X, h0 + bottom)
    # ax[0].plot(X, h + bottom)
    # ax[0].plot(X, bottom)
    # ax[1].plot(X, u0)
    # ax[1].plot(X, u)
    # plt.show()
