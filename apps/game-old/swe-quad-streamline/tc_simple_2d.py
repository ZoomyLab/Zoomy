import jax.numpy as np
import matplotlib.pyplot as plt

import swe._2d.model as model
import swe._2d.bc as bc
import swe._2d.initial_conditions as ic
import swe._2d.reconstruction as reconstruction
import swe._2d.numerics as numerics
import swe._2d.solver as solver

# dimension of the problem
n_dim = 2

# discretization
# we assume 2 ghost points
n_elements = 100 + 2
xL = -10.
xR = 10.
dx = (xR-xL)/n_elements
x = np.linspace(xL-dx/2, xR+dx/2, n_elements)
y = np.array(x)
xv, xy = np.meshgrid(x, y)

X = np.array([xv, xy])



# time stepping
dtmin = 0.0001
dtmax = 0.5
dt = dx / 10.
CFL = 0.9
t_end = 1.

# output
dt_snapshot = 1.0

# model
n_dof = 3

# parameters
wet_tol = 0.000001
g = 9.81
nu = 0


# bottom topography
bottom_gradient = np.gradient(ic.bottom_constant(X), dx)[1:-1]


def apply_boundary_conditions(Q):
    return bc.extrapolation(Q)


def apply_initial_conditions(x):
    # return ic.rarefaction_shock(x)
    # return ic.rarefaction_rarefaction(x)
    return ic.shock_shock(x)



def compute_reconstruction(Q):
    return reconstruction.constant(Q)


def compute_flux(Q):
    return model.flux(Q, g)


def compute_source(Q):
    # return model.source_chezy_friction(Q, nu) + model.source_bottom_topograhy(Q, bottom_gradient, g)
    return model.source_chezy_friction(Q, nu) 


def compute_numerical_flux(Ql, Qr, Fl, Fr, max_speed):
    return numerics.lax_friedrichs(Ql, Qr, Fl, Fr, max_speed)

def compute_max_abs_eigenvalue(Q):
    return model.max_abs_eigenvalue(Q, g)


if __name__ == '__main__':
     
    Q, T = solver.simulate()
    fig, ax = plt.subplots(2)
    h0 = Q[0][0]
    u0 = Q[0][1] / h0
    v0 = Q[0][2] / h0
    h = Q[-1][0]
    u = Q[-1][1] / h
    v = Q[-1][2] / h
    # ax[0].imshow(h0)
    # ax[1].imshow(h)
    ax[0].plot(X[0][0], h0[0])
    ax[0].plot(X[0][0], h[0])
    ax[1].plot(X[0][0], u0[0])
    ax[1].plot(X[0][0], u[0])
    plt.show()
