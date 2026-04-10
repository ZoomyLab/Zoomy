import jax.numpy as np
from time import time as get_time

import apps.game.swe.swe._2d.model as model
import apps.game.swe.swe._2d.bc as bc
import apps.game.swe.swe._2d.initial_conditions as ic
import apps.game.swe.swe._2d.reconstruction as reconstruction
import apps.game.swe.swe._2d.numerics as numerics
import apps.game.swe.swe._2d.solver as solver
from apps.game.stream.parameters import Nx

# dimension of the problem
n_dim = 4

# discretization
# we assume 2 ghost points
n_elements = Nx + 2
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
dt = dx / 100.
CFL = 0.45
t_end = 1.

# output
dt_snapshot = 1.0

# model
n_dof = 4

# parameters
wet_tol = 0.000001
g = 9.81
nu = 0


# bottom topography
bottom_gradient = np.gradient(ic.bottom_constant(X), dx)[1:-1]


def apply_boundary_conditions(Q):
    #return bc.periodic(Q)
    return bc.inflow(Q)


def apply_initial_conditions(x):
    # return ic.rarefaction_shock(x)
    # return ic.rarefaction_rarefaction(x)
    #return ic.shock_shock(x)
    #return ic.dam_break(x)
    return ic.flat(x)



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

if __name__ == "__main__":


    Q, step  = solver.setup()
    for i in range(1000):
        start = get_time()
        Q = step(Q)
        print(f'time elapsed: {get_time()-start}')




