# import numpy as np
import jax.numpy as np
import numpy.random as random


def flux(Q, g):
    h = Q[0]
    u = Q[1] / h
    v = Q[2] / h
    flux_x =  np.array([h*u, h*u**2 + g * h**2 / 2, h * u * v ])
    flux_y =  np.array([h*u, h*u*v , h * v**2 + g * h**2 / 2 ])
    return [flux_x, flux_y]


def source_bottom_topograhy(Q, bottom_gradient, g):
    h = Q[0]
    zeros = np.zeros_like(h)
    return np.array([zeros, - g* h * bottom_gradient[0], - g * h * bottom_gradient[1]])


def source_chezy_friction(Q, nu):
    h = Q[0]
    u = Q[1] / h
    v = Q[2] / h
    zeros = np.zeros_like(h)
    u_norm = np.sqrt(u**2 + v**2)
    return np.array([zeros, - nu * h * u_norm * u, - nu * h * u_norm * v ])

def max_abs_eigenvalue(Q, g):
    h = Q[0]
    u = Q[1] / h
    v = Q[2] / h
    max_x =  (np.abs(u) + np.sqrt(g * h)).max()
    max_y =  (np.abs(v) + np.sqrt(g * h)).max()
    return max(max_x, max_y)

