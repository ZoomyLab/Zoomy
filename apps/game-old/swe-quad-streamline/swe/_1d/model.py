# import numpy as np
import jax.numpy as np
import numpy.random as random


def flux(Q, g):
    h = Q[0]
    u = np.where(h > 0, Q[1] / h, 0)
    out = np.zeros_like(Q)
    out = out.at[0].set(h * u)
    out = out.at[1].set(h * u**2)
    return out


def source_bottom_topograhy(Q, bottom_gradient, g):
    h = Q[0]
    zeros = np.zeros_like(h)
    return np.array([zeros, - g* h * bottom_gradient])


def source_chezy_friction(Q, nu):
    h = Q[0]
    u = np.where(h > 0, Q[1] / h, 0)
    zeros = np.zeros_like(h)
    out = np.zeros_like(Q)
    out = out.at[1].set(- nu * h * np.abs(u) * u)
    return out

def max_abs_eigenvalue(Q, g):
    h = Q[0]
    u = np.where(h > 0, Q[1] / h, 0)
    return (np.abs(u) + np.sqrt(g * h)).max()

