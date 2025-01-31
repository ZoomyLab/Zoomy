import numpy as nnp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
from copy import deepcopy

# from parameters import *
# from model import *
# from bc import periodic
# from streamline import *
# import visualization as visu

import tc_Roe_1d as tc
# import tc_LF_1d as tc

def wet_dry_fix(Q):
    h = Q[0]
    hu = Q[1]
    h = np.where(h > tc.wet_tol, h, 0)
    hu = np.where(h > tc.wet_tol, hu, 0)
    Q = Q.at[0].set(h)
    Q = Q.at[1].set(hu)
    return Q

def step_fvm_conservative(Q):

    # aliases
    dt = tc.dt
    dx = tc.dx

    # cut off unphysical cells with h < 0
    Q = wet_dry_fix(Q)

    Qi, Qip, Qim = tc.compute_reconstruction(Q)
    Fi  = tc.compute_flux(Qi)
    Fip = tc.compute_flux(Qip)
    Fim = tc.compute_flux(Qim)

    # LF
    max_speed = dx/dt
    # Rusanov
    max_speed = tc.compute_max_abs_eigenvalue(Q)
    dt = (np.max(np.array([np.min(np.array([tc.CFL *  dx / max_speed, tc.dtmax])), tc.dtmin])))
    # if max_speed * dt / dx > tc.CFL * 1.001:
    #     print(f"CFL condition violated with value {max_speed * dt / dx}")
    #     assert False


    F_left_interface = tc.compute_numerical_flux(Qim, Qi, Fim, Fi, max_speed, dt, dx, sign=+1)
    F_right_interface = tc.compute_numerical_flux(Qi, Qip, Fi, Fip, max_speed, dt, dx, sign=-1)

    filter_left = np.where(np.logical_and(Qim <= 0, F_left_interface > 0), 0, 1)
    filter_left2 = np.where(np.logical_and(Qi <= 0, F_left_interface < 0), 0, 1)
    filter_right = np.where(np.logical_and(Qip <= 0, F_right_interface > 0), 0, 1)
    filter_right2 = np.where(np.logical_and(Qi <= 0, F_right_interface < 0), 0, 1)

    F_left_interface *= filter_left * filter_left2
    F_right_interface *= filter_right * filter_right2


    flux_contribution = dt / dx * (F_left_interface + F_right_interface)


    Qnew = Q[:, 1:-1] - flux_contribution

    source_contribution = dt * tc.compute_source(Qnew)

    Qnew += source_contribution


    Q = Q.at[:, 1:tc.n_elements-1].set(Qnew)

    Q = wet_dry_fix(Q)

    Q = tc.apply_boundary_conditions(Q)

    return Q


def simulate():
    t = 0.0
    t_save = tc.dt_snapshot
    iteration = 0

    X = tc.X

    Q = np.zeros((tc.n_elements, tc.n_dof))
    Q = tc.apply_initial_conditions(X)
    Q = tc.apply_boundary_conditions(Q)

    step = jax.jit(step_fvm_conservative)
    # step = step_fvm_conservative

    Q_list = [Q.copy()]
    T_list = [t]

    while t < tc.t_end:
        iteration += 1

        Q = step(Q)

        Q_list.append(Q.copy())
        T_list.append(t)

        print(f"Time: {t}")

        t = t + tc.dt

    return Q_list, T_list

