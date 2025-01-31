import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
from copy import deepcopy


import tc_simple_2d as tc

def wet_dry_fix(Q):
    h = Q[0]
    hu = Q[1]
    hv = Q[2]
    h = np.where(h > tc.wet_tol, h, 0)
    hu = np.where(h > tc.wet_tol, hu, 0)
    hv = np.where(h > tc.wet_tol, hv, 0)
    Q = np.array([h, hu, hv])
    return Q

def step_fvm_conservative(Q):

    # aliases
    dt = tc.dt
    dx = tc.dx

    # cut off unphysical cells with h < 0
    Q = wet_dry_fix(Q)

    Qi, Qn, Qs, Qe, Qw = tc.compute_reconstruction(Q)
    Fi, Gi  = tc.compute_flux(Qi)
    Fn, Gn = tc.compute_flux(Qn)
    Fs, Gs = tc.compute_flux(Qs)
    Fw, Gw = tc.compute_flux(Qw)
    Fe, Ge = tc.compute_flux(Qe)

    # LF
    # max_speed = dx/dt
    # Rusanov
    max_speed = tc.compute_max_abs_eigenvalue(Q)
    dt = (np.max(np.array([np.min(np.array([tc.CFL *  dx / max_speed, tc.dtmax])), tc.dtmin])))
    if max_speed * dt / dx > tc.CFL * 1.001:
        print(f"CFL condition violated with value {max_speed * dt / dx}")
        assert False


    F_west_interface = tc.compute_numerical_flux(Qw, Qi, Fw, Fi, max_speed)
    F_east_interface = tc.compute_numerical_flux(Qi, Qe, Fi, Fe, max_speed)
    F_north_interface = tc.compute_numerical_flux(Qi, Qn, Gi, Gn, max_speed)
    F_south_interface = tc.compute_numerical_flux(Qs, Qi, Gs, Gi, max_speed)

    flux_contribution = dt / dx * (F_west_interface - F_east_interface + F_south_interface - F_north_interface)
    source_contribution = dt * tc.compute_source(Qi)

    Qnew = Q[:, 1:-1, 1:-1] + flux_contribution + source_contribution

    Q = Q.at[:, 1:-1, 1:-1].set(Qnew)

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

    # step = jax.jit(step_fvm_conservative)
    step = step_fvm_conservative

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

def setup_simulation():
    X = tc.X

    Q = np.zeros((tc.n_elements, tc.n_dof))
    Q = tc.apply_initial_conditions(X)
    Q = tc.apply_boundary_conditions(Q)

    # step = jax.jit(step_fvm_conservative)
    step = step_fvm_conservative

    return X, Q, step

