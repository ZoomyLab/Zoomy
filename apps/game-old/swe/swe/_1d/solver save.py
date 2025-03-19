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

# import settings["s"settings["]"]

# from dataclasses import dataclass

# @dataclass(frozen=True)
# class Data:
#     dt: float = 0.01

# data = {'dt': 0.1}

def wet_dry_fix(Q, settings):
    h = Q[0]
    hu = Q[1]
    hu = np.where(h > settings["wet_tol"], hu, 0)
    Q = np.array([h, hu])
    return Q

def step_fvm_conservative(Q, settings):

    # aliases
    dt = settings["dt"]
    dx = settings["dx"]

    # cut off unphysical cells with h < 0
    Q = wet_dry_fix(Q)

    Qi, Qip, Qim = settings["compute_reconstruction"](Q)
    Fi  = settings["compute_flux"](Qi)
    Fip = settings["compute_flux"](Qip)
    Fim = settings["compute_flux"](Qim)

    # LF
    max_speed = dx/dt
    # Rusanov
    max_speed = settings["compute_max_abs_eigenvalue"](Q)
    dt = (np.max(np.array([np.min(np.array([settings["CFL"] *  dx / max_speed, settings["dtmax"]])), settings["dtmin"]])))
    # if max_speed * dt / dx > settings["CFL"] * 1.001:
    #     print(f"CFL condition violated with value {max_speed * dt / dx}")
    #     assert False


    F_left_interface = settings["compute_numerical_flux"](Qim, Qi, Fim, Fi, max_speed)
    F_right_interface = settings["compute_numerical_flux"](Qi, Qip, Fi, Fip, max_speed)
    flux_contribution = dt / dx * (F_left_interface - F_right_interface)
    source_contribution = dt * settings["compute_source"](Qi)

    Qnew = Q[:, 1:-1] + flux_contribution + source_contribution

    Q = Q.at[:, 1:settings["n_elements"]-1].set(Qnew)

    Q = wet_dry_fix(Q)

    Q = settings["apply_boundary_conditions"](Q)

    return Q


def simulate(settings):
    t = 0.0
    t_save = settings["dt_snapshot"]
    iteration = 0

    X = settings["X"]

    Q = np.zeros((settings["n_elements"], settings["n_dof"]))
    Q = settings["apply_initial_conditions"](X)
    Q = settings["apply_boundary_conditions"](Q)

    step = jax.jit(step_fvm_conservative)
    # step = step_fvm_conservative

    Q_list = [Q.copy()]
    T_list = [t]

    # data = Data()

    while t < settings["t_end"]:
        iteration += 1

        Q = step(Q, settings)

        Q_list.append(Q.copy())
        T_list.append(t)

        print(f"Time: {t}")

        t = t + settings["dt"]

    print(params.dt)
    return Q_list, T_list

