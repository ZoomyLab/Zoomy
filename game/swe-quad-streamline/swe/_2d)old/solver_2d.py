import numpy as nnp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from time import time
from copy import deepcopy

from parameters import *
from model import *
from bc import *
from streamline import *
import visualization as visu

def step_fvm_conservative(Q, iteration):
    h = Q[0]
    hu = Q[1]
    hv = Q[2]
    b = Q[3]
    u = np.where(h > wet_tol, hu / h, 0)
    v = np.where(h > wet_tol, hv / h, 0)
    U = np.array([h, u, v, b])

    # Compute fluxes
    F = flux_f(U)
    G = flux_g(U)

    Qm = Q[:, 1:-1, 1:-1]
    Qt = Q[:, 2:, 1:-1]
    Qb = Q[:, :-2, 1:-1]
    Qr = Q[:, 1:-1, 2:]
    Ql = Q[:, 1:-1, :-2]

    Fm = F[:, 1:-1, 1:-1]
    Ft = F[:, 2:, 1:-1]
    Fb = F[:, :-2, 1:-1]
    Fr = F[:, 1:-1, 2:]
    Fl = F[:, 1:-1, :-2]

    Gm = G[:, 1:-1, 1:-1]
    Gt = G[:, 2:, 1:-1]
    Gb = G[:, :-2, 1:-1]
    Gr = G[:, 1:-1, 2:]
    Gl = G[:, 1:-1, :-2]

    max_speed = (np.max(np.array([np.abs(u).max(), np.abs(v).max()]))).astype(float)
    cfl = 0.45
    # if max_speed == 0:
    #     dt = dtmax
    # else:
    #     dt = (np.max(np.array([np.min(np.array([cfl *  dx / max_speed, dtmax])), dtmin]))).asarray(float)
    # if iteration < 3:
    #     dt = dtmin
    dt = (np.max(np.array([np.min(np.array([cfl *  dx / max_speed, dtmax])), dtmin])))

    flux_x = dt / dx * (rusanov(Ql, Qm, Fl, Fm, max_speed) - rusanov(Qm, Qr, Fm, Fr, max_speed) ) 
    flux_y = dt / dy * (rusanov(Qb, Qm, Gb, Gm, max_speed) - rusanov(Qm, Qt, Gm, Gt, max_speed) )
    nc_flux_x = -dt / dx * (nc_flux(Ql, Qm) + nc_flux(Qm, Qr))
    nc_flux_y = -dt / dy * (nc_flux(Qb, Qm) + nc_flux(Qm, Qt))

    ## multiply with normal
    # flux_x[2] = 0
    # flux_y[1] = 0
    nc_flux_x = nc_flux_x.at[2, :, :].set(0)
    nc_flux_y = nc_flux_y.at[1, :, :].set(0)

    source = dt * S(Qm, xv, yv)


    Qnew = Q[:, 1:-1, 1:-1] + (flux_x + flux_y + nc_flux_x + nc_flux_y)  + source
    Q = Q.at[:, 1:M-1, 1:N-1].set(Qnew)

    h = Q[0]
    h = np.where(h > wet_tol, h, 0)
    hu = Q[1]
    hv = Q[2]
    b = Q[3]
    u = np.where(h > wet_tol, hu / h, 0)
    v = np.where(h > wet_tol, hv / h, 0)
    hu = h*u
    hv = h*v

    h, hu, hv = apply_bc(h, hu, hv)

    Q = np.array([h, hu, hv, b])
    return Q, u, v, dt

def interpolate(Q_list, T_list, streamlines, time):
    if time > T_list[-1]:
        return Q_list[-1], streamlines, time
    for i, t in enumerate(T_list):
        if t > time:
            break
    if i == 0:
        return Q_list[0], streamlines, T_list[0]    
    t0 = T_list[i-1] 
    t1 = T_list[i]
    Q0 = Q_list[i-1]
    Q1 = Q_list[i]
    fraction = (time - t0) / (t1 - t0)
    assert fraction > 0.
    assert fraction < 1.
    streamlines_int = streamlines
    for streamline in streamlines_int:
        streamline[0] = streamline[0][:i] + [streamline[0][i-1] + (streamline[0][i] - streamline[0][i-1]) * fraction]
        streamline[1] = streamline[1][:i] + [streamline[1][i-1] + (streamline[1][i] - streamline[1][i-1]) * fraction]
    Q_int = Q0 + (Q1 - Q0) * fraction
    time_int = t0 + (t1 - t0) * fraction
    return Q_int, streamlines_int, time_int

def simulate(fig, ax, streamlines, hdisplay=None, images=None):
    Q = IC(xv, yv)

    # image = visu.update(streamlines, fig, ax, hdisplay, Q, 0, images=images)

    # Time integration loop
    t = 0.0
    t_save = dt_snapshot
    iteration = 0
    compute_finished = False
    animation_finished = False

    Q_list = [Q.copy()]
    T_list = [t]

    plot_start = 0.
    plot_last = 0.
    global_time_start = time()
    global_time_last = time()
    time_elapsed_total = 0
    time_elapsed_last = 0
    time_next_plot = 0
    time_plot = 0
    b_plot = False
    b_first = True
    time_finish = np.inf

    step_jit = jax.jit(step)

    while not (compute_finished and animation_finished):
        iteration += 1

        Q, u, v, dt = step_jit(Q, iteration)

        Q_list.append(Q.copy())
        t = t + dt
        T_list.append(t)
        streamlines, compute_finished = integrate_streamlines(streamlines.copy(), u, v, dt, X, Y)
        if compute_finished and np.isinf(time_finish):
            time_finish = t

        # assert len(T_list) == len(streamlines[0][0])

        # ax['a'].set_title(f'Who is first? \n Game starts in {5 - time() + global_time_start} seconds')

        if time() - global_time_start > 0 and b_first:
            b_plot = True
            plot_start = time() 
            plot_start = time() 
            plot_last = time()
            b_first = False
        if time() - global_time_last > dt_snapshot :
            global_time_last = time()
            print(t, dt, time() - global_time_start)
        #     print(time_elapsed_total, time_finish)
        if b_plot:
            time_elapsed_total = time() - plot_start
            time_elapsed_last = time() - plot_last
            if time_elapsed_last > dt_frame:
                plot_last = time()
                # Q_int, streamlines_int, time_int = interpolate(Q_list.copy(), T_list.copy(), deepcopy(streamlines), time_elapsed_total)
                Q_int, streamlines_int, time_int = interpolate(Q_list.copy(), T_list.copy(), deepcopy(streamlines), time_elapsed_total)
                image = visu.update(streamlines_int, fig, ax, hdisplay, Q_int, time_int, images=images)
        if compute_finished and time_elapsed_total >= time_finish:
            animation_finished = True
    return image
