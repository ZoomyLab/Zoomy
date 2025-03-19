# import numpy as np
import jax.numpy as np

def inflow_outflow(h, hu, hv):
    alpha = 1.0

    # bottom
    h0 = 0.1
    h = h.at[0,:].set(h0)
    hu  = hu.at[0, :].set(hu[1,:])
    hv = hv.at[0,:].set(10.)

    # top
    h = h.at[-1,:].set(h0)
    hu = hu.at[-1,:].set(hu[-2, :])
    hv = hv.at[-1,:].set(hv[-2, :])

    # left
    h  = h.at[:,0].set(h0)
    hu = hu.at[:,0].set(-alpha * hu[:, 1])
    hv = hv.at[:,0].set(alpha * hv[:, 1])

    # right
    h =  h.at[:,-1].set(h0)
    hu = hu.at[:,-1].set(-alpha * hu[:, -2])
    hv = hv.at[:,-1].set(alpha * hv[:, -2])
    return h, hu, hv


def outflow(h, hu, hv):
    alpha = 0.5

    # bottom
    h0 = 1.
    h[0, :] = h0
    hu[0, :]  = alpha * hu[1,:]

    # top
    h[-1, :] = h0
    hu[-1, :] = alpha  * hu[-2, :]

    # left
    h[:, 0]  = h0
    hv[:, 0] = alpha * hv[:, 1]

    # right
    h[:, -1] = h0
    hv[:, -1] = alpha * hv[:, -2]

    # bottom
    hv[0, :] = np.where(hv[1,:] < 0, hv[1, :], 0)
    # top
    hv[-1, :] = np.where(hv[-2,:] > 0, hv[-2, :], 0)
    # left
    hu[:, 0] = np.where(hu[:,1] < 0, hu[:, 1], 0)
    # right
    hu[:, -1] = np.where(hu[:,-2] > 0, hu[:, -2], 0)


def wall(h, hu, hv): 
    noslip = 0.

    # bottom
    h[0, :] =  h[1, :]
    hu[0, :] = (1-noslip) * hu[1,:] 
    hv[0, :] = -hv[1,:]

    # top
    h[-1, :] =  h[-2, :]
    hu[-1, :] = (1-noslip) * hu[-2, :]
    hv[-1, :] = - hv[-2,:]

    # left
    h[:, 0]  =  h[:, 1]
    hu[:, 0] = - hu[:, 1]
    hv[:, 0] = (1-noslip) * hv[:, 1]

    # right
    h[:, -1] = h[:, -2]
    hu[:, -1] = - hu[:, -2]
    hv[:, -1] = (1-noslip) *  hv[:, -2]


def slide(h, hu, hv):
    noslip = 0.
    h0 = 0.1
    v0 = 0.

    # top
    h = h.at[-1, :].set(h0)
    hu = hu.at[-1, :].set(hu[-2, :])
    # h[-1, :] = h0
    # hu[-1, :] =  hu[-2, :]

    # bottom
    h = h.at[0, :].set(h[1, :])
    hu = hu.at[0, :].set(hu[1, :])
    # h[0, :] = h[1, :]
    # hu[0, :]  = hu[1,:]

    # left
    # h[:, 0]  =  h[:, 1]
    # hu[:, 0] = - hu[:, 1]
    # hv[:, 0] = (1-noslip) * hv[:, 1]
    h = h.at[:, 0].set(h[:, 1])
    hu = hu.at[:, 0].set(- hu[:, 1])
    hv = hv.at[:, 0].set((1-noslip) * hv[:, 1])

    # right
    # h[:, -1] = h[:, -2]
    # hu[:, -1] = - hu[:, -2]
    # hv[:, -1] = (1-noslip) *  hv[:, -2]
    h = h.at[:, -1].set(h[:, -2])
    hu = hu.at[:, -1].set(- hu[:, -2])
    hv = hv.at[:, -1].set((1-noslip) *  hv[:, -2])

    # top
    # hv[-1, :] = np.where(hv[-2,:] > 0, hv[-2, :], - v0 * h[-1, :])

    # hv[-1, :] = - v0 * h0 *np.ones_like(h)[-2, :]
    hv = hv.at[-1, :].set(- v0 * h0 *np.ones_like(h)[-2, :])

    # bottom
    # hv[0, :] = np.where(hv[1,:] < 0, hv[1, :], 0)
    hv = hv.at[0, :].set(np.where(hv[1,:] < 0, hv[1, :], 0))
    # hv[0, :] = - v0 * np.ones_like(h)[1, :]

    return h, hu, hv

def apply_bc(h, hu, hv):
    return inflow_outflow(h, hu, hv)
