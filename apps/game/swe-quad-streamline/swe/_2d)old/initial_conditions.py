import jax.numpy as np

def rarefaction_shock(x):
    n_cells = x.shape[0]
    n_dofs = 2
    Q = np.zeros(n_cells, n_dofs, dtype=float)
    h = np.where(x[0] < 0, 2, 1)
    u = np.where(x[0] < 0, 0, 0)
    Q[:, 0] = h
    Q[:, 1] = h * u
    return Q

def shock_shock(x):
    n_cells = x.shape[0]
    n_dofs = 2
    Q = np.zeros(n_cells, n_dofs, dtype=float)
    h = np.where(x[0] < 0, 1, 1)
    u = np.where(x[0] < 0, 1, -1)
    Q[:, 0] = h
    Q[:, 1] = h * u
    return Q

def rarefaction_rarefaction(x):
    n_cells = x.shape[0]
    n_dofs = 2
    Q = np.zeros(n_cells, n_dofs, dtype=float)
    h = np.where(x[0] < 0, 1, 1)
    u = np.where(x[0] < 0, -1, 1)
    Q[:, 0] = h
    Q[:, 1] = h * u
    return Q

def slope(x, y):
    return 0.04*y

def bump(x, y):
    b = np.zeros_like(x)
    delta_b = 1.

    n_bumps = 400
    # Generate random locations for the bumps
    x_bumps = random.uniform(x.min(), x.max(), n_bumps)
    y_bumps = random.uniform(0.0 * y.max(), 0.9*y.max(), n_bumps)

    bump_height = random.uniform(-delta_b, delta_b, n_bumps)
    bump_width = random.uniform(1/30, 1/300, n_bumps)

    for i in range(n_bumps):
        bump =  bump_height[i]* np.exp(- bump_width[i] * ((x - x_bumps[i])**2 + (y - y_bumps[i])**2) )
        b += bump
    return np.asarray(b)

def radial_drop(x0, y0, xv, yv, r):
    return np.where((x0-xv[1:-1, 1:-1])**2 + (y0-yv[1:-1, 1:-1])**2 < r**2, 1., 0.)

def radial_drop_full(x0, y0, xv, yv, r):
    return np.where((x0-xv)**2 + (y0-yv)**2 < r**2, 1., 0.)

def topo(x, y):
    # return slope(x, y) + bump(x, y)
    # return slope(x, y)
    return radial_drop_full(50, 50, x, y, 5)



def IC_vortex(xv, yv):
    N = xv.shape[0]
    M = yv.shape[1]
    h = 0.1*np.ones((N, M))  # water height
    hu = 1*np.zeros((N, M))  # velocity in x direction
    hv = 10*np.ones((N, M))  # velocity in y direction
    z = np.zeros((N, M))  # topography 
    z = topo(xv, yv)
    print(h.shape)
    print(z.shape)
    Q = np.array([h, hu, hv, z])
    return Q

def IC_downhill(xv, yv):
    N = xv.shape[0]
    M = yv.shape[1]
    h = np.zeros((N, M))  # water height
    u = np.zeros((N, M))  # velocity in x direction
    v = np.zeros((N, M))  # velocity in y direction
    z = np.zeros((N, M))  # topography 

    u0 = 0.0
    v0 = 0.
    h0 = 0.

    # h[1:-1, 1:-1] += 0.1*radial_drop(50, 50, xv, yv, 10)
    z = topo(xv, yv)
    # h = h0 * h
    # h = np.where(yv > 0.99 * yv.max(), 0.1, 0.00)
    # h0 = 0.9 * (z-slope(xv, yv)).max()
    # h = h0 * h
    # h = h - topo(xv, yv)
    # h = np.where(h > 0, h, 0.0)
    # hu = h * u0
    # hv = -h * v0
    hu = np.zeros_like(h)
    hv = np.zeros_like(h)

    z = topo(xv, yv)

    h_bed = 0.0
    # h = np.where(h - z > 0, h - z, 0)
    # h = np.where(np.logical_and(np.logical_and(h - z > 0 ,xv > 0.4 * 2 * np.pi),  xv < 0.6 * 2 * np.pi), (h - z) + 1., h)
    # h = np.where(np.logical_and(np.logical_and(np.logical_and(xv > 0.4 * 2 * np.pi, xv < 0.6 * 2 * np.pi), yv > 0.4 * 2 * np.pi), yv < 0.6 * 2 * np.pi), h_bed + 1., h_bed)
    Q = np.array([h, hu, hv, z])
    return Q
