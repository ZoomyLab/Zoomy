import jax.numpy as np

def rarefaction_shock(x):
    n_cells = x.shape[0]
    n_dofs = 2
    Q = np.zeros((n_dofs, n_cells))
    h = np.where(x[0] < 0, 2, 1)
    u = np.where(x[0] < 0, 0, 0)
    Q = Q.at[0, :, :].set(h)
    Q = Q.at[1, :, :].set(h*u)
    return Q

def shock_shock(x):
    n_cells = x.shape[1] 
    n_dofs = 3
    Q = np.zeros((n_dofs, n_cells, n_cells))
    h = np.where(x[0] < 0, 1, 1)
    u = np.where(x[0] < 0, 1, -1)
    Q = Q.at[0, :, :].set(h)
    Q = Q.at[1, :, :].set(h*u)
    return Q
def rarefaction_rarefaction(x):
    n_cells = x.shape[0]
    n_dofs = 2
    Q = np.zeros((n_dofs, n_cells))
    h = np.where(x[0] < 0, 1, 1)
    u = np.where(x[0] < 0, -1, 1)
    Q = Q.at[0, :, :].set(h)
    Q = Q.at[1, :, :].set(h*u)
    return Q

def dam_break(x):
    n_x = x.shape[1]
    n_y = x.shape[2]
    n_dofs = 2
    Q = np.zeros((n_dofs, n_x, n_y))
    h = np.where(x[0] < 0, 2, 1)
    u = np.where(x[0] < 0, 0, 0)
    Q = Q.at[0, :, :].set(h)
    Q = Q.at[1, :, :].set(h*u)
    return Q

def flat(x):
    n_x = x.shape[1]
    n_y = x.shape[2]
    n_dofs = 4
    Q = np.zeros((n_dofs, n_x, n_y))
    h = np.where(x[0] < 0, 0.1, 0.1)
    u = np.where(x[0] < 0, 0, 0)
    Q = Q.at[0, :, :].set(h)
    Q = Q.at[1, :, :].set(h*u)
    return Q

def bottom_constant(x):
    return 0.0 * x[0]

def bottom_slope(x):
    return 0.05 * x[0]

def bottom_basins(x):
    return np.sin(x[0]) + 2.

