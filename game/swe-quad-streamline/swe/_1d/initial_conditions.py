import jax.numpy as np

def rarefaction_shock(x):
    n_cells = x.shape[0]
    n_dofs = 2
    Q = np.zeros((n_dofs, n_cells))
    h = np.where(x < 0, 2, 1)
    u = np.where(x < 0, 0, 0)
    Q = Q.at[0, :].set(h)
    Q = Q.at[1, :].set(h*u)
    return Q

def shock_shock(x):
    n_cells = x.shape[0]
    n_dofs = 2
    Q = np.zeros((n_dofs, n_cells))
    h = np.where(x < 0, 1, 1)
    u = np.where(x < 0, 1, -1)
    Q = Q.at[0, :].set(h)
    Q = Q.at[1, :].set(h*u)
    return Q

def rarefaction_rarefaction(x):
    n_cells = x.shape[0]
    n_dofs = 2
    Q = np.zeros((n_dofs, n_cells))
    h = np.where(x < 0, 1, 1)
    u = np.where(x < 0, -1, 1)
    Q = Q.at[0, :].set(h)
    Q = Q.at[1, :].set(h*u)
    return Q

def dam_break_w_bottom(x):
    n_cells = x.shape[0]
    n_dofs = 3
    Q = np.zeros((n_dofs, n_cells))
    h = np.where(x > -1 , np.where(x < 1, 2, 1), 1)
    u = np.where(x < 0, 0, 0)
    Q = Q.at[0, :].set(h)
    Q = Q.at[1, :].set(h*u)
    return Q

def dam_break(x):
    n_cells = x.shape[0]
    n_dofs = 2
    Q = np.zeros((n_dofs, n_cells))
    h = np.where(x > -1 , np.where(x < 1, 3, 2), 2)
    u = np.where(x < 0, 0, 0)
    Q = Q.at[0, :].set(h)
    Q = Q.at[1, :].set(h*u)
    return Q

def bottom_constant(x):
    return 0.0 * x

def bottom_slope(x):
    return 0.05 * x

def bottom_basins(x):
    # return np.zeros_like(x)
    # return 0.5 * np.sin(x*2*np.pi/20) + 0.
    return np.where(x < 2, 0, np.where(x > 4, 0, 2.5))
    # return 1.0 * x**2

