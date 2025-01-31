import jax.numpy as np

def periodic(Q):
    # east boundary
    Q = Q.at[:, :, -1].set(Q[:, :, 1])

    # west boundary
    Q = Q.at[:, :,  0].set(Q[:, :,  -2])

    # north boundary
    Q = Q.at[:, -1, :].set(Q[:, 1, :])

    # south boundary
    Q = Q.at[:, 0, :].set(Q[:, -2, :])

    return Q

def extrapolation(Q):

    # east boundary
    Q = Q.at[:, :, -1].set(Q[:, :, -2])

    # west boundary
    Q = Q.at[:, :,  0].set(Q[:, :,  1])

    # north boundary
    Q = Q.at[:, -1, :].set(Q[:, -2, :])

    # south boundary
    Q = Q.at[:, 0, :].set(Q[:, 1, :])

    return Q

def inflow_west(Q):

    # east boundary
    Q = Q.at[:, :, -1].set(Q[:, :, -2])

    # west boundary
    Q = Q.at[:, :,  0].set(Q[:, :,  1])
    Q = Q.at[0, :,  0].set(0.1)
    Q = Q.at[0, :,  2].set(0.05)
    Q = Q.at[0, :,  3].set(0.05)

    # north boundary
    Q = Q.at[:, -1, :].set(Q[:, -2, :])

    # south boundary
    Q = Q.at[:, 0, :].set(Q[:, 1, :])

    return Q

