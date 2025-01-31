import jax.numpy as np

def periodic(Q):
    # right boundary
    Q = Q.at[:, -1].set(Q[:, 1])

    # left boundary
    Q = Q.at[:, 0].set(Q[:, -2])

    return Q

def extrapolation(Q):
    # right boundary
    Q = Q.at[:, -1].set(Q[:, -2])

    # left boundary
    Q = Q.at[:, 0].set(Q[:, 1])

    return Q

