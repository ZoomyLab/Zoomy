import jax.numpy as np
from apps.game.stream.parameters import o_in, o_top, o_out, o_bot


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

def inflow(Q):

    # east boundary
    Q = Q.at[0, :, -1].set(Q[0, :, -2])

    #wall
    Q = Q.at[1, :, -1].set(-Q[1, :, -2])
    # outflow
    for [o0, o1] in o_out:
        Q = Q.at[1, o0:o1, -1].set(Q[1, o0:o1, -2])

    Q = Q.at[2, :, -1].set(Q[2, :, -2])
    Q = Q.at[3, :, -1].set(Q[3, :, -2])

    # west boundary
    Q = Q.at[0, :,  0].set(Q[0, :, 1])
    
    #wall
    Q = Q.at[1, :,  0].set(-Q[1, :, 1])
    #inflow
    for [o0, o1] in o_in:
        Q = Q.at[1, o0:o1,  0].set(np.where(Q[1, o0:o1, 1] >= 0, 0.05, Q[1, o0:o1, 1]))
    
    Q = Q.at[2, :,  0].set(Q[2, :, 1])
    Q = Q.at[3, :,  0].set(Q[3, :, 1])

    # north boundary
    Q = Q.at[0, -1, :].set(Q[0, -2, :])
    Q = Q.at[1, -1, :].set(Q[1, -2, :])
    Q = Q.at[2, -1, :].set(-Q[2, -2, :])
    # outflow
    for [o0, o1] in o_top:
        Q = Q.at[2, -1, o0:o1].set(np.where(Q[2, -2,  o0:o1] >= 0, Q[2, -2,  o0:o1], 0))
    Q = Q.at[3, -1, :].set(Q[3, -2, :])

    # south boundary
    Q = Q.at[0, 0, :].set(Q[0, 1, :])
    Q = Q.at[1, 0, :].set(Q[1, 1, :])
    Q = Q.at[2, 0, :].set(-Q[2, 1, :])
    for [o0, o1] in o_bot:
        Q = Q.at[2, 0, o0:o1].set(np.where(Q[2, 1, o0:o1] > 0, 0, Q[2, 1,  o0:o1]))
    Q = Q.at[3, 0, :].set(Q[3, 1, :])

    return Q
