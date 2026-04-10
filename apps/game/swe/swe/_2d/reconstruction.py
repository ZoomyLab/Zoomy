import jax.numpy as np


def constant(Q):
    Q_i = Q[:, 1:-1, 1:-1]
    Q_n = Q[:, 2:, 1:-1]
    Q_s = Q[:, :-2, 1:-1]
    Q_e = Q[:, 1:-1, 2:]
    Q_w = Q[:, 1:-1, :-2]
    return Q_i, Q_n, Q_s, Q_e, Q_w
