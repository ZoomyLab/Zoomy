import jax.numpy as np


def constant(A):
    A_i = A[:, 1:-1]
    A_ip = A[:, 2:]
    A_im = A[:, :-2]
    return A_i, A_ip, A_im
