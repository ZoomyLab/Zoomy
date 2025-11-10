import numpy as np


def RK1(func, Q, Qaux, param, dt, func_jac=None, func_bc=None):
    dQ = np.zeros_like(Q)
    dQ = func(dt, Q, Qaux, param, dQ)
    return Q + dt * dQ


def RK2(func, Q, Qaux, param, dt, func_jac=None, func_bc=None):
    """
    heun scheme
    """
    dQ = np.zeros_like(Q)
    Q0 = np.array(Q)
    dQ = func(dt, Q, Qaux, param, dQ)
    Q1 = Q + dt * dQ
    dQ = func(dt, Q1, Qaux, param, dQ)
    Q2 = Q1 + dt * dQ
    return 0.5 * (Q0 + Q2)


def RK3(func, Q, Qaux, param, dt, func_jac=None, func_bc=None):
    """ """
    dQ = np.zeros_like(Q)
    Q0 = np.array(Q)
    dQ = func(dt, Q, Qaux, param, dQ)
    Q1 = Q + dt * dQ
    dQ = func(dt, Q1, Qaux, param, dQ)
    Q2 = 3.0 / 4 * Q0 + 1.0 / 4 * (Q1 + dt * dQ)
    dQ = func(dt, Q2, Qaux, param, dQ)
    Q3 = 1.0 / 3 * Q0 + 2 / 3 * (Q2 + dt * dQ)
    # TODO see old implementation
    # func(dt, Q3, Qaux, param, dQ)
    return Q3


def RKimplicit(func, Q, Qaux, param, dt, func_jac=None, func_bc=None):
    """
    implicit euler
    """
    assert func_jac is not None
    Jac = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]), dtype=float)
    dQ = np.zeros_like(Q)
    I = np.eye(Q.shape[0])

    dQ = func(dt, Q, Qaux, param, dQ)
    Jac = func_jac(dt, Q, Qaux, param, Jac)

    b = Q + dt * dQ
    for i in range(Q.shape[1]):
        A = I - dt * Jac[:, :, i]
        b[:, i] += -dt * np.dot(Jac[:, :, i], Q[:, i])
        Q[:, i] = np.linalg.solve(A, b[:, i])
    return Q
