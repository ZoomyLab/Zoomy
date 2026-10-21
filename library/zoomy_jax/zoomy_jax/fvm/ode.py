"""Module `zoomy_jax.fvm.ode`."""

import numpy as np
import jax.numpy as jnp


# REQ-185: the integrated operator ``func`` is called as
# ``func(dt, time, Q, Qaux, param, dQ)`` — the current simulation ``time`` is
# threaded from ``step`` INTO the timestepping so a time-dependent source
# (rain hyetograph, manufactured ``S(x, t)``) binds ``t``.  ``time`` defaults
# to 0.0 so existing call sites that omit it are exact for autonomous
# operators (the flux operator already declares the ``(dt, time, ...)``
# signature; the default keeps the RK API backward-compatible).
def RK1(func, Q, Qaux, param, dt, time=0.0, func_jac=None, func_bc=None):
    """RK1."""
    dQ = jnp.zeros_like(Q)
    dQ = func(dt, time, Q, Qaux, param, dQ)
    return Q + dt * dQ


def RK2(func, Q, Qaux, param, dt, time=0.0, func_jac=None, func_bc=None):
    """SSP-RK2 (Heun's method). JIT-compatible."""
    dQ = jnp.zeros_like(Q)
    Q0 = Q
    dQ = func(dt, time, Q, Qaux, param, dQ)
    Q1 = Q + dt * dQ
    dQ = jnp.zeros_like(Q)
    dQ = func(dt, time, Q1, Qaux, param, dQ)
    Q2 = Q1 + dt * dQ
    return 0.5 * (Q0 + Q2)


def RK3(func, Q, Qaux, param, dt, time=0.0, func_jac=None, func_bc=None):
    """SSP-RK3. JIT-compatible."""
    dQ = jnp.zeros_like(Q)
    Q0 = Q
    dQ = func(dt, time, Q, Qaux, param, dQ)
    Q1 = Q + dt * dQ
    dQ = jnp.zeros_like(Q)
    dQ = func(dt, time, Q1, Qaux, param, dQ)
    Q2 = 3.0 / 4 * Q0 + 1.0 / 4 * (Q1 + dt * dQ)
    dQ = jnp.zeros_like(Q)
    dQ = func(dt, time, Q2, Qaux, param, dQ)
    Q3 = 1.0 / 3 * Q0 + 2 / 3 * (Q2 + dt * dQ)
    return Q3


def RKimplicit(func, Q, Qaux, param, dt, time=0.0, func_jac=None, func_bc=None):
    """
    implicit euler
    """
    assert func_jac is not None
    Jac = np.zeros((Q.shape[0], Q.shape[0], Q.shape[1]), dtype=float)
    dQ = np.zeros_like(Q)
    I = np.eye(Q.shape[0])

    dQ = func(dt, time, Q, Qaux, param, dQ)
    Jac = func_jac(dt, Q, Qaux, param, Jac)

    b = Q + dt * dQ
    for i in range(Q.shape[1]):
        A = I - dt * Jac[:, :, i]
        b[:, i] += -dt * np.dot(Jac[:, :, i], Q[:, i])
        Q[:, i] = np.linalg.solve(A, b[:, i])
    return Q
