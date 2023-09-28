import numpy as np
import os
import sys
from scipy.integrate import ode

# from numba import jit
from copy import deepcopy

from library.solver.baseclass import BaseYaml  # nopep8
import library.solver.misc as misc  # nopep8

main_dir = os.getenv("SMPYTHON")


class ODE(BaseYaml):
    yaml_tag = "!ODE"

    def set_default_parameters(self):
        self.scheme = "RK"
        self.order = 1
        # TODO implement relaxation
        self.timestep_relaxation = 1.0

    def evaluate(self, func, t, Q, **kwargs):
        return getattr(sys.modules[__name__], self.scheme)(
            func, t, Q, self.order, **kwargs
        )


# TODO this is a real problem, since nc terms using dQdx depend on
# neighboring cells.
# This however, it not 'intended' by ODE solvers of the form dQdt = R(Q).
# "ODE" solvers of the form dQdt = R(Q, dQdx) are an initial boundary value
# problem -right?
# So I can only support equations that do not depend on dQdx
def scipy(func, t, Q, order, **kwargs):
    assert "func_jac" in kwargs
    dt = kwargs["dt"]
    jac = kwargs["func_jac"]
    kwargs2 = deepcopy(kwargs)
    del kwargs2["func_jac"]
    return scipy_vode(func, t, Q, jac, **kwargs2)


# TODO enable Isoda?
# TODO enable numba
# @jit(nopython=False)
def scipy_vode(func, t, Q, func_jac, **kwargs):
    Qnew = np.array(Q)
    dt = kwargs["dt"]
    # dQdx is not supported, see above!
    dx = kwargs["dx"]
    # Qr = np.roll(Q, -1, axis=1)
    # Ql = np.roll(Q, 1, axis=1)
    # dQdx = (Qr-Ql)/2/dx
    for i in range(Q.shape[1]):
        kwargs2 = {"dx": dx[:, i], "dHdx": kwargs["dHdx"][i]}
        func2 = lambda t, Q: func(t, Q, **kwargs2)
        res = ode(func2, func_jac).set_integrator("vode", method="BDF")
        # res = ode(f, jac).set_integrator('Isoda')
        res.set_initial_value(Qnew[:, i], t)
        Qnew[:, i] = res.integrate(res.t + dt)
        assert res.successful()
    return Qnew


def RK(func, t, Q_, order, **kwargs):
    Q = np.array(Q_)
    if order == 1:
        # explicit euler
        dt = kwargs["dt"]
        dQ = func(t, Q, **kwargs)

        return Q + dt * dQ
    elif order == 2:
        # heun scheme
        dt = kwargs["dt"]
        Q0 = np.array(Q)
        dQ = func(t, Q, **kwargs)
        Q1 = Q + dt * dQ

        dQ = func(t, Q1, **kwargs)
        Q2 = Q1 + dt * dQ

        return 0.5 * (Q0 + Q2)
    elif order == 3:
        dt = kwargs["dt"]
        Q0 = np.array(Q)
        dQ = func(t, Q, **kwargs)
        Q1 = Q + dt * dQ

        dQ = func(t, Q1, **kwargs)
        Q2 = 3.0 / 4 * Q0 + 1.0 / 4 * (Q1 + dt * dQ)

        dQ = func(t, Q2, **kwargs)

        result = 1.0 / 3 * Q0 + 2 / 3 * (Q2 + dt * dQ)
        _ = func(t, result, **kwargs)
        return result
    elif order == -1:
        dt = kwargs["dt"]
        func_jac = kwargs["func_jac"]
        # TODO hack
        if Q[0] <= 0:
            return Q
        # implicit euler
        # identity matrix vectorized
        I = np.repeat(np.eye(Q.shape[0])[:, :, np.newaxis], Q.shape[1], axis=2)
        # I = np.repeat(massmatrix[:,:,np.newaxis], Q.shape[1], axis=2)
        Jac = (func_jac(t, Q, **kwargs)).reshape((Q.shape[0], Q.shape[0], Q.shape[1]))
        A = I - dt * Jac
        # b = dt * func(t, Q, **kwargs) + np.einsum("ij...,j...->i...", A, Q)
        b = Q + dt * func(t, Q, **kwargs) - dt * np.einsum("ij...,j...->i...", Jac, Q)
        for i in range(A.shape[2]):
            Q[:, i] = np.linalg.solve(A[:, :, i], b[:, i])
        # b = dt * func(t, Q, **kwargs)
        # for i in range(A.shape[2]):
        #     Q[:,i] = np.linalg.solve(A[:,:,i], b[:,i]) + Q[:,i]
        return Q
    else:
        assert False
