import numpy as np

def RK1(func, Q, Qaux, param, dt):
    dQ = np.zeros_like(Q)
    func(dt, Q, Qaux, param, dQ)
    return Q + dt * dQ