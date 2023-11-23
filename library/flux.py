import numpy as np

"""
Lax-Friedrichs flux implementation
"""
def LF(Qi, Qj, Qauxi, Qauxj, param, normal, model_functions, EVi=None, EVj=None, mesh_props = None):
    assert mesh_props is not None
    dt_dx = mesh_props.dt_dx
    Qout = np.zeros_like(Qi)
    flux = model_functions.flux
    dim = normal.shape[0]
    num_eq = Qi.shape[0]
    Fi = np.zeros((num_eq))
    Fj = np.zeros((num_eq))
    for n_i in range(dim):
        flux[n_i](Qi, Qauxi, param, Fi)  
        flux[n_i](Qj, Qauxj, param, Fj)  
        Qout += 0.5 * (Fi + Fj) * normal[n_i]
    Qout -= 0.5 * dt_dx * (Qj - Qi)
    return Qout, False


"""
Rusanov (local Lax-Friedrichs) flux implementation
"""
def LLF(Qi, Qj, nij, model_functions, EVi=None, EVj=None, mesh_props = None):
    assert EVi is not None
    assert EVj is not None
    smax = np.max(np.abs(np.vstack([EVi, EVj])))
    Qout = np.zeros_like(Qi)
    F = model_functions.flux
    dim = nij[0].shape
    for i in range(Qi.shape[0]):
        for n_i in range(dim):
            Qout[i] += 0.5 * (F(Qi[i]) + F(Qj[i])) * nij[i, n_i]
        Qout[i] -= 0.5 * smax * (Qj[i] - Qi[i])
    return Qout, False


