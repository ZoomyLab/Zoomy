import numpy as np

"""
Lax-Friedrichs flux implementation
"""
def LF(Qi, Qj, Qauxi, Qauxj, param, normal, model_functions, mesh_props = None):
    assert mesh_props is not None
    dt_dx = mesh_props.dt_dx
    Qout = np.zeros_like(Qi)
    flux = model_functions.flux
    dim = normal.shape[0]
    num_eq = Qi.shape[0]
    Fi = np.zeros((num_eq))
    Fj = np.zeros((num_eq))
    for d in range(dim):
        flux[d](Qi, Qauxi, param, Fi)  
        flux[d](Qj, Qauxj, param, Fj)  
        Qout += 0.5 * (Fi + Fj) * normal[d]
    Qout -= 0.5 * dt_dx * (Qj - Qi)
    return Qout, False


"""
Rusanov (local Lax-Friedrichs) flux implementation
"""
def LLF(Qi, Qj, Qauxi, Qauxj, param, normal, model_functions, mesh_props = None):
    EVi = np.zeros_like(Qi)
    EVj = np.zeros_like(Qj)
    model_functions.eigenvalues(Qi, Qauxi, param, normal, EVi )
    model_functions.eigenvalues(Qj, Qauxj, param, normal, EVj )
    smax = np.max(np.abs(np.vstack([EVi, EVj])))
    Qout = np.zeros_like(Qi)
    flux = model_functions.flux
    dim = normal.shape[0]
    num_eq = Qi.shape[0]
    Fi = np.zeros((num_eq))
    Fj = np.zeros((num_eq))
    for d in range(dim):
        flux[d](Qi, Qauxi, param, Fi)  
        flux[d](Qj, Qauxj, param, Fj)  
        Qout += 0.5 * (Fi + Fj) * normal[d]
    Qout -= 0.5 * smax * (Qj - Qi)
    return Qout, False


