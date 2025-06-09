import numpy as np

"""
Dummy flux
"""


def Zero():
    def flux(Qi, Qj, Qauxi, Qauxj, param, normal, model_functions, mesh_props=None):
        Qout = np.zeros_like(Qi)
        return Qout, False

    return flux


"""
Lax-Friedrichs flux implementation
"""


def LF():
    def flux(Qi, Qj, Qauxi, Qauxj, param, normal, model_functions, mesh_props=None):
        assert mesh_props is not None
        dt_dx = mesh_props.dt_dx
        Qout = np.zeros_like(Qi)
        flux = model_functions.flux
        dim = normal.shape[0]
        num_eq = Qi.shape[0]
        for d in range(dim):
            Fi = flux[d](Qi, Qauxi, param)
            Fj = flux[d](Qj, Qauxj, param)
            Qout += 0.5 * (Fi + Fj) * normal[d]
        Qout -= 0.5 * dt_dx * (Qj - Qi)
        return Qout, False

    return flux


"""
Rusanov (local Lax-Friedrichs) flux implementation
"""


def LLF():
    def flux(Qi, Qj, Qauxi, Qauxj, param, normal, model_functions, mesh_props=None):
        EVi = model_functions.eigenvalues(Qi, Qauxi, param, normal)
        EVj = model_functions.eigenvalues(Qj, Qauxj, param, normal)
        assert not np.isnan(EVi).any()
        assert not np.isnan(EVj).any()
        smax = np.max(np.abs(np.vstack([EVi, EVj])))
        Qout = np.zeros_like(Qi)
        flux = model_functions.flux
        dim = normal.shape[0]
        num_eq = Qi.shape[0]
        for d in range(dim):
            Fi = flux[d](Qi, Qauxi, param)
            Fj = flux[d](Qj, Qauxj, param)
            Qout += 0.5 * (Fi + Fj) * normal[d]
        Qout -= 0.5 * smax * (Qj - Qi)
        return Qout, False

    return flux


"""
Rusanov (local Lax-Friedrichs) flux implementation
with topography fix (e.g. for model SWEtopo)
with WB fix for lake at rest
"""


def LLF_wb():
    def flux(Qi, Qj, Qauxi, Qauxj, param, normal, model_functions, mesh_props=None):
        IWB = np.eye(Qi.shape[0])
        IWB[-1, :] = 0.0
        IWB[0, -1] = 0.0
        EVi = np.zeros_like(Qi)
        EVj = np.zeros_like(Qj)
        EVi = model_functions.eigenvalues(Qi, Qauxi, param, normal)
        EVj = model_functions.eigenvalues(Qj, Qauxj, param, normal)
        assert not np.isnan(EVi).any()
        assert not np.isnan(EVj).any()
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
        Qout -= 0.5 * smax * IWB @ (Qj - Qi)
        return Qout, False

    return flux
