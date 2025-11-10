import numpy as np


class Flux:
    def get_flux_operator(self, model):
        pass


class Zero(Flux):
    def get_flux_operator(self, model):
        def compute(Qi, Qj, Qauxi, Qauxj, parameters, normal, Vi, Vj, Vij, dt):
            return np.zeros_like(Qi)

        return compute


class LaxFriedrichs(Flux):    
    """
    Lax-Friedrichs flux implementation
    """
    def get_flux_operator(self, model):
        def compute(Qi, Qj, Qauxi, Qauxj, parameters, normal, Vi, Vj, Vij, dt):
            Fi = np.einsum("id..., d...-> i...", model.flux(Qi, Qauxi, parameters), normal)
            Fj = np.einsum("id..., d...-> i...", model.flux(Qj, Qauxj, parameters), normal)
            Qout = 0.5 * (Fi + Fj)
            Qout -= 0.5 * dt / (Vi + Vj) * (Qj - Qi)
            return Qout
        return compute
    
class Rusanov(Flux):
    def __init__(self, identity_matrix=None):
        if not identity_matrix:
            self.Id = lambda n: np.eye(n)
        else:
            self.Id = identity_matrix
    """
    Rusanov (local Lax-Friedrichs) flux implementation
    """
    def get_flux_operator(self, model):
        Id_single = self.Id(model.n_variables)
        def compute(Qi, Qj, Qauxi, Qauxj, parameters, normal, Vi, Vj, Vij, dt):
            EVi = model.eigenvalues(Qi, Qauxi, parameters, normal)
            EVj = model.eigenvalues(Qj, Qauxj, parameters, normal)
            smax = np.max(np.abs(np.hstack([EVi, EVj])))
            Id = np.stack([Id_single]*Qi.shape[1], axis=2)  # (n_eq, n_eq, n_points)
            Fi = np.einsum("id..., d...-> i...", model.flux(Qi, Qauxi, parameters), normal)
            Fj = np.einsum("id..., d...-> i...", model.flux(Qj, Qauxj, parameters), normal)
            Qout = 0.5 * (Fi + Fj)
            Qout -= 0.5 * smax * np.einsum("ij..., jk...-> ik...", Id, (Qj - Qi))
            return Qout
        return compute 
            
