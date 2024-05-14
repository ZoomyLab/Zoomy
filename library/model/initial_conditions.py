import numpy as np
import sys
import attr
from attr import define
from typing import Callable, Optional

from library.misc.custom_types import FArray
from library.mesh.fvm_mesh import Mesh
import library.misc.io as io
import library.misc.interpolation as interpolate_mesh

# from library.mesh import Mesh2D


@define(slots=True, frozen=True)
class InitialConditions:
    def apply(self, X, Q):
        assert False
        return Q


@define(slots=True, frozen=True)
class Constant(InitialConditions):
    constants: Callable[[int], FArray] = lambda n_fields: np.array([1.] + [0. for i in range(n_fields-1)])
    def apply(self, X, Q):
        n_fields = Q.shape[0]
        for i in range(Q.shape[1]):
            Q[:,i] = self.constants(n_fields)
        return Q


@define(slots=True, frozen=False)
class RP(InitialConditions):
    low: Callable[[int], FArray] = lambda n_fields: np.array([1.0 * (i == 0) for i in range(n_fields)])
    high: Callable[[int], FArray] = lambda n_fields: np.array([2.0 * (i == 0) for i in range(n_fields)])
    jump_position_x: float = 0.0

    def apply(self, X, Q):
        assert X.shape[1] == Q.shape[1]
        n_fields = Q.shape[0]
        for i in range(Q.shape[1]):
            if X[0, i] < self.jump_position_x:
               Q[:,i]  = self.high(n_fields)
            else:
                Q[:,i] = self.low(n_fields)
        return Q

@define(slots=True, frozen=False)
class RP2d(InitialConditions):
    low: Callable[[int], FArray] = lambda n_fields: np.array([1.0 * (i == 0) for i in range(n_fields)])
    high: Callable[[int], FArray] = lambda n_fields: np.array([2.0 * (i == 0) for i in range(n_fields)])
    jump_position_x: float = 0.0
    jump_position_y: float = 0.0

    def apply(self, X, Q):
        assert X.shape[1] == Q.shape[1]
        n_fields = Q.shape[0]
        for i in range(Q.shape[1]):
            if X[0, i] < self.jump_position_x and X[1,i] < self.jump_position_y:
                Q[:,i] = self.high(n_fields)
            else:
                Q[:,i] = self.low(n_fields)
        return Q

@define(slots=True, frozen=False)
class RadialDambreak(InitialConditions):
    low: Callable[[int], FArray] = lambda n_fields: np.array([1.0 * (i == 0) for i in range(n_fields)])
    high: Callable[[int], FArray] = lambda n_fields: np.array([2.0 * (i == 0) for i in range(n_fields)])
    radius: float = 0.1

    def apply(self, X, Q):
        dim = X.shape[0]
        center = np.zeros(dim)
        for d in range(dim):
            center[d] = X[d, :].mean()
        assert X.shape[1] == Q.shape[1]
        n_fields = Q.shape[0]
        for i in range(Q.shape[1]):
            if np.linalg.norm(X[:, i]-center) <= self.radius:
                Q[:,i] = self.high(n_fields)
            else:
                Q[:,i] = self.low(n_fields)
        return Q


@define(slots=True, frozen=True)
class UserFunction(InitialConditions):
    function: Optional[Callable[[FArray], FArray]] = None

    def apply(self, X, Q):
        assert X.shape[1] == Q.shape[1]
        if self.function is None:
            self.function = lambda x: np.zeros(Q.shape[1])
        for i, x in enumerate(X.T):
            Q[:, i] = self.function(x)
        return Q

#TODO do time interpolation
@define(slots=True, frozen=True)
class RestartFromHdf5(InitialConditions):
    path_to_fields: Optional[str] = None
    mesh_new: Optional[Mesh] = None
    mesh_identical: bool = False
    path_to_old_mesh: Optional[str] = None
    snapshot: Optional[int] = -1
    map_fields: Optional[dict] = None


    def apply(self, X, Q):
        assert self.mesh_new is not None
        assert self.path_to_fields is not None
        assert X.shape[0] == Q.shape[0]
        if self.map_fields is None:
            map_fields = {i: i for i in range(Q.shape[1])}
        else:
            map_fields = self.map_fields
        mesh = Mesh.from_hdf5(self.path_to_old_mesh)
        _Q, _Qaux, time = io.load_fields_from_hdf5(self.path_to_fields, i_snapshot=self.snapshot)
        Q = np.zeros_like(Q)
        if self.mesh_identical:
            Q[:, list(map_fields.values())] = _Q[:, list(map_fields.keys())]
        else:
            assert self.path_to_old_mesh is not None
            Q[:, list(map_fields.values())] = interpolate_mesh.to_new_mesh(_Q, mesh, self.mesh_new)[:, list(map_fields.keys())]
        return Q

