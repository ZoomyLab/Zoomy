import numpy as np
from time import time as get_time


import sympy
from sympy import Matrix

from attr import define, field
from typing import Callable, List

from zoomy_core.misc.misc import Zstruct, ZArray

from zoomy_core.model.basefunction import Function


@define(slots=True, frozen=False, kw_only=True)
class BoundaryCondition:
    tag: str

    """ 
    Default implementation. The required data for the 'ghost cell' is the data from the interior cell. Can be overwritten e.g. to implement periodic boundary conditions.
    """

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        print("BoundaryCondition is a virtual class. Use one if its derived classes!")
        assert False


@define(slots=True, frozen=False, kw_only=True)
class Extrapolation(BoundaryCondition):
    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        return ZArray(Q)


@define(slots=True, frozen=False, kw_only=True)
class InflowOutflow(BoundaryCondition):
    prescribe_fields: dict[int, float]

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        Qout = ZArray(Q)
        for k, v in self.prescribe_fields.items():
            Qout[k] = eval(v)
        return Qout


@define(slots=True, frozen=False, kw_only=True)
class Lambda(BoundaryCondition):
    prescribe_fields: dict[int, float]

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        Qout = ZArray(Q)
        for k, v in self.prescribe_fields.items():
            Qout[k] = v(time, X, dX, Q, Qaux, parameters, normal)
        return Qout


def _sympy_interpolate_data(time, timeline, data):
    assert timeline.shape[0] == data.shape[0]
    conditions = (((data[0], time <= timeline[0])),)
    for i in range(timeline.shape[0] - 1):
        t0 = timeline[i]
        t1 = timeline[i + 1]
        y0 = data[i]
        y1 = data[i + 1]
        conditions += (
            (-(time - t1) / (t1 - t0) * y0 + (time - t0) / (t1 - t0) * y1, time <= t1),
        )
    conditions += (((data[-1], time > timeline[-1])),)
    return sympy.Piecewise(*conditions)


@define(slots=True, frozen=False, kw_only=True)
class FromData(BoundaryCondition):
    prescribe_fields: dict[int, np.ndarray]
    timeline: np.ndarray

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        # Extrapolate all fields
        Qout = ZArray(Q)

        # Set the fields which are prescribed in boundary condition dict
        time_start = get_time()
        for k, v in self.prescribe_fields.items():
            interp_func = _sympy_interpolate_data(time, self.timeline, v)
            Qout[k] = 2 * interp_func - Q[k]
        return Qout


@define(slots=True, frozen=False, kw_only=True)
class Wall(BoundaryCondition):
    """
    momentum_field_indices: list(int): indicate which fields need to be mirrored at the wall
    permeability: float : 1.0 corresponds to a perfect reflection (impermeable wall)
    """

    momentum_field_indices: List[List[int]] = [[1, 2]]
    permeability: float = 0.0
    wall_slip: float = 1.0

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        q = ZArray(Q)
        n_variables = q.shape[0]
        momentum_list = [Matrix([q[k] for k in l]) for l in self.momentum_field_indices]
        dim = momentum_list[0].shape[0]
        n = Matrix(normal[:dim])
        out = ZArray.zeros(n_variables)
        out = q
        momentum_list_wall = []
        for momentum in momentum_list:
            normal_momentum_coef = momentum.dot(n)
            transverse_momentum = momentum - normal_momentum_coef * n
            momentum_wall = (
                self.wall_slip * transverse_momentum
                - (1 - self.permeability) * normal_momentum_coef * n
            )
            momentum_list_wall.append(momentum_wall)
        for l, momentum_wall in zip(self.momentum_field_indices, momentum_list_wall):
            for i_k, k in enumerate(l):
                out[k] = momentum_wall[i_k]
        return out


@define(slots=True, frozen=False, kw_only=True)
class RoughWall(Wall):
    CsW: float = 0.5  # roughness constant
    Ks: float = 0.001  # roughness height

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        slip_length = dX * sympy.ln((dX * self.CsW) / self.Ks)
        # wall_slip = (1-2*dX/slip_length)
        # wall_slip = sympy.Min(sympy.Max(wall_slip, 0.0), 1.0)
        f = dX / slip_length
        wall_slip = (1 - f) / (1 + f)
        self.wall_slip = wall_slip
        return super().compute_boundary_condition(
            time, X, dX, Q, Qaux, parameters, normal
        )


@define(slots=True, frozen=False, kw_only=True)
class Periodic(BoundaryCondition):
    periodic_to_physical_tag: str

    def compute_boundary_condition(self, time, X, dX, Q, Qaux, parameters, normal):
        return ZArray(Q)


@define(slots=True, frozen=True)
class BoundaryConditions:
    _boundary_conditions: List[BoundaryCondition]
    _boundary_functions: List[Callable] = field(init=False)
    _boundary_tags: List[str] = field(init=False)

    def __attrs_post_init__(self):
        tags_unsorted = [bc.tag for bc in self._boundary_conditions]
        order = np.argsort(tags_unsorted)
        object.__setattr__(
            self,
            "_boundary_functions",
            [self._boundary_conditions[i].compute_boundary_condition for i in order],
        )
        object.__setattr__(
            self,
            "_boundary_tags",
            [self._boundary_conditions[i].tag for i in order],
        )
        object.__setattr__(
            self, "_boundary_conditions", [self._boundary_conditions[i] for i in order]
        )

    def get_boundary_condition_function(self, time, X, dX, Q, Qaux, parameters, normal):
        bc_idx = sympy.Symbol("bc_idx", integer=True)
        bc_func = sympy.Piecewise(
            *(
                (func(time, X.get_list(), dX, Q.get_list(), Qaux.get_list(), parameters.get_list(), normal.get_list()), sympy.Eq(bc_idx, i))
                for i, func in enumerate(self._boundary_functions)
            )
        )
        func = Function(
            name="boundary_conditions",
            args=Zstruct(
                idx=bc_idx,
                time=time,
                position=X,
                distance=dX,
                variables=Q,
                aux_variables=Qaux,
                parameters=parameters,
                normal=normal,
            ),
            definition=bc_func,
        )
        return func
