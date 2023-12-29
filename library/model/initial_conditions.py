import numpy as np
import sys
import attr
from attr import define
from typing import Callable, Optional

from library.misc.custom_types import FArray

# from library.mesh import Mesh2D


@define(slots=True, frozen=True)
class InitialConditions:
    def apply(self, X, Q):
        assert False


@define(slots=True, frozen=True)
class Constant(InitialConditions):
    def apply(self, X, Q):
        Q = np.ones_like(Q)


@define(slots=True, frozen=False)
class RP(InitialConditions):
    left: Callable[[int], FArray] = lambda n_fields: np.ones(n_fields, dtype=float)
    right: Callable[[int], FArray] = lambda n_fields: 2.0 * np.ones(
        n_fields, dtype=float
    )
    jump_position_x: float = 0.0

    def apply(self, X, Q):
        assert X.shape[0] == Q.shape[0]
        n_fields = Q.shape[1]
        for i, q in enumerate(Q):
            if X[i, 0] < self.jump_position_x:
                Q[i] = self.left(n_fields)
            else:
                Q[i] = self.right(n_fields)

@define(slots=True, frozen=False)
class RP2d(InitialConditions):
    low: Callable[[int], FArray] = lambda n_fields: np.ones(n_fields, dtype=float)
    high: Callable[[int], FArray] = lambda n_fields: 2.0 * np.ones(
        n_fields, dtype=float
    )
    jump_position_x: float = 0.0
    jump_position_y: float = 0.0

    def apply(self, X, Q):
        assert X.shape[0] == Q.shape[0]
        n_fields = Q.shape[1]
        for i, q in enumerate(Q):
            if X[i, 0] < self.jump_position_x and X[i,1] < self.jump_position_y:
                Q[i] = self.high(n_fields)
            else:
                Q[i] = self.low(n_fields)


@define(slots=True, frozen=True)
class UserFunction(InitialConditions):
    function: Optional[Callable[[FArray], FArray]] = None

    def apply(self, X, Q):
        assert X.shape[0] == Q.shape[0]
        if self.function is None:
            self.function = lambda x: np.zeros(Q.shape[1])
        for i, x in enumerate(X):
            Q[i] = self.function(x)
