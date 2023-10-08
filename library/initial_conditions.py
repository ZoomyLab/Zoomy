import numpy as np
import sys
import attr
from attr import define
from typing import Callable, Optional

from library.custom_types import FArray

# from library.mesh import Mesh2D


@define(slots=True, frozen=True)
class InitialConditions:
    def apply(self, Q, X):
        assert False


@define(slots=True, frozen=True)
class Constant:
    def apply(self, Q, X):
        return np.ones_like(Q)


@define(slots=True, frozen=False)
class RP(InitialConditions):
    left: Callable[[int], FArray] = lambda n_fields: np.ones(n_fields, dtype=float)
    right: Callable[[int], FArray] = lambda n_fields: 2.0 * np.ones(
        n_fields, dtype=float
    )
    jump_position_x: float = 0.0

    def apply(self, Q, X):
        n_fields = Q.shape[1]
        for i, q in enumerate(Q):
            if X[i, 0] < self.jump_position_x:
                Q[i] = self.left(n_fields)
            else:
                Q[i] = self.right(n_fields)
        return Q


@define(slots=True, frozen=True)
class UserFunction(InitialConditions):
    function: Optional[Callable[[FArray], FArray]] = None

    def apply(self, Q, X):
        if self.function is None:
            self.function = lambda x: np.zeros(Q.shape[1])
        for i, x in enumerate(X):
            Q[i] = self.function(x)
        return Q
