import numpy as np
import jax.numpy as jnp
import jax
from attr import define
from typing import Callable
from functools import partial
import timeit

from zoomy_core.misc.static_class import register_static_pytree


N = 1000000
NT = 1000


@register_static_pytree
@define(slots=True, frozen=True)
class Mesh:
    n_elements = N
    y = np.ndarray
    x = np.linspace(0, 1, n_elements)

    @classmethod
    def create_1d(cls, y: np.ndarray):
        return cls(4, y, y)


class SpaceOperator:
    @partial(jax.jit, static_argnames=["self"])
    def solve(self, Q, mesh):
        return Q + mesh.x


if __name__ == "__main__":
    mesh = Mesh.create_1d(y=np.linspace(0, 1, 10))
    Q = jnp.zeros(mesh.n_elements)
    so = SpaceOperator()
    start = timeit.default_timer()
    for i in range(NT):
        Q = so.solve(Q, mesh)
    end = timeit.default_timer()
    print(f"time: {end - start}")
