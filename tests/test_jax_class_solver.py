import numpy as np
import jax.numpy as jnp
import jax
from attr import define
from typing import Callable
from functools import partial
import timeit

from library.misc.static_class import register_static_pytree


N = 1000000
NT = 1000

@register_static_pytree
@define(slots=True, frozen=True)
class Mesh:
    n_elements = N
    x = np.linspace(0, 1, n_elements)


class SpaceOperator():

    @partial(jax.jit, static_argnames=['self', 'mesh'])
    def solve(self, Q, mesh):
        return Q + mesh.x


if __name__ == "__main__":

    mesh = Mesh()
    Q = jnp.zeros(mesh.n_elements)
    so = SpaceOperator()
    start = timeit.default_timer()
    for i in range(NT):
        Q = so.solve(Q, mesh)
    end = timeit.default_timer()
    print(f'time: {end-start}')

