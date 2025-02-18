import jax.numpy as np
import jax
from attr import define
from typing import Callable
import timeit

N = 1000
NT = 100

@define(slots=True, frozen=True)
class Mesh:
    n_elements = N
    x = np.linspace(0, 1, n_elements)


def func(q: float, x: float) -> float:
    # print(q + x)
    return np.array(q + x)


def get_space_operator(mesh, func: jax.lib.xla_extension.PjitFunction):
    def space_operator(
        Q: np.ndarray ) -> np.ndarray:
        for i in range(mesh.n_elements):
            if i < int(mesh.n_elements / 2):
                Q = Q.at[i].set(func(Q[i], mesh.x[i]))
        return Q
    return space_operator


def get_vectorized_space_operator(mesh, func: jax.lib.xla_extension.PjitFunction):
    I_validate = np.linspace(0, mesh.n_elements-1, mesh.n_elements, dtype=np.int32)[:int(mesh.n_elements / 2)]
    def space_operator(
        Q: np.ndarray) -> np.ndarray:
        Q = Q.at[I_validate].set(func(Q[I_validate], mesh.x[I_validate]))
        return Q

    return space_operator

def get_vectorized2_space_operator(mesh, func: jax.lib.xla_extension.PjitFunction):
    I = np.array(list(range(mesh.n_elements)), dtype=int)
    def space_operator(
        Q: np.ndarray ) -> np.ndarray:
        #for i in range(mesh.n_elements):
        #    if i < int(mesh.n_elements / 2):
        #        Q = Q.at[i].set(func(Q[i], mesh.x[i]))
        Q = np.where(I < mesh.n_elements / 2, func(Q, mesh.x), Q)
        return Q
    return space_operator


def solver():
    mesh = Mesh()
    Q = np.zeros(mesh.n_elements)
    operator = get_space_operator(mesh, func)
    for i in range(NT):
        Q = operator(Q)
    return Q

def solver_vectorized():
    mesh = Mesh()
    Q = np.zeros(mesh.n_elements)
    func_vec = jax.vmap(func)
    operator = get_vectorized2_space_operator(mesh, func_vec)
    for i in range(NT):
        Q = operator(Q)
    return Q

def solver_jit():
    mesh = Mesh()
    Q = np.zeros(mesh.n_elements)
    #func_jit = jax.jit(func, inline=True)
    #func_jit = jax.jit(func, inline=False)
    #func_vec = jax.vmap(func)
    func_vec = func
    operator = get_vectorized2_space_operator(mesh, func_vec)
    #operator = get_space_operator(mesh, func_jit)
    operator_jit = jax.jit(operator)
    for i in range(NT):
        Q = operator_jit(Q)
    return Q

def solver_vmap():
    mesh = Mesh()
    Q = np.zeros(mesh.n_elements)
    func_jit = jax.vmap(func, inline=True)
    operator = get_space_operator(mesh, func_jit)
    operator_jit = jax.vmap(operator)
    for i in range(NT):
        Q = operator_jit(Q)
    return Q


if __name__ == "__main__":

    #start = timeit.default_timer()
    #Q = solver()
    #elapsed = timeit.default_timer() - start
    #print(f"Normal: Elapsed time: {elapsed:.2f} s")

    start = timeit.default_timer()
    Q = solver_vectorized()
    elapsed = timeit.default_timer() - start
    print(f"Vectorized: Elapsed time: {elapsed:.2f} s")

    start = timeit.default_timer()
    Q = solver_jit()
    elapsed = timeit.default_timer() - start
    print(f"JIT: Elapsed time: {elapsed:.2f} s")

    # start = timeit.default_timer()
    # Qjit = solver_vmap()
    # elapsed = timeit.default_timer() - start
    # print(f"VMAP: Elapsed time: {elapsed:.2f} s")

