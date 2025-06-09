import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres, cg

jax.config.update("jax_enable_x64", True)


# Parameters
N = 50  # number of grid points per variable
dx = 1.0 / (N - 1)
dt = 0.1
alpha1 = 1.0

u_left = 1.0
u_right = 2.0


# Residual for one heat eq, backward Euler in time
def constraint1(u_new, u_old, alpha):
    res = jnp.zeros_like(u_new)

    def laplacian(u):
        u_ghost_left = u_left
        u_ghost_right = u_right
        u_pad = jnp.concatenate(
            [jnp.array([u_ghost_left]), u[0], jnp.array([u_ghost_right])]
        )
        return (u_pad[:-2] - 2 * u_pad[1:-1] + u_pad[2:]) / dx**2

    res = res.at[0].set((u_new[0] - u_old[0]) / dt - alpha * laplacian(u_new))
    # res = - alpha * laplacian(u_new)
    return res


def constraint2(u_new, u_old, alpha):
    res = jnp.zeros_like(u_new)

    def laplacian(u):
        u_ghost_left = u_left
        u_ghost_right = u_right
        u_pad = jnp.concatenate(
            [jnp.array([u_ghost_left]), u[1], jnp.array([u_ghost_right])]
        )
        return (u_pad[:-2] - 2 * u_pad[1:-1] + u_pad[2:]) / dx**2

    res = res.at[1].set(-alpha * laplacian(u_new))
    return res


# Full residual stacking two heat eqs
def residual(q, q_old):
    res = constraint1(q, q_old, alpha1)
    res += constraint2(q, q_old, alpha1)
    return res


# Jacobian-vector product helper
def Jv(u, v, u_old1):
    return jax.jvp(lambda x: residual(x, u_old1), (u,), (v,))[1]


# Newton solver using CG for linear solve
def newton_solve(u0, u_old1, tol=1e-8, maxiter=20):
    u = u0
    for i in range(maxiter):
        r = residual(u, u_old1)
        res_norm = jnp.linalg.norm(r)
        print(f"Iter {i} , residual norm = {res_norm:.3e}")
        if res_norm < tol:
            break

        def lin_op(v):
            return Jv(u, v, u_old1)

        delta, info = gmres(
            lin_op,
            -r,
            x0=jnp.zeros_like(u),
            maxiter=100,
            solve_method="incremental",
            tol=10 ** (-8),
        )

        alpha = 1.0
        for _ in range(10):
            u_new = u + alpha * delta
            r_new = residual(u_new, u_old1)
            if jnp.linalg.norm(r_new) < jnp.linalg.norm(r):
                u = u_new
                break
            alpha *= 0.5

    return u


# Initial conditions (zeros)
u_old1 = jnp.ones(2 * N).reshape((2, N))


# Initial guess for Newton (previous timestep)
u0 = u_old1

r0 = residual(u0, u_old1)
print("Initial residual norm:", jnp.linalg.norm(r0))
print("Initial residual vector:", r0)

# Run Newton
u_new = newton_solve(u0, u_old1)

# Print solution snapshots for first variable
print("Final u1:", u_new)
