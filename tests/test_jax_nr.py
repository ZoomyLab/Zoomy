import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres, cg
jax.config.update("jax_enable_x64", True)



# Parameters
N = 50            # number of grid points per variable
dx = 1.0 / (N-1)
dt = 0.1
alpha1 = 1.0
alpha2 = 0.5

u_left = 1.0
u_right = 2.0

# Laplacian operator with Dirichlet BCs
def laplacian(u):
    #u_ghost_left = 2*u_left - u[0]
    #u_ghost_right = 2*u_right - u[-1]
    u_ghost_left = u_left
    u_ghost_right =u_right
    u_pad = jnp.concatenate([jnp.array([u_ghost_left]), u, jnp.array([u_ghost_right])])
    return (u_pad[:-2] - 2 * u_pad[1:-1] + u_pad[2:]) / dx**2

# Residual for one heat eq, backward Euler in time
def heat_residual(u_new, u_old, alpha):
    #res = (u_new - u_old) / dt - alpha * laplacian(u_new)
    res = - alpha * laplacian(u_new)
    return res


# Full residual stacking two heat eqs
def residual(u, u_old1, u_old2):
    u1 = u[:N]
    u2 = u[N:]

    r1 = heat_residual(u1, u_old1, alpha1) 
    r2 = heat_residual(u2, u_old2, alpha2)
    return jnp.concatenate([r1, r2])

# Jacobian-vector product helper
def Jv(u, v, u_old1, u_old2):
    return jax.jvp(lambda x: residual(x, u_old1, u_old2), (u,), (v,))[1]

def compute_diagonal_of_jacobian(u, u_old1, u_old2):
    diag = []
    n = u.shape[0]
    for i in range(n):
        e = jnp.zeros_like(u).at[i].set(1.0)
        J_e = Jv(u, e, u_old1, u_old2)
        diag.append(J_e[i])
    return jnp.array(diag)

# Newton solver using CG for linear solve
def newton_solve(u0, u_old1, u_old2, tol=1e-8, maxiter=20):
    u = u0
    for i in range(maxiter):
        r = residual(u, u_old1, u_old2)
        res_norm = jnp.linalg.norm(r)
        print(f"Iter {i} , residual norm = {res_norm:.3e}")
        if res_norm < tol:
            break

        lin_op = lambda v: Jv(u, v, u_old1, u_old2)
        ## CG
        #delta, info = cg(lin_op, -r, x0=jnp.zeros_like(u), maxiter=50)
        ## GMRES
        #delta, info = gmres(lin_op, -r, x0=jnp.zeros_like(u), maxiter=100, solve_method='incremental', tol = 10**(-8))
        ## GMRES with preconditioner
        diag_J = compute_diagonal_of_jacobian(u, u_old1, u_old2)
        def preconditioner(v):
            return v / diag_J
        delta, info = gmres(lin_op, -r, x0=jnp.zeros_like(u), maxiter=100, solve_method='incremental', tol = 10**(-8), M=preconditioner)

        alpha = 1.0
        for _ in range(10):
            u_new = u + alpha * delta
            r_new = residual(u_new, u_old1, u_old2)
            if jnp.linalg.norm(r_new) < jnp.linalg.norm(r):
                u = u_new
                break
            alpha *= 0.5

        #u = u + delta
    return u

# Initial conditions (zeros)
u_old1 = jnp.ones(N)
u_old2 = jnp.ones(N)
#u_old1 = jnp.linspace(u_left, u_right, N+2)[1:-1]
#u_old2 = jnp.linspace(u_left, u_right, N+2)[1:-1]


# Initial guess for Newton (previous timestep)
u0 = jnp.concatenate([u_old1, u_old2])

r0 = residual(u0, u_old1, u_old2)
print("Initial residual norm:", jnp.linalg.norm(r0))
print("Initial residual vector:", r0)

 #Run Newton
u_new = newton_solve(u0, u_old1, u_old2)

# Print solution snapshots for first variable
print("Final u1:", u_new[:N])
print("Final u2:", u_new[N:])

