"""Growth check of the toy schemes over all wavenumbers at CFL 0.9, long time."""
import math
import fourier_projection as _fp
from fourier_projection import add, mv, norm
def setup(*a): return _fp.setup(*a)

def growth(theta, dx, cfl, nsteps, mode):
    A, G, D, DG, La = setup(theta, dx)
    dt = cfl * dx
    Pex = lambda U: add(U, G, -sum(D[i] * U[i] for i in range(3)) / DG)
    U = Pex([1.0, 0.3, 0.2]); n0 = norm(U)
    p = sum(D[i] * mv(A, U)[i] for i in range(3)) / DG if mode == "incr" else 0.0
    mx = 1.0
    for _ in range(nsteps):
        Us = U
        for st in range(2):
            if mode == "incr":
                Y = add(add(Us, mv(A, Us), dt), G, -dt * p)
                dp = sum(D[i] * Y[i] for i in range(3)) / (dt * La)
                Us, p = add(Y, G, -dt * dp), p + dp
            else:
                Y = add(Us, mv(A, Us), dt)
                op = DG if mode == "exact" else La
                Us = add(Y, G, -sum(D[i] * Y[i] for i in range(3)) / op)
        U = [(a + b) / 2 for a, b in zip(U, Us)]
        mx = max(mx, norm(U) / n0)
    return mx

dx = 0.1
for cfl in (0.45, 0.9):
    for mode in ("exact", "approx", "incr"):
        g = [growth(j * math.pi / 16, dx, cfl, 4000, mode) for j in range(1, 16)]
        print(f"cfl {cfl} {mode:6s} max growth over theta=j*pi/16: {max(g):.3e} at j={1+g.index(max(g))}")
