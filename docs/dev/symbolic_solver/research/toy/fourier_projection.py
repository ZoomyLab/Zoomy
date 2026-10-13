"""Single-Fourier-mode model of a linear one-layer non-hydrostatic system
(h, q, w), periodic, collocated central differences, to compare time orders of
SSP-RK2 with: exact projection, approximate (compact-Laplacian) projection, and
the approximate projection with an incremental (lagged) pressure.
Pure-python, stdlib only.
  h' = -dx q,  q' = -dx h - dx p,  w' = 2 p,  constraint D U = dx q + 2 w = 0
  G p = (0, dx p, -2 p);  D G = dx dx - 4  (wide);  L_a = compact dxx - 4
"""
import cmath
import math
import sys


def mat(n, m, f): return [[f(i, j) for j in range(m)] for i in range(n)]
def mm(A, B): return mat(len(A), len(B[0]), lambda i, j: sum(A[i][k] * B[k][j] for k in range(len(B))))
def mv(A, x): return [sum(A[i][k] * x[k] for k in range(len(x))) for i in range(len(A))]
def add(x, y, a=1.0): return [xi + a * yi for xi, yi in zip(x, y)]
def norm(x): return math.sqrt(sum(abs(v) ** 2 for v in x))


def expm(A, t):
    n = len(A); s = 0; nrm = max(sum(abs(v) for v in r) for r in A) * abs(t)
    while nrm > 0.5: nrm /= 2; s += 1
    B = mat(n, n, lambda i, j: A[i][j] * t / 2 ** s)
    E = mat(n, n, lambda i, j: 1.0 if i == j else 0.0); T = [r[:] for r in E]
    for k in range(1, 25):
        T = mat(n, n, lambda i, j: sum(T[i][l] * B[l][j] for l in range(n)) / k)
        E = mat(n, n, lambda i, j: E[i][j] + T[i][j])
    for _ in range(s): E = mm(E, E)
    return E


def setup(theta, dx):
    d = 1j * math.sin(theta) / dx                     # central first derivative
    lap_c = -4 * math.sin(theta / 2) ** 2 / dx ** 2   # compact second derivative
    A = [[0, -d, 0], [-d, 0, 0], [0, 0, 0]]           # hyperbolic part F = A U
    G = [0, d, -2]                                    # G p
    D = [0, d, 2]                                     # D U
    DG = sum(D[i] * G[i] for i in range(3))           # d^2 - 4  (wide)
    La = lap_c - 4                                    # compact
    return A, G, D, DG, La


def run(theta, dx, T, m, mode):
    A, G, D, DG, La = setup(theta, dx)
    dt = T / m
    Pex = lambda U: add(U, G, -sum(D[i] * U[i] for i in range(3)) / DG)
    U0 = Pex([1.0, 0.3, 0.2])
    ref = mv(expm(mm(mat(3, 3, lambda i, j: (1 if i == j else 0) - G[i] * D[j] / DG), A), T), U0)
    U, p = U0[:], 0.0
    if mode == "incr":                                # initial pressure: p with P_ex(A U0) = A U0 - G p
        p = sum(D[i] * mv(A, U0)[i] for i in range(3)) / DG
    for _ in range(m):
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
    return norm(add(U, ref, -1.0))


if __name__ == "__main__":
    T = 2.0
    for ppw in (20, 40, 80, 4):
        theta = 2 * math.pi / ppw; dx = theta     # k = 1
        s = 1 - (-math.sin(theta) ** 2 / dx ** 2 - 4) / (-4 * math.sin(theta / 2) ** 2 / dx ** 2 - 4)
        print(f"points/wavelength {ppw}: dx {dx:.4f}  s = 1 - DG/L_a = {s:.3e}")
        for mode in ("exact", "approx", "incr"):
            ms = (100, 200, 400, 800)
            e = [run(theta, dx, T, m, mode) for m in ms]
            o = [math.log2(e[i] / e[i + 1]) for i in range(len(e) - 1)]
            print(f"  {mode:6s} err " + " ".join(f"{x:.2e}" for x in e) + "  orders " + " ".join(f"{x:.2f}" for x in o))
