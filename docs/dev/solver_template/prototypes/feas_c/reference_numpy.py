"""Independent numpy reference of the SAME scheme the C++ stubs implement
(wall ghosts, minmod MUSCL, Rusanov, SSP-RK2 Shu-Osher with a FROZEN dt).
Written without the emitted orchestration, so agreement validates the
emitted stage sequence / Shu-Osher combination / dt freezing."""
import sys, pathlib
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
g, N, L = 9.81, 100, 1.0
dx = L / N
dt = 2.0 ** -10

def minmod(a, b):
    return np.where(a * b <= 0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))

def residual(h, q, order):
    hg = np.concatenate([[h[0]], h, [h[-1]]])
    qg = np.concatenate([[-q[0]], q, [-q[-1]]])
    if order == 2:
        sh = minmod((hg[1:-1] - hg[:-2]) / dx, (hg[2:] - hg[1:-1]) / dx)
        sq = minmod((qg[1:-1] - qg[:-2]) / dx, (qg[2:] - qg[1:-1]) / dx)
    else:
        sh = sq = np.zeros(N)
    hL = np.concatenate([[hg[0]], h + 0.5 * dx * sh]); qL = np.concatenate([[qg[0]], q + 0.5 * dx * sq])
    hR = np.concatenate([h - 0.5 * dx * sh, [hg[-1]]]); qR = np.concatenate([q - 0.5 * dx * sq, [qg[-1]]])
    uL, uR = qL / hL, qR / hR
    s = np.maximum(np.abs(uL) + np.sqrt(g * hL), np.abs(uR) + np.sqrt(g * hR))
    Fh = 0.5 * (qL + qR) - 0.5 * s * (hR - hL)
    Fq = 0.5 * (qL * uL + 0.5 * g * hL**2 + qR * uR + 0.5 * g * hR**2) - 0.5 * s * (qR - qL)
    return -(Fh[1:] - Fh[:-1]) / dx, -(Fq[1:] - Fq[:-1]) / dx

def run(order):
    x = (np.arange(N) + 0.5) * dx
    h = np.where(x < 0.5, 2.0, 1.0); q = np.zeros(N)
    tab = {1: [(0.0, 1.0)], 2: [(0.0, 1.0), (0.5, 1.0)]}[order]
    for _ in range(10):
        h0, q0 = h.copy(), q.copy()
        for a, b in tab:
            rh, rq = residual(h, q, order)
            hc, qc = h + b * dt * rh, q + b * dt * rq
            h, q = a * h0 + (1 - a) * hc, a * q0 + (1 - a) * qc
    return h, q

for order, v in ((1, "o1_plain"), (2, "o2_plain"), (2, "o2_all")):
    h, q = run(order)
    c = np.loadtxt(HERE / "fixed" / v / "final_state.txt")
    print(f"{v}: max|h_C++ - h_ref| = {np.abs(c[:,0]-h).max():.3e}   "
          f"max|q_C++ - q_ref| = {np.abs(c[:,1]-q).max():.3e}   mass_ref={h.sum()*dx:.15f}")
