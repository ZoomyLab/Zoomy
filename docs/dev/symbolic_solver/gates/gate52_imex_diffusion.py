"""Gate 52: IMEX-SSP2(2,2,2) with the implicit diffusion in its stages.

Constant dt = DT0 / 2^k, k = 0..3, on a fixed mesh; errors against the exact solution of
the semi-discrete system (the same mesh: the time error alone):

  heat      gate 41's u_t = div(nu grad u) (diffusion_matrix, the implicit slot), nu 1,
            t 0.05: `per` (periodic) and `dir0` (u = 0 on the walls, mirror ghosts) on
            1-D and quads, nx 32; stiff: nu dt / dx^2 = 12.8 at k = 0.  The initial
            field is an eigenvector of the two-point operator (the sine with the mesh's
            own mirror ghost / period), exact u0 exp(lambda t),
            lambda = -sum_d 4 nu / dx^2 sin^2(m pi dx / 2), m = 2 (per), 1 (dir0).
  advdiff   u_t + a u_x = nu u_xx on [0, 1], u = 0 at both ends (gate 41's `dir0`), nx 64,
            a 1 (the Rusanov flux of a scalar: first-order upwind), t 0.25,
            u0 = sin(pi x); exact expm(t L) u0, L the semi-discrete operator (linear,
            homogeneous BCs: column j = one explicit Euler step of the unit vector e_j on
            the NumPy backend, (u_1 - e_j) / dt).  Walls, so the upwind and the two-point
            operators do not commute (periodic: both circulant, Lie splitting exact).
            Schemes: ssp2 (IMEX-SSP2: the flux explicit, the diffusion implicit, in
            every stage), lie1 / lie2 (SSP-RK1 / RK2 flux step, then Crank-Nicolson).
            nu 0.02 (nu dt / dx^2 = 1.0 at k = 0): the orders.  nu 0.2 (10.2, stiff):
            the solver against the same scheme in matrix form, built from L_a (nu = 0)
            and L_d = L - L_a: the stage solves (I - A_ii dt L_d) Y_i = R_i of
            Pareschi, Russo 2005, Table II, and Lie's (I - dt/2 L_d)^-1 (I + dt/2 L_d).

Pass: heat order (k 2 -> 3) >= 1.8 (theory 2); advdiff ssp2 >= 1.8 (theory 2), lie1 >= 0.8,
lie2 <= 1.4 (theory 1: the Lie splitting caps SSP-RK2); solver vs matrix scheme <= 1e-10
(Newton 1e-12, GMRES 1e-13).  At nu 0.2 IMEX-SSP2 itself shows the order reduction of a
DIRK that is not stiffly accurate (orders 1.23 / 1.46 / 1.65, reaching 1.94 at dt / 64 in
matrix form): printed, not a criterion.

    python gate52_imex_diffusion.py [jax|numpy]
"""
import sys

import numpy as np

import common  # noqa: F401  (x64)
import gate41_diffusion_bc as G
from zoomy_core.mesh import LSQMesh
from zoomy_core.symsolver.discretization.time import IMEX_RK, SSP_RK
from zoomy_core.symsolver.settings import Imex

DT0, KS = 0.0125, (0, 1, 2, 3)
T_END = dict(heat=0.05, advdiff=0.25)
NX = dict(heat=32, advdiff=64)
NU_HEAT, NU_ADV, NU_STIFF, A_ADV = 1.0, 0.02, 0.2, 1.0
#: scheme -> (imex, time_order)
SCHEMES = {"ssp2": (Imex.ssp2, None), "lie1": (Imex.lie, 1), "lie2": (Imex.lie, 2)}
TIGHT = dict(implicit_tol=1e-12, gmres_tol=1e-13, implicit_maxiter=10, gmres_maxiter=200)


def heat_case(kind, case, nx=NX["heat"]):
    """(mesh, nsm): gate 41's heat equation, nu = NU_HEAT."""
    nsm = G.heat(1 if kind == "1d" else 2, case, "impl")
    nsm.sm.parameter_values.nu = NU_HEAT
    return G.mesh_of(kind, nx), nsm


def advdiff_case(nu=NU_ADV, a=A_ADV, nx=NX["advdiff"]):
    """(mesh, nsm): gate 41's 1-D `dir0` model with the advection speed `a`."""
    nsm = G.heat(1, "dir0", "impl")
    nsm.sm.parameter_values.nu, nsm.sm.parameter_values.a = nu, a
    return LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=nx), nsm


def semi_discrete_operator(mesh, nsm, dt=1e-3):
    """The matrix L of u' = L u (a linear model, homogeneous BCs): column j from one
    explicit Euler step of the unit vector e_j on the NumPy backend."""
    from zoomy_core.symsolver.numpy_backend import SymbolicHyperbolicSolverNumpy
    s = SymbolicHyperbolicSolverNumpy(time_end=dt, dt=dt, cfl_check_every=0)
    Q, A = s.setup(mesh, nsm)
    n = np.asarray(Q).shape[1]
    L = np.empty((n, n))
    for j in range(n):
        e = np.zeros((1, n))
        e[0, j] = 1.0
        _, Q1, _ = s.run(e, s.initial_aux(e, A))
        L[:, j] = (np.asarray(Q1)[0] - e[0]) / dt
    return L


def heat_exact(kind, case, xc, t, nx=NX["heat"]):
    """The semi-discrete exact solution at the cell centres `xc` (rows x, y)."""
    dim = 1 if kind == "1d" else 2
    m = 2 if case == "per" else 1
    dx = 1.0 / nx
    lam = -dim * 4 * NU_HEAT / dx ** 2 * np.sin(m * np.pi * dx / 2) ** 2
    return np.prod(np.sin(m * np.pi * np.asarray(xc)[:dim]), axis=0) * np.exp(lam * t)


_OPS = {}


def operators(nu):
    """(L_a, L_d): the semi-discrete advection (nu = 0) and diffusion parts."""
    if nu not in _OPS:
        La = semi_discrete_operator(*advdiff_case(0.0))
        _OPS[nu] = (La, semi_discrete_operator(*advdiff_case(nu)) - La)
    return _OPS[nu]


def u0_advdiff(xc):
    """The initial field sin(pi x)."""
    return np.sin(np.pi * np.asarray(xc)[0])


def advdiff_exact(xc, t, nu=NU_ADV):
    """The semi-discrete exact solution expm(t L) u0."""
    from scipy.linalg import expm
    La, Ld = operators(nu)
    return expm(t * (La + Ld)) @ u0_advdiff(xc)


def matrix_scheme(scheme, u, dt, n, nu):
    """`n` steps of `scheme` on u' = L_a u + L_d u in matrix form (the reference of the
    solver's discretization)."""
    La, Ld = operators(nu)
    eye = np.eye(len(u))
    for _ in range(n):
        if scheme == "ssp2":
            tab = IMEX_RK[Imex.ssp2]
            At = [[float(x) for x in r] for r in tab.At]
            A = [[float(x) for x in r] for r in tab.A]
            F, Gd = [], []
            for i in range(len(tab.c)):
                r = u + sum(dt * (At[i][j] * F[j] + A[i][j] * Gd[j]) for j in range(i))
                Y = np.linalg.solve(eye - dt * A[i][i] * Ld, r)
                F.append(La @ Y)
                Gd.append(Ld @ Y)
            u = u + dt * sum(float(wt) * f + float(w) * g
                             for wt, w, f, g in zip(tab.wt, tab.w, F, Gd))
        else:
            u0 = u
            for a, _ in SSP_RK[int(scheme[-1])]:
                u = float(a) * u0 + (1 - float(a)) * (u + dt * La @ u)
            u = np.linalg.solve(eye - 0.5 * dt * Ld, u + 0.5 * dt * Ld @ u)
    return u


def run(backend, mesh, nsm, imex, time_order, dt, t_end):
    """(t, u, cell centres) of one march at constant dt."""
    s = G.solver_class(backend)(time_end=t_end, dt=dt, cfl_check_every=0, imex=imex,
                                time_order=time_order, **TIGHT)
    Q, A = s.setup(mesh, nsm)
    t, Q, _ = s.run(Q, A)
    nc = np.asarray(Q).shape[1]
    return float(t), np.asarray(Q)[0], np.asarray(s.mesh_np.cell_centers)[:, :nc]


def heat_errors(backend, kind, case):
    """Max error vs the semi-discrete exact solution / max |exact|, k = KS (IMEX-SSP2)."""
    out = []
    for k in KS:
        mesh, nsm = heat_case(kind, case)
        t, u, xc = run(backend, mesh, nsm, Imex.ssp2, None, DT0 / 2 ** k, T_END["heat"])
        ex = heat_exact(kind, case, xc, t)
        out.append(float(np.abs(u - ex).max() / np.abs(ex).max()))
    return out


def advdiff_errors(backend, scheme, nu=NU_ADV, ks=KS):
    """(errors vs the semi-discrete exact solution, max rel |solver - matrix scheme|)."""
    imex, to = SCHEMES[scheme]
    err, dev = [], []
    for k in ks:
        dt = DT0 / 2 ** k
        mesh, nsm = advdiff_case(nu)
        t, u, xc = run(backend, mesh, nsm, imex, to, dt, T_END["advdiff"])
        ex = advdiff_exact(xc, t, nu)
        ref = matrix_scheme(scheme, u0_advdiff(xc), dt, int(round(t / dt)), nu)
        err.append(float(np.abs(u - ex).max() / np.abs(ex).max()))
        dev.append(float(np.abs(u - ref).max() / np.abs(ref).max()))
    return err, max(dev)


def main(backend="jax"):
    fails = 0

    def report(label, value, op, bound, info=""):
        nonlocal fails
        ok = value >= bound if op == ">=" else value <= bound
        fails += not ok
        print(f"gate52 {backend} {label}: {value:.3g} ({op} {bound:g})  "
              f"{'PASS' if ok else 'FAIL'}  {info}", flush=True)
    for kind in ("1d", "quad"):
        for case in ("per", "dir0"):
            e = heat_errors(backend, kind, case)
            report(f"heat {kind} {case} ssp2 order", np.log2(e[-2] / e[-1]), ">=", 1.8,
                   " ".join(f"{x:.2e}" for x in e))
    for scheme, (op, bound) in (("ssp2", (">=", 1.8)), ("lie1", (">=", 0.8)),
                                ("lie2", ("<=", 1.4))):
        e, dev = advdiff_errors(backend, scheme)
        report(f"advdiff nu {NU_ADV} {scheme} order", np.log2(e[-2] / e[-1]), op, bound,
               " ".join(f"{x:.2e}" for x in e))
        report(f"advdiff nu {NU_ADV} {scheme} vs matrix scheme", dev, "<=", 1e-10)
        e, dev = advdiff_errors(backend, scheme, NU_STIFF)
        report(f"advdiff nu {NU_STIFF} {scheme} vs matrix scheme", dev, "<=", 1e-10,
               "errors " + " ".join(f"{x:.2e}" for x in e)
               + f", order {np.log2(e[-2] / e[-1]):.2f}")
    print(f"gate52 {backend}: {fails} FAIL", flush=True)
    return fails


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1] if len(sys.argv) > 1 else "jax") else 0)
