"""Gate 46: Hand-written viscous SWE SystemModel (no Model derivation):
q_t + div F(q) = div(nu h (grad u + grad u^T)) + 2 nu grad(h div u), Marche 2007 eq. (5.19)
(https://doi.org/10.1016/j.euromechflu.2006.04.007); in 1-D Gerbeau, Perthame 2001 eq. (1),
4 nu d_x(h d_x u).  The second term is the viscous normal stress tau_zz = -2 nu div u in the
pressure, i.e. div(h tau') with tau' = nu (grad u + grad u^T) + 2 nu (div u) I.

The viscous term goes in conservative form into diffusion_matrix A[i, j, a, b] (RHS d_a(A d_b Q_j)),
with h d_b u = d_b q - (q / h) d_b h.  One implicit-Lie step with nu = 1 minus nu = 0, divided by dt,
is the discrete viscous operator; it must converge (order 2) to the exact div(h tau').
"""
import sys
import numpy as np
import sympy as sp
import jax.numpy as jnp
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.misc.misc import Zstruct

from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel, face_normal_symbols
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

X, Y = sp.symbols("x y", real=True)
h, qx, qy = sp.symbols("h q_x q_y", real=True)
g, nu = sp.Symbol("g", positive=True), sp.Symbol("nu", nonnegative=True)


def viscous_swe():
    space = [X, Y]
    params = Zstruct(g=g, nu=nu)
    params._symbolic_name = "p"
    n = face_normal_symbols(2)
    F = sp.Matrix([[qx, qy],
                   [qx**2 / h + g * h**2 / 2, qx * qy / h],
                   [qx * qy / h, qy**2 / h + g * h**2 / 2]])
    A = sp.MutableDenseNDimArray.zeros(3, 3, 2, 2)      # rows/cols: h, q_x, q_y; a: outer, b: inner
    # x-momentum: d_x(2 nu h u_x) + d_y(nu h (u_y + v_x)) + d_x(2 nu h (u_x + v_y))
    A[1, 1, 0, 0], A[1, 0, 0, 0] = 4 * nu, -4 * nu * qx / h
    A[1, 2, 0, 1], A[1, 0, 0, 1] = 2 * nu, -2 * nu * qy / h
    A[1, 1, 1, 1], A[1, 0, 1, 1] = nu, -nu * qx / h
    A[1, 2, 1, 0], A[1, 0, 1, 0] = nu, -nu * qy / h
    # y-momentum: d_x(nu h (u_y + v_x)) + d_y(2 nu h v_y) + d_y(2 nu h (u_x + v_y))
    A[2, 1, 0, 1], A[2, 0, 0, 1] = nu, -nu * qx / h
    A[2, 2, 0, 0], A[2, 0, 0, 0] = nu, -nu * qy / h
    A[2, 1, 1, 0], A[2, 0, 1, 0] = 2 * nu, -2 * nu * qx / h
    A[2, 2, 1, 1], A[2, 0, 1, 1] = 4 * nu, -4 * nu * qy / h
    un = (qx * n[0] + qy * n[1]) / h
    c = sp.sqrt(g * h)
    sm = SystemModel(
        time=sp.Symbol("t", real=True), space=space, state=[h, qx, qy], aux_state=[],
        parameters=params, flux=F, hydrostatic_pressure=sp.zeros(3, 2),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(3, 3, 2),
        source=sp.zeros(3, 1), mass_matrix=sp.eye(3),
        eigenvalues=sp.Matrix([un - c, un, un + c]),
        parameter_values=Zstruct(g=9.81, nu=1.0), diffusion_matrix=A)
    sm.declare_roles(depth=h, momentum=[[qx], [qy]])
    sm.declare_admissible([])
    tags = ("left", "right", "bottom", "top")
    sm.attach_boundary_conditions(BC.BoundaryConditions(
        [BC.Periodic(tag=tags[i], periodic_to_physical_tag=tags[i ^ 1]) for i in range(4)]))
    sm.initial_conditions = IC.Constant(constants=lambda n_: np.array([1.0, 0.0, 0.0]))
    return sm


x, y = sp.symbols("x y")
tp = 2 * sp.pi
H = 1 + sp.Rational(1, 10) * sp.sin(tp * x) * sp.cos(tp * y)
QX = sp.Rational(1, 10) * sp.cos(tp * x) * sp.sin(tp * y) + sp.Rational(1, 20)
QY = sp.Rational(1, 20) * sp.sin(tp * (x + y))
u, v = QX / H, QY / H
txy = sp.diff(u, y) + sp.diff(v, x)
divu = sp.diff(u, x) + sp.diff(v, y)
EXACT = {"q_x": sp.diff(2 * H * sp.diff(u, x) + 2 * H * divu, x) + sp.diff(H * txy, y),
         "q_y": sp.diff(H * txy, x) + sp.diff(2 * H * sp.diff(v, y) + 2 * H * divu, y)}
fH, fQX, fQY = (sp.lambdify((x, y), e, "numpy") for e in (H, QX, QY))
fEX = {k: sp.lambdify((x, y), e, "numpy") for k, e in EXACT.items()}


def operator(nx, dt=1e-7):
    out = {}
    for nuv in (1.0, 0.0):
        sm = viscous_swe()
        sm.parameter_values.nu = nuv
        nsm = NumericalSystemModel.from_system_model(sm, reconstruction=ReconstructionSpec(order=2, limiter="none"))
        mesh = LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=nx, ny=nx)
        s = SymbolicHyperbolicSolverJax(time_end=dt, dt=dt, cfl_check_every=0, imex="lie", mood=False)
        Q, A = s.setup(mesh, nsm)
        names = [str(v_) for v_ in s.nsm.state]
        xc = np.asarray(s.mesh_np.cell_centers)[:, :Q.shape[1]]
        Q0 = np.stack([fH(xc[0], xc[1]), fQX(xc[0], xc[1]), fQY(xc[0], xc[1])])
        Q = jnp.asarray(Q0)
        A = s.initial_aux(Q, A)
        out[nuv] = np.asarray(s.run(Q, A)[1])
    L = (out[1.0] - out[0.0]) / dt
    return [float(np.abs(L[names.index(k)] - fEX[k](xc[0], xc[1])).max()
                  / np.abs(fEX[k](xc[0], xc[1])).max()) for k in ("q_x", "q_y")]


if __name__ == "__main__":
    res = [(n, operator(n)) for n in (16, 32, 64)]
    for (n, e), nxt in zip(res, res[1:] + [None]):
        o = "" if nxt is None else "  order " + " / ".join(f"{np.log2(a / b):.2f}" for a, b in zip(e, nxt[1]))
        print(f"viscous SWE (hand-written) nx {n}: discrete vs div(h tau'): q_x {e[0]:.3e}  q_y {e[1]:.3e}{o}", flush=True)
