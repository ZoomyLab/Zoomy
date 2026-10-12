"""Gate 45: the in-plane viscous operator of SME(0) + NewtonianInPlane, measured on the JAX
symbolic solver, against (a) the physics div(nu h (grad u + grad u^T)) and h div(tau), and
(b) the SystemModel's own operator (diffusion_matrix + the nu-part of the source).

Findings 2026-10-13 (ANALYSIS 45): the discrete operator equals the SystemModel operator WITHOUT its
mixed-index diffusion entries A[i, j, x, y] (TPFA drops them; second order to that), and the
SystemModel operator itself matches neither physical form (3-11 %).

Periodic smooth h, qx, qy on [0,1]^2; one implicit-Lie step of dt with nu = 1 and nu = 0;
(Q_nu - Q_0) / dt on the momentum rows is the discrete viscous operator (O(dt) accurate).
"""
import sys
import numpy as np
import sympy as sp
import zoomy_core.model.boundary_conditions as BC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree, NewtonianInPlane
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel import SystemModel
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
import jax.numpy as jnp

x, y = sp.symbols("x y")
tp = 2 * sp.pi
H = 1 + sp.Rational(1, 10) * sp.sin(tp * x) * sp.cos(tp * y)
QX = sp.Rational(1, 10) * sp.cos(tp * x) * sp.sin(tp * y) + sp.Rational(1, 20)
QY = sp.Rational(1, 20) * sp.sin(tp * (x + y))
u, v = QX / H, QY / H
P = {"q_x_0": sp.diff(2 * H * sp.diff(u, x), x) + sp.diff(H * (sp.diff(u, y) + sp.diff(v, x)), y),
     "q_y_0": sp.diff(H * (sp.diff(u, y) + sp.diff(v, x)), x) + sp.diff(2 * H * sp.diff(v, y), y)}
Hf, QXf, QYf = (sp.lambdify((x, y), e, "numpy") for e in (H, QX, QY))
gu, gv = (sp.diff(u, x), sp.diff(u, y)), (sp.diff(v, x), sp.diff(v, y))
txx, txy, tyy = 2 * gu[0], gu[1] + gv[0], 2 * gv[1]
CAND = {
    "div(h tau)":      {"q_x_0": P["q_x_0"], "q_y_0": P["q_y_0"]},
    "h div(tau)":      {"q_x_0": H * (sp.diff(txx, x) + sp.diff(txy, y)), "q_y_0": H * (sp.diff(txy, x) + sp.diff(tyy, y))},
    "div(h grad u)":   {"q_x_0": sp.diff(H * gu[0], x) + sp.diff(H * gu[1], y), "q_y_0": sp.diff(H * gv[0], x) + sp.diff(H * gv[1], y)},
    "lap(q)":          {"q_x_0": sp.diff(QX, x, 2) + sp.diff(QX, y, 2), "q_y_0": sp.diff(QY, x, 2) + sp.diff(QY, y, 2)},
    "div(h tau) transposed-x": {"q_x_0": sp.diff(2 * H * gu[0], x) + sp.diff(H * gu[1], y) + sp.diff(H * gv[1], x), "q_y_0": P["q_y_0"]},
}
CF = {n: {k: sp.lambdify((x, y), e, "numpy") for k, e in c.items()} for n, c in CAND.items()}
_sm = SystemModel.from_model(SME(level=0, dimension=3, boundary_conditions=BC.BoundaryConditions([BC.Wall(tag=t) for t in ("left", "right", "bottom", "top")]),
                                  closures=[Newtonian(), ManningFriction(), StressFree(), NewtonianInPlane()]))
_nu = _sm.parameters.nu
_F = {"h": H, "q_x_0": QX, "q_y_0": QY, "b": sp.S.Zero}
_St = list(_sm.state)
def _tof(e): return e.subs({a: _F[str(a)] for a in e.free_symbols if str(a) in _F}).subs(_nu, 1)
_aux = {}
for a in _sm.aux_state:
    n_ = str(a)
    if n_[-2:] in ("_x", "_y"):
        base = sp.sympify(n_[:-2], locals={**{k: sp.Symbol(k) for k in _F}, "nu": _nu})
        _aux[a] = sp.diff(_tof(base), x if n_[-1] == "x" else y)
print("unmapped aux:", [str(a) for a in _sm.aux_state if a not in _aux])
_A = _sm.diffusion_matrix
SMOP = {}
for i, k in ((2, "q_x_0"), (3, "q_y_0")):
    R = sum(sp.diff(_tof(_A[i, j, a, b]) * sp.diff(_F[str(_St[j])], (x, y)[b]), (x, y)[a]) for j in range(4) for a in range(2) for b in range(2))
    src = sp.expand(_sm.source[i, 0]); Snu = sum((t for t in sp.Add.make_args(src) if t.has(_nu)), sp.S.Zero)
    SMOP[k + " diff"] = sp.lambdify((x, y), R, "numpy")
    Rs = sum(sp.diff(_tof(_A[i, j, a, b]) * sp.diff(_F[str(_St[j])], (x, y)[a]), (x, y)[b]) for j in range(4) for a in range(2) for b in range(2))
    SMOP[k + " swap"] = sp.lambdify((x, y), Rs, "numpy")
    Rd = sum(sp.diff(_tof(_A[i, j, a, a]) * sp.diff(_F[str(_St[j])], (x, y)[a]), (x, y)[a]) for j in range(4) for a in range(2))
    SMOP[k + " diag"] = sp.lambdify((x, y), Rd, "numpy")
    SMOP[k + " src"] = sp.lambdify((x, y), _tof(Snu.subs(_aux)), "numpy")

bcs = [BC.Periodic(tag="left", periodic_to_physical_tag="right"),
       BC.Periodic(tag="right", periodic_to_physical_tag="left"),
       BC.Periodic(tag="bottom", periodic_to_physical_tag="top"),
       BC.Periodic(tag="top", periodic_to_physical_tag="bottom")]


def operator(nx, dt=1e-7):
    out = {}
    for nu in (1.0, 0.0):
        sm = SystemModel.from_model(SME(level=0, dimension=3, boundary_conditions=BC.BoundaryConditions(bcs),
                                        closures=[Newtonian(), ManningFriction(), StressFree(), NewtonianInPlane()]))
        sm.parameter_values.nu = nu
        for k in ("n", "n_m"):
            if hasattr(sm.parameter_values, k):
                setattr(sm.parameter_values, k, 0.0)
        nsm = NumericalSystemModel.from_system_model(sm, reconstruction=ReconstructionSpec(order=2, limiter="none"))
        mesh = LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=nx, ny=nx)
        s = SymbolicHyperbolicSolverJax(time_end=dt, dt=dt, cfl_check_every=0, imex="lie", mood=False)
        Q, A = s.setup(mesh, nsm)
        names = [str(v) for v in s.nsm.state]
        xc = np.asarray(s.mesh_np.cell_centers)[:, :Q.shape[1]]
        Q0 = np.zeros(Q.shape)
        Q0[names.index("h")] = Hf(xc[0], xc[1])
        Q0[names.index("q_x_0")] = QXf(xc[0], xc[1])
        Q0[names.index("q_y_0")] = QYf(xc[0], xc[1])
        Q = jnp.asarray(Q0)
        A = s.initial_aux(Q, A)
        out[nu] = np.asarray(s.run(Q, A)[1])
    L = (out[1.0] - out[0.0]) / dt
    res = {}
    for k in ("q_x_0", "q_y_0"):
        Lk = L[names.index(k)]
        D, S_ = SMOP[k + " diff"](xc[0], xc[1]), SMOP[k + " src"](xc[0], xc[1])
        W = SMOP[k + " swap"](xc[0], xc[1])
        Dg = SMOP[k + " diag"](xc[0], xc[1])
        Pk = CF["div(h tau)"][k](xc[0], xc[1]); Hk = CF["h div(tau)"][k](xc[0], xc[1])
        res[f"{k}: SM(full) vs div(h tau)"] = (np.array([]), float(np.abs(D + S_ - Pk).max() / np.abs(Pk).max()))
        res[f"{k}: SM(full) vs h div(tau)"] = (np.array([]), float(np.abs(D + S_ - Hk).max() / np.abs(Hk).max()))
        for lab, ref in (("SM diagonal-index diff + src", Dg + S_),):
            ref = ref * np.ones_like(Lk)
            res[f"{k} vs {lab}"] = (np.array([]), float(np.abs(Lk - ref).max() / np.abs(ref).max()))
        Lk = L[names.index(k)]
        B = np.stack([CF[n][k](xc[0], xc[1]) for n in ("div(h tau)", "h div(tau)", "div(h grad u)", "lap(q)")], 1)
        c, *_ = np.linalg.lstsq(B, Lk, rcond=None)
        res[k] = (c, float(np.abs(B @ c - Lk).max() / np.abs(Lk).max()))
        c1 = float(Lk @ B[:, 0] / (B[:, 0] @ B[:, 0]))
        res[k + " scale-only"] = (np.array([c1]), float(np.abs(c1 * B[:, 0] - Lk).max() / np.abs(Lk).max()))
    return res


for n in (32, 64):
    r = operator(n)
    for k, (c, e) in r.items():
        print(f"nx {n} {k}: coeffs [div(h tau), h div(tau), div(h grad u), lap q] = {np.round(c, 3)}  resid {e:.2e}", flush=True)
