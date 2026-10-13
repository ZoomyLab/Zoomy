"""Gate 45: the in-plane viscous operator of SME(0) + Newtonian (stress=Stress.full), measured on the symbolic
solvers, against (a) the SystemModel's own operator (diffusion_matrix A[i, j, a, b] with its
mixed-index entries a != b, plus the nu-part of the source) and (b) the exact div(h tau'),
tau' = nu (grad u + grad u^T) + 2 nu (div u) I (the in-plane stress minus the viscous normal
stress tau_zz = -2 nu div u; Marche 2007 eq. (5.19)), u = q / h.

Symbolic part: with generic smooth fields h, q_x, q_y, b of (x, y), the SystemModel operator minus
div(h tau') and minus gate 46's hand-written operator, flat and varying bed.

Periodic smooth h, qx, qy on [0,1]^2; one implicit-Lie step of dt with nu = 1 and nu = 0;
(Q_nu - Q_0) / dt on the momentum rows is the discrete viscous operator (O(dt) accurate).

Findings (ANALYSIS 45, 45c, 45d): with two-point fluxes only, the discrete operator was the
SystemModel operator WITHOUT its mixed-index entries (16 % / 38 % off the full one); the face
gradient (mean of the cell LSQ gradients + normal correction) fixed that.  The derivation gave
h div(tau) until the bed / surface traction carried the in-plane stress (45e) and div(h tau)
until tau_zz entered the pressure (45f); now the SystemModel operator is div(h tau') exactly
and the discrete one converges to it at order 2.

    python gate45_inplane_viscous_operator.py [jax|foam|amrex ...]
"""
import os
import sys

import numpy as np
import sympy as sp

import common  # noqa: F401  (x64)
import zoomy_core.model.boundary_conditions as BC
from zoomy_core.mesh import LSQMesh, ensure_lsq_mesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import ManningFriction, Newtonian, StressFree
from zoomy_core.model.models.equations import Stress
from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel import SystemModel
from zoomy_core.symsolver.settings import Imex

WORK = os.environ.get("ZOOMY_GATE_WORK", "/tmp/zoomy_gates")
NXS = (16, 32, 64)
DT = 1e-7
TIGHT = dict(implicit_tol=1e-12, gmres_tol=1e-13, implicit_maxiter=10, gmres_maxiter=60)
ROWS = ("q_x_0", "q_y_0")

x, y = sp.symbols("x y")
tp = 2 * sp.pi
H = 1 + sp.Rational(1, 10) * sp.sin(tp * x) * sp.cos(tp * y)
QX = sp.Rational(1, 10) * sp.cos(tp * x) * sp.sin(tp * y) + sp.Rational(1, 20)
QY = sp.Rational(1, 20) * sp.sin(tp * (x + y))
u, v = QX / H, QY / H
Hf, QXf, QYf = (sp.lambdify((x, y), e, "numpy") for e in (H, QX, QY))
gu, gv = (sp.diff(u, x), sp.diff(u, y)), (sp.diff(v, x), sp.diff(v, y))
txx, txy, tyy = 2 * gu[0], gu[1] + gv[0], 2 * gv[1]
dv = gu[0] + gv[1]                                    # tau' = tau - tau_zz I, tau_zz = -2 div u
PHYS = {"div(h tau')": {"q_x_0": sp.diff(H * (txx + 2 * dv), x) + sp.diff(H * txy, y),
                        "q_y_0": sp.diff(H * txy, x) + sp.diff(H * (tyy + 2 * dv), y)},
        "div(h tau)": {"q_x_0": sp.diff(H * txx, x) + sp.diff(H * txy, y),
                       "q_y_0": sp.diff(H * txy, x) + sp.diff(H * tyy, y)}}


def model(bcs):
    """SME(0) + Newtonian with the full stress and the boundary conditions ``bcs``."""
    return SystemModel.from_model(SME(level=0, dimension=3, boundary_conditions=BC.BoundaryConditions(bcs),
                                      stress=Stress.full,
                                      closures=[Newtonian(), ManningFriction(), StressFree()]))


def sm_operator():
    """``{row: (full, diagonal-index)}`` lambdas of the SystemModel operator at nu = 1: the
    diffusion ``d_a(A_ijab d_b Q_j)`` (all a, b / only a = b) plus the nu-part of the source, the
    aux derivatives exact.
    """
    sm = model([BC.Wall(tag=t) for t in ("left", "right", "bottom", "top")])
    nu, St = sm.parameters.nu, list(sm.state)
    F = {"h": H, "q_x_0": QX, "q_y_0": QY, "b": sp.S.Zero}

    def tof(e):
        return e.subs({a: F[str(a)] for a in e.free_symbols if str(a) in F}).subs(nu, 1)
    aux = {}
    for a in sm.aux_state:
        n_ = str(a)
        if n_[-2:] in ("_x", "_y"):
            base = sp.sympify(n_[:-2], locals={**{k: sp.Symbol(k) for k in F}, "nu": nu})
            aux[a] = sp.diff(tof(base), x if n_[-1] == "x" else y)
        elif n_.startswith("d") and n_[-2:] in ("dx", "dy"):     # dhdx, dq_x0dx, ...
            base = {"q_x0": "q_x_0", "q_y0": "q_y_0"}.get(n_[1:-2], n_[1:-2])
            aux[a] = sp.diff(F[base], x if n_[-1] == "x" else y)
    missing = [str(a) for a in sm.aux_state if a not in aux]
    assert not missing, f"unmapped aux {missing}"
    A, out = sm.diffusion_matrix, {}
    for i, k in ((2, "q_x_0"), (3, "q_y_0")):
        src = sp.expand(sm.source[i, 0])
        S = tof(sum((t for t in sp.Add.make_args(src) if t.has(nu)), sp.S.Zero).subs(aux))

        def div(pairs):
            return sum(sp.diff(tof(A[i, j, a, b]) * sp.diff(F[str(St[j])], (x, y)[b]), (x, y)[a])
                       for j in range(len(St)) for a, b in pairs)
        out[k] = tuple(sp.lambdify((x, y), e, "numpy")
                       for e in (div([(a, b) for a in range(2) for b in range(2)]) + S,
                                 div([(0, 0), (1, 1)]) + S))
    return out


def symbolic_check():
    """``{name: difference}`` of the SystemModel operator (generic smooth fields, aux = their
    exact derivatives) against div(h tau') (varying bed) and gate 46 (flat and varying bed);
    every difference must be 0.
    """
    import gate46_viscous_swe_reference as G46
    sm = model([BC.Wall(tag=t) for t in ("left", "right", "bottom", "top")])
    nu, St = sm.parameters.nu, [str(s_) for s_ in sm.state]
    X = (x, y)
    F = {n_: sp.Function("F_" + n_)(x, y) for n_ in St}

    def tof(e):
        return e.subs({a: F[str(a)] for a in e.free_symbols if str(a) in F})
    aux = {e["aux_symbol"]: sp.diff(F[e["target_name"]], *[v for v, k in zip(X, e["multi_index"])
                                                             for _ in range(k)])
           for e in sm.aux_registry if e.get("kind") == "derivative"}
    assert all(a in aux for a in sm.aux_state), "non-derivative aux"
    A, H = sm.diffusion_matrix, F["h"]
    U = [F["q_x_0"] / H, F["q_y_0"] / H]
    divU = sum(sp.diff(U[e], X[e]) for e in range(2))
    tau = [[nu * (sp.diff(U[e], X[d]) + sp.diff(U[d], X[e])) + (2 * nu * divU if d == e else 0)
            for e in range(2)] for d in range(2)]
    sm46 = G46.viscous_swe()
    sub46 = {G46.h: H, G46.qx: F["q_x_0"], G46.qy: F["q_y_0"], G46.nu: nu}
    Q46 = [H, F["q_x_0"], F["q_y_0"]]
    out = {}
    for d, k in enumerate(ROWS):
        i = St.index(k)
        src = sp.expand(sm.source[i, 0])
        op = tof(sum((t_ for t_ in sp.Add.make_args(src) if t_.has(nu)), sp.S.Zero).subs(aux))
        op += sum(sp.diff(tof(sp.sympify(A[i, j, a, b])) * sp.diff(F[St[j]], X[b]), X[a])
                  for j in range(len(St)) for a in range(2) for b in range(2))
        op46 = sum(sp.diff(sp.sympify(sm46.diffusion_matrix[d + 1, j, a, b]).subs(sub46)
                           * sp.diff(Q46[j], X[b]), X[a])
                   for j in range(3) for a in range(2) for b in range(2))
        ref = sum(sp.diff(H * tau[d][e], X[e]) for e in range(2))
        out[f"{k} SM - div(h tau'), varying bed"] = sp.simplify(sp.expand(op - ref))
        out[f"{k} SM - gate 46, varying bed"] = sp.simplify(sp.expand(op - op46))
        out[f"{k} SM - gate 46, flat bed"] = sp.simplify(sp.expand((op - op46).subs(F["b"], 0).doit()))
    return out


def setup(nx, nu):
    """(mesh, nsm, Q0 in mesh order) of the periodic case at viscosity ``nu``."""
    sides = (("left", "right"), ("right", "left"), ("bottom", "top"), ("top", "bottom"))
    sm = model([BC.Periodic(tag=a, periodic_to_physical_tag=b) for a, b in sides])
    sm.parameter_values.nu = nu
    for k in ("n", "n_m"):
        if hasattr(sm.parameter_values, k):
            setattr(sm.parameter_values, k, 0.0)
    nsm = NumericalSystemModel.from_system_model(sm, reconstruction=ReconstructionSpec(order=2, limiter=Limiter.none))
    mesh = ensure_lsq_mesh(LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=nx, ny=nx), nsm)
    names = [str(s) for s in nsm.state]
    xc = np.asarray(mesh.cell_centers)[:2, :int(mesh.n_inner_cells)]
    Q0 = np.zeros((len(names), xc.shape[1]))
    Q0[names.index("h")] = Hf(xc[0], xc[1])
    Q0[names.index("q_x_0")] = QXf(xc[0], xc[1])
    Q0[names.index("q_y_0")] = QYf(xc[0], xc[1])
    return mesh, nsm, Q0


def step(backend, nx, nu, work=None):
    """One implicit-Lie step of ``DT`` on ``backend``: ``(Q, names, xc)`` in mesh order."""
    mesh, nsm, Q0 = setup(nx, nu)
    names = [str(s) for s in nsm.state]
    xc = np.asarray(mesh.cell_centers)[:2, :Q0.shape[1]]
    kw = dict(time_end=DT, dt=DT, cfl_check_every=0, imex=Imex.lie, mood=False, **TIGHT)
    if backend in ("jax", "numpy"):
        import jax.numpy as jnp
        from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
        if backend == "numpy":
            from zoomy_core.symsolver.numpy_backend import (
                SymbolicHyperbolicSolverNumpy as SymbolicHyperbolicSolverJax)
            jnp = np
        s = SymbolicHyperbolicSolverJax(**kw)
        _, A = s.setup(mesh, nsm)
        Q = jnp.asarray(Q0)
        return np.asarray(s.run(Q, s.initial_aux(Q, A))[1]), names, xc
    if backend == "foam":
        from zoomy_foam.symsolver import SymbolicHyperbolicSolverFoam
        s = SymbolicHyperbolicSolverFoam(**kw)
    else:
        from zoomy_amrex.symsolver import SymbolicHyperbolicSolverAmrex
        s = SymbolicHyperbolicSolverAmrex(max_grid_size=8, **kw)
    _, Q = s.run(mesh, nsm, Q0, f"{work or WORK}/gate45_{backend}_n{nx}_nu{nu:g}")
    return np.asarray(Q), names, xc


def operator(backend, nx, work=None):
    """The discrete operator ``{row: L}`` on ``backend`` and the cell centres."""
    Q1, names, xc = step(backend, nx, 1.0, work)
    Q0, _, _ = step(backend, nx, 0.0, work)
    return {k: (Q1 - Q0)[names.index(k)] / DT for k in ROWS}, xc


EXACT = {k: sp.lambdify((x, y), e, "numpy") for k, e in PHYS["div(h tau')"].items()}


def errors(L, xc, SM):
    """Max errors relative to max |operator| per row: vs the exact div(h tau') (``exact``), the
    full SystemModel operator (``full``) and its diagonal-index part (``diag``).
    """
    out = {}
    for k in ROWS:
        full, diag = (f(xc[0], xc[1]) * np.ones(xc.shape[1]) for f in SM[k])
        ex = EXACT[k](xc[0], xc[1]) * np.ones(xc.shape[1])
        out[f"{k} exact"] = float(np.abs(L[k] - ex).max() / np.abs(ex).max())
        out[f"{k} full"] = float(np.abs(L[k] - full).max() / np.abs(full).max())
        out[f"{k} diag"] = float(np.abs(L[k] - diag).max() / np.abs(diag).max())
    return out


def main(backends):
    for name, dd in symbolic_check().items():  # the SystemModel vs the physics, symbolically
        print(f"gate45 symbolic {name}: {dd}", flush=True)
    SM = sm_operator()
    xs = np.random.default_rng(0).random((2, 200))
    for name, c in PHYS.items():
        for k in ROWS:
            P, full = sp.lambdify((x, y), c[k], "numpy")(*xs), SM[k][0](*xs)
            print(f"gate45 SM(full) {k} vs {name}: {np.abs(full - P).max() / np.abs(P).max():.2e}")
    res = {}
    for b in backends:
        for nx in NXS:
            L, xc = operator(b, nx)
            res[b, nx] = (L, errors(L, xc, SM))
            print(f"gate45 {b} nx {nx}: " + "  ".join(f"{k} {e:.2e}" for k, e in res[b, nx][1].items()),
                  flush=True)
        for k in ROWS:
            for ref in ("exact", "full"):
                e = [res[b, nx][1][f"{k} {ref}"] for nx in NXS]
                print(f"gate45 {b} {k} vs {"div(h tau')" if ref == 'exact' else 'SM(full)'}: "
                      + " / ".join(f"{v:.2e}" for v in e) + "  orders "
                      + " / ".join(f"{np.log2(e[i] / e[i + 1]):.2f}" for i in range(len(e) - 1)),
                      flush=True)
        if b != "jax" and ("jax", NXS[-1]) in res:
            for nx in NXS:
                d = max(float(np.abs(res[b, nx][0][k] - res["jax", nx][0][k]).max()
                              / np.abs(res["jax", nx][0][k]).max()) for k in ROWS)
                print(f"gate45 {b} nx {nx}: max rel |L_{b} - L_jax| {d:.2e}", flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1:] or ["jax"])
