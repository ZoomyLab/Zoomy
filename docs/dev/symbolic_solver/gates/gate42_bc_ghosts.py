"""Gate 42: boundary ghosts through the runtime blocks (ANALYSIS 36/37).

The templates take the BC GHOST kernel (``nsm.call.boundary_ghost_state``,
decision D5: the value at the mirror image m = c + 2 d n of the boundary
cell centre c, d = (x_f - c).n) for every ghost consumer, and the FACE kernel
(``nsm.call.boundary_conditions``) for the Riemann problem only.  This gate
takes REAL BC objects on SME(0) (b, h, q) through the JAX runtime blocks the
printed solver calls (``boundary_states``, ``gradient``, ``derivative_aux``
with ghosts, ``boundary_traces``) on 1-D lines, 2-D quads and 2-D triangles
(quads cut along alternating diagonals), with a LINEAR field consistent with
the BC, and checks:

  ghost   the ghost value itself, against an independent numpy formula:
          Dirichlet 2 v(x_foot) - u_c, Neumann u_c + 2 d g, Extrapolation u_c,
          Wall q_n -> -q_n (q_t, h kept), Periodic the partner cell's value
  grad    the degree-1 LSQ gradient (``gradient``) at boundary cells = the
          field's exact gradient
  daux    the model's derivative aux rows of the BC'd state rows
          (``derivative_aux(spec, Q, Qaux, mesh, Qg, rows)``) at boundary
          cells = the exact derivatives
  trace   the Riemann outer state (``boundary_traces`` with the FACE kernel)
          of the exact face trace = that trace (a consistent field has no
          jump at the face)
Sensitivity (must be > tol, i.e. the gate sees the old bug): ``grad`` with
the FACE kernel's value used as the ghost (the pre-fix templates) -- column
"face-as-ghost"; ``daux`` with the pre-fix lift 2 Qb - Q of the face
kernel's value (walls: -3 q_n) -- "old-daux-lift"; ``trace`` with the GHOST
kernel at the face -- "ghost-at-face" (Neumann: u + 2 d g, not the face
value).

Cases (domain [0,1]^d; u = the BC'd row):
  dir-const  Dirichlet(value) on left/right, u = 0.7 + 2 x; others Lambda
  dir-pos    Lambda (position-dependent Dirichlet) on every side,
             u = 1 + 2 x + 3 y
  neumann    Flux(gradient = du/dn) on every side, u = 1 + 2 x + 3 y
  extrap     Extrapolation on left/right, u = 1 + 3 y; others Lambda
  wall       Wall on left (q_n odd, q_t and h even): q_x = 2 x, q_y = 1 + 3 y,
             h = 1 + 3 y; others Lambda on every row
  periodic   Periodic left <-> right, u = 1 + 3 y; others Lambda
(1-D: y-terms dropped; extrap / periodic then a constant.)

Run:  source env_new.sh; python gate42_bc_ghosts.py   (exit 0 iff all PASS)
GATE42_STORE=<file.json>: also store the errors (gate42_figure.py plots them).
"""
import json
import os
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import zoomy_core.model.boundary_conditions as BC
from zoomy_core.mesh import LSQMesh, ensure_lsq_mesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
from gate41_diffusion_bc import triangles

TOL = 1e-10
NX = {"1d": 8, "quad": 6, "tri": 6}
TAGS = {1: ("left", "right"), 2: ("left", "right", "bottom", "top")}
NORMAL = {"left": (-1.0, 0.0), "right": (1.0, 0.0),
          "bottom": (0.0, -1.0), "top": (0.0, 1.0)}


# ── the cases: BCs + the linear field per row ───────────────────────────────
def lin(c0, cx, cy):
    """u = c0 + cx x + cy y as (numpy f(X), gradient, sympy f(X))."""
    return (lambda X: c0 + cx * X[0] + (cy * X[1] if len(X) > 1 else 0.0),
            np.array([cx, cy]),
            lambda t, X, dX, Q, Qaux, p, n: c0 + cx * X[0] + (cy * X[1] if len(X) > 1 else 0))


def case(name, dim, names):
    """(BCs, {row: (f, grad)}, BC'd rows, the tested tags)."""
    h = names.index("h")
    q = [i for i, s in enumerate(names) if s.startswith("q")]
    tags = TAGS[dim]
    cy = 3.0 if dim == 2 else 0.0
    if name == "dir-const":
        f = lin(0.7, 2.0, 0.0)
        bcs = [BC.Dirichlet("left", on="h", value=0.7), BC.Dirichlet("right", on="h", value=2.7)]
        bcs += [BC.Lambda(tag=t, prescribe_fields={h: f[2]}) for t in tags[2:]]
        return bcs, {h: f}, (h,), ("left", "right")
    if name == "dir-pos":
        f = lin(1.0, 2.0, cy)
        return ([BC.Lambda(tag=t, prescribe_fields={h: f[2]}) for t in tags],
                {h: f}, (h,), tags)
    if name == "neumann":
        f = lin(1.0, 2.0, cy)
        bcs = [BC.Flux(t, on="h", gradient=float(np.dot(f[1][:dim], NORMAL[t][:dim])))
               for t in tags]
        return bcs, {h: f}, (h,), tags
    if name == "extrap":
        f = lin(1.0, 0.0, cy)
        bcs = [BC.Extrapolation(tag=t) for t in ("left", "right")]
        bcs += [BC.Lambda(tag=t, prescribe_fields={h: f[2]}) for t in tags[2:]]
        return bcs, {h: f}, (h,), ("left", "right")
    if name == "wall":
        fields = {h: lin(1.0, 0.0, cy), q[0]: lin(0.0, 2.0, 0.0)}
        if dim == 2:
            fields[q[1]] = lin(1.0, 0.0, 3.0)
        bcs = [BC.Wall(tag="left")]
        bcs += [BC.Lambda(tag=t, prescribe_fields={r: f[2] for r, f in fields.items()})
                for t in tags[1:]]
        return bcs, fields, tuple(fields), ("left",)
    if name == "periodic":
        f = lin(1.0, 0.0, cy)
        bcs = [BC.Periodic(tag="left", periodic_to_physical_tag="right"),
               BC.Periodic(tag="right", periodic_to_physical_tag="left")]
        bcs += [BC.Lambda(tag=t, prescribe_fields={h: f[2]}) for t in tags[2:]]
        return bcs, {h: f}, (h,), ("left", "right")
    raise ValueError(name)


CASES = ("dir-const", "dir-pos", "neumann", "extrap", "wall", "periodic")


NAMES = {1: ["b", "h", "q_0"], 2: ["b", "h", "q_x_0", "q_y_0"]}     # SME(0)


def build(kind, name):
    dim = 1 if kind == "1d" else 2
    names = NAMES[dim]
    bcs, fields, rows, tested = case(name, dim, names)
    sm = SystemModel.from_model(SME(level=0, dimension=dim + 1,
                                    closures=[Newtonian(), StressFree()],
                                    boundary_conditions=BC.BoundaryConditions(bcs)))
    assert [str(v) for v in sm.state] == names, sm.state
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2, limiter=Limiter.none))
    if kind == "1d":
        mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=NX[kind])
    elif kind == "quad":
        mesh = LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=NX[kind], ny=NX[kind])
    else:
        mesh = ensure_lsq_mesh(triangles(NX[kind]), nsm)
    s = SymbolicHyperbolicSolverJax(time_end=0.1, dt=1e-3, cfl_check_every=0)
    s.setup(mesh, nsm)
    return s, dim, names, fields, rows, tested


# ── one (mesh, case) ────────────────────────────────────────────────────────
def check(kind, name):
    s, dim, names, fields, rows, tested = build(kind, name)
    m, g, mnp = s.module, s.mesh, s.mesh_np
    mis = [tuple(int(k) for k in mi) for mi in s.mesh_j.lsq_monomial_multi_index]
    degree = max(sum(mi) for mi in mis)
    nc = int(mnp.n_inner_cells)
    cc = np.asarray(g.cell_centers)[:dim, :nc]
    fcen = np.asarray(g.face_centers)[:, :dim].T                      # (dim, nf)
    nrm = np.asarray(g.face_normals)[:dim]
    bf_face = np.asarray(mnp.boundary_face_face_indices, int)
    bf_cell = np.asarray(mnp.boundary_face_cells, int)               # periodic: partner
    inner = np.asarray(mnp.face_cells[0], int)[bf_face]
    btag = np.asarray(mnp.boundary_conditions_sorted_names)[
        np.asarray(mnp.boundary_face_function_numbers, int)[:len(bf_face)]]
    n_b = nrm[:, bf_face]
    c_b = cc[:, inner]
    dn = ((fcen[:, bf_face] - c_b) * n_b).sum(0)
    foot = c_b + dn * n_b
    periodic = bf_cell != inner
    on_tested = np.isin(btag, tested)

    # the state: the case's linear fields, every other row a constant
    Q = np.zeros((len(names), nc))
    for r in range(len(names)):
        Q[r] = fields[r][0](cc) if r in fields else (0.0 if names[r] == "b" else 0.5)
    Qj = jnp.asarray(Q)
    Qaux = jnp.zeros((len(s.nsm.aux_state), nc))
    p, t0 = s.parameters, 0.0
    ghost_k, face_k = s.rt.boundary_ghost_state, s.rt.boundary_conditions

    Qg = np.asarray(m["boundary_states"](ghost_k, Qj, Qaux, p, t0, g)[0])
    Qf = np.asarray(m["boundary_states"](face_k, Qj, Qaux, p, t0, g)[0])   # the old ghosts

    # ghost: the independent formula per BC kind
    exp = Q[:, inner].copy()
    for r, (f, grad, _) in fields.items():
        if name in ("dir-const", "dir-pos"):
            exp[r] = np.where(on_tested, 2 * f(foot) - Q[r, inner], 2 * f(foot) - Q[r, inner])
        elif name == "neumann":
            exp[r] = Q[r, inner] + 2 * dn * (grad[:dim] @ n_b)
        else:                             # Lambda off the tested tags
            exp[r] = np.where(on_tested, Q[r, inner], 2 * f(foot) - Q[r, inner])
    if name == "wall":
        qr = [i for i, sname in enumerate(names) if sname.startswith("q")]
        qv = Q[qr][:, inner]
        qn = (qv * n_b).sum(0)
        refl = qv - 2 * qn * n_b
        for k, r in enumerate(qr):
            exp[r] = np.where(on_tested, refl[k], exp[r])
    if name == "periodic":
        exp[:, periodic] = Q[:, bf_cell[periodic]]
    scale = max(np.abs(Q).max(), 1.0)
    e_ghost = np.abs(Qg - exp)[list(rows)].max() / scale

    # gradient and derivative aux at boundary cells (``tcells``: those whose
    # every boundary face carries the BC under test)
    bcells = np.unique(inner)
    tcells = np.setdiff1d(np.unique(inner[on_tested]), inner[~on_tested])
    G = np.asarray(m["gradient"](Qj, jnp.asarray(Qg), g))              # (n, dim, nc)
    Gf = np.asarray(m["gradient"](Qj, jnp.asarray(Qf), g))
    exact = {r: f[1][:dim] for r, f in fields.items()}
    e_grad = max(np.abs(G[r, :, bcells] - exact[r]).max() for r in rows) / scale
    e_face = max(np.abs(Gf[r, :, bcells] - exact[r]).max() for r in rows) / scale
    e_face_t = max(np.abs(Gf[r, :, tcells] - exact[r]).max() for r in rows) / scale
    spec = [e for e in s.rt.aux_derivatives
            if e["kind"] == "derivative" and e["target_kind"] == "state"
            and e["state_index"] in rows and sum(e["multi_index"]) == 1]
    A = np.asarray(m["derivative_aux"](spec, Qj, Qaux, g, jnp.asarray(Qg), rows))
    # ... and with the pre-fix lift of the face kernel's value, 2 Qb - Q
    A_old = np.asarray(m["derivative_aux"](spec, Qj, Qaux, g,
                                           jnp.asarray(2 * Qf - Q[:, bf_cell]), rows))
    e_daux = e_lift = e_lift_t = 0.0
    for e in spec:
        axis = list(e["multi_index"]).index(1)
        ex = exact[e["state_index"]][axis]
        e_daux = max(e_daux, np.abs(A[e["row"], bcells] - ex).max() / scale)
        e_lift = max(e_lift, np.abs(A_old[e["row"], bcells] - ex).max() / scale)
        e_lift_t = max(e_lift_t, np.abs(A_old[e["row"], tcells] - ex).max() / scale)

    # the Riemann outer state of the exact face trace (FACE kernel), and the
    # ghost kernel misused there
    QL = np.zeros((len(names), int(mnp.n_faces)))
    for r in range(len(names)):
        QL[r] = fields[r][0](fcen) if r in fields else Q[r, 0]
    QLj = jnp.asarray(QL)
    QR = np.asarray(m["boundary_traces"](face_k, Qj, QLj, QLj, Qaux, p, t0, g))
    QRg = np.asarray(m["boundary_traces"](ghost_k, Qj, QLj, QLj, Qaux, p, t0, g))
    e_trace = np.abs(QR - QL)[:, bf_face].max() / scale
    e_trace_g = np.abs(QRg - QL)[:, bf_face].max() / scale
    return dict(degree=degree, nbf=int(on_tested.sum()), nb=len(bcells), ghost=e_ghost,
                grad=e_grad, daux=e_daux, trace=e_trace, n_daux=len(spec),
                face_as_ghost=e_face, ghost_at_face=e_trace_g, lift=e_lift,
                face_as_ghost_tested=e_face_t, lift_tested=e_lift_t,
                grad_tested=max(np.abs(G[r, :, tcells] - exact[r]).max() for r in rows) / scale)


def main():
    t0 = time.time()
    n_pass = n_fail = 0
    fails, store = [], {}
    for kind in ("1d", "quad", "tri"):
        for name in CASES:
            r = check(kind, name)
            store[f"{kind} {name}"] = {k: float(v) for k, v in r.items()}
            ok = all(r[k] <= TOL for k in ("ghost", "grad", "daux", "trace")) and r["degree"] == 1
            n_pass += ok
            n_fail += not ok
            if not ok:
                fails.append(f"{kind} {name}")
            # sensitivity: the old semantics must be visible where they differ
            sens = []
            if name in ("dir-const", "dir-pos", "neumann"):
                sens.append(f"face-as-ghost {r['face_as_ghost']:.1e}"
                            + ("" if r["face_as_ghost"] > TOL else " (NOT SEEN)"))
            if name in ("wall", "neumann"):
                sens.append(f"old-daux-lift {r['lift']:.1e}"
                            + ("" if r["lift"] > TOL else " (NOT SEEN)"))
            if name == "neumann":
                sens.append(f"ghost-at-face {r['ghost_at_face']:.1e}"
                            + ("" if r["ghost_at_face"] > TOL else " (NOT SEEN)"))
            print(f"gate42 {kind:4s} {name:9s} deg {r['degree']}  bfaces {r['nbf']:2d}  "
                  f"ghost {r['ghost']:.1e}  grad {r['grad']:.1e}  daux({r['n_daux']}) "
                  f"{r['daux']:.1e}  trace {r['trace']:.1e}  {'PASS' if ok else 'FAIL'}"
                  + (f"   [{'; '.join(sens)}]" if sens else ""), flush=True)
    for f in fails:
        print("FAIL:", f)
    if os.environ.get("GATE42_STORE"):
        with open(os.environ["GATE42_STORE"], "w") as f:
            json.dump(store, f, indent=1)
    print(f"SUMMARY gate42: PASS={n_pass} FAIL={n_fail} (tol {TOL:g}) runtime "
          f"{time.time() - t0:.0f}s -> {'OK' if not n_fail else 'FAILED'}", flush=True)
    return 0 if not n_fail else 1


if __name__ == "__main__":
    sys.exit(main())
