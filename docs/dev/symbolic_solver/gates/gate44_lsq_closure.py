"""Gate 44: convergence of the LSQ derivatives with the boundary closure
(ANALYSIS 40), through the JAX runtime.

Real BC objects on SME(0), the BC'd row h = f(x, y) (a smooth trigonometric
field, the prototype's):
  dir   Lambda on every side: h = f at the face (position-dependent Dirichlet)
  neu   Flux on every side: dh/dn = n . grad f (position-dependent Neumann)
on quads (create_2d) and triangles (gate16 ``tri_bump``), domain (0, 10)^2,
nx = 12, 24, 48; fit degree p = 2, 3 (the NSM declares a p-th derivative).
The ghosts come from the printed ghost kernel (``boundary_ghosts``), every
derivative of h of order 1..p from the runtime's ``derivative_aux``:
  closure  the classes the topology gave h (Dirichlet / Neumann rows close
           by a constraint at the foot point)
  ghost    the same call with no class (the ghost as a fit sample: the
           pre-closure fit) -- for comparison
Errors: max |D - exact| per derivative order k over the boundary cells (a
boundary face) and the interior cells; order = log2 of the last two ratios.
PASS iff the closure's boundary order of every k >= p + 1 - k - 0.2 and its
interior order the same (the interior fit is unchanged).

Arguments: none (quads only, the quick default), ``tri``, or ``all`` (quads +
triangles).  The SME derivation is done once per BC kind (``system_for``).

GATE44_STORE=<file.json>: store the error tables (deliverable figure).
"""
import functools
import json
import os
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import sympy as sp
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import zoomy_core.model.boundary_conditions as BC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
from gate16_parallel import tri_bump

NX = (12, 24, 48)
TAGS = ("left", "right", "bottom", "top")
SLACK = 0.2
K = 2 * np.pi / 10


def field(x, y, m):
    """The prototype's field and its derivatives (m: sympy or numpy)."""
    return (m.sin(0.9 * K * x + 0.3) * m.cos(0.7 * K * y + 0.2)
            + 0.2 * m.cos(0.5 * K * x * 1.3))


@functools.lru_cache(maxsize=None)
def symbols():
    sm = SystemModel.from_model(SME(level=0, dimension=3,
                                    closures=[Newtonian(), StressFree()]))
    return sm.position_struct().get_list()[:2], sm.normal.get_list()[:2]


@functools.lru_cache(maxsize=None)
def system_for(bc):
    """The derived SME(0) SystemModel with the ``bc`` BCs -- once per BC kind
    (the SME derivation is ~5 s; the fit degree p is the NSM's, not its)."""
    (x, y), (n0, n1) = symbols()
    f = field(x, y, sp)
    if bc == "dir":
        bcs = [BC.Lambda(tag=t, prescribe_fields={1: lambda t_, X, *a: field(X[0], X[1], sp)})
               for t in TAGS]
    else:
        g = n0 * sp.diff(f, x) + n1 * sp.diff(f, y)
        bcs = [BC.Flux(t, on="h", gradient=g) for t in TAGS]
    return SystemModel.from_model(SME(level=0, dimension=3,
                                      closures=[Newtonian(), StressFree()],
                                      boundary_conditions=BC.BoundaryConditions(bcs)))


def nsm_for(bc, p):
    nsm = NumericalSystemModel.from_system_model(
        system_for(bc), reconstruction=ReconstructionSpec(order=1))
    # the derivative a source of order p would declare -> LSQ degree p
    nsm.source_derivative_specs = [SimpleNamespace(axes=("x",) * p)]
    return nsm, symbols()[0]


def mesh_for(kind, nx):
    if kind == "quad":
        return LSQMesh.create_2d(domain=(0, 10, 0, 10), nx=nx, ny=nx)
    return tri_bump(2, nx=nx)[0]


def errors(kind, bc, p, nx):
    nsm, (x, y) = nsm_for(bc, p)
    s = SymbolicHyperbolicSolverJax(time_end=0.1, dt=1e-3, cfl_check_every=0)
    s.setup(mesh_for(kind, nx), nsm)
    m, g, mnp, topo = s.module, s.mesh, s.mesh_np, s.topo
    nc = int(mnp.n_inner_cells)
    cc = np.asarray(g.cell_centers)[:2, :nc]
    names = [str(v) for v in s.nsm.state]
    h = names.index("h")
    Q = np.zeros((len(names), nc))
    Q[h] = field(cc[0], cc[1], np)
    Q[names.index("b")] = 0.0
    Qj = jnp.asarray(Q)
    Qaux = jnp.zeros((len(s.nsm.aux_state), nc))
    Qg = m["boundary_ghosts"](s.rt.boundary_ghost_state, Qj, Qaux, s.parameters, 0.0, g)
    mis = [tuple(int(k) for k in mi) for mi in topo.lsq_monomial_multi_index]
    cls = topo.lsq_closure_class[("nsm", "state")][h]
    fx = sp.lambdify((x, y), field(x, y, sp), "numpy")
    exact = {mi: np.broadcast_to(sp.lambdify((x, y), sp.diff(field(x, y, sp), x, mi[0], y, mi[1]),
                                             "numpy")(cc[0], cc[1]), (nc,)) for mi in mis}
    bnd = np.zeros(nc, bool)
    bfc = np.asarray(mnp.boundary_face_cells)
    bnd[bfc[bfc < nc]] = True
    out = {}
    for method, c in (("closure", cls), ("ghost", -1)):
        spec = [dict(kind="derivative", target_kind="state", state_index=h,
                     multi_index=mi, row=r, closure=c) for r, mi in enumerate(mis)]
        A = np.asarray(m["derivative_aux"](spec, Qj, jnp.zeros((len(mis), nc)), g, Qg, (h,)))
        for k in range(1, p + 1):
            rows = [r for r, mi in enumerate(mis) if sum(mi) == k]
            e = np.max([np.abs(A[r] - exact[mis[r]]) for r in rows], axis=0)
            out[(method, k, "boundary")] = float(e[bnd].max())
            out[(method, k, "interior")] = float(e[~bnd].max())
    assert cls >= 0, "h has no closure class"
    return out


def main():
    t0 = time.time()
    store, ok_all, lines = {}, True, []
    # quick default: the quads (~90 s); `tri` the triangles, `all` both (~3 min)
    kinds = sys.argv[1:] or ["quad"]
    kinds = [k for a in kinds for k in (["quad", "tri"] if a == "all" else [a])]
    for kind in kinds:
        for p in (2, 3):
            for bc in ("dir", "neu"):
                E = [errors(kind, bc, p, nx) for nx in NX]
                for method in ("closure", "ghost"):
                    parts, ok = [], True
                    for where in ("boundary", "interior"):
                        for k in range(1, p + 1):
                            e = [Ei[(method, k, where)] for Ei in E]
                            o = np.log2(e[-2] / e[-1])
                            need = p + 1 - k - SLACK
                            good = o >= need
                            if method == "closure":
                                ok &= good
                            store[f"{kind} p{p} {bc} {method} {where} d{k}"] = e
                            parts.append(f"{where[0]}d{k} {e[0]:.1e}>{e[-1]:.1e} o{o:4.1f}"
                                         + ("" if good or method != "closure" else "!"))
                    line = (f"gate44 {kind:4s} p={p} {bc} {method:7s} " + "  ".join(parts)
                            + (f"  {'PASS' if ok else 'FAIL'}" if method == "closure" else ""))
                    print(line, flush=True)
                    ok_all &= ok
    if os.environ.get("GATE44_STORE"):
        with open(os.environ["GATE44_STORE"], "w") as f:
            json.dump({"nx": NX, "errors": store}, f, indent=1)
    print(f"SUMMARY gate44: {'OK' if ok_all else 'FAILED'} (closure boundary/interior order "
          f">= p+1-k-{SLACK}) runtime {time.time() - t0:.0f}s", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
