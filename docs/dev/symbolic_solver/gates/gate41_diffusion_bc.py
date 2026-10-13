"""Gate 41: the diffusion blocks at boundaries (Dirichlet / Neumann ghosts).

Scalar heat equation u_t = div(nu grad u), a SystemModel written directly
(diffusion tensor nu I; the template needs a flux: advection a u with a = 0,
an exactly zero Rusanov flux), on [0, 1]^d; exact solutions, L_inf error
at t_end on nx = 16 / 32 / 64, observed order from the last two:

  dir0   u = 0 on the walls        u = exp(-d nu pi^2 t) prod sin(pi x_i)
  neu    du/dn = 0 (ZeroNeumann)   u = exp(-d nu pi^2 t) prod cos(pi x_i)
  dirL   u = L on the walls        u = L + exp(-d nu pi^2 t) prod sin(pi x_i),
         L = 1 + 2 x + 3 y (a position-dependent Dirichlet value, Lambda BC)
  per    periodic in every direction  u = exp(-4 d nu pi^2 t) prod sin(2 pi x_i)
         (the seam's two-point flux: the partner cell)
  steady u = L, u = L on the walls: kept to round-off (two-point fluxes with
         mirror ghosts are exact for a linear field)

Paths: 'expl' (diffusion_matrix_explicit, Euler, dt = 0.4 / spectral radius) and
'impl' (diffusion_matrix + IMEX: Crank-Nicolson, dt = h / 32).  Meshes:
1-D, 2-D quads, 2-D triangles (quads cut along alternating diagonals:
centroid lines are normal to the faces, so two-point fluxes are consistent).
Pass: order >= 1.8; steady <= 1e-12.

    python gate41_diffusion_bc.py               # serial table
    mpiexec -n 2 python gate41_diffusion_bc.py par   # 2 ranks vs serial
GATE41_STORE=<file.json>: also store the errors (deliverable.py plots them).
"""
import json
import os
import sys
import time as clock

import numpy as np
import sympy as sp
import jax
jax.config.update("jax_enable_x64", True)

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh, ensure_lsq_mesh
from zoomy_core.mesh.base_mesh import BaseMesh
from zoomy_core.misc.misc import Zstruct
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel, face_normal_symbols
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

NU, T_END = 1.0, 0.05
NXS = (16, 32, 64)
TIGHT = dict(implicit_tol=1e-10, gmres_tol=1e-13, implicit_maxiter=4,
             gmres_maxiter=200)
TAGS = {1: ("left", "right"), 2: ("left", "right", "bottom", "top")}


def lin(x):
    return 1.0 + 2.0 * x[0] + (3.0 * x[1] if len(x) > 1 else 0.0)


def exact(case, dim, x, t):
    x = np.asarray(x)[:dim]
    decay = np.exp(-dim * NU * np.pi ** 2 * t)
    if case == "per":
        return decay ** 4 * np.prod(np.sin(2 * np.pi * x), axis=0)
    if case == "neu":
        return decay * np.prod(np.cos(np.pi * x), axis=0)
    base = decay * np.prod(np.sin(np.pi * x), axis=0)
    if case == "dir0":
        return base
    if case == "dirL":
        return lin(x) + base
    return lin(x)                                            # steady


def heat(dim, case, path):
    """The scalar heat equation as a SystemModel (no Model derivation)."""
    space = [sp.Symbol(c, real=True) for c in "xy"[:dim]]
    u = sp.Symbol("u", real=True)
    nu, a = sp.Symbol("nu", positive=True), sp.Symbol("a", real=True)
    params = Zstruct(nu=nu, a=a)
    params._symbolic_name = "p"
    n = face_normal_symbols(dim)
    A = sp.MutableDenseNDimArray.zeros(1, 1, dim, dim)
    for d in range(dim):
        A[0, 0, d, d] = nu
    slot = "diffusion_matrix_explicit" if path == "expl" else "diffusion_matrix"
    sm = SystemModel(
        time=sp.Symbol("t", real=True), space=space, state=[u], aux_state=[],
        parameters=params, flux=sp.Matrix([[a * u] * dim]),
        hydrostatic_pressure=sp.zeros(1, dim),
        nonconservative_matrix=sp.MutableDenseNDimArray.zeros(1, 1, dim),
        source=sp.zeros(1, 1), mass_matrix=sp.eye(1),
        eigenvalues=sp.Matrix([a * sum(n)]),
        parameter_values=Zstruct(nu=NU, a=0.0),
        **{slot: A})
    sm.declare_roles(depth=None, momentum=[])
    sm.declare_admissible([])
    if case == "neu":
        bcs = [BC.ZeroNeumann(tag=t) for t in TAGS[dim]]
    elif case == "dir0":
        bcs = [BC.Dirichlet(t, value=0.0) for t in TAGS[dim]]
    elif case == "per":
        tags = TAGS[dim]
        bcs = [BC.Periodic(tag=tags[i], periodic_to_physical_tag=tags[i ^ 1])
               for i in range(len(tags))]
    else:                         # the linear field at the BC's position
        f = (lambda t, X, dX, Q, Qaux, p, n: 1 + 2 * X[0]
             + (3 * X[1] if dim == 2 else 0))
        bcs = [BC.Lambda(tag=t, prescribe_fields={0: f}) for t in TAGS[dim]]
    sm.attach_boundary_conditions(BC.BoundaryConditions(bcs))
    sm.initial_conditions = IC.UserFunction(
        function=lambda x: np.array([exact(case, dim, x, 0.0)]))
    return NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=1))


def triangles(nx):
    """[0, 1]^2, each grid quad cut along alternating diagonals (gate16)."""
    xs = np.linspace(0, 1, nx + 1)
    vid = lambda i, k: i * (nx + 1) + k                          # noqa: E731
    X = np.array([[x, y] for x in xs for y in xs]).T
    tris = []
    for i in range(nx):
        for k in range(nx):
            a, b, c, d = vid(i, k), vid(i + 1, k), vid(i + 1, k + 1), vid(i, k + 1)
            tris += ([(a, b, c), (a, c, d)] if (i + k) % 2 else [(a, b, d), (b, c, d)])
    tags = {}
    for m in range(nx):
        tags[tuple(sorted((vid(0, m), vid(0, m + 1))))] = 0
        tags[tuple(sorted((vid(nx, m), vid(nx, m + 1))))] = 1
        tags[tuple(sorted((vid(m, 0), vid(m + 1, 0))))] = 2
        tags[tuple(sorted((vid(m, nx), vid(m + 1, nx))))] = 3
    return BaseMesh.from_arrays(X, np.array(tris).T, "triangle", face_tags=tags,
                                tag_names={0: "left", 1: "right", 2: "bottom", 3: "top"})


def mesh_of(kind, nx):
    if kind == "1d":
        return LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=nx)
    if kind == "quad":
        return LSQMesh.create_2d(domain=(0, 1, 0, 1), nx=nx, ny=nx)
    return triangles(nx)


def run(kind, case, path, nx, comm=None):
    """Returns (u at t_end in global cell order or None, cell centres, wall)."""
    dim = 1 if kind == "1d" else 2
    nsm = heat(dim, case, path)
    mesh = mesh_of(kind, nx)
    if kind == "tri":
        mesh = ensure_lsq_mesh(mesh, nsm)
    h = 1.0 / nx
    # explicit Euler: spectral radius 4 d / h^2 on quads, 24 / h^2 on these
    # triangles
    rho = 4 * dim / h ** 2 if kind != "tri" else 24 / h ** 2
    dt = 0.4 / (rho * NU) if path == "expl" else h / 32
    s = SymbolicHyperbolicSolverJax(time_end=T_END, dt=dt, cfl_check_every=0,
                                    imex="lie" if path == "impl" else False, comm=comm, **TIGHT)
    Q, A = s.setup(mesh, nsm)
    t0 = clock.perf_counter()
    t, Q, A = s.run(Q, A)
    getattr(Q, "block_until_ready", lambda: None)()     # numpy: none
    wall = clock.perf_counter() - t0
    if s.partition is not None:
        Q = s.partition.gather(Q)
        if Q is None:
            return None, None, wall, float(t)
    gm = s.global_mesh if s.partition is None else s.global_mesh
    nc = int(np.asarray(Q).shape[1])
    xc = np.asarray(gm.cell_centers)[:, :nc]
    return np.asarray(Q)[0], xc, wall, float(t)


def error(kind, case, path, nx):
    dim = 1 if kind == "1d" else 2
    u, xc, wall, t = run(kind, case, path, nx)
    return np.abs(u - exact(case, dim, xc, t)).max(), wall


def serial():
    fails, store = 0, {"nx": list(NXS)}
    print(f"gate41 heat eq., nu={NU}, t_end={T_END}, L_inf at nx {NXS}", flush=True)
    for kind in ("1d", "quad", "tri"):
        for path in ("expl", "impl"):
            for case in ("dir0", "neu", "dirL", "per"):
                errs, walls = zip(*(error(kind, case, path, nx) for nx in NXS))
                store[f"{kind} {path} {case}"] = [float(e) for e in errs]
                order = np.log2(errs[-2] / errs[-1])
                ok = order >= 1.8
                tag = "PASS" if ok else "FAIL"
                fails += not ok
                print(f"gate41 {kind:4s} {path} {case:4s}  L_inf "
                      + " ".join(f"{e:.2e}" for e in errs)
                      + f"  order {np.log2(errs[0] / errs[1]):.2f} {order:.2f}  {tag}"
                      f"  [max run {max(walls):.1f}s]", flush=True)
            e, wall = error(kind, "steady", path, 16)
            ok = e <= 1e-12
            fails += not ok
            print(f"gate41 {kind:4s} {path} steady linear Dirichlet  max|u-L| "
                  f"{e:.1e}  {'PASS' if ok else 'FAIL'}", flush=True)
    print(f"gate41 checks done: {fails} FAIL", flush=True)
    if os.environ.get("GATE41_STORE"):
        with open(os.environ["GATE41_STORE"], "w") as f:
            json.dump(store, f, indent=1)
    return fails


def parallel():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, fails = comm.Get_rank(), 0
    for kind in ("1d", "quad", "tri"):
        for path in ("expl", "impl"):
            for case in ("dir0", "dirL", "per"):
                up, _, wall, _ = run(kind, case, path, 32, comm)
                if rank == 0:
                    us, _, _, _ = run(kind, case, path, 32)
                    d = np.abs(up - us).max() / np.abs(us).max()
                    ok = d <= 1e-12
                    fails += not ok
                    print(f"gate41 {kind:4s} {path} {case:4s} nx 32: ranks "
                          f"{comm.Get_size()} max rel |par-ser| {d:.1e}  "
                          f"{'PASS' if ok else 'FAIL'}  [run {wall:.1f}s]", flush=True)
    if rank == 0:
        print(f"gate41 parallel checks done: {fails} FAIL", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "par":
        parallel()
    else:
        sys.exit(1 if serial() else 0)
