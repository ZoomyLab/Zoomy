"""F4: numpy indexed BC kernel called BATCHED over all boundary faces vs the
per-face Python loop of solver_numpy.py:956-977 (and 991-1002).  Bitwise?
(1) the kernel as shipped; (2) the SAME sympy kernel re-lambdified with a
broadcasting ``array`` in the module dict (scratch-only; no repo edit)."""
import time, numpy as np
from common import *
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
import zoomy_core.model.initial_conditions as IC
from zoomy_core.fvm.solver_numpy import HyperbolicSolver as NPSolver

def barray(x, *a, **k):
    """broadcasting np.array: scalar Piecewise-branch constants -> face axis."""
    if isinstance(x, (list, tuple)):
        el = [barray(e) for e in x]
        return np.array(np.broadcast_arrays(*el), *a, **k)
    return np.asarray(x, *a, **k)

def f11_bcs():
    return [BC.Dirichlet("left", on="h", value=1.2), BC.Dirichlet("left", on="q_x_0", value=0.6),
            BC.Extrapolation(tag="right"), BC.Wall(tag="bottom"), BC.Wall(tag="top")]
def setups():
    yield "SME0 2-D F11 set (Dirichlet-perfield/Extrap/Wall/Wall)", SME(level=0, dimension=3, closures=[Newtonian(), NavierSlip(), StressFree()],
            parameters={"nu": 1e-6, "lambda_s": 0.0}, boundary_conditions=BC.BoundaryConditions(f11_bcs())), 2
    lam = {2: lambda t, X, dX, Q, Qa, p, n: 0.5 + 0.1 * X[1] + t}
    yield "SWE 2-D Lambda(X,t)/InflowOutflow/RoughWall/FromModel", SWE(dimension=2, boundary_conditions=BC.BoundaryConditions([
            BC.Lambda(tag="left", prescribe_fields=lam), BC.InflowOutflow(tag="right", prescribe_fields={1: 1.1}),
            BC.RoughWall(tag="bottom"), BC.FromModel(tag="top", definition="wall")])), 2
    yield "SWE 1-D Characteristic(opaque eig)/CharacteristicWall", SWE(dimension=1, boundary_conditions=BC.BoundaryConditions([
            BC.Characteristic(tag="left", prescribe_fields={2: lambda *a: 0.8}), BC.CharacteristicWall(tag="right")])), 1
    yield "SWE 1-D Dirichlet(h)/FromData", SWE(dimension=1, boundary_conditions=BC.BoundaryConditions([
            BC.Dirichlet("left", on="h", value=1.3),
            BC.FromData(tag="right", timeline=np.array([0.0, 1.0]), prescribe_fields={1: np.array([1.0, 2.0])})])), 1

rng = np.random.default_rng(0)
mod = dict(NumpyRuntimeModel.module); mod["array"] = barray
for label, model, dim in setups():
    nsm = build_nsm(model, order=1)
    sm = nsm.sm
    sm.initial_conditions = IC.Constant(constants=lambda n: np.array([0.0, 1.0] + [0.1] * (n - 2)))
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    rt_b = NumpyRuntimeModel.from_system_model(sm, module=mod)
    for N in ((8, 200) if dim == 2 else (50,)):
        mesh = LSQMesh.create_2d((0, 1, 0, 1), N, N) if dim == 2 else LSQMesh.create_1d((0, 1), N)
        s = NPSolver(time_end=1.0)
        nc = mesh.n_inner_cells
        s.setup_simulation(mesh, nsm, write_output=False)
        bc_fn, idx, cells, fidx, d = s._bc_fn, s._bc_indices, s._bf_cells, s._bf_fidx, s._d_face
        n_bf = s._n_bf; msh = s._sim_mesh
        normals = msh.face_normals[:dim, :]; fcs = msh.face_centers; p = s._sim_parameters
        nv = len(sm.state); na = sm.n_aux_variables
        Q = np.vstack([rng.uniform(-0.1, 0.1, nc), rng.uniform(0.5, 1.5, nc)] + [rng.uniform(-1, 1, nc) for _ in range(nv - 2)])
        Qa = rng.uniform(0.5, 2.0, (na, nc)); T = 0.37
        t0 = time.perf_counter()
        loop = np.zeros((nv, n_bf))
        for i in range(n_bf):
            loop[:, i] = bc_fn(idx[i], T, fcs[fidx[i], :], d[i], Q[:, cells[i]], Qa[:, cells[i]], p, normals[:, fidx[i]])
        t_loop = time.perf_counter() - t0
        args = (idx, T, fcs[fidx, :].T, d, Q[:, cells], Qa[:, cells], p, normals[:, fidx])
        for tag, fn in (("as-shipped", bc_fn), ("broadcasting-array", rt_b.boundary_conditions)):
            try:
                t0 = time.perf_counter(); bat = np.asarray(fn(*args), dtype=float); t_bat = time.perf_counter() - t0
                shp = bat.shape; bat = bat.reshape(nv, n_bf)
                print(f"{label} N={N} n_bf={n_bf} [{tag}]: out {shp}; bitwise={np.array_equal(bat, loop)}; "
                      f"max|d|={np.abs(bat - loop).max():.1e}; loop {t_loop*1e3:.2f} ms vs batched {t_bat*1e3:.3f} ms ({t_loop/t_bat:.0f}x)")
            except Exception as e:
                print(f"{label} N={N} [{tag}]: batched FAILED {type(e).__name__}: {str(e)[:110]}")
        # scalar per-face eval through the broadcasting module must equal the shipped one
        loop_b = np.array([np.asarray(rt_b.boundary_conditions(idx[i], T, fcs[fidx[i], :], d[i], Q[:, cells[i]], Qa[:, cells[i]], p, normals[:, fidx[i]]), float).ravel() for i in range(n_bf)]).T
        print(f"    per-face eval with broadcasting module == shipped: {np.array_equal(loop_b, loop)}")
