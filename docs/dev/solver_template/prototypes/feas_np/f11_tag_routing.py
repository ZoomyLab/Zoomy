"""F11/D25: per-tag BC routing on create_2d, numpy vs jax HyperbolicSolver."""
import sys, time, copy, numpy as np
from common import *
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
import zoomy_core.model.initial_conditions as IC
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver as NPSolver
from zoomy_jax.fvm.solver_jax import HyperbolicSolver as JXSolver

ORDER = int(sys.argv[1]) if len(sys.argv) > 1 else 1
NX = NY = 8
H_IN, QX_IN = 1.2, 0.6

def make_nsm():
    bcs = BC.BoundaryConditions([
        BC.Dirichlet("left", on="h", value=H_IN),
        BC.Dirichlet("left", on="q_x_0", value=QX_IN),
        BC.Extrapolation(tag="right"),
        BC.Wall(tag="bottom"),
        BC.Wall(tag="top")])
    m = SME(level=0, dimension=3, closures=[Newtonian(), NavierSlip(), StressFree()],
            parameters={"nu": 1e-6, "lambda_s": 0.0}, boundary_conditions=bcs)
    sm = SystemModel.from_model(m)
    def ic(x):
        out = np.zeros(4)
        out[1] = 1.0 + 0.1 * np.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
        out[2] = 0.3 * out[1]; out[3] = 0.1 * out[1]
        return out
    sm.initial_conditions = IC.UserFunction(function=ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    return NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=ORDER, limiter="venkatakrishnan"))

def mk_mesh():
    return LSQMesh.create_2d(domain=(0.0, 1.0, 0.0, 1.0), nx=NX, ny=NY)

nsm = make_nsm()
print("state", [str(s) for s in nsm.state])
bc_tags = list(nsm.sm._bc_source.list_sorted_function_names)
bc_cls = [type(b).__name__ for b in nsm.sm._bc_source.boundary_conditions_list]
print("BC branch order (tag-sorted):", list(zip(range(len(bc_tags)), bc_tags, bc_cls)))

# ---- numpy solver: its own (remapped) bc indices -------------------------------
mesh_np = mk_mesh()
print("mesh function-number order:", list(mesh_np.boundary_conditions_sorted_names))
snp = NPSolver(time_end=1.0, compute_dt=timestepping.adaptive(CFL=0.4))
snp.setup_simulation(mesh_np, nsm, write_output=False)
idx_np = np.asarray(snp._bc_indices)
# ---- jax solver: raw boundary_face_function_numbers ----------------------------
mesh_jx = mk_mesh()
sjx = JXSolver(time_end=np.inf, compute_dt=timestepping.adaptive(CFL=0.4))
Qj, Qauxj = sjx.setup_simulation(mesh_jx, nsm)
idx_jx = np.asarray(sjx._rt_mesh.boundary_face_function_numbers)

geo = bf_geometry(mesh_np, 2)
rt_np, rt_jx, p = runtimes(nsm)
probe = np.array([0.0, 1.0, 0.3, 0.2])
aux = np.zeros(nsm.sm.n_aux_variables); aux[-1] = 1.0
def expected_branch(tag):
    return bc_tags.index(tag)
print("\nper-tag routing  (tag : numpy branch -> BC  |  jax branch -> BC)   kernel(probe=[b,h,qx,qy]=[0,1,.3,.2]) numpy vs jax")
rows = {}
for i in range(geo["n_bf"]):
    tag = geo["tag"][i]
    rows.setdefault(tag, set()).add((int(idx_np[i]), int(idx_jx[i])))
for tag, s in rows.items():
    for (bn, bj) in sorted(s):
        i = next(k for k in range(geo["n_bf"]) if geo["tag"][k] == tag)
        on = np.asarray(rt_np.boundary_conditions(bn, 0.0, geo["fc"][i], geo["d"][i], probe, aux, p, geo["nrm"][:, i]), float).ravel()
        oj = np.asarray(rt_jx.boundary_conditions(bj, 0.0, geo["fc"][i], geo["d"][i], probe, aux, p, geo["nrm"][:, i]), float).ravel()
        ok_n = bn == expected_branch(tag); ok_j = bj == expected_branch(tag)
        print(f"  {tag:6s}: numpy {bn}->{bc_tags[bn]:6s}({bc_cls[bn]}) {'OK' if ok_n else 'WRONG'} | "
              f"jax {bj}->{bc_tags[bj]:6s}({bc_cls[bj]}) {'OK' if ok_j else 'WRONG'}   "
              f"n={np.round(on, 3)} j={np.round(oj, 3)}  normal={geo['nrm'][:, i]}")
n_wrong = int(sum(idx_jx[i] != expected_branch(geo['tag'][i]) for i in range(geo['n_bf'])))
print(f"jax misrouted boundary faces: {n_wrong}/{geo['n_bf']}")

# ---- 20-step march, fixed dt -------------------------------------------------
DT, NSTEP = 0.004, 20
def march_np():
    m = mk_mesh(); s = NPSolver(time_end=1.0, compute_dt=timestepping.adaptive(CFL=0.4))
    s.setup_simulation(m, nsm, write_output=False); t = 0.0
    for _ in range(NSTEP):
        s._sim_time = t; s.step(DT); t += DT
    return np.asarray(s._sim_Q)
def march_jx(remap=False):
    m = mk_mesh()
    if remap:   # CONTROL: give jax the numpy name-remapped branch numbers (scratch copy of the mesh only)
        names = list(m.boundary_conditions_sorted_names)
        r = np.array([bc_tags.index(nm) for nm in names])
        m.boundary_face_function_numbers = r[np.asarray(m.boundary_face_function_numbers)]
    s = JXSolver(time_end=np.inf, compute_dt=timestepping.adaptive(CFL=0.4))
    Q, Qa = s.setup_simulation(m, nsm); t = 0.0
    for _ in range(NSTEP):
        Qn = s.step(DT, t, Q, Qa); Q, Qa = s.post_step(t + DT, DT, Qn, Q, Qa); t += DT
    return np.asarray(Q)[:, :m.n_inner_cells]
Qn = march_np(); Qj = march_jx(False); Qjr = march_jx(True)
nc = NX * NY
cc = np.asarray(mesh_np.cell_centers)[:2, :nc]
def region(mask): return mask
bnd = {"left col": cc[0] < 1 / NX, "right col": cc[0] > 1 - 1 / NX,
       "bottom row": cc[1] < 1 / NY, "top row": cc[1] > 1 - 1 / NY,
       "interior(2+ cells from wall)": (cc[0] > 2 / NX) & (cc[0] < 1 - 2 / NX) & (cc[1] > 2 / NY) & (cc[1] < 1 - 2 / NY)}
names = [str(s) for s in nsm.state]
print(f"\n20-step march, order={ORDER}, dt={DT}: max|Q_numpy - Q_jax| per row")
for lab, Qx in (("jax (raw fn numbers, as shipped)", Qj), ("jax (CONTROL: name-remapped fn numbers)", Qjr)):
    d = np.abs(Qn - Qx)
    print(f"  {lab:42s} all={d.max():.3e}  " + "  ".join(f"{n}={d[k].max():.2e}" for k, n in enumerate(names)))
    print("     by region: " + "  ".join(f"{r}={d[:, m].max():.2e}" for r, m in bnd.items()))
print(f"  mass numpy={Qn[1].sum()/nc:.12f} jax={Qj[1].sum()/nc:.12f} jax_ctrl={Qjr[1].sum()/nc:.12f}")
np.savez(f"f11_states_o{ORDER}.npz", Qn=Qn, Qj=Qj, Qjr=Qjr)
