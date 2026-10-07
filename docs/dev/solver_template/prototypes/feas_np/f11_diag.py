"""Diagnose the residual numpy-vs-jax gap of the F11 control run."""
import sys, numpy as np
from common import *
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
import zoomy_core.model.initial_conditions as IC
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver as NPSolver
from zoomy_jax.fvm.solver_jax import HyperbolicSolver as JXSolver
NX = NY = 8
def make_nsm(kind):
    if kind == "extrap": bl = [BC.Extrapolation(tag=t) for t in ("left", "right", "bottom", "top")]
    elif kind == "wall": bl = [BC.Wall(tag=t) for t in ("left", "right", "bottom", "top")]
    else: bl = [BC.Dirichlet("left", on="h", value=1.2), BC.Dirichlet("left", on="q_x_0", value=0.6),
                BC.Extrapolation(tag="right"), BC.Wall(tag="bottom"), BC.Wall(tag="top")]
    m = SME(level=0, dimension=3, closures=[Newtonian(), NavierSlip(), StressFree()],
            parameters={"nu": 1e-6, "lambda_s": 0.0}, boundary_conditions=BC.BoundaryConditions(bl))
    sm = SystemModel.from_model(m)
    def ic(x):
        out = np.zeros(4); out[1] = 1.0 + 0.1 * np.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
        out[2] = 0.3 * out[1]; out[3] = 0.1 * out[1]; return out
    sm.initial_conditions = IC.UserFunction(function=ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    return NumericalSystemModel.from_system_model(sm, reconstruction=ReconstructionSpec(order=1))
for kind in ("extrap", "wall", "mixed"):
    nsm = make_nsm(kind); bc_tags = list(nsm.sm._bc_source.list_sorted_function_names)
    m1 = LSQMesh.create_2d((0, 1, 0, 1), NX, NY); s = NPSolver(time_end=1.0); s.setup_simulation(m1, nsm, write_output=False)
    Q0n, A0n = np.array(s._sim_Q), np.array(s._sim_Qaux)
    m2 = LSQMesh.create_2d((0, 1, 0, 1), NX, NY)
    r = np.array([bc_tags.index(nm) for nm in m2.boundary_conditions_sorted_names])
    m2.boundary_face_function_numbers = r[np.asarray(m2.boundary_face_function_numbers)]
    j = JXSolver(time_end=np.inf); Q0j, A0j = j.setup_simulation(m2, nsm)
    print(kind, "init |dQ|", np.abs(Q0n - np.asarray(Q0j)[:, :64]).max(), "init |dQaux| per row", np.abs(A0n - np.asarray(A0j)[:, :64]).max(axis=1))
    fn = s._sim_flux_operator(0.004, 0.0, Q0n, A0n, s._sim_parameters, np.zeros_like(Q0n))
    fj = np.asarray(j._rt_flux_op(0.004, 0.0, Q0j, A0j, j._rt_parameters, jnp_zero := np.zeros_like(Q0n)))
    sn = s._sim_source_operator(0.004, 0.0, Q0n, A0n, s._sim_parameters, np.zeros_like(Q0n))
    sj = np.asarray(j._rt_source_op(0.004, 0.0, Q0j, A0j, j._rt_parameters, np.zeros_like(Q0n)))
    print("   flux-op |dQ| per row", np.abs(fn - fj[:, :64]).max(axis=1), " source |d| per row", np.abs(sn - sj[:, :64]).max(axis=1))
    # same flux operator but with numpy's aux fed to jax
    fj2 = np.asarray(j._rt_flux_op(0.004, 0.0, Q0j, A0n, j._rt_parameters, np.zeros_like(Q0n)))
    print("   flux-op with numpy Qaux |dQ| per row", np.abs(fn - fj2[:, :64]).max(axis=1))

# --- confirm: residual = numpy recomputes ghost-side aux (hinv) from the BC state; jax reuses inner aux
nsm = make_nsm("mixed"); bc_tags = list(nsm.sm._bc_source.list_sorted_function_names)
m1 = LSQMesh.create_2d((0, 1, 0, 1), NX, NY); s = NPSolver(time_end=1.0); s.setup_simulation(m1, nsm, write_output=False)
Q0n, A0n = np.array(s._sim_Q), np.array(s._sim_Qaux)
rt = s._sim_model; saved = rt.update_aux_variables; rt.update_aux_variables = None
flux_noghostaux = s.get_flux_operator(s._sim_mesh, rt); rt.update_aux_variables = saved
m2 = LSQMesh.create_2d((0, 1, 0, 1), NX, NY)
r = np.array([bc_tags.index(nm) for nm in m2.boundary_conditions_sorted_names])
m2.boundary_face_function_numbers = r[np.asarray(m2.boundary_face_function_numbers)]
j = JXSolver(time_end=np.inf); Q0j, A0j = j.setup_simulation(m2, nsm)
fn = flux_noghostaux(0.004, 0.0, Q0n, A0n, s._sim_parameters, np.zeros_like(Q0n))
fj = np.asarray(j._rt_flux_op(0.004, 0.0, Q0j, A0j, j._rt_parameters, np.zeros_like(Q0n)))
print("mixed, numpy WITHOUT ghost-aux recompute vs jax(remapped): flux-op |dQ| per row", np.abs(fn - fj[:, :64]).max(axis=1))
