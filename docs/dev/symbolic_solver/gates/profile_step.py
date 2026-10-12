"""One time step, old vs symbolic: wall time per step and XLA cost analysis."""
import sys, time as wall
import common as C
import jax, jax.numpy as jnp, numpy as np, sympy as sp
from zoomy_core.symsolver import PyPrinter
from zoomy_core.symsolver.discretization.time import step_size
from zoomy_core.symsolver.strategies.hyperbolic import HyperbolicTemplate
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.symsolver import runtime as R
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
order = int(sys.argv[1]) if len(sys.argv) > 1 else 1
NX = int(sys.argv[2]) if len(sys.argv) > 2 else 160
make = lambda: C.bump_2d(order, nx=NX)
mesh, nsm = make()
o = HyperbolicSolver(time_end=1.0, compute_dt=C.adaptive(0.9, mesh), march_log_interval=0)
Qo, Ao = o.setup_simulation(mesh, nsm)
p = o._rt_parameters
def old_step(Q, Qaux):
    dt = o.compute_timestep(Q, Qaux, p)
    Qn = o.step(dt, 0.0, Q, Qaux, p)
    return o.post_step(dt, dt, Qn, Q, Qaux, p)
mesh, nsm = make()
s = SymbolicHyperbolicSolverJax(time_end=1.0, cfl=0.9)
Q, A = s.setup(mesh, nsm)
Qs, Qauxs, ps, time, mesh_s = sp.symbols("Q Qaux p time mesh")
template = HyperbolicTemplate(s.nsm, cfl=0.9, dimension=2)
src = PyPrinter().procedure("one", (Qs, Qauxs, ps, time, mesh_s), lambda Q, Qaux, p, time, mesh: template.step(
    Q, Qaux, p, time, step_size(s.nsm, Q, Qaux, p, mesh, 0.9, 2, None, time), mesh))
one = R.build(src, "one", s.module)
geometry = s.mesh
new_step = lambda Q, Qaux: one(Q, Qaux, s.parameters, 0.0, geometry, jnp.int64(0))[:2]
import collections, re
for name, f, args in (("old", old_step, (Qo, Ao)), ("new", new_step, (Q, A))):
    jf = jax.jit(f)
    c = jf.lower(*args).compile()
    ca = c.cost_analysis()
    ca = ca[0] if isinstance(ca, list) else ca
    txt = c.as_text()
    ops = collections.Counter(re.findall(r"= \S+ (\w[\w-]*)\(", txt))
    for ln in txt.splitlines():
        if "custom_call_target" in ln and ("eig" in ln or "geev" in ln):
            print("      ", name, re.search(r"custom-call\(([^)]*)\)", ln).group(1)[:80] if "custom-call(" in ln else "", re.search(r"= (\(?[^ ]+)", ln).group(1)[:70])
    ops["eig_calls"] = len(re.findall(r"custom_call_target=\"[^\"]*(?:geev|eig)[^\"]*\"", txt))
    out = jf(*args); jax.block_until_ready(out)
    best = 1e9
    for _ in range(3):
        t0 = wall.perf_counter()
        for _ in range(10):
            out = jf(*args)
        jax.block_until_ready(out)
        best = min(best, (wall.perf_counter() - t0) / 10)
    print(f"{name}: {1e3 * best:.2f} ms/step, flops {ca.get('flops', 0):.3e}, bytes {ca.get('bytes accessed', 0):.3e}", flush=True)
    print("   ", {k: v for k, v in sorted(ops.items()) if k in ("fusion", "scatter", "gather", "while", "dynamic-slice", "dynamic-update-slice", "sort", "custom-call", "call", "conditional", "reduce", "copy", "transpose", "concatenate", "eig_calls")})
