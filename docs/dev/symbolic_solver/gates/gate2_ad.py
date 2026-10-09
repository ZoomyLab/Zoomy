"""Gate 2: AD through the symbolic solver.  Forward (jvp) vs central FD, and
reverse (checkpointed while, jax.grad) vs forward.  J = sum(h^2 |V|) at T."""
import numpy as np
import jax
import jax.numpy as jnp
import common as C
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax


def objective(make, ad_mode, t_end, fixed_dt):
    mesh, nsm = make()
    s = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=0.9, dt=fixed_dt,
                                    ad_mode=ad_mode, ad_checkpoints=16)
    Q0, A0 = s.setup(mesh, nsm)
    h = s.nsm.sm.roles.depth
    V = s.mesh.cell_volumes[:Q0.shape[1]]

    def J(p):
        _, Q, _ = s.run(Q0, s.initial_aux(Q0, A0, p), parameters=p)
        return jnp.sum(Q[h] ** 2 * V)
    return jax.jit(J), s.parameters, s.rt.parameter_names


for label, make, t_end, fixed_dt in (
        ("SWE 2-D bump O2, walls, background flow",
         lambda: C.bump_2d(2, nx=20, flow=(0.1, 0.05)), 0.3, 2e-2),
        ("SME(1) 1-D dam break O2", lambda: C.dambreak_1d(1, 2, nx=100), 0.05, 1e-3)):
    Jf, p0, names = objective(make, "forward", t_end, fixed_dt)
    Jr, _, _ = objective(make, "reverse", t_end, fixed_dt)
    p0 = p0.at[names.index("n")].set(0.03)          # Manning friction on
    grad = jax.grad(Jr)(p0)
    print(f"{label}: J = {float(Jf(p0)):.10e}, parameters {names}")
    for i, nm in enumerate(names):
        if float(p0[i]) == 0.0:
            continue
        e = jnp.zeros_like(p0).at[i].set(1.0)
        d = float(jax.jvp(Jf, (p0,), (e,))[1])
        if d == 0.0:
            print(f"   dJ/d{nm} = 0 (forward), reverse {float(grad[i]):.1e}")
            continue
        fd = min(abs(float((Jf(p0 + r * p0[i] * e) - Jf(p0 - r * p0[i] * e))
                           / (2 * r * p0[i])) - d)
                 for r in (1e-3, 1e-4, 1e-5, 1e-6)) / abs(d)
        print(f"   dJ/d{nm} = {d: .10e}: |FD-jvp| {fd:.1e}, "
              f"|rev-jvp| {abs(float(grad[i]) - d) / abs(d):.1e}")
