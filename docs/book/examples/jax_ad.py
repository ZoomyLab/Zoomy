"""Derivatives through the JAX march: forward (jvp) and reverse (grad) w.r.t. the Manning n."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from dambreak import dambreak  # noqa: E402
from zoomy_jax.symsolver.solver import AdMode, SymbolicHyperbolicSolverJax  # noqa: E402


def objective(ad_mode):
    """J(p) = sum(h^2 |V|) at t_end, as a function of the parameter vector p."""
    mesh, nsm = dambreak(order=2, n=60)
    s = SymbolicHyperbolicSolverJax(time_end=0.2, dt=1e-2, mood=False,
                                    ad_mode=ad_mode, ad_checkpoints=16)
    Q0, Qaux0 = s.setup(mesh, nsm)
    h = s.nsm.sm.roles.depth                      # row index of the depth
    V = s.mesh.cell_volumes[:Q0.shape[1]]

    def J(p):
        _, Q, _ = s.run(Q0, s.initial_aux(Q0, Qaux0, p), parameters=p)
        return jnp.sum(Q[h] ** 2 * V)
    return jax.jit(J), s.parameters, list(s.rt.parameter_names)


J_fwd, p0, names = objective(AdMode.forward)      # jax.jvp, jax.jacfwd
J_rev, _, _ = objective(AdMode.reverse)           # jax.grad, jax.vjp
i = names.index("n")
p0 = p0.at[i].set(0.03)                           # Manning friction on
e = jnp.zeros_like(p0).at[i].set(1.0)

forward = float(jax.jvp(J_fwd, (p0,), (e,))[1])
reverse = float(jax.grad(J_rev)(p0)[i])
r = 1e-4 * float(p0[i])
central = float(J_fwd(p0 + r * e) - J_fwd(p0 - r * e)) / (2 * r)
print(f"parameters {names}")
print(f"dJ/dn: forward {forward:.8e}, reverse {reverse:.8e}, central difference {central:.8e}")
