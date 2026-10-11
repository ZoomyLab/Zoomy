"""Gate 3d-e: AD through the symbolic solver on 3-D hexahedra (Euler
SystemModel, gate2 style): dJ/dgamma, forward (jvp) vs central FD and
reverse (checkpointed while) vs forward.  J = sum(rho^2 |V|) at T, wall box
10^3 with an off-centre pulse.
    python gate3d_e_ad.py"""
import jax
import jax.numpy as jnp
import numpy as np

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from euler_common import TAGS, conserved, nsm_of
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel.euler import compressible_euler
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax


def make(order, n=10):
    def ic(x):
        r2 = sum((x[d] - 0.4) ** 2 for d in range(3))
        b = np.exp(-r2 / 0.02)
        return conserved(1.0 + 0.5 * b, np.zeros(3), 1.0 + 2.0 * b)
    sm = compressible_euler(3, 1.4, boundary_conditions=BC.BoundaryConditions(
        [BC.Wall(tag=t) for t in TAGS[3]]), initial_conditions=IC.UserFunction(function=ic))
    return (LSQMesh.create_3d(domain=(0, 1, 0, 1, 0, 1), nx=n, ny=n, nz=n),
            nsm_of(sm, order))


def objective(order, ad_mode, t_end, dt):
    mesh, nsm = make(order)
    s = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=0.45, dt=dt,
                                    ad_mode=ad_mode, ad_checkpoints=16)
    Q0, A0 = s.setup(mesh, nsm)
    V = s.mesh.cell_volumes[:Q0.shape[1]]

    def J(p):
        _, Q, _ = s.run(Q0, s.initial_aux(Q0, A0, p), parameters=p)
        return jnp.sum(Q[0] ** 2 * V)
    return jax.jit(J), s.parameters, s.rt.parameter_names


for label, order, dt in (("O1, constant dt", 1, 5e-3), ("O1, adaptive dt", 1, None),
                         ("O2 + MOOD, constant dt", 2, 5e-3)):
    Jf, p0, names = objective(order, "forward", 0.1, dt)
    Jr, _, _ = objective(order, "reverse", 0.1, dt)
    i = names.index("gamma")
    e = jnp.zeros_like(p0).at[i].set(1.0)
    d = float(jax.jvp(Jf, (p0,), (e,))[1])
    g = float(jax.grad(Jr)(p0)[i])
    fds = {r: abs(float((Jf(p0 + r * p0[i] * e) - Jf(p0 - r * p0[i] * e))
                        / (2 * r * p0[i])) - d) / abs(d)
           for r in (1e-3, 1e-4, 1e-5, 1e-6)}
    best = min(fds, key=fds.get)
    print(f"Euler 3-D {label}: J = {float(Jf(p0)):.10e}  dJ/dgamma = {d: .10e}  "
          f"|FD-jvp| {fds[best]:.1e} (eps {best:g})  |rev-jvp| {abs(g - d) / abs(d):.1e}",
          flush=True)
