"""Gate 1: symbolic hyperbolic solver vs the existing solver_jax, same NSM,
mesh, CFL.  Cases: SWE 2-D bump (walls) O1/O2; SME(1) 1-D dam break O2
(friction source + moment rows + hinv aux); radial dry dam break O2 eta+MOOD."""
import time as wall
import numpy as np
import jax.numpy as jnp
import common as C
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
from zoomy_core.symsolver.settings import CflWaveSpeed


def old_march(make, t_end, cfl, **kw):
    """The existing solver, stepped exactly like its tests do."""
    mesh, nsm = make()
    s = HyperbolicSolver(time_end=t_end, compute_dt=C.adaptive(cfl, mesh),
                         march_log_interval=0, **kw)
    Q, Qaux = s.setup_simulation(mesh, nsm)
    t, n = 0.0, 0
    while t < t_end:
        dt = min(float(s.compute_timestep(Q, Qaux)), t_end - t)
        Qn = s.step(dt, t, Q, Qaux)
        Q, Qaux = s.post_step(t + dt, dt, Qn, Q, Qaux)
        t, n = t + dt, n + 1
    return np.asarray(Q), n


def new_march(make, t_end, cfl, **kw):
    mesh, nsm = make()
    steps = []
    s = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=cfl, **kw)
    Q, Qaux = s.setup(mesh, nsm)
    t, Q, Qaux = s.run(Q, Qaux)
    return np.asarray(Q), float(t), s


def eta_kw(make):
    mesh, nsm = make()
    names = [str(s) for s in nsm.state]
    return dict(reconstruction_variables="eta",
                free_surface_b_index=names.index("b"),
                free_surface_h_index=names.index("h"),
                free_surface_momentum_indices=[i for i, s in enumerate(names)
                                               if s.startswith("q_")],
                mood=True)


if __name__ == "__main__":
    import sys
    # matched with the old solver: its CFL takes the cell states; the old
    # conservative path has no MOOD
    off = dict(mood=False, cfl_wave_speed=CflWaveSpeed.cells)
    dry_old = lambda: C.dry_dambreak_2d(ties="old")      # noqa: E731
    CASES = [
        ("SWE 2-D bump, walls, O1", lambda: C.bump_2d(1), 0.5, 0.9, {}, off),
        ("SWE 2-D bump, walls, O2", lambda: C.bump_2d(2), 0.5, 0.9, {}, off),
        ("SME(1) 1-D dam break, O2", lambda: C.dambreak_1d(1, 2), 0.1, 0.9, {}, off),
        # NOT a parity case any more: the template sets round-off negatives
        # of h to 0, so MOOD (strict h < 0) stops firing on them; the old
        # solver falls back to a whole O1 step in ~14 of its 22 steps
        ("radial DRY dam break, O2 eta + MOOD, old ties", dry_old, 0.02,
         0.45, eta_kw(dry_old), dict(cfl_wave_speed=CflWaveSpeed.cells)),
    ]
    CASES = [c for c in CASES if not sys.argv[1:] or any(a in c[0] for a in sys.argv[1:])]

    for name, make, t_end, cfl, old_kw, new_kw in CASES:
        t0 = wall.time()
        Qo, n_old = old_march(make, t_end, cfl, **old_kw)
        t1 = wall.time()
        Qn, t_new, s = new_march(make, t_end, cfl, **new_kw)
        t2 = wall.time()
        nc = Qo.shape[1]
        diff = np.max(np.abs(Qn - Qo))
        scale = np.max(np.abs(Qo))
        h = s.nsm.sm.roles.depth
        print(f"{name}: max|new-old| = {diff:.2e} (rel {diff / scale:.1e}), "
              f"t = {t_new:.6g}, old steps {n_old}, h_min {Qn[h].min():.3e}, "
              f"mass new-old {abs(Qn[h].sum() - Qo[h].sum()):.1e}  "
              f"[{t1 - t0:.0f}s / {t2 - t1:.0f}s]")
