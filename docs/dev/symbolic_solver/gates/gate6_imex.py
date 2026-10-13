"""Gate 6: IMEX (explicit flux, implicit diffusion, implicit-Euler source by
cell-local Newton) vs the existing IMEXSourceSolverJax; then AD through it
(forward vs FD, reverse vs forward) w.r.t. the friction coefficient n."""
import numpy as np
import jax
import jax.numpy as jnp
import common as C
from gate1_vs_solver_jax import old_march
from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax
from zoomy_jax.symsolver.solver import AdMode, SymbolicHyperbolicSolverJax
from zoomy_core.symsolver.settings import CflWaveSpeed, Imex

STIFF = dict(n=0.05, nu=0.01)


def stiff(make):
    def m():
        mesh, nsm = make()
        for k, v in STIFF.items():
            setattr(nsm.sm.parameter_values, k, v)
        return mesh, nsm
    return m


CASES = [("SME(1) 1-D dam break O2", stiff(lambda: C.dambreak_1d(1, 2, nx=100)), 0.1, 0.9),
         ("SWE 2-D bump O2, flow", stiff(lambda: C.bump_2d(2, nx=20, flow=(0.1, 0.05))), 0.3, 0.9)]

if __name__ == "__main__":
    for name, make, t_end, cfl in CASES:
        from gate1_vs_solver_jax import old_march as _om
        import zoomy_jax.fvm.solver_jax as SJ
        mesh, nsm = make()
        s = IMEXSourceSolverJax(time_end=t_end, compute_dt=C.adaptive(cfl, mesh),
                                march_log_interval=0)
        Q, Qaux = s.setup_simulation(mesh, nsm)
        t, n = 0.0, 0
        while t < t_end:
            dt = min(float(s.compute_timestep(Q, Qaux)), t_end - t)
            Qn = s.step(dt, t, Q, Qaux)
            Q, Qaux = s.post_step(t + dt, dt, Qn, Q, Qaux)
            t, n = t + dt, n + 1
        Qo = np.asarray(Q)
        mesh, nsm = make()
        sn = SymbolicHyperbolicSolverJax(time_end=t_end, cfl=cfl, imex=Imex.lie,
                                         cfl_wave_speed=CflWaveSpeed.cells)   # as the old
        Q0, A0 = sn.setup(mesh, nsm)
        _, Qn_, _ = sn.run(Q0, A0)
        Qn_ = np.asarray(Qn_)
        print(f"{name}: max|new-old| {np.abs(Qn_ - Qo).max():.2e} (rel "
              f"{np.abs(Qn_ - Qo).max() / np.abs(Qo).max():.1e}), old steps {n}, "
              f"diffusion block: {sn.nsm.call.diffusion_matrix is not None}")

    # AD through the implicit source (fixed dt)
    mesh, nsm = stiff(lambda: C.dambreak_1d(1, 2, nx=60))()
    names = list(SymbolicHyperbolicSolverJax().param.values())  # noqa (param check)
    res = {}
    for mode in (AdMode.forward, AdMode.reverse):
        mesh, nsm = stiff(lambda: C.dambreak_1d(1, 2, nx=60))()
        s = SymbolicHyperbolicSolverJax(time_end=0.05, dt=1e-3, imex=Imex.lie, ad_mode=mode,
                                        ad_checkpoints=16)
        Q0, A0 = s.setup(mesh, nsm)

        def J(p, s=s, Q0=Q0, A0=A0):
            _, Q, _ = s.run(Q0, s.initial_aux(Q0, A0, p), parameters=p)
            return jnp.sum(Q[1] ** 2 + Q[2] ** 2)
        res[mode] = (jax.jit(J), s.parameters, s.rt.parameter_names)
    Jf, p0, pnames = res["forward"]
    Jr = res["reverse"][0]
    grad = jax.grad(Jr)(p0)
    for nm in ("n", "nu", "g"):
        i = pnames.index(nm)
        e = jnp.zeros_like(p0).at[i].set(1.0)
        d = float(jax.jvp(Jf, (p0,), (e,))[1])
        fd = min(abs(float((Jf(p0 + r * p0[i] * e) - Jf(p0 - r * p0[i] * e)) / (2 * r * p0[i])) - d)
                 for r in (1e-3, 1e-4, 1e-5, 1e-6)) / abs(d)
        print(f"IMEX AD dJ/d{nm} = {d: .6e}: |FD-jvp| {fd:.1e}, |rev-jvp| "
              f"{abs(float(grad[i]) - d) / abs(d):.1e}")
