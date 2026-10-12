"""Runs that are not "march Q0 to t_end": AD (C15) and preCICE coupling
(C16).  Each returns {metric: value} (rank 0; None on the other ranks)."""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

GATES = Path(__file__).resolve().parent.parent / "gates"


# ── C15: AD (JAX) ───────────────────────────────────────────────────────────
def _ad_solver(kind, ad_mode, comm):
    import common as C
    from zoomy_jax.symsolver.solver import (SymbolicChorinSolverJax,
                                            SymbolicHyperbolicSolverJax)
    kw = dict(ad_mode=ad_mode, ad_checkpoints=16, comm=comm)
    if kind == "chorin":
        from cases import vam_wave, wave_q0
        import jax.numpy as jnp
        mesh, (pred, press, corr) = vam_wave(1, 30)
        pred.spectral_radius_method, pred._call = "eig", None      # as cases.prepare
        s = SymbolicChorinSolverJax(time_end=0.05, pressure_linear_solver="gmres", **kw)
        Q, A = s.setup(mesh, pred, press, corr)
        names = [str(v) for v in s.nsm.state]
        Q = jnp.asarray(wave_q0(names, np.asarray(s.mesh_np.cell_centers)[:, :Q.shape[1]]))
        return s, Q, s.initial_aux(Q, A)
    mesh, nsm = C.bump_2d(2, nx=12, flow=(0.1, 0.05))
    nsm.sm.parameter_values.n = 0.5 if kind == "imex" else 0.03
    nsm.spectral_radius_method, nsm._call = "eig", None             # as cases.prepare
    s = SymbolicHyperbolicSolverJax(time_end=0.2, imex=kind == "imex", **kw)
    Q, A = s.setup(mesh, nsm)
    return s, Q, A


def ad(kind, part, comm):
    """``part`` "fwd": J, dJ/dg by jvp and by central FD; "rev": dJ/dg by
    jax.grad.  J = sum over owned cells of h^2 vol; distributed: the ranks'
    shares summed (gate 17)."""
    import jax
    import jax.numpy as jnp
    s, Q0, A0 = _ad_solver(kind, "forward" if part == "fwd" else "reverse", comm)
    h = s.nsm.sm.roles.depth
    n = s.partition.n_own if s.partition else Q0.shape[1]
    total = (lambda x: x) if s.partition is None else (lambda x: comm.allreduce(x))

    def J(p):
        _, Q, _ = s.run(Q0, s.initial_aux(Q0, A0, p), parameters=p)
        return jnp.sum(Q[h, :n] ** 2 * s.mesh.cell_volumes[:n])
    p0 = jnp.asarray(s.parameters)
    e = jnp.zeros_like(p0).at[list(s.rt.parameter_names).index("g")].set(1.0)
    if part == "rev":
        g = np.asarray(jax.grad(J)(p0))
        if s.partition is not None:
            g = s.partition.sum_to_global(g, None, None)
        return dict(rev=float(g @ np.asarray(e)))
    Jf = jax.jit(J)
    fwd = total(float(jax.jvp(Jf, (p0,), (e,))[1]))
    fd = [total(float((Jf(p0 + eps * e) - Jf(p0 - eps * e)) / (2 * eps)))
          for eps in (1e-3, 1e-4, 1e-5)]
    return dict(J=total(float(Jf(p0))), fwd=fwd,
                fd_rel=min(abs(x - fwd) for x in fd) / abs(fwd))


def ad3d(variant, part):
    """C22 (gate 3d-e): Euler 3-D wall box 10^3, off-centre pulse, J = sum
    rho^2 |V| at t 0.1, dJ/dgamma; variant o1_dt, o1_adaptive, o2_dt."""
    import jax
    import jax.numpy as jnp
    from cases import pulse
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
    order, dt = {"o1_dt": (1, 5e-3), "o1_adaptive": (1, None), "o2_dt": (2, 5e-3)}[variant]
    mesh, nsm = pulse(order, 10, 0.4)
    nsm.spectral_radius_method, nsm._call = "eig", None
    s = SymbolicHyperbolicSolverJax(time_end=0.1, cfl=0.45, dt=dt, ad_checkpoints=16,
                                    ad_mode="forward" if part == "fwd" else "reverse")
    Q0, A0 = s.setup(mesh, nsm)
    V = s.mesh.cell_volumes[:Q0.shape[1]]

    def J(p):
        _, Q, _ = s.run(Q0, s.initial_aux(Q0, A0, p), parameters=p)
        return jnp.sum(Q[0] ** 2 * V)
    p0 = jnp.asarray(s.parameters)
    i = list(s.rt.parameter_names).index("gamma")
    e = jnp.zeros_like(p0).at[i].set(1.0)
    if part == "rev":
        return dict(rev=float(jax.grad(J)(p0)[i]))
    Jf = jax.jit(J)
    fwd = float(jax.jvp(Jf, (p0,), (e,))[1])
    fd = [float((Jf(p0 + r * p0[i] * e) - Jf(p0 - r * p0[i] * e)) / (2 * r * p0[i]))
          for r in (1e-3, 1e-4, 1e-5, 1e-6)]
    return dict(J=float(Jf(p0)), fwd=fwd, fd_rel=min(abs(x - fwd) for x in fd) / abs(fwd))


# ── C16: preCICE (the gate scripts, as subprocesses) ────────────────────────
def _gate(args, timeout=280):
    env = dict(os.environ, PYTHONPATH=f"{GATES}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    r = subprocess.run([sys.executable, *args], cwd=GATES, env=env, capture_output=True,
                       text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def couple(pair, ranks):
    """jax|jax (gate 28), foam|foam (gate 26), jax|foam (gate 10, serial),
    amrex|amrex and amrex|jax (gate 38): max|coupled - single domain| and the
    mass drift, O1, window 1e-3."""
    if pair == "amrex_amrex":
        rc, out = _gate(["gate38_amrex_precice.py", "aa", str(ranks), str(ranks)])
    elif pair == "amrex_jax":
        rc, out = _gate(["gate38_amrex_precice.py", "aj", str(ranks)])
    elif pair == "jax_foam":
        if ranks > 1:
            raise NotImplementedError("no jax<->foam driver with MPI ranks (gate 10 is serial)")
        rc, out = _gate(["gate10_xcouple.py", "jax"])
    elif pair == "foam_foam":
        rc, out = _gate(["gate26_foam_couple_parallel.py", str(ranks), str(ranks)])
    else:
        rc, out = _gate(["gate28_jax_couple_parallel.py", "1d", "1", "0.001", "window",
                         str(ranks), str(ranks)])
        if rc == 0:
            from gate3_precice import single, h, dx
            Q0, Qs = single(1, fixed_dt=1e-3)
            Qc = np.load(GATES / f"Q_1d_o1_0.001_window_L{ranks}_R{ranks}.npy")
            return dict(coupled_vs_single=float(np.abs(Qc - Qs).max()),
                        mass=float(abs(Qc[h].sum() - Q0[h].sum()) * dx))
    m = re.search(r"max\|coupled ?- ?single\| ([\d.e+-]+), mass drift ([\d.e+-]+)", out)
    if rc != 0 or m is None:
        raise RuntimeError(f"gate failed (rc {rc}):\n{out[-2000:]}")
    return dict(coupled_vs_single=float(m.group(1)), mass=float(m.group(2)))


# ── C17: AMReX AMR (gate 37) ────────────────────────────────────────────────
AMR_CONFIG = {"serial": "cpu1", "mpi2": "cpu2", "gpu1": "gpu1"}


def amr(case, config, work):
    """One gate-37 AMR run; its measure (as gate 37's table) and, off cpu1,
    max rel vs the cpu1 run.  The fine single-level reference (cover, wave)
    is made with the cpu1 run."""
    import gate37_amrex_amr as G
    from euler_common import wave_exact
    # one directory for every config: off cpu1 a run is compared with the cpu1 one
    out = Path(work).parents[2] / "serial" / Path(work).parent.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = AMR_CONFIG[config]
    G.run(out, case, cfg)
    r = np.load(out / f"{case}_{cfg}.npz")
    res = {}
    if cfg != "cpu1":
        c1 = np.load(out / f"{case}_cpu1.npz")
        res["vs_cpu1"] = float(G.rel(r["Q"], c1["Q"]))
        return res
    n = G.base_n(case)
    if case.startswith(("cover", "wave")):
        G.ref(out, case)
        f = np.load(out / f"ref_{case}.npz")
    if case.startswith("cover"):
        res["fine_vs_single"] = float(G.rel(r["L1"], f["Qa"]))
    elif case.startswith("wave"):
        t = float(r["t"])
        res["L1"] = float(G.composite(case, n, r["Q"], r["L1"],
                                      lambda Q, x: np.abs(Q[0] - wave_exact(x, t, G.VEL))))
    elif case.startswith("box"):
        res["mass"] = float(abs(r["Q"][0].sum() / r["I0"][0].sum() - 1))
        res["energy"] = float(abs(r["Q"][-1].sum() / r["I0"][-1].sum() - 1))
    elif case == "lake":
        res["u"] = float(max(np.nanmax(np.abs(Q[2:4]) / Q[1]) for Q in (r["Q"], r["L1"])))
    else:
        res["mass"] = float(abs(r["Q"][1].sum() / r["I0"][1].sum() - 1))
        res["h_min"] = float(min(np.nanmin(r["Q"][1]), np.nanmin(r["L1"][1])))
    return res
