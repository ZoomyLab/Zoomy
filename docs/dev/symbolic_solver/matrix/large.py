"""The LARGE tier (never part of the default run; every command <= 5 min).

    python matrix.py run --tier large L1 BACKEND CONFIG     weak scaling, 200^2 cells per rank
    python matrix.py run --tier large L2 BACKEND CONFIG     strong scaling, 1600^2 = 2.56M cells
    python matrix.py run --tier large L2P BACKEND CONFIG    strong scaling, pressure (VAM(1), ~1M cells)
    python matrix.py run --tier large L3 BACKEND CONFIG [T] catchment (JAX: 14.8M cells, 1 m;
                                                            AMReX: its 10 m grid driver)
    python matrix.py run --tier large L4 BACKEND NX         setup / compile / memory vs mesh size
    python matrix.py run --tier large L5 RANKS              parallel mesh + field write, 14.8M cells

L3 JAX reads the fine mesh that L5 writes: run L5 first.

CONFIG: serial mpi2 mpi4 gpu1 gpu2 gpu4.  Results: STORE/large/<L>/<backend>_<config>[_<n>].json;
`python matrix.py table` prints them in their own section.  Multi-node / Slurm: untestable here.
"""
import json
import os
import re
import resource
import subprocess
import sys
import time as clock
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GATES = HERE.parent / "gates"
STORE = Path(os.environ.get("MATRIX_STORE", HERE / "store")) / "large"
WORK = Path(os.environ.get("MATRIX_WORK", HERE / "work")) / "large"
CATCH = Path(os.environ.get("MATRIX_CATCH", "/tmp/claude-765404697/-mnt-userdrive-Users-home-"
                            "adam-obbpb5az1dhsjzf-git-zotero-rag/b5667895-053e-413b-8d69-"
                            "8137a92fc71c/scratchpad/catch"))      # dem_1m.npy, coarse.npz (read only)
N_STEPS = 20


def ranks(config):
    return int(config[3:]) if config[:3] in ("mpi", "gpu") else 1


def save(level, name, res):
    STORE.joinpath(level).mkdir(parents=True, exist_ok=True)
    STORE.joinpath(level, f"{name}.json").write_text(json.dumps(res))
    print(level, name, res, flush=True)


def maxrss_gb():
    return max(resource.getrusage(r).ru_maxrss for r in
               (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)) / 1e6


# ── hyperbolic timing: C1 bump O2, constant dt, N_STEPS steps ───────────────
def bump(nx):
    import common as C
    return C.bump_2d(2, nx=nx)        # the default wave speed (Gelfand bound) everywhere


def time_jax(make, n_steps, dt, chorin=None, comm=None, q0=None):
    """(setup s, compile s, march s) of n_steps constant steps."""
    import jax.numpy as jnp
    from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax, SymbolicHyperbolicSolverJax
    t0 = clock.perf_counter()
    mesh, systems = make()
    if chorin:
        s = SymbolicChorinSolverJax(time_end=dt, dt=dt, cfl_check_every=0, comm=comm,
                                    pressure_linear_solver=chorin)
        Q, A = s.setup(mesh, *systems)
    else:
        s = SymbolicHyperbolicSolverJax(time_end=dt, dt=dt, cfl_check_every=0, comm=comm,
                                        mood=False)
        Q, A = s.setup(mesh, systems)
    if q0 is not None:
        Q = jnp.asarray(q0([str(v) for v in s.nsm.state],
                           np.asarray(s.mesh_np.cell_centers)[:, :Q.shape[1]]))
        A = s.initial_aux(Q, A)
    bar = comm.Barrier if comm is not None else (lambda: None)
    bar()
    t1 = clock.perf_counter()
    s.run(Q, A)[1].block_until_ready()                    # compile + 1 step
    bar()
    t2 = clock.perf_counter()
    s.time_end = n_steps * dt
    s.run(Q, A)[1].block_until_ready()
    bar()
    t3 = clock.perf_counter()
    n = int(s.partition.n_global) if s.partition else int(Q.shape[1])
    res = dict(cells=n, setup=t1 - t0, compile=t2 - t1 - (t3 - t2) / n_steps,
               ms_step=1e3 * (t3 - t2) / n_steps)
    if chorin:
        res["solver"] = s._pressure_backend()
    if chorin and res["solver"] == "hypre":    # the pressure solves (both runs), host ms/solve
        from zoomy_jax.symsolver.hypre import STATS as st
        k = max(st["solves"], 1)
        res.update(solves=st["solves"], its_solve=st["iterations"] / k, amg_setups=st["setups"],
                   **{f"ms_{p}": 1e3 * st[f"t_{q}"] / k for p, q in
                      (("asm", "assemble"), ("setup", "setup"), ("solve", "solve"))})
    return res


def time_cxx(backend, make, n_steps, dt, config, work, chorin=False, q0=None):
    """foam / AMReX: (setup + build + IO s, march s) from the solver's own
    'march wall time'."""
    import cases
    t0 = clock.perf_counter()
    r = cases.Run("large", make, n_steps * dt, dt=dt, chorin=chorin, q0=q0)
    mesh, systems, Q0, _ = cases.prepare(r)
    kw = dict(time_end=n_steps * dt, dt=dt, cfl_check_every=0, mood=False)
    if backend == "amrex":
        from zoomy_amrex.symsolver import SymbolicChorinSolverAmrex, SymbolicHyperbolicSolverAmrex
        cls = SymbolicChorinSolverAmrex if chorin else SymbolicHyperbolicSolverAmrex
        s = cls(ranks=ranks(config), gpu=config.startswith("gpu"), **kw)
        s.run(mesh, systems, Q0, work)
        march, log = s.wall, s.log
    else:
        from zoomy_foam.symsolver import SymbolicChorinSolverFoam, SymbolicHyperbolicSolverFoam
        cls = SymbolicChorinSolverFoam if chorin else SymbolicHyperbolicSolverFoam
        cls(nprocs=ranks(config), **kw).run(mesh, systems, Q0, str(work))
        log = (Path(work) / "log.symFoam").read_text()
        march = float(re.findall(r"march wall time ([\d.eE+-]+)", log)[-1])
    total = clock.perf_counter() - t0
    res = dict(cells=int(Q0.shape[1]), setup=total - march, ms_step=1e3 * march / n_steps,
               maxrss_gb=maxrss_gb())
    m = re.search(r"linsolve summary: ([^,]+), (\d+) solves, ([\d.eE+-]+) iterations/solve, "
                  r"ms/solve: assembly ([\d.eE+-]+), setup ([\d.eE+-]+), solve ([\d.eE+-]+)", log)
    if m:      # the pressure solves (C++ backends' own counters)
        res.update(linsolve=m.group(1), solves=int(m.group(2)), its_solve=float(m.group(3)),
                   **{f"ms_{k}": float(m.group(i)) for i, k in enumerate(("asm", "setup", "solve"), 4)})
    return res


def hyperbolic(level, backend, config, nx, n_steps=N_STEPS):
    dt = 0.05 * 10.0 / nx                                 # CFL ~ 0.1
    name = f"{backend}_{config}"
    if backend == "jax":
        comm = None
        if ranks(config) > 1:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        res = time_jax(lambda: bump(nx), n_steps, dt, comm=comm)
        if comm is None or comm.Get_rank() == 0:
            save(level, name, dict(res, maxrss_gb=maxrss_gb(), ranks=ranks(config)))
    else:
        res = time_cxx(backend, lambda: bump(nx), n_steps, dt, config,
                       WORK / level / name)
        save(level, name, dict(res, ranks=ranks(config)))


# ── pressure strong scaling: VAM(1) Chorin channel, ~1M cells ───────────────
def vam_large(nx):
    from cases import predictor_nsm, vam_split
    from zoomy_core.mesh import LSQMesh
    pred, press, corr = vam_split()
    return (LSQMesh.create_2d(domain=(-1.5, 1.5, 0.0, 0.3), nx=nx, ny=nx // 10),
            (predictor_nsm(pred, press, corr), press, corr))


def pressure(backend, config, nx=3200):
    dt, n = 1e-4, 5
    name = f"{backend}_{config}"
    from cases import wave_q0
    if backend == "jax":
        comm = None
        if ranks(config) > 1:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        # auto: hypre on CPU ranks, pyamg serial
        from zoomy_core.symsolver.settings import LinearSolver
        solver = LinearSolver.amgx if config.startswith("gpu") else LinearSolver.auto
        res = time_jax(lambda: vam_large(nx), n, dt, chorin=solver, comm=comm, q0=wave_q0)
        if comm is None or comm.Get_rank() == 0:
            save("L2P", name, dict(res, maxrss_gb=maxrss_gb(), ranks=ranks(config)))
    else:
        res = time_cxx(backend, lambda: vam_large(nx), n, dt, config, WORK / "L2P" / name,
                       chorin=True, q0=wave_q0)
        save("L2P", name, dict(res, ranks=ranks(config)))


# ── catchment and parallel I/O (bench_catchment.py; JAX) ────────────────────
def _sub(argv, env_name, n, timeout=290):
    pre = ["micromamba", "run", "-n", env_name]
    if n > 1:
        pre += ["mpiexec", "-n", str(n)]
    env = dict(os.environ, PYTHONPATH=f"{GATES}{os.pathsep}{os.environ['PYTHONPATH']}")
    if env_name != "zoomy":
        env.pop("JAX_PLATFORMS", None)
    r = subprocess.run(pre + argv, cwd=GATES, env=env, capture_output=True, text=True,
                       timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def catchment(backend, config, T):
    out = WORK / "catch"
    if backend == "amrex":
        drv = GATES / "catchment_amrex.py"
        cfg = {"serial": "cpu1", "mpi4": "cpu4"}.get(config, config)
        if not (out / "amrex" / f"jax_{T:g}.npz").exists():   # the same mesh on JAX
            _sub(["python", str(drv), str(out / "amrex"), str(T), "jax"], "zoomy", 1)
        rc, txt = _sub(["python", str(drv), str(out / "amrex"), str(T), cfg], "zoomy", 1)
        m = re.search(r"catchment .*", txt)
        save("L3", f"amrex_{config}", dict(rc=rc, line=m.group(0) if m else txt[-500:]))
        return
    gpu = config.startswith("gpu")
    rc, txt = _sub(["python", "-m", "zoomy_jax.gpurank"] * gpu + ["bench_catchment.py", "run",
                   str(out), str(T)], "zoomy-cuda-ompi" if gpu else "zoomy", ranks(config))
    m = re.search(r"fine run: .*", txt)
    save("L3", f"jax_{config}", dict(rc=rc, line=m.group(0) if m else txt[-800:]))


def io(n):
    out = WORK / "catch"
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "coarse.npz").exists():
        os.symlink(CATCH / "coarse.npz", out / "coarse.npz")
    rc, txt = _sub(["python", "bench_catchment.py", "mesh", str(out), str(CATCH / "dem_1m.npy")],
                   "zoomy-cuda-ompi", n)
    m = re.search(r"fine mesh .*", txt)
    save("L5", f"write_{n}ranks", dict(rc=rc, line=m.group(0) if m else txt[-800:]))


# ── setup / compile / memory vs mesh size ───────────────────────────────────
def scaling_setup(backend, nx):
    nx = int(nx)
    if backend == "jax":
        res = time_jax(lambda: bump(nx), 5, 0.05 * 10.0 / nx)
        save("L4", f"jax_{nx}", dict(res, maxrss_gb=maxrss_gb()))
    else:
        res = time_cxx(backend, lambda: bump(nx), 5, 0.05 * 10.0 / nx, "serial",
                       WORK / "L4" / f"{backend}_{nx}")
        save("L4", f"{backend}_{nx}", res)


def launch(args):
    """JAX on MPI ranks / GPUs: re-run this command under mpiexec / the CUDA env."""
    if os.environ.get("MATRIX_CHILD") or args[0] not in ("L1", "L2", "L2P") or args[1] != "jax":
        return False
    config, n = args[2], ranks(args[2])
    if config == "serial":
        return False
    pre = (["mpiexec", "-n", str(n), sys.executable] if config.startswith("mpi") else
           ["env", "JAX_PLATFORMS=cuda", "XLA_PYTHON_CLIENT_PREALLOCATE=false", "micromamba", "run",
            "-n", "zoomy-cuda"] + (["mpiexec", "-n", str(n)] if n > 1 else [])
           + ["python", "-m", "zoomy_jax.gpurank"])
    subprocess.run(pre + [str(HERE / "matrix.py"), "run", "--tier", "large", *args],
                   env=dict(os.environ, MATRIX_CHILD="1"))
    return True


def main(args):
    if launch(args):
        return
    level = args[0]
    if level == "L1":
        hyperbolic("L1", args[1], args[2], int(round(200 * np.sqrt(ranks(args[2])))))
    elif level == "L2":
        hyperbolic("L2", args[1], args[2], 1600, 5)      # steps of seconds: 5 fit 5 min
    elif level == "L2P":
        pressure(args[1], args[2])
    elif level == "L3":
        catchment(args[1], args[2], float(args[3]) if len(args) > 3 else 5.0)
    elif level == "L4":
        scaling_setup(args[1], args[2])
    elif level == "L5":
        io(int(args[1]))


def table():
    """Markdown lines of the large tier."""
    lines = ["", "## Large tier", "", "| level | run | result |", "|---|---|---|"]
    for f in sorted(STORE.glob("*/*.json")):
        r = json.loads(f.read_text())
        txt = r.get("line") or ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                                         for k, v in r.items())
        lines.append(f"| {f.parent.name} | {f.stem} | {txt} |")
    return lines
