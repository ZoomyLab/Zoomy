"""Measurement of one timing case on one backend and config (README.md).

``measure`` marches a case ``reps`` times and returns the phase times and the
final state.  JAX on a GPU runs in a child process (the CUDA env):
``python timing_run.py worker DIR`` is that child."""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time as clock
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

import timing_cases
import timing_paths
import timing_records


@dataclass
class Measurement:
    """cells, final time ``t`` and state ``Q`` (rows = state), the state ``names``,
    setup / compile seconds (first repetition), the march seconds of every
    repetition, the JAX device (None: C++), the pressure solver of a Chorin run."""
    cells: int
    t: float
    Q: np.ndarray
    names: list
    setup: float
    compile: float
    march: list
    device: str = None
    steps_run: int = None
    detail: str = None

    def meta(self):
        return {k: v for k, v in self.__dict__.items() if k != "Q"}


# ── JAX ─────────────────────────────────────────────────────────────────────
def _jax(run, reps):
    """setup = model + mesh + printing; compile = first call minus one step;
    march = ``reps`` marches of the compiled solver."""
    import jax
    import jax.numpy as jnp
    import cases
    from backends import _solver_kw
    from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax, SymbolicHyperbolicSolverJax
    t0 = clock.perf_counter()
    mesh, systems, Q0, names = cases.prepare(run)
    cls = SymbolicChorinSolverJax if run.chorin else SymbolicHyperbolicSolverJax
    kw = dict(_solver_kw(run, cls), cfl_check_every=0)
    if run.chorin:
        s = cls(pressure_linear_solver="auto", **kw)         # pyamg on a CPU, AmgX on a GPU
        Q, A = s.setup(mesh, *systems)
    else:
        s = cls(**kw)
        Q, A = s.setup(mesh, systems)
    if run.q0 is not None:
        Q = jnp.asarray(run.q0([str(v) for v in s.nsm.state],
                               np.asarray(s.mesh_np.cell_centers)[:, :Q.shape[1]]))
        A = s.initial_aux(Q, A)
    else:
        Q = jnp.asarray(Q0)
    jax.block_until_ready((Q, A))
    t1 = clock.perf_counter()
    s.time_end = run.dt
    jax.block_until_ready(s.run(Q, A)[:2])                    # compile + 1 step
    t2 = clock.perf_counter()
    s.time_end, march = run.t_end, []
    for _ in range(reps):
        t = clock.perf_counter()
        out = s.run(Q, A)
        jax.block_until_ready(out[:2])
        march.append(clock.perf_counter() - t)
    steps = round(run.t_end / run.dt)
    return Measurement(cells=int(Q0.shape[1]), t=float(out[0]), Q=np.asarray(out[1]), names=names,
                       setup=t1 - t0, compile=max(0.0, t2 - t1 - min(march) / steps), march=march,
                       device=jax.default_backend(),
                       detail=f"pressure={s._pressure_backend()}" if run.chorin else None)


# ── foam / AMReX ────────────────────────────────────────────────────────────
def _pin_amrex(module):
    """AMReX runs on the first core of this process's affinity, builds on all of it."""
    cpus = sorted(os.sched_getaffinity(0))
    module.RUN_CPUS, module.BUILD_CPUS = str(cpus[0]), timing_records._cpulist(cpus)


def _timed(cls):
    """``cls`` with ``build_s``, the seconds ``build`` took (compile, or the cache hit)."""
    class Timed(cls):
        def build(self, *a, **k):
            t = clock.perf_counter()
            try:
                return super().build(*a, **k)
            finally:
                self.build_s = clock.perf_counter() - t
    return Timed


def _cxx(run, backend, config, reps, work):
    """setup = everything but compile and march (model, printing, case files,
    container start, read-back); compile = the build (a hash-cached binary: ~0
    after the first); march = the solver's own 'march wall time'."""
    import cases
    from backends import _solver_kw
    kw = dict(cfl_check_every=0)
    if backend == "amrex":
        from zoomy_amrex import symsolver as S
        _pin_amrex(S)
        cls = S.SymbolicChorinSolverAmrex if run.chorin else S.SymbolicHyperbolicSolverAmrex
        kw.update(ranks=1, gpu=config.startswith("gpu"))
        if run.chorin:
            kw["max_grid_size"] = 8
    else:
        from zoomy_foam import symsolver as S
        cls = S.SymbolicChorinSolverFoam if run.chorin else S.SymbolicHyperbolicSolverFoam
    kw.update(_solver_kw(run, cls))
    Timed = _timed(cls)
    first = clock.perf_counter()
    prepared = cases.prepare(run)
    mesh, systems, Q0, names = prepared
    march, setup, comp, steps_run = [], None, None, None
    for rep in range(reps):
        t0 = first if rep == 0 else clock.perf_counter()
        d = Path(work) / f"rep{rep}"
        s = Timed(**kw)
        t, Q = s.run(mesh, systems, Q0, d if backend == "amrex" else str(d))
        total = clock.perf_counter() - t0
        if backend == "amrex":
            wall, steps_run = s.wall, s.steps
        else:
            wall = float(re.findall(r"march wall time ([\d.eE+-]+)",
                                    (d / "log.symFoam").read_text())[-1])
        march.append(wall)
        if rep == 0:
            setup, comp = total - wall - s.build_s, s.build_s
    return Measurement(cells=int(Q0.shape[1]), t=float(t), Q=np.asarray(Q), names=names, setup=setup,
                       compile=comp, march=march, steps_run=steps_run)


# ── entry points ────────────────────────────────────────────────────────────
def _measure(tc, backend, config, reps, work):
    run = tc.run()
    return _jax(run, reps) if backend == "jax" else _cxx(run, backend, config, reps, work)


def _child(tc, backend, config, reps, work):
    """The measurement in a child process: JAX on a GPU needs the CUDA env
    (backends.Jax.launch); ZOOMY_TIMING_SUBPROCESS=1 forces it for any config."""
    from backends import Jax
    prefix = Jax().launch(config) or [sys.executable]
    out = Path(tempfile.mkdtemp(prefix="child_", dir=_mk(timing_paths.work_dir())))
    (out / "spec.json").write_text(json.dumps(dict(case=tc.key, backend=backend, config=config,
                                                   reps=reps, work=str(work))))
    r = subprocess.run(prefix + [str(Path(__file__).resolve()), "worker", str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"timing worker failed ({r.returncode}):\n{(r.stdout + r.stderr)[-3000:]}")
    meta = json.loads((out / "measure.json").read_text())
    m = Measurement(Q=np.load(out / "Q.npy"), **meta)
    shutil.rmtree(out, ignore_errors=True)
    return m


def _mk(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def measure(tc, backend, config, reps, work):
    """March ``tc`` ``reps`` times on ``backend`` / ``config``; returns a Measurement."""
    if backend == "jax" and (config.startswith("gpu") or os.environ.get("ZOOMY_TIMING_SUBPROCESS")):
        return _child(tc, backend, config, reps, work)
    return _measure(tc, backend, config, reps, _mk(work))


# ── the JAX serial reference (cached on disk across tests and processes) ────
@lru_cache(maxsize=1)
def _digest():
    """Library commits and the mtimes of the zoomy_core / zoomy_jax sources and of the cases."""
    h = hashlib.sha1(json.dumps(timing_records.metadata()["libs"], sort_keys=True).encode())
    for pkg in ("zoomy_core", "zoomy_jax"):
        root = Path(__import__(pkg).__file__).parent
        for f in sorted(root.rglob("*.py")):
            st = f.stat()
            h.update(f"{f.relative_to(root)}{st.st_size}{st.st_mtime_ns}".encode())
    h.update(Path(timing_cases.__file__).read_bytes())
    return h.hexdigest()[:12]


def _ref_path(tc):
    return Path(timing_paths.work_dir()) / "ref" / f"{tc.key}_{_digest()}.npz"


def store_reference(tc, m):
    p = _ref_path(tc)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, Q=m.Q, t=m.t, names=np.array(m.names))


def reference(tc):
    """(Q, names) of the JAX serial run of ``tc``: from the cache, else run once."""
    p = _ref_path(tc)
    if not p.exists():
        store_reference(tc, measure(tc, "jax", "serial", 1, timing_paths.work_dir() / "ref_run" / tc.key))
    d = np.load(p)
    return d["Q"], [str(n) for n in d["names"]]


def agreement(spec, Q, names, ref):
    """{metric, value, tol, ok} of ``Q`` against the reference ``ref`` = (Q, names) by the
    matrix's own measure and threshold."""
    import matrix
    value = getattr(matrix, spec.metric)(dict(Q=Q, names=names), dict(Q=ref[0], names=ref[1]))
    op, bound, source = spec.threshold()
    ok = bool(value <= bound if op == "<=" else value >= bound)
    return dict(metric=spec.metric, value=float(value), tol=f"{op} {bound:g}", source=source, ok=ok)


def _worker(d):
    spec = json.loads((d / "spec.json").read_text())
    m = _measure(timing_cases.CASES[spec["case"]], spec["backend"], spec["config"], spec["reps"],
                 Path(spec["work"]))
    np.save(d / "Q.npy", m.Q)
    (d / "measure.json").write_text(json.dumps(m.meta()))


if __name__ == "__main__" and sys.argv[1:2] == ["worker"]:
    _worker(Path(sys.argv[2]))
