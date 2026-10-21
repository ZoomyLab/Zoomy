"""Timing records: one JSON line per test in timings.jsonl, the latest record
of every (case, backend, config) in timings_latest.json (README.md).

Inputs: the fields of a measurement.  Outputs: the files in ``timing_paths.results_dir()``."""
import fcntl
import importlib.util
import json
import os
import socket
import statistics
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import timing_paths

SCHEMA = 1
LIBS = ("zoomy_core", "zoomy_jax", "zoomy_foam", "zoomy_amrex")


def _out(*cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _commit(path):
    top = _out("git", "-C", str(path), "rev-parse", "--show-toplevel")
    if top is None:
        return None
    sha = _out("git", "-C", top, "rev-parse", "--short", "HEAD")
    dirty = _out("git", "-C", top, "status", "--porcelain", "--untracked-files=no")
    return None if sha is None else sha + ("+dirty" if dirty else "")


def _cpulist(cpus):
    out, cpus = [], sorted(cpus)
    while cpus:
        a = b = cpus.pop(0)
        while cpus and cpus[0] == b + 1:
            b = cpus.pop(0)
        out.append(str(a) if a == b else f"{a}-{b}")
    return ",".join(out)


@lru_cache(maxsize=1)
def metadata():
    """Host, CPU, GPU, pinning, library commits and wave-speed method (cached per process)."""
    cpu = next((ln.split(":", 1)[1].strip() for ln in (_out("lscpu") or "").splitlines()
                if ln.startswith("Model name")), None)
    gpus = (_out("nvidia-smi", "--query-gpu=name", "--format=csv,noheader") or "").splitlines()
    libs = {}
    for name in LIBS:
        spec = importlib.util.find_spec(name)
        libs[name] = _commit(Path(spec.origin).parent) if spec and spec.origin else None
    libs["superrepo"] = _commit(timing_paths.ROOT)
    from zoomy_core.systemmodel.system_model import SystemModel
    wave = str(SystemModel.spectral_radius_method)   # the default without a closed form
    cpus = os.sched_getaffinity(0)
    return dict(host=socket.gethostname(), cpu=cpu, gpu=", ".join(sorted(set(gpus))) or None,
                n_gpus=len(gpus), cores_pinned=len(cpus), cpus=_cpulist(cpus), libs=libs,
                wave_speed=wave)


def record(case, backend, config, *, cells, steps, dt, march, setup, compile, status="ok",
           device=None, detail=None, diff=None):
    """One record.  ``march``: the march times (s) of the repetitions; min and
    median, ms/step and us/cell/step (of the min) are derived.  ``diff``:
    {metric, value, tol} of the agreement with JAX serial, or None."""
    best = min(march)
    return dict(schema=SCHEMA, date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                **metadata(), case=case, backend=backend, config=config, device=device,
                detail=detail, cells=cells, steps=steps, dt=dt, reps=len(march), march_s=list(march),
                min_s=best, median_s=statistics.median(march), ms_step=1e3 * best / steps,
                us_cell_step=1e6 * best / steps / cells, setup_s=setup, compile_s=compile,
                status=status, diff=diff)


def failure(case, backend, config, status):
    """A record of a test that produced no times (the run raised)."""
    return dict(schema=SCHEMA, date=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                **metadata(), case=case, backend=backend, config=config, status=status)


def key(rec):
    return f"{rec['case']}/{rec['backend']}/{rec['config']}"


def append(rec, directory=None):
    """Append ``rec`` to timings.jsonl and rewrite timings_latest.json from the
    whole file (the last record of every key); safe for concurrent writers."""
    d = Path(directory or timing_paths.results_dir())
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "timings.jsonl", "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + "\n")
        f.flush()
        f.seek(0)
        latest = {}
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            latest[key(r)] = r
        rows = [latest[k] for k in sorted(latest)]
        tmp = d / f".timings_latest.{os.getpid()}.json"
        tmp.write_text(json.dumps(dict(schema=SCHEMA, updated=rec["date"], rows=rows), indent=1) + "\n")
        os.replace(tmp, d / "timings_latest.json")
    return d / "timings.jsonl"
