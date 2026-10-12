"""Gate preCICE-amrex: gate 3's 1-D dam break split at x = 0.5 into two
preCICE participants (parallel explicit, window 1e-3 = the step, O1), vs the
single-domain march with the same fixed step (JAX).

    aa NL NR   AMReX(Left, NL ranks) <-> AMReX(Right, NR ranks)
    aj NL      AMReX(Left, NL ranks) <-> JAX(Right)

    python gate38_amrex_precice.py aa 1 1 | aa 2 2 | aj 1 | aj 2
Needs ZOOMY_AMREX_CONTAINERS pointing at the images with libprecice 3.4.0.
"""
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

import common as C
from gate3_precice import single
from zoomy_amrex.symsolver import SymbolicHyperbolicSolverAmrex
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

GATES = Path(C.__file__).resolve().parent
T, WINDOW, h, dx = 0.05, 1e-3, 1, 1 / 200
mode, NL = sys.argv[1], int(sys.argv[2])
NR = int(sys.argv[3]) if mode == "aa" else 1
WORK = Path(os.environ.get("GATE_WORK", "/tmp")) / f"precice_amrex_{mode}{NL}{NR}"
shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True)
cfg = (GATES / "precice-config.xml").read_text()
cfg = re.sub(r'time-window-size value="[^"]*"', f'time-window-size value="{WINDOW}"', cfg)
cfg = re.sub(r'max-time value="[^"]*"', f'max-time value="{T}"', cfg)
cfg = cfg.replace('exchange-directory="."', f'exchange-directory="{WORK}"')
# a point interface (1-D): zero-size bounding boxes, the geometric filter would drop it
cfg = re.sub(r'(<receive-mesh [^/]*)/>', r'\1geometric-filter="no-filter" />', cfg)
config = WORK / "precice-config.xml"
config.write_text(cfg)


def amrex_side(name, ranks):
    left = name == "Left"
    mesh, nsm = C.dambreak_1d(0, 1, domain=(0.0, 0.5) if left else (0.5, 1.0),
                              nx=100, x_dam=1.0 if left else 0.0)
    me, peer = ("L", "R") if left else ("R", "L")
    sj = SymbolicHyperbolicSolverJax(time_end=T)
    Q0, _ = sj.setup(mesh, nsm)
    s = SymbolicHyperbolicSolverAmrex(cfl=0.9, ranks=ranks, couple=dict(
        participant=name, config=str(config), mesh=f"{name}-Mesh",
        write=[f"{me}{i}" for i in range(3)], read=[f"{peer}{i}" for i in range(3)],
        tag="right" if left else "left", side=0 if left else 1))
    s.build(sj.nsm, 1)                          # compile before the peer waits
    return lambda: s.run(sj.mesh_np, sj.nsm, np.asarray(Q0), WORK / name)[1]


def jax_right():
    env = {**os.environ, "PYTHONPATH": f"{GATES}{os.pathsep}{os.environ['PYTHONPATH']}"}
    p = subprocess.run([sys.executable, str(GATES / "participant.py"), "Right", "1",
                        "Right.npy", "window"], cwd=WORK, env=env, capture_output=True,
                       text=True)
    if p.returncode:
        raise RuntimeError(p.stdout[-2000:] + p.stderr[-2000:])
    return np.load(WORK / "Right.npy")


runs = {"Left": amrex_side("Left", NL),
        "Right": amrex_side("Right", NR) if mode == "aa" else jax_right}
res, err = {}, {}


def go(n, f):
    try:
        res[n] = f()
    except Exception as e:                      # noqa: BLE001
        err[n] = e


threads = [threading.Thread(target=go, args=(n, f)) for n, f in runs.items()]
for th in threads:
    th.start()
for th in threads:
    th.join()
if err:
    raise SystemExit(f"FAILED {err}")
Qc = np.concatenate([res["Left"], res["Right"]], 1)
Q0s, Qs = single(1, fixed_dt=WINDOW)
peer = f"AMReX({NR})" if mode == "aa" else "JAX"
print(f"AMReX(Left, {NL}) <-> {peer}(Right), O1, window {WINDOW}: "
      f"max|coupled - single| {np.abs(Qc - Qs).max():.2e}, "
      f"mass drift {abs(Qc[h].sum() * dx - Q0s[h].sum() * dx):.1e}", flush=True)
