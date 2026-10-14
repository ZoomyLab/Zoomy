"""Gate preCICE-amrex: gate 3's 1-D dam break split at x = 0.5 into two
preCICE participants (parallel explicit, window 1e-3 = the step, O1), vs the
single-domain march with the same fixed step (JAX).

    aa NL NR   AMReX(Left, NL ranks) <-> AMReX(Right, NR ranks)
    aj NL      AMReX(Left, NL ranks) <-> JAX(Right)

    python gate38_amrex_precice.py aa 1 1 | aa 2 2 | aj 1 | aj 2
Needs ZOOMY_AMREX_CONTAINERS pointing at the images with libprecice 3.4.0.
"""
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

import couple_side as CS

GATES = CS.HERE
WINDOW = CS.WINDOW
mode, NL = sys.argv[1], int(sys.argv[2])
NR = int(sys.argv[3]) if mode == "aa" else 1
WORK = Path(os.environ.get("GATE_WORK", "/tmp")) / f"precice_amrex_{mode}{NL}{NR}"
shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True)
config = CS.write_config(WORK)


def amrex_side(name, ranks):
    run = CS.cxx_side("amrex", name, config, WORK, ranks)
    return lambda: run()[1]


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
m = CS.vs_single(Qc)
peer = f"AMReX({NR})" if mode == "aa" else "JAX"
print(f"AMReX(Left, {NL}) <-> {peer}(Right), O1, window {WINDOW}: "
      f"max|coupled - single| {m['coupled_vs_single']:.2e}, mass drift {m['mass']:.1e}", flush=True)
