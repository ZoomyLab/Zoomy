"""Gate 10: preCICE with the C++ backend, both running the printed
hyperbolic_precice template.  O1, one step per window (window 1e-3 <= CFL
step) vs the single-domain JAX march.
    python gate10_xcouple.py jax    Left = JAX, Right = foam
    python gate10_xcouple.py foam   Left = foam, Right = foam"""
import os
import shutil
import subprocess
import sys
import threading

import numpy as np

import couple_side as CS

from gate9_foam import WORK as GATE_WORK

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(GATE_WORK, "xcouple")
WINDOW = CS.WINDOW

shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
config = str(CS.write_config(WORK))

LEFT = sys.argv[1] if len(sys.argv) > 1 else "jax"


def foam_side(name):
    return CS.cxx_side("foam", name, config, WORK)


res, runs = {}, {"Right": foam_side("Right")}
if LEFT == "foam":
    runs["Left"] = foam_side("Left")
else:
    env = {**os.environ, "PYTHONPATH": HERE + os.pathsep + os.environ["PYTHONPATH"]}
    # LEFT_CMD (e.g. "micromamba run -n zoomy-cuda python") + LEFT_ENV_* run
    # the JAX participant elsewhere, e.g. on the GPU
    cmd = os.environ.get("LEFT_CMD", sys.executable).split()
    env.update({k[len("LEFT_ENV_"):]: v for k, v in os.environ.items()
                if k.startswith("LEFT_ENV_")})
    if "LEFT_ENV_PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = HERE + os.pathsep + os.environ["LEFT_ENV_PYTHONPATH"]
    left = subprocess.Popen([*cmd, os.path.join(HERE, "participant.py"),
                             "Left", "1", "Left.npy", "window"], cwd=WORK, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
threads = [threading.Thread(target=lambda n=n, f=f: res.update({n: f()}))
           for n, f in runs.items()]
for th in threads:
    th.start()
if LEFT != "foam":
    out = left.communicate(timeout=1200)[0]
    if left.returncode:
        print(out[-3000:])
    print([ln for ln in out.splitlines() if ln.startswith("Left:")])
    res["Left"] = (None, np.load(os.path.join(WORK, "Left.npy")))
for th in threads:
    th.join()
tR, QR = res["Right"]
QL = res["Left"][1]
Qc = np.concatenate([QL, QR], 1)
m = CS.vs_single(Qc)
print(f"{LEFT}(Left) <-> foam(Right), O1, window {WINDOW}: t foam {tR:.6g}, "
      f"max|coupled-single| {m['coupled_vs_single']:.2e}, mass drift {m['mass']:.1e}")
print("G10_DONE")
