"""Gate 10: preCICE with the C++ backend, both running the printed
hyperbolic_precice template.  O1, one step per window (window 1e-3 <= CFL
step) vs the single-domain JAX march.
    python gate10_xcouple.py jax    Left = JAX, Right = foam
    python gate10_xcouple.py foam   Left = foam, Right = foam"""
import os
import re
import shutil
import subprocess
import sys
import threading

import common as C
import numpy as np

from gate3_precice import single
from zoomy_foam.symsolver import SymbolicHyperbolicSolverFoam
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = ("/tmp/claude-765404697/-mnt-userdrive-Users-home-adam-obbpb5az1dhsjzf-git-"
        "zotero-rag/b5667895-053e-413b-8d69-8137a92fc71c/scratchpad/foam_cases/xcouple")
T, WINDOW, h, dx = 0.05, 1e-3, 1, 1 / 200

shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
cfg = open(os.path.join(HERE, "precice-config.xml")).read()
cfg = re.sub(r'time-window-size value="[^"]*"', f'time-window-size value="{WINDOW}"', cfg)
cfg = re.sub(r'max-time value="[^"]*"', f'max-time value="{T}"', cfg)
cfg = cfg.replace('exchange-directory="."', f'exchange-directory="{WORK}"')
config = os.path.join(WORK, "precice-config.xml")
open(config, "w").write(cfg)

LEFT = sys.argv[1] if len(sys.argv) > 1 else "jax"


def foam_side(name):
    left = name == "Left"
    mesh, nsm = C.dambreak_1d(0, 1, domain=(0.0, 0.5) if left else (0.5, 1.0),
                              nx=100, x_dam=1.0 if left else 0.0)
    me, peer = ("L", "R") if left else ("R", "L")
    sj = SymbolicHyperbolicSolverJax(time_end=T)
    Q0, _ = sj.setup(mesh, nsm)
    sf = SymbolicHyperbolicSolverFoam(cfl=0.9, couple=dict(
        participant=name, config=config, mesh=f"{name}-Mesh",
        write=[f"{me}{i}" for i in range(3)], read=[f"{peer}{i}" for i in range(3)],
        tag="right" if left else "left", side=0 if left else 1))
    sf.build(sj.nsm, 1)                             # compile before the peer waits
    return lambda: sf.run(sj.mesh_np, sj.nsm, np.asarray(Q0), os.path.join(WORK, name))


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
Q0s, Qs = single(1, fixed_dt=WINDOW)
print(f"{LEFT}(Left) <-> foam(Right), O1, window {WINDOW}: t foam {tR:.6g}, "
      f"max|coupled-single| {np.abs(Qc - Qs).max():.2e}, "
      f"mass drift {abs(Qc[h].sum() * dx - Q0s[h].sum() * dx):.1e}")
print("G10_DONE")
