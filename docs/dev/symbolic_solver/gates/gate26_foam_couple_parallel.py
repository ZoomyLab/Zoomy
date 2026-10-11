"""Gate 26: gate 10 (foam <-> foam, preCICE) with each participant on N ranks:
each rank registers only its own coupling vertices (preCICE rank/size).
    python gate26_foam_couple_parallel.py NLEFT NRIGHT"""
import os
import re
import shutil
import sys
import threading

import common as C
import numpy as np

from gate3_precice import single
from gate9_foam import FAILED, WORK as GATE_WORK, check
from zoomy_foam.symsolver import SymbolicHyperbolicSolverFoam
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(GATE_WORK, "xcouple_par")
T, WINDOW, h, dx = 0.05, 1e-3, 1, 1 / 200

shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
cfg = open(os.path.join(HERE, "precice-config.xml")).read()
cfg = re.sub(r'time-window-size value="[^"]*"', f'time-window-size value="{WINDOW}"', cfg)
cfg = re.sub(r'max-time value="[^"]*"', f'max-time value="{T}"', cfg)
cfg = cfg.replace('exchange-directory="."', f'exchange-directory="{WORK}"')
# a point interface (1-D): zero-size bounding boxes, the geometric filter would drop it
cfg = re.sub(r'(<receive-mesh [^/]*)/>', r'\1geometric-filter="no-filter" />', cfg)
config = os.path.join(WORK, "precice-config.xml")
open(config, "w").write(cfg)

NP = dict(Left=int(sys.argv[1]), Right=int(sys.argv[2]))   # ranks per participant


def foam_side(name):
    left = name == "Left"
    mesh, nsm = C.dambreak_1d(0, 1, domain=(0.0, 0.5) if left else (0.5, 1.0),
                              nx=100, x_dam=1.0 if left else 0.0)
    me, peer = ("L", "R") if left else ("R", "L")
    sj = SymbolicHyperbolicSolverJax(time_end=T)
    Q0, _ = sj.setup(mesh, nsm)
    sf = SymbolicHyperbolicSolverFoam(cfl=0.9, nprocs=NP[name], couple=dict(
        participant=name, config=config, mesh=f"{name}-Mesh",
        write=[f"{me}{i}" for i in range(3)], read=[f"{peer}{i}" for i in range(3)],
        tag="right" if left else "left", side=0 if left else 1))
    sf.build(sj.nsm, 1)                             # compile before the peer waits
    return lambda: sf.run(sj.mesh_np, sj.nsm, np.asarray(Q0), os.path.join(WORK, name))


res, runs = {}, {"Right": foam_side("Right"), "Left": foam_side("Left")}
threads = [threading.Thread(target=lambda n=n, f=f: res.update({n: f()}))
           for n, f in runs.items()]
for th in threads:
    th.start()
for th in threads:
    th.join()
tR, QR = res["Right"]
QL = res["Left"][1]
Qc = np.concatenate([QL, QR], 1)
Q0s, Qs = single(1, fixed_dt=WINDOW)
print(f"foam(Left, {NP['Left']} ranks) <-> foam(Right, {NP['Right']} ranks), O1, "
      f"window {WINDOW}: t foam {tR:.6g}, "
      f"max|coupled-single| {np.abs(Qc - Qs).max():.2e}, "
      f"mass drift {abs(Qc[h].sum() * dx - Q0s[h].sum() * dx):.1e}")
check("coupled vs single max", np.abs(Qc - Qs).max(), 1e-14)
check("mass drift", abs(Qc[h].sum() * dx - Q0s[h].sum() * dx), 1e-14)
print("G26_DONE" if not FAILED else f"G26_FAILED {FAILED}")
