"""Gate 26: gate 10 (foam <-> foam, preCICE) with each participant on N ranks:
each rank registers only its own coupling vertices (preCICE rank/size).
    python gate26_foam_couple_parallel.py NLEFT NRIGHT"""
import os
import shutil
import sys
import threading

import numpy as np

import couple_side as CS
from gate9_foam import FAILED, WORK as GATE_WORK, check

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(GATE_WORK, "xcouple_par")
WINDOW = CS.WINDOW

shutil.rmtree(WORK, ignore_errors=True)
os.makedirs(WORK)
config = str(CS.write_config(WORK))

NP = dict(Left=int(sys.argv[1]), Right=int(sys.argv[2]))   # ranks per participant


def foam_side(name):
    return CS.cxx_side("foam", name, config, WORK, NP[name])


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
m = CS.vs_single(Qc)
print(f"foam(Left, {NP['Left']} ranks) <-> foam(Right, {NP['Right']} ranks), O1, "
      f"window {WINDOW}: t foam {tR:.6g}, "
      f"max|coupled-single| {m['coupled_vs_single']:.2e}, mass drift {m['mass']:.1e}")
check("coupled vs single max", m["coupled_vs_single"], 1e-14)
check("mass drift", m["mass"], 1e-14)
print("G26_DONE" if not FAILED else f"G26_FAILED {FAILED}")
