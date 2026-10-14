"""Malpasset on the thesis configuration (ANALYSIS 48): the GUI session model
(`library/zoomy_gui/projects/tools/symsolver/malpasset_jax_model.py`: SME(1) mixing length, closed-form
spectrum, HLL with the hydrostatic reconstruction, order 2 Venkatakrishnan + front detector + MOOD,
point-implicit source, CFL 0.4) on the JAX solver, the CFL wave-speed rule `--rule`, a write every
`--dt-write` s.  Stores `LABEL.npz` beside this file: per write t, h_min, the water volume, the wet
area (h > 0.05 m), max |q_1| / h over h > 1e-3 m; the max-depth envelope and the library commits
(the PYTHONPATH entries; `--note` where they have no git).

    PYTHONPATH=<library>/zoomy_core:<library>/zoomy_jax python run.py LABEL --rule faces|cells \
        [--t-end 2000] [--dt-write 1]
"""
import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "library", "zoomy_gui", "projects", "tools", "symsolver"))
MESH = os.environ.get("MALPASSET_MESH", os.path.join(ROOT, "data", "malpasset",
                                                     "geo_malpasset-small.msh"))
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--rule", required=True, choices=["faces", "cells"])
ap.add_argument("--t-end", type=float, default=2000.0)
ap.add_argument("--dt-write", type=float, default=1.0)
ap.add_argument("--note", default="", help="the libraries, where PYTHONPATH has no git")
a = ap.parse_args()

from zoomy_core.mesh import LSQMesh  # noqa: E402
from zoomy_core.symsolver.discretization.time import CflWaveSpeed  # noqa: E402
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax  # noqa: E402
import malpasset_jax_model as M  # noqa: E402

settings = {"model": {"type": "MalpassetSME", "level": 1, "dimension": 3, "closure": "mixing",
                      "k_s": 0.05, "zeta_p": 0.15, "quadrature_order": 5, "wet_dry_eps": 0.001,
                      "u_max": 30.0, "desingularize": True, "clamp": True},
            "numerics": {"order": 2, "limiter": "venkatakrishnan", "front_tol": 0.5, "mood": True,
                         "riemann": "hr", "source_treatment": "linearized", "cfl": 0.4},
            "mesh": MESH}
mesh = LSQMesh.from_msh(MESH)
model = M.build_model(settings, mesh, case_dir="/")
idx = model["idx"]
ih, iq1 = idx["h"], [idx["q_x_1"], idx["q_y_1"]]
nc = mesh.n_inner_cells
vol = np.asarray(mesh.cell_volumes)[:nc]
rows, env = [], {}


def on_write(t, Q, Qaux):
    """One row per write and the running max-depth envelope."""
    Q = np.asarray(Q)
    h = Q[ih]
    a1 = np.abs(Q[iq1]).max(0) / np.maximum(h, 1e-14)
    rows.append((float(t), float(h.min()), float(h @ vol), float(vol[h > 0.05].sum()),
                 float(np.where(h > 1e-3, a1, 0.0).max())))
    env["hmax"] = h.copy() if "hmax" not in env else np.maximum(env["hmax"], h)


solver = SymbolicHyperbolicSolverJax(
    time_end=a.t_end, cfl=0.4, mood=True, cfl_wave_speed=CflWaveSpeed(a.rule),
    dt_write=a.dt_write, on_write=on_write)
t0 = time.time()
Q, Qaux = solver.setup(mesh, model["nsm"])
t_end, Q, _ = solver.run(Q, Qaux)
lib = {os.path.basename(d): subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"],
                                           capture_output=True, text=True).stdout.strip()
       for d in os.environ.get("PYTHONPATH", "").split(os.pathsep) if d}
r = np.array(rows)
np.savez(os.path.join(HERE, f"{a.label}.npz"), t=r[:, 0], h_min=r[:, 1], volume=r[:, 2],
         wet_area=r[:, 3], alpha1_max=r[:, 4], hmax=env["hmax"].astype(np.float32),
         t_end=float(t_end), rule=a.rule, wall=time.time() - t0, lib=json.dumps(lib), note=a.note)
print(f"{a.label}: t_end {float(t_end)}  min h_min {r[:, 1].min():.3e}  "
      f"max alpha1 {r[:, 4].max():.3e}  wall {time.time() - t0:.0f} s  {lib}")
