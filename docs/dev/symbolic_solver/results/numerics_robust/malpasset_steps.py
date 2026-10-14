"""ANALYSIS 49b: Malpasset on the thesis configuration (`../wavespeed_bound/run.py`'s settings), cells
rule, to --t-end: the step count (check.march's outputs of the JAX program) and the march wall
time; --old-gate adds the former dt gate h > wet_dry_eps (1e-3).  Environment as malpasset_order1.py.
"""
import argparse, json, os, sys, time
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("label"); ap.add_argument("--t-end", type=float, default=300.0)
ap.add_argument("--old-gate", action="store_true"); a = ap.parse_args()
import malpasset_jax_model as M
from zoomy_core.mesh import LSQMesh
from zoomy_core.symsolver.discretization.time import CflWaveSpeed
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
MESH = os.environ["MALPASSET_MESH"]
settings = {"model": {"type": "MalpassetSME", "level": 1, "dimension": 3, "closure": "mixing",
                      "k_s": 0.05, "zeta_p": 0.15, "quadrature_order": 5, "wet_dry_eps": 0.001,
                      "u_max": 30.0, "desingularize": True, "clamp": True},
            "numerics": {"order": 2, "limiter": "venkatakrishnan", "front_tol": 0.5, "mood": True,
                         "riemann": "hr", "source_treatment": "linearized", "cfl": 0.4},
            "mesh": MESH}
mesh = LSQMesh.from_msh(MESH)
if a.old_gate:     # the former CFL gate h > wet_dry_eps (1e-3), for comparison
    from zoomy_core.numerics import numerical_system_model as NSMmod
    from zoomy_core.systemmodel.operations import gate_eigenvalues_dry
    orig = NSMmod.NumericalSystemModel.from_system_model.__func__
    def fsm(cls, sm, **kw):
        kw["extra_operations"] = list(kw.get("extra_operations") or []) + [gate_eigenvalues_dry(1e-3)]
        return orig(cls, sm, **kw)
    NSMmod.NumericalSystemModel.from_system_model = classmethod(fsm)
model = M.build_model(settings, mesh, case_dir="/")
s = SymbolicHyperbolicSolverJax(time_end=a.t_end, cfl=0.4, mood=True, cfl_wave_speed=CflWaveSpeed.cells,
                                dt_write=a.t_end)
Q, A = s.setup(mesh, model["nsm"])
import jax
tok = jax.numpy.int64(0)
t0 = time.time()
out = s._jit_fixed(Q, A, s.parameters, 0.0, a.t_end, tok); jax.block_until_ready(out)
w1 = time.time() - t0          # compile + run
t0 = time.time()
out = s._jit_fixed(Q, A, s.parameters, 0.0, a.t_end, tok); jax.block_until_ready(out)
w2 = time.time() - t0          # run only
n = int(out[7]) if len(out) > 4 else -1
print(json.dumps(dict(label=a.label, t=float(out[0]), steps=n, wall_first=w1, wall_run=w2,
                      per_step_ms=(1e3 * w2 / n if n > 0 else None))), flush=True)
