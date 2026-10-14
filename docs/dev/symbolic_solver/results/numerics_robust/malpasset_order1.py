"""ANALYSIS 49c: Malpasset SME(1) mixing length, order 1, CFL 0.4, HLL + hydrostatic reconstruction
('hr'), point-implicit source, faces rule (--rule), to t = 120 s: closed-form vs numerical spectrum,
the numerical one with HLL's Davis bounds or the former Rusanov pair (--hll rusanov).  Prints h_min
over the writes, the water in the initially dry cells ("released") and the volumes.

    PYTHONPATH=<library>/zoomy_core:<library>/zoomy_jax:<library>/zoomy_gui/projects/tools/symsolver \
    MALPASSET_MESH=<data>/malpasset/geo_malpasset-small.msh python malpasset_order1.py C --spectrum closed
"""
import argparse, json, os, sys, time
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("label"); ap.add_argument("--spectrum", choices=["closed", "numerical"], required=True)
ap.add_argument("--hll", choices=["davis", "rusanov"], default="davis")
ap.add_argument("--t-end", type=float, default=120.0)
ap.add_argument("--rule", choices=["faces", "cells"], default="faces")
ap.add_argument("--no-clean", action="store_true")
ap.add_argument("--dt-write", type=float, default=5.0)
a = ap.parse_args()
import malpasset_jax_model as M
from zoomy_core.fvm import riemann_solvers as RS
if a.hll == "rusanov":
    RS.HLL.wave_speed_bounds = RS.Numerics.wave_speed_bounds
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.numerics import Limiter, NumericalSystemModel, ReconstructionSpec, SourceTreatment
from zoomy_core.symsolver.discretization.time import CflWaveSpeed
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
MESH = os.environ["MALPASSET_MESH"]
mesh = LSQMesh.from_msh(MESH)
closures, params, quad = M.closures_and_parameters({"closure": "mixing", "k_s": 0.05, "zeta_p": 0.15,
                                                    "quadrature_order": 5, "wet_dry_eps": 1e-3})
sme = M.MalpassetSME(level=1, dimension=3, closures=closures, desingularize=True, clamp=True, u_max=30.0,
                     parameters=params, quadrature_order=quad, symbolic_spectrum=a.spectrum == "closed")
sm = sme.system_model
ic, idx = M.build_ic(sm, mesh_path=MESH)
bcs, tags = M.wall_bcs(sm, idx, mesh)
sm.attach_boundary_conditions(bcs)
sm.initial_conditions = ic
sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
nsm = NumericalSystemModel.from_system_model(sm, riemann=M._RIEMANN["hr"],
                                             reconstruction=ReconstructionSpec(order=1),
                                             source_treatment=SourceTreatment.linearized,
                                             **({'depth_round_off': None} if a.no_clean else {}))
print("eigenvalues closed:", nsm.eigenvalues is not None, flush=True)
ih = idx["h"]; nc = mesh.n_inner_cells; vol = np.asarray(mesh.cell_volumes)[:nc]
rows = []
def on_write(t, Q, A):
    h = np.asarray(Q)[ih]
    rows.append((float(t), float(h.min()), float(h @ vol), h.copy()))
s = SymbolicHyperbolicSolverJax(time_end=a.t_end, cfl=0.4, mood=False, cfl_wave_speed=CflWaveSpeed(a.rule),
                                dt_write=a.dt_write, on_write=on_write)
t0 = time.time()
Q, A = s.setup(mesh, nsm)
t, Q, _ = s.run(Q, A)
h0 = rows[0][3]; h = rows[-1][3]
released = float(h[h0 <= 0] @ vol[h0 <= 0])
np.savez(f"order1_{a.label}.npz", t=[r[0] for r in rows], hmin=[r[1] for r in rows], vol=[r[2] for r in rows])
print(json.dumps(dict(label=a.label, t=float(t), h_min=min(r[1] for r in rows), released=released,
                      vol0=rows[0][2], vol_end=rows[-1][2], wall=time.time() - t0)), flush=True)
