"""Speed: symbolic foam (symFoam) vs zoomyFoam, Stoker wet dam break (SWASHES
domain, SME(0) 1-D, extrapolation ends), same model, same Riemann solver
(PositiveNonconservativeRusanov), same CFL.  Solver wall time only (no
apptainer start, no build).

    python bench_foam.py N ORDER [T_END]
"""
import os
import re
import sys

import common  # noqa: F401
import numpy as np

T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../library/zoomy_foam/tests")
sys.path.insert(0, T)
import foam_cases as FC  # noqa: E402
import foam_models as FM  # noqa: E402

from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov  # noqa: E402
from zoomy_core.mesh import LSQMesh  # noqa: E402
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec  # noqa: E402
from zoomy_core.systemmodel.system_model import SystemModel  # noqa: E402
from zoomy_foam.symsolver import SymbolicHyperbolicSolverFoam  # noqa: E402

WORK = ("/tmp/claude-765404697/-mnt-userdrive-Users-home-adam-obbpb5az1dhsjzf-git-"
        "zotero-rag/b5667895-053e-413b-8d69-8137a92fc71c/scratchpad/foam_bench")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
ORDER = int(sys.argv[2]) if len(sys.argv) > 2 else 1
T_END = float(sys.argv[3]) if len(sys.argv) > 3 else FC.SWASHES_T_END
DOMAIN, CFL = FC.SWASHES_DOMAIN, 0.9

import zoomy_core.model.initial_conditions as IC  # noqa: E402
import zoomy_core.model.boundary_conditions as BC  # noqa: E402
from zoomy_core.model.models import SME  # noqa: E402
from zoomy_core.model.models.closures import (  # noqa: E402
    ManningFriction, Newtonian, StressFree)
model = SME(level=0, dimension=2,
            closures=[Newtonian(), ManningFriction(), StressFree()],
            boundary_conditions=BC.BoundaryConditions(
                [BC.Extrapolation(tag=t) for t in ("left", "right")]),
            initial_conditions=IC.UserFunction(function=FC.stoker_ic),
            aux_initial_conditions=IC.Constant(constants=lambda n: np.zeros(n)))
Qz, _, info = FC.march(model, f"{WORK}/zoomyfoam_{N}_{ORDER}", n_inner_cells=N,
                       domain=DOMAIN, t_end=T_END, cfl=CFL, order=ORDER)
t_z = float(open(os.path.join(info["case"], "solver_wall.txt")).read())

sm = SystemModel.from_model(model)
nsm = NumericalSystemModel.from_system_model(
    sm, riemann=PositiveNonconservativeRusanov,
    reconstruction=ReconstructionSpec(          # as zoomyFoam: cell values at O1,
        order=ORDER, limiter="minmod",           # the model's W at O2
        variables="conservative" if ORDER == 1 else "model"))
mesh = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=N)
cc = np.asarray(mesh.cell_centers)[:, :N]
Q0 = model.initial_conditions.apply(cc, np.zeros((len(sm.state), N)))
sf = SymbolicHyperbolicSolverFoam(time_end=T_END, cfl=CFL, mood=False)   # zoomyFoam: none
case = f"{WORK}/symfoam_{N}_{ORDER}"
sf.build(nsm, 1)                                       # compile outside the timing
tf, Qf = sf.run(mesh, nsm, Q0, case)
t_s = float(re.search(r"march wall time ([0-9.eE+-]+)",
                      open(f"{case}/log.symFoam").read()).group(1))
d = np.abs(Qf - Qz[:, :N]).max()
print(f"Stoker 1-D, {N} cells, O{ORDER}, t={T_END:g}: zoomyFoam {t_z:.2f}s "
      f"(steps {info.get('n_steps', '?')}) | symFoam {t_s:.2f}s | speedup {t_z / t_s:.2f}x "
      f"| max|sym-zoomyFoam| {d:.1e} (rel {d / np.abs(Qz).max():.1e})", flush=True)
