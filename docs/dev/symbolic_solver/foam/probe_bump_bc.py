import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../thesis/cases/escalante_vam_bump"))
import numpy as np, sympy as sp
from run_derived import build_model
from zoomy_core.systemmodel.system_model import SystemModel, _DT_SYMBOL
from zoomy_core.numerics import NumericalSystemModel
from zoomy_jax.transformation.jax_runtime import JaxRuntime
m = build_model()
sm = SystemModel.from_model(m)
r = m.chorin_split(_DT_SYMBOL, system_model=sm)
for name, s in (("pred", r.SM_pred), ("press", r.SM_press), ("corr", r.SM_corr)):
    nsm = NumericalSystemModel.from_system_model(s)
    print(name, "bc tags", list(nsm._bc_source.list_sorted_function_names), "bc kernel", nsm.call.boundary_conditions is not None,
          "params", list(nsm.parameters.keys()))
    rt = JaxRuntime.from_nsm(nsm)
    q = np.array([0.1, 0.3, 0.05, 0.01, 0.02, 0.03, 1.0, 2.0])
    for idx, tag in enumerate(nsm._bc_source.list_sorted_function_names):
        g = rt.boundary_conditions(idx, 0.0, np.zeros(3), 0.025, q, np.zeros(len(nsm.aux_state)), np.asarray(rt.parameters), np.array([1.0 if tag=="right" else -1.0]))
        print("   ", tag, "ghost of", q.tolist(), "->", np.round(np.asarray(g), 4).tolist())
