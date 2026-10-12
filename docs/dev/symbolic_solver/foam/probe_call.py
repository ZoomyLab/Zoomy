import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../docs/dev/symbolic_solver/gates"))
import common
mesh, nsm = common.dambreak_1d(0, 2)
call = nsm.call
from zoomy_core.systemmodel.system_model import OPERATOR_ARG_SLOTS
fns = {}
for name in OPERATOR_ARG_SLOTS:
    f = nsm.operator(name)
    if f is not None: fns[f.name] = f
fns[nsm.boundary_conditions.name] = nsm.boundary_conditions
fns.update(nsm.build_numerics().functions)
for k, f in fns.items():
    if call[k] is None: continue
    args = {a: (type(v).__name__, len(v.values()) if hasattr(v, "values") else "-") for a, v in f.args.items()}
    print(f"{k:42s} shape={getattr(f.definition,'shape',None)} args={args}")
print("aux_derivatives", call.aux_derivatives)
print("riemann", type(nsm.build_numerics()).__name__)
