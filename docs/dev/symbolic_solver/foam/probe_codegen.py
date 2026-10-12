import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../docs/dev/symbolic_solver/gates"))
import common
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_foam._pipeline import _codegen, FOAM_ROOT
m = common.sme(0, 2, ("left", "right"), "extrapolation")
sm = _codegen(m)
print("FOAM_ROOT", FOAM_ROOT)
print("state", [str(s) for s in sm.state], "aux", [str(s) for s in sm.aux_state])
