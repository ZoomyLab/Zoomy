import time, numpy as np
t0=time.time()
from zoomy_core.model.models import SME
from zoomy_core.model.models.swe import SWE
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
import zoomy_core.model.boundary_conditions as BC
for name, build in [("SWE1", lambda: SWE(dimension=1)), ("SWE2", lambda: SWE(dimension=2)),
                    ("SME0_3", lambda: SME(level=0, dimension=3, closures=[Newtonian(), NavierSlip(), StressFree()], parameters={"nu":1e-6,"lambda_s":0.0}))]:
    try:
        m = build()
        sm = SystemModel.from_model(m)
        nsm = NumericalSystemModel.from_system_model(sm, reconstruction=ReconstructionSpec(order=2, limiter="venkatakrishnan"))
        print(name, "state", [str(s) for s in nsm.state], "aux", [str(s) for s in nsm.sm.aux_state] if hasattr(nsm.sm,'aux_state') else None,
              "n_aux", nsm.sm.n_aux_variables, "dim", nsm.sm.dimension, "params", dict(nsm.sm.parameter_values.items()) if hasattr(nsm.sm.parameter_values,'items') else nsm.sm.parameter_values,
              "riemann", getattr(nsm,'riemann',None), "upd", nsm.update_variables is not None, time.time()-t0)
        print("  source", nsm.sm.source)
    except Exception as e:
        import traceback; traceback.print_exc()
