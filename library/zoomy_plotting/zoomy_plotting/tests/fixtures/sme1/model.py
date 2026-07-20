import numpy as np
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.systemmodel.system_model import SystemModel

# Open ends + a small dam break (h=2 left / h=1 right) on x in (0, 10).
model = SystemModel.from_model(SME(
    level=1,
    dimension=2,
    closures=[Newtonian(), NavierSlip(), StressFree()],
    boundary_conditions=[BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")],
    initial_conditions=IC.RP(high=lambda n: np.array([0.0, 2.0] + [0.0] * (n - 2)),
                             low=lambda n: np.array([0.0, 1.0] + [0.0] * (n - 2)),
                             jump_position_x=5.0),
))
