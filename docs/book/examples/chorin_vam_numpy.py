"""The Chorin projection solver on the NumPy backend: the VAM(1) channel of chorin_vam.py."""
import numpy as np

from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import Dirichlet, Extrapolation
from zoomy_core.model.initial_conditions import Constant
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.model.models.vam import VAM
from zoomy_core.symsolver.numpy_backend import SymbolicChorinSolverNumpy
from zoomy_core.symsolver.settings import LinearSolver
from zoomy_core.symsolver.strategies.chorin import predictor_nsm
from zoomy_core.systemmodel.system_model import _DT_SYMBOL, SystemModel

bcs = ([Dirichlet("left", on=f, value=v)
        for f, v in (("q_x_0", 0.11197), ("q_x_1", 0.0), ("r_0", 0.0), ("r_1", 0.0))]
       + [Dirichlet("right", on=f"P_{i}", value=0.0) for i in range(2)]
       + [Extrapolation(tag=t) for t in ("bottom", "top")])
model = VAM(level=1, dimension=3, boundary_conditions=bcs, closures=[Newtonian(), StressFree()])
sm = SystemModel.from_model(model)
sm.initial_conditions = Constant(constants=lambda n: np.zeros(n))
sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
split = model.chorin_split(_DT_SYMBOL, system_model=sm)     # zoomy's dt symbol
press, corr = split.SM_press, split.SM_corr
pred = predictor_nsm(split.SM_pred, press, corr)            # the predictor's numerics
names = [str(s) for s in pred.state]

mesh = LSQMesh.create_2d(domain=(-1.5, 1.5, 0.0, 0.3), nx=60, ny=6)
solver = SymbolicChorinSolverNumpy(time_end=8 * 2.0 ** -10, dt=2.0 ** -10, time_order=1,
                                   pressure_linear_solver=LinearSolver.gmres)
Q, Qaux = solver.setup(mesh, pred, press, corr)

x = solver.mesh_np.cell_centers[0, :Q.shape[1]]
Q0 = np.zeros(Q.shape)
b = 0.20 * np.exp(-x ** 2 / (2 * 0.20 ** 2))
Q0[names.index("b")] = b
Q0[names.index("h")] = np.where(x < 0.0, 0.34 - b, 0.015)  # reservoir, shallow downstream
Qaux = solver.initial_aux(Q0, Qaux)
time, Q, Qaux = solver.run(Q0, Qaux)
h = Q[names.index("h")]
print(f"t = {time:.6f}, cells = {h.size}, h in [{h.min():.5f}, {h.max():.5f}]")
