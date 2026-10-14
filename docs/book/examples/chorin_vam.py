"""The Chorin projection solver: VAM(1) channel (gate 21 / matrix C8), JAX."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from zoomy_core.mesh import LSQMesh  # noqa: E402
from zoomy_core.model.boundary_conditions import Dirichlet, Extrapolation  # noqa: E402
from zoomy_core.model.initial_conditions import Constant  # noqa: E402
from zoomy_core.model.models.closures import Newtonian, StressFree  # noqa: E402
from zoomy_core.model.models.vam import VAM  # noqa: E402
from zoomy_core.systemmodel.system_model import _DT_SYMBOL, SystemModel  # noqa: E402
from zoomy_core.symsolver.settings import LinearSolver  # noqa: E402
from zoomy_core.symsolver.strategies.chorin import predictor_nsm  # noqa: E402
from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax  # noqa: E402

bcs = ([Dirichlet("left", on=f, value=v)
        for f, v in (("q_x_0", 0.11197), ("q_x_1", 0.0), ("r_0", 0.0), ("r_1", 0.0))]
       + [Dirichlet("right", on=f"P_{i}", value=0.0) for i in range(2)]
       + [Extrapolation(tag=t) for t in ("bottom", "top")])
model = VAM(level=1, dimension=3, boundary_conditions=bcs, closures=[Newtonian(), StressFree()])
sm = SystemModel.from_model(model)
sm.initial_conditions = Constant(constants=lambda n: np.zeros(n))
sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
split = model.chorin_split(_DT_SYMBOL, system_model=sm)     # zoomy's dt symbol
pred, press, corr = split.SM_pred, split.SM_press, split.SM_corr

# The predictor's numerics on the full (padded) state: positive Rusanov, Audusse-rescaled momenta.
pred = predictor_nsm(pred, press, corr)
names = [str(s) for s in pred.state]

mesh = LSQMesh.create_2d(domain=(-1.5, 1.5, 0.0, 0.3), nx=60, ny=6)
solver = SymbolicChorinSolverJax(time_end=8 * 2.0 ** -10, dt=2.0 ** -10, time_order=1,
                                 pressure_linear_solver=LinearSolver.gmres)
Q, Qaux = solver.setup(mesh, pred, press, corr)

x = np.asarray(solver.mesh_np.cell_centers)[0, :Q.shape[1]]
Q0 = np.zeros(Q.shape)
b = 0.20 * np.exp(-x ** 2 / (2 * 0.20 ** 2))
Q0[names.index("b")] = b
Q0[names.index("h")] = np.where(x < 0.0, 0.34 - b, 0.015)  # reservoir, shallow downstream
Qaux = solver.initial_aux(jax.numpy.asarray(Q0), Qaux)
time, Q, Qaux = solver.run(jax.numpy.asarray(Q0), Qaux)
h = np.asarray(Q)[names.index("h")]
print(f"state {names}")
print(f"t = {float(time):.6f}, cells = {h.size}, h in [{h.min():.5f}, {h.max():.5f}]")
