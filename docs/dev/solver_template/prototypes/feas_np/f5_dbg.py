import warnings, numpy as np
from common import *
from zoomy_core.fvm.solver_numpy import _build_free_surface_numerics
from zoomy_core.fvm.riemann_solvers import NonconservativeRusanov
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
warnings.simplefilter("ignore")
exec(open("f5_pos_rusanov.py").read().split("for label, model in")[0].split("rng = np.random.default_rng(1)")[1].replace("def states", "def states"), globals())
rng = np.random.default_rng(1)
model = SME(level=1, dimension=2, closures=[Newtonian(), NavierSlip(), StressFree()], parameters={"nu": 1e-6, "lambda_s": 0.0})
nsm = build_nsm(model, order=1); sm = nsm.sm
print("state", [str(s) for s in sm.state], "aux", [str(s) for s in sm.aux_state], dict(sm.parameter_values.items()))
rt_model = NumpyRuntimeModel.from_system_model(sm)
p = np.array(list(sm.parameter_values.values()), float); nv = len(sm.state); na = sm.n_aux_variables
numerics = _build_free_surface_numerics(sm); rn = numerics.to_runtime_numpy()
nf = 3000
qA, qB = states(nv, nf, 1), states(nv, nf, 1); qB[0, : nf // 2] = qA[0, : nf // 2]
def aux_of(Q):
    A0 = np.zeros((na, Q.shape[1]))
    return np.asarray(rt_model.update_aux_variables(Q, A0, p, 0.0, np.zeros(3)), float).reshape(na, -1)
aA, aB = aux_of(qA), aux_of(qB)
n = rng.choice([-1.0, 1.0], (1, nf))
fn = rn.numerical_fluctuations
loop = np.array([np.asarray(fn(qA[:, i], qB[:, i], aA[:, i], aB[:, i], p, n[:, i]), float).reshape(-1) for i in range(nf)]).T
bat = np.asarray(fn(qA, qB, aA, aB, p, n), float).reshape(2 * nv, -1)
d = np.abs(bat - loop).max(axis=0)
idx = np.argsort(-d)[:6]
for i in idx:
    print(f"face {i}: maxdiff {d[i]:.2e}  qA={qA[:, i]}  qB={qB[:, i]} aA={aA[:, i]} aB={aB[:, i]} n={n[0, i]}")
    print("    loop", loop[:, i]); print("    bat ", bat[:, i])
# is the difference input-dependent only through the face itself? re-evaluate face i alone in a batch of 1 and of 2
i = idx[0]
one = np.asarray(fn(qA[:, [i]], qB[:, [i]], aA[:, [i]], aB[:, [i]], p, n[:, [i]]), float).reshape(-1)
two = np.asarray(fn(qA[:, [i, idx[1]]], qB[:, [i, idx[1]]], aA[:, [i, idx[1]]], aB[:, [i, idx[1]]], p, n[:, [i, idx[1]]]), float).reshape(2 * nv, -1)[:, 0]
print("batch-of-1 == loop:", np.array_equal(one, loop[:, i]), " batch-of-2 col0 == loop:", np.array_equal(two, loop[:, i]), " ==full batch:", np.array_equal(two, bat[:, i]))
