"""ML-VAM: every derivative aux of the model (and of its Chorin split) targets a state, and the
w rows' interface-slope term d_x(l_1 h) reads the finite difference of l_1 h on a smooth field.

The layer depths are h_l = l_l h with constant fractions; substituting them into an unevaluated
d_x(h_1) left the compound d_x(l_1 h), lifted to an aux of the non-state target 'h*l_1'
(`target_kind='unknown'`) that no backend computes: JAX and NumPy left it 0 (NumPy with boundary
ghosts raised).  The split's predictor had d_x(0) (the B d_x P term with P = 0) the same way.
"""
import numpy as np
import pytest
import sympy as sp

from zoomy_core.fvm.solver_numpy import Solver
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models.closures import NavierSlip, Newtonian, StressFree
from zoomy_core.model.models.ml_vam import MLVAM
from zoomy_core.systemmodel.system_model import SystemModel, _DT_SYMBOL

pytestmark = [pytest.mark.model, pytest.mark.small]

L1 = 0.4


@pytest.fixture(scope="module")
def mlvam():
    m = MLVAM(n_layers=2, level=0, dimension=2,
              closures=[Newtonian(), NavierSlip(), StressFree()])
    sm = SystemModel.from_model(m)
    sm.parameter_values.l_1 = L1
    return m, sm


def _unknown(sm):
    return [e["name"] for e in sm.aux_registry or [] if e.get("target_kind") != "state"]


def test_every_derivative_aux_targets_a_state(mlvam):
    m, sm = mlvam
    assert _unknown(sm) == []
    split = m.chorin_split(_DT_SYMBOL, system_model=sm)
    for sub in (split.SM_pred, split.SM_press, split.SM_corr):
        assert _unknown(sub) == []


def test_w_rows_read_the_finite_difference_of_l1_h(mlvam):
    """The r rows' source with the LSQ aux equals the source with exact derivatives where
    d_x(l_1 h) = l_1 h' (second order in dx), on a smooth non-trivial state."""
    _, sm = mlvam
    names = [str(s) for s in sm.state]
    xs = sp.Symbol("x")
    exact = {"b": 0.1 * sp.exp(-xs ** 2), "h": 1.0 + 0.2 * sp.sin(xs),
             "q_1_0": L1 * 0.5 * (1 + 0.3 * sp.cos(xs)), "q_2_0": (1 - L1) * 0.7 * (1 + 0.1 * xs),
             "r_1_0": 0.05 * sp.sin(2 * xs), "r_2_0": 0.03 * sp.cos(xs)}
    f = {n: sp.lambdify(xs, exact.get(n, sp.S.Zero)) for n in names}
    # the exact x-derivative of any registered target, a compound one included
    fields = {sp.Symbol(n): exact.get(n, sp.S.Zero) for n in names}
    fields.update({sp.Symbol(str(p)): float(sm.parameter_values[str(p)]) for p in sm.parameters})

    def dfx(target):
        return sp.lambdify(xs, sp.diff(sp.sympify(target).subs(fields), xs))
    errs = []
    for nx in (64, 128):
        mesh = LSQMesh.create_1d(domain=(-2.0, 2.0), n_inner_cells=nx)
        nc = mesh.n_inner_cells
        xc = np.asarray(mesh.cell_centers)[0, :nc]
        Q = np.array([np.broadcast_to(f[n](xc), xc.shape) for n in names], dtype=float)
        A = Solver._walk_derivative_aux(None, sm, np.zeros((len(sm.aux_state), nc)), Q, mesh)
        A_ex = np.zeros_like(A)
        for e in sm.aux_registry:
            if e["kind"] == "derivative" and tuple(e["multi_index"]) == (1,):
                A_ex[e["row"]] = np.broadcast_to(dfx(e["target_name"])(xc), xc.shape)
        par = [float(sm.parameter_values[str(p)]) for p in sm.parameters]
        src = sp.lambdify([list(sm.state), list(sm.aux_state), list(sm.parameters)],
                          [sp.sympify(sm.source[names.index(r), 0]) for r in ("r_1_0", "r_2_0")])
        inner = slice(3, nc - 3)
        num = np.array([np.broadcast_to(v, xc.shape) for v in src(Q, A, par)])[:, inner]
        ex = np.array([np.broadcast_to(v, xc.shape) for v in src(Q, A_ex, par)])[:, inner]
        assert np.abs(ex).max() > 1e-3          # the rows carry slope terms on this field
        errs.append(np.abs(num - ex).max() / np.abs(ex).max())
    assert errs[1] < 1e-3 and errs[0] / errs[1] > 3.5, errs   # second order, nothing read as 0
