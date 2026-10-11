"""The compressible Euler system built directly as a SystemModel."""
import numpy as np
import pytest
import sympy as sp

import zoomy_core.model.boundary_conditions as BC
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.systemmodel.euler import compressible_euler, pressure

TAGS = {1: ("left", "right"), 2: ("left", "right", "bottom", "top"),
        3: ("left", "right", "front", "back", "bottom", "top")}


@pytest.mark.parametrize("dim", (1, 2, 3))
def test_spectrum_is_the_quasilinear_spectrum(dim):
    sm = compressible_euler(dim)
    rng = np.random.default_rng(dim)
    vals = {sm.state[0]: 1.3, sm.state[-1]: 4.0, sm.parameters.gamma: 1.4}
    for k in range(dim):
        vals[sm.state[1 + k]] = float(rng.uniform(-0.5, 0.5))
    n = rng.normal(size=dim)
    n /= np.linalg.norm(n)
    nsub = {s: v for s, v in zip(sm.normal.values(), n)}
    QL = sm.quasilinear_matrix
    A = sp.Matrix(dim + 2, dim + 2, lambda i, j: sum(
        QL[i, j, d] * n[d] for d in range(dim)))
    ev_num = np.sort(np.linalg.eigvals(np.array(A.subs(vals), float)).real)
    ev_sym = np.sort(np.array(sm.eigenvalues.subs({**vals, **nsub}), float).ravel())
    assert np.allclose(ev_num, ev_sym, atol=1e-12)


def test_wall_reflects_declared_momentum():
    sm = compressible_euler(3, boundary_conditions=BC.BoundaryConditions(
        [BC.Wall(tag=t) for t in TAGS[3]]))
    wall = sm._bc_source.boundary_conditions_list[0]
    assert wall.momentum_field_indices == [[1, 2, 3]]
    q = np.array([1.0, 0.3, -0.2, 0.5, 3.0])
    n = np.array([1.0, 0.0, 0.0])
    ghost = wall.face_value(q, np.zeros(0), n, 0.1, 0.0, np.array([1.4]))
    assert np.allclose(ghost, [1.0, -0.3, -0.2, 0.5, 3.0])


def test_admissibility_and_roles():
    sm = compressible_euler(2)
    nsm = NumericalSystemModel.from_system_model(sm)
    assert nsm.roles.depth is None
    assert [str(e) for e in nsm.admissible_constraints()][0] == "rho"
    p = pressure(sm)
    assert sp.simplify(p.subs({sm.state[1]: 0, sm.state[2]: 0})
                       - (sm.parameters.gamma - 1) * sm.state[-1]) == 0


def test_reconstruction_pair_round_trips():
    sm = compressible_euler(3)
    W = list(sm.reconstruction_variables)
    Wsym = [sp.Symbol(f"WB_{s.name}", real=True) for s in sm.state]
    back = [e.subs(dict(zip(Wsym, W))) for e in sm.state_from_reconstruction]
    assert all(sp.simplify(b - s) == 0 for b, s in zip(back, sm.state))
