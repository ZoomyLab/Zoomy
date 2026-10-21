"""JAX's Newton on an implicit Manning source (AD Jacobian) is finite where the state is exactly
at rest: the source's |u| has the derivative 0 there (zoomy_core `norm_derivative_at_zero`,
ANALYSIS 49, item 4); before, 0 * inf = NaN in every cell at rest after the first step.
"""
import os
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from zoomy_core.symsolver.discretization.time import Imex
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

pytestmark = [pytest.mark.small, pytest.mark.jax]
GATES = os.path.join(os.path.dirname(__file__), *[".."] * 4, "docs", "dev", "symbolic_solver",
                     "gates")


@pytest.mark.parametrize("imex", [Imex.lie, Imex.ssp2])
def test_implicit_manning_at_rest(imex):
    if not os.path.isdir(GATES):
        pytest.skip("superrepo gates not found")
    sys.path.insert(0, os.path.abspath(GATES))
    import common as C
    mesh, nsm = C.bump_2d(1, nx=8, flow=(0.0, 0.0))
    nsm.sm.parameter_values.n = 0.5
    s = SymbolicHyperbolicSolverJax(time_end=0.05, cfl=0.4, imex=imex)
    Q, A = s.setup(mesh, nsm)
    xc = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    Q = jnp.asarray(Q).at[1].set(1.0).at[2].set(jnp.where(xc < 3, 2.0, 0.0))
    t, Q, _ = s.run(Q, s.initial_aux(Q, A))
    assert float(t) == pytest.approx(0.05) and np.isfinite(np.asarray(Q)).all()
    assert (np.asarray(Q)[2] == 0).any()        # cells still exactly at rest
