"""X01 — the ONE numpy solver golden (approved spec §A, gate member).

End-to-end numpy FVM path: SME(1) lake-at-rest over a bump bed, Audusse
equilibrium-reconstruction hook, PERIODIC wrap, order-1, 10 cells, 2 adaptive
steps.  Hard asserts (inside the builder): WB <= 1e-11, model-declared IC
threads to the SystemModel (REQ-103), + the folded Bernoulli
moving-equilibrium round-trip / exact-discharge check (absorbs
test_equilibrium_wb).  The golden pins the final state values.
"""
import pytest

import goldenlib

pytestmark = pytest.mark.solver


@pytest.mark.parametrize("name", goldenlib.golden_params("solver"))
def test_solver_golden(name):
    builder, _family, _tier = goldenlib.GOLDENS[name]
    goldenlib.assert_matches_golden(name, builder())
