"""A lake at rest in a partially wet pit stays at rest at order 2 with the Venkatakrishnan
limiter (JAX runtime): the hydrostatic extension of the free surface in the LSQ stencil and the
limiter bounds (``zoomy_core.symsolver.discretization.reconstruction.shore_extension``).
Without it the lake accelerates from rest to 0.40 m/s within 1 s.  The NumPy twin is
``zoomy_core/tests/symsolver/test_shore_lake_at_rest.py`` (same case)."""
import numpy as np
import pytest

from zoomy_core.numerics import Limiter


@pytest.mark.small
@pytest.mark.jax
@pytest.mark.parametrize("tilt", [0.0, 0.05])
def test_pit_lake_stays_at_rest_jax(tilt):
    import jax
    jax.config.update("jax_enable_x64", True)
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
    from zoomy_core.mesh import LSQMesh  # noqa: F401
    import importlib.util, pathlib
    core_test = (pathlib.Path(__import__("zoomy_core").__file__).parent.parent / "tests"
                 / "symsolver" / "test_shore_lake_at_rest.py")
    spec = importlib.util.spec_from_file_location("core_shore", core_test)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    mesh, nsm = T.pit(0.5, Limiter.venkatakrishnan, tilt)
    s = SymbolicHyperbolicSolverJax(time_end=1.0, cfl=0.4, mood=True)
    Q0, A = s.setup(mesh, nsm)
    t, Q, _ = s.run(Q0, A)
    Q = np.asarray(Q)
    h = Q[1]
    wet = h > 1e-3
    u = np.hypot(Q[2], Q[3])[wet] / h[wet]
    assert float(t) == pytest.approx(1.0, abs=1e-12)
    assert 0 < wet.sum() < h.size
    assert u.max() < 1e-12, f"the lake moved: max |u| = {u.max():.3e}"
    assert np.ptp((Q[0] + h)[wet]) < 1e-12
