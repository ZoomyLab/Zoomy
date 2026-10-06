"""IMEXSourceSolverJax runs on the base HyperbolicSolver time loop and aux
path (``update_qaux`` is inherited, not rebuilt), so the aux legs are covered by
the base solver's tests.  This pins the public ``solve(mesh, model)`` entry
point end to end on SWE: tracing + execution of a full IMEX march.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax


def test_imex_jax_solves_swe_end_to_end():
    """A full IMEXSourceSolverJax SWE dam-break traces, executes, and keeps a
    positive depth with mass in a sane band."""
    import zoomy_core.fvm.timestepping as timestepping
    import zoomy_core.model.boundary_conditions as BC
    import zoomy_core.model.initial_conditions as IC
    from zoomy_core.mesh import LSQMesh
    from zoomy_core.model.models.swe import SWE

    bcs = BC.BoundaryConditions(
        [BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")])

    def ic(x):
        out = np.zeros(3)
        out[1] = 0.2 if x[0] < 0.5 else 0.1        # h step (b=0, hu=0)
        return out

    # IMEXSourceSolverJax.solve wants the symbolic model (it reads
    # model.boundary_conditions and builds the NSM itself), not a prebuilt NSM.
    model = SWE(boundary_conditions=bcs,
                initial_conditions=IC.UserFunction(function=ic))
    mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=40)
    solver = IMEXSourceSolverJax(
        time_end=0.02, compute_dt=timestepping.adaptive(CFL=0.3))

    Q, Qaux = solver.solve(mesh, model, write_output=False)
    h = np.asarray(Q[1, :40])
    assert np.all(np.isfinite(np.asarray(Q)))
    assert h.min() > 0.0                            # depth stays positive
    assert 0.09 < h.mean() < 0.21                   # mass in a sane band


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
