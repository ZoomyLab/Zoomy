"""IMEXSourceSolverJax must apply the SAME aux path as the canonical jax
HyperbolicSolver: a local ``update_aux_variables`` prefix leg + an
``aux_registry`` spatial-derivative leg, plus ``update_variables`` (update_q).

Before the unification fix the bespoke ``_build_update_qaux`` only walked the
legacy ``derivative_specs`` structure (so any canonical aux_registry model had
its aux frozen at the IC), and ``update_variables`` was never applied in the
IMEX time loop.  These tests pin the composed closure (legs that need no mesh
are unit-tested directly) and a real end-to-end SWE run (tracing + execution).

The historical jax-IMEX target models (advection-diffusion, Green-Naghdi) were
moved to legacy / deleted, so their old tests are broken at import; this file
deliberately depends only on the still-present ``SWE`` model.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax


class _FakeRuntime:
    """Minimal stand-in for JaxRuntime exposing only what _build_update_qaux
    reads: update_aux_variables, sm (.aux_registry), model (.derivative_specs),
    parameters."""

    def __init__(self, update_aux_variables=None, sm=None, model=None,
                 parameters=None):
        self.update_aux_variables = update_aux_variables
        self.sm = sm
        self.model = model
        self.parameters = parameters


def test_build_update_qaux_applies_local_leg():
    """The local update_aux_variables leg (e.g. KP hinv) must be applied —
    previously absent from the IMEX aux path."""
    def local(Q, Qaux, p):
        return jnp.stack([2.0 * Q[0]])            # aux row 0 := 2*h

    rt = _FakeRuntime(update_aux_variables=local, parameters=jnp.array([9.81]))
    uq = IMEXSourceSolverJax._build_update_qaux(rt, mesh=None)

    Q = jnp.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    Qaux = jnp.zeros((1, 3))
    out = uq(Q, Qaux, Q, jnp.asarray(0.1))
    assert np.allclose(np.asarray(out[0]), 2.0 * np.asarray(Q[0]))


def test_build_update_qaux_identity_when_nothing_declared():
    """No local formula, no registry, no specs => exact passthrough."""
    rt = _FakeRuntime(parameters=jnp.array([1.0]))
    uq = IMEXSourceSolverJax._build_update_qaux(rt, mesh=None)
    Qaux = jnp.ones((1, 3))
    out = uq(jnp.zeros((2, 3)), Qaux, jnp.zeros((2, 3)), jnp.asarray(0.1))
    assert np.allclose(np.asarray(out), np.asarray(Qaux))


def test_build_update_qaux_preserves_time_derivative_leg():
    """The legacy derivative_specs (Q-Qold)/dt time derivative — the one thing
    aux_registry cannot express (Green-Naghdi u_t) — must survive."""
    class _Spec:
        key, field, axes = "dth", "h", ("t",)

    class _Model:
        variables = {"h": 0, "q": 1}
        derivative_specs = [_Spec()]
        derivative_key_to_index = {"dth": 0}

    rt = _FakeRuntime(model=_Model(), parameters=jnp.array([1.0]))
    uq = IMEXSourceSolverJax._build_update_qaux(rt, mesh=None)

    Q = jnp.array([[3.0, 3.0, 3.0], [0.0, 0.0, 0.0]])
    Qold = jnp.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    out = uq(Q, jnp.zeros((1, 3)), Qold, jnp.asarray(0.5))
    assert np.allclose(np.asarray(out[0]), (3.0 - 1.0) / 0.5)   # = 4.0


def test_imex_jax_solves_swe_end_to_end():
    """End-to-end: the rewritten closure + the new update_variables leg trace
    and execute through a full IMEXSourceSolverJax SWE run (no aux on SWE, so
    this guards the loop/tracing, not the aux legs)."""
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
