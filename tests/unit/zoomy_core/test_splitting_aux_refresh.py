"""SplittingSolver.step must refresh Qaux every step (update_q + update_qaux),
not freeze it at the IC.

Before the unification fix ``SplittingSolver.step`` committed only ``_sim_Q``
and ``_sim_pressure``; the auxiliary vector was read into every substep but
never written back, so any free-surface model (``FSFSplittingSolver``) whose
flux/source read aux (hinv / derivative-aux) ran on stale aux.  The fix appends
the canonical post-step pair (``update_q`` then ``update_qaux``) and commits
``_sim_Qaux``.

The Splitting solver's historical target models (INS / free-surface pressure
projection) were deleted in the model restructure, so there is no live
end-to-end fixture; this drives ``step()`` directly with stubbed substeps and
asserts the commit path now invokes both legs and stores the refreshed aux.
"""
import numpy as np
import pytest

from zoomy_core.fvm.solver_splitting_numpy import (
    SplittingSolver, FSFSplittingSolver)


def _bare(cls, nc=4, n_vars=3, n_aux=2):
    """Construct a solver without setup_simulation and wire the minimal
    _sim_* state + stubbed substeps that step() consumes."""
    s = cls.__new__(cls)
    Q = np.ones((n_vars, nc))
    object.__setattr__(s, "_sim_Q", Q.copy())
    object.__setattr__(s, "_sim_Qaux", np.zeros((n_aux, nc)))
    object.__setattr__(s, "_sim_parameters", np.array([1.0]))
    object.__setattr__(s, "_sim_time", 0.0)
    object.__setattr__(s, "_sim_mesh", object())
    object.__setattr__(s, "_sim_model", object())
    # stub the three substeps: flux -> 0, viscous -> identity, pressure -> (Q, p)
    object.__setattr__(s, "_sim_flux_operator",
                       lambda dt, t, Q, Qaux, p, dQ: np.zeros_like(Q))
    object.__setattr__(s, "_apply_viscous_diffusion",
                       lambda Qs, Qaux, p, t, dt: Qs)
    object.__setattr__(s, "_pressure_correction",
                       lambda Qs, Qaux, p, t, dt: (Qs, np.full(Qs.shape[1], 0.5)))
    return s


def test_splitting_step_applies_update_q_and_update_qaux():
    s = _bare(SplittingSolver)
    calls = {}

    def fake_update_q(Q, Qaux, mesh, model, parameters):
        calls["q"] = True
        return Q + 1.0                              # state-hygiene sentinel

    def fake_update_qaux(Q, Qaux, Qold, Qauxold, mesh, model,
                         parameters, time, dt):
        calls["qaux"] = True
        return Qaux + 7.0                           # refreshed-aux sentinel

    object.__setattr__(s, "update_q", fake_update_q)
    object.__setattr__(s, "update_qaux", fake_update_qaux)

    s.step(0.1)

    assert calls.get("q") is True                   # update_variables applied
    assert calls.get("qaux") is True                # aux refreshed (was frozen)
    assert np.allclose(s._sim_Qaux, 7.0)            # refreshed aux COMMITTED
    assert np.allclose(s._sim_Q, 2.0)               # update_q result committed
    assert s._sim_pressure is not None              # pressure still committed


def test_fsf_splitting_inherits_the_refresh():
    """FSFSplittingSolver (the variant that actually carries aux) inherits the
    fixed step() unchanged."""
    assert FSFSplittingSolver.step is SplittingSolver.step
    s = _bare(FSFSplittingSolver)
    object.__setattr__(s, "update_q",
                       lambda Q, Qaux, mesh, model, parameters: Q)
    object.__setattr__(
        s, "update_qaux",
        lambda Q, Qaux, Qold, Qauxold, mesh, model, parameters, time, dt:
        Qaux + 3.0)
    s.step(0.1)
    assert np.allclose(s._sim_Qaux, 3.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
