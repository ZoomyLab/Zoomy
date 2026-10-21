"""Two silent-ignore defects of the symbolic JAX solver are errors (ANALYSIS 66).

(a) `ad_mode` is read once, at `setup`: assigning another mode afterwards used to be ignored
    (the program kept its loop, `jax.grad` then failed on the plain `lax.while_loop`); `run`
    now raises and names the fix.
(b) `AdMode.reverse` has no host callbacks: `on_write` / `output` used to be skipped silently;
    `setup` now raises.  Forward mode still writes, and the results are unchanged.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("equinox")

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, StressFree
from zoomy_core.numerics import NumericalSystemModel, PositiveNonconservativeHLL
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_jax.symsolver.solver import AdMode, SymbolicHyperbolicSolverJax

pytestmark = [pytest.mark.small, pytest.mark.jax]


def _dambreak(n=20):
    """1-D shallow water (SME level 0) dam break between walls."""
    model = SME(level=0, dimension=2, closures=[Newtonian(), StressFree()],
                boundary_conditions=BoundaryConditions([FromModel(tag=t, definition="wall")
                                                        for t in ("left", "right")]))
    sm = SystemModel.from_model(model)
    sm.initial_conditions = IC.UserFunction(
        function=lambda x: np.array([0.0, 1.0 + (x[0] < 0.5), 0.0]))
    return (NumericalSystemModel.from_system_model(sm, riemann=PositiveNonconservativeHLL),
            LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=n))


def _solver(**kw):
    return SymbolicHyperbolicSolverJax(time_end=0.05, cfl=0.4, dt=0.004, ad_checkpoints=4, **kw)


@pytest.mark.parametrize("built, later", [(AdMode.forward, AdMode.reverse),
                                          (AdMode.reverse, AdMode.forward)])
def test_ad_mode_changed_after_setup_raises(built, later):
    nsm, mesh = _dambreak()
    s = _solver(ad_mode=built)
    Q0, A = s.setup(mesh, nsm)
    _, Q_ref, _ = s.run(Q0, A)
    s.ad_mode = later                                   # ignored by the built program
    with pytest.raises(ValueError, match="Set ad_mode before setup"):
        s.run(Q0, A)
    s.ad_mode = built                                   # back to the built mode: runs, bitwise
    assert np.array_equal(np.asarray(s.run(Q0, A)[1]), np.asarray(Q_ref))


def test_ad_mode_set_before_setup_differentiates():
    """The documented route: the mode is set before setup, `jax.grad` runs on the checkpointed
    loop and equals the forward jvp."""
    def J_of(mode):
        nsm, mesh = _dambreak()
        s = _solver(ad_mode=mode)
        Q0, A = s.setup(mesh, nsm)
        h = s.nsm.sm.roles.depth
        return (lambda p: jnp.sum(s.run(Q0, s.initial_aux(Q0, A, p), parameters=p)[1][h] ** 2),
                s.parameters)
    Jf, p0 = J_of(AdMode.forward)
    Jr, _ = J_of(AdMode.reverse)
    e = jnp.zeros_like(p0).at[0].set(1.0)
    jvp = float(jax.jvp(Jf, (p0,), (e,))[1])
    grad = float(jax.grad(Jr)(p0)[0])
    assert grad == pytest.approx(jvp, rel=1e-8, abs=1e-12)


@pytest.mark.parametrize("opts, named", [(dict(on_write=lambda t, Q, Qaux: None), "on_write"),
                                         (dict(output="ad_traps_unused.h5"), "output")])
def test_reverse_with_snapshots_raises_at_setup(opts, named):
    from zoomy_core.misc import misc
    nsm, mesh = _dambreak()
    s = _solver(ad_mode=AdMode.reverse, **opts)
    with pytest.raises(ValueError, match=rf"AdMode\.reverse writes no snapshots.*{named}"):
        s.setup(mesh, nsm)
    # refused before anything is written
    assert not os.path.exists(os.path.join(misc.get_main_directory(), "ad_traps_unused.h5"))


def test_forward_with_on_write_unchanged():
    """Forward mode still calls `on_write`; with the same `dt_write` the final state does not
    depend on it."""
    nsm, mesh = _dambreak()
    frames = []
    s = _solver(on_write=lambda t, Q, Qaux: frames.append(float(t)), dt_write=0.025)
    Q0, A = s.setup(mesh, nsm)
    _, Q_w, _ = s.run(Q0, A)
    jax.effects_barrier()
    assert len(frames) >= 2
    nsm, mesh = _dambreak()
    s0 = _solver(dt_write=0.025)
    Q0, A = s0.setup(mesh, nsm)
    assert np.array_equal(np.asarray(Q_w), np.asarray(s0.run(Q0, A)[1]))
