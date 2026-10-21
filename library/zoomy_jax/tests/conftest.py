"""zoomy_jax test-suite conftest — determinism, tiers, the march helpers.

Determinism:

* ``JAX_PLATFORMS=cpu`` is pinned HERE, at module top, BEFORE any jax /
  zoomy_jax import anywhere in the suite (conftest imports first).
* ``XLA_FLAGS=--xla_force_host_platform_device_count=4`` gives the SPMD tier
  four real CPU devices — also before the first jax import, because a
  test-file-level setdefault would be too late.  It USED to be 2 (enough for
  the retired ``runtime/test_parallel_two_devices.py``); the ``tests/spmd``
  suite moved in from the superrepo asserts device-count transparency across
  {3, 4} devices and would SKIP itself into vacuity at 2.  A conftest that
  imports first wins, so THIS number — not the ``setdefault`` at the top of
  each spmd test module — is the one that takes effect.
* x64 is ASSERTED, not set: ``zoomy_jax/__init__.py`` enables it by default,
  but a stray ``ZOOMY_JAX_ENABLE_X64=0`` in the environment silently flips the
  whole suite to float32 (LSQ error differs ~9 orders).  We hard-fail loudly
  instead of setdefault-ing over it.

Tiers:

* default run = the ``small`` gate tier (seconds per test);
* ``-m regression`` runs the reference marches; ``--run-large`` adds the
  ``@pytest.mark.large`` ones;
* an explicit ``-m`` expression takes over selection entirely.

References: every test compares the FULL ``Q`` and ``Qaux`` against
``tests/refs/<name>.npz`` and records its wall time in
``tests/refs/timings.json`` (see ``refs.py``).  ``--overwrite-results``
regenerates instead of comparing.

DEVIATION FROM THE PROPOSAL (import style, deliberate): the proposal writes
``from tests import models, refs``.  The top-level package name ``tests`` is
already taken by the superrepo (``~/git/Zoomy/tests/__init__.py`` EXISTS,
verified) and this directory has no ``__init__.py``, so ``tests.models`` would
resolve to the superrepo package under a superrepo-wide run.  This directory
is therefore injected onto ``sys.path`` and the shared modules import bare
(``import models, refs``), exactly as ``zoomy_core/tests`` exposes
``goldenlib``.  Module and symbol names are otherwise EXACTLY the proposal's.

Models: ALL derived (user mandate), ALL from the derivation cache.  Shallow
water is SME(level=0).  Wet/dry cap OFF; nothing floors or clips h — the only
depth regularization is the automatic NSM-level KP hinv sweep.
"""
from __future__ import annotations

import os
import pathlib
import sys

# ── determinism: BEFORE any jax import ──────────────────────────────────────
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# 4 CPU devices for the SPMD tier — BEFORE the first jax import.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

# shared test modules (refs / models / cases) import bare
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import pytest

pytest.importorskip("jax")

import jax
import zoomy_jax  # noqa: F401  — flips x64 on import (default-on env knob)

if not jax.config.jax_enable_x64:
    raise RuntimeError(
        "zoomy_jax test suite requires float64 but jax_enable_x64 is OFF. "
        "Most likely a stray ZOOMY_JAX_ENABLE_X64=0 in the environment "
        "(zoomy_jax/__init__.py:35) — the whole suite would silently run in "
        "float32 (LSQ error differs ~9 orders). Unset it and re-run."
    )

from loguru import logger

logger.remove()


# ── the CFL law (user mandate — no augmentation, ever) ──────────────────────
# ONE number, in EVERY dimension.  ``timestepping.adaptive`` already carries the
# spatial-dimension factor INSIDE the formula
#
#     dt <= CFL * 2*r_in / (d * (2k+1) * |lambda|_max)      (d = mesh dimension)
#
# so ``CFL`` is a pure safety factor in (0, 1] and the law "effective 0.9 in
# 1-D, 0.45 in 2-D" falls out of the ``1/d`` by construction.  A separate
# ``CFL_2D = 0.45`` constant encoded that SAME dimensional factor a SECOND
# time and silently quartered the 2-D step (measured effective 0.225); passing no
# ``dimension`` at all left the default d=2 on 1-D meshes and halved those.
# Do NOT "fix" this back into a per-dimension split — pass the MESH's
# ``dimension`` into :func:`_adaptive` instead.
CFL = 0.9                           # user law — no augmentation, ever
ORDER_FLOOR = {1: 0.9, 2: 1.9}      # measured smooth rates are 1.95-2.11

G = 9.81


# ── tiers + the results option ──────────────────────────────────────────────
def pytest_addoption(parser):
    # zoomy_core/tests/conftest.py registers the same option; when a
    # superrepo-wide run collects both suites the second registration would
    # raise — tolerate it (either registration serves both).
    group = parser.getgroup("zoomy test tiers")
    for flag, helptext in (
        ("--run-large", "run @pytest.mark.large tests (real time-march)."),
        ("--overwrite-results",
         "regenerate reference .npz / timings instead of comparing"),
    ):
        try:
            group.addoption(flag, action="store_true", default=False,
                            help=helptext)
        except ValueError:
            pass


def pytest_collection_modifyitems(config, items):
    # An explicit -m expression owns selection; don't second-guess it.
    if config.option.markexpr:
        return
    run_large = config.getoption("--run-large", default=False)
    selected, deselected = [], []
    for item in items:
        drop = "large" in item.keywords and not run_large
        (deselected if drop else selected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture
def overwrite(request):
    return (request.config.getoption("--overwrite-results")
            or os.environ.get("ZOOMY_OVERWRITE_RESULTS") == "1")


# ── the one shared runner ───────────────────────────────────────────────────
def march(nsm, mesh, cfl, t_end=None, n_steps=None, n_devices=1, **solver_kw):
    """Run the jax HyperbolicSolver; return the INNER ``(Q, Qaux)`` block.

    Exactly one of ``t_end`` / ``n_steps`` is given.  ``n_steps`` stops after
    that many adaptive steps (the small-twin idiom: identical construction to
    the regression march, two steps, full state stored).  ``n_devices > 1``
    routes to the SPMD path — see :func:`march_sharded`.

    ``solver_kw`` is forwarded verbatim to ``HyperbolicSolver``.  This is how
    a test asks for the wet/dry numerics: ``ReconstructionSpec`` (the NSM
    slot) carries ONLY ``order`` / ``limiter`` as far as the jax backend is
    concerned — the free-surface reconstruction variables, the a-priori front
    pre-detector and the a-posteriori MOOD corrector are all *solver* params
    (``zoomy_jax/fvm/solver_jax.py:184-226``), not NSM ones.  A test that
    builds ``ReconstructionSpec(order=2, positivity="mood")`` and marches gets
    plain conservative LSQ-MUSCL with NO positivity mechanism whatsoever: the
    spec field is never read on this path.  See ``wet_dry_o2`` below.
    """
    if (t_end is None) == (n_steps is None):
        raise TypeError("march() takes exactly one of t_end / n_steps")
    if n_devices != 1:
        return march_sharded(nsm, mesh, cfl, n_devices,
                             t_end=t_end, n_steps=n_steps, **solver_kw)

    from zoomy_jax.fvm.solver_jax import HyperbolicSolver

    n = mesh.n_inner_cells
    if t_end is not None:
        solver = HyperbolicSolver(time_end=t_end, compute_dt=_adaptive(cfl, mesh),
                                  **solver_kw)
        Q, Qaux = solver.solve(mesh, nsm, write_output=False)
        return np.asarray(Q)[:, :n], np.asarray(Qaux)[:, :n]

    # n_steps: drive step/post_step directly so the count is EXACT.  Marching
    # to a t_end guessed from dt would silently take a different number of
    # steps as soon as the wave speed changes.
    solver = HyperbolicSolver(time_end=np.inf, compute_dt=_adaptive(cfl, mesh),
                              **solver_kw)
    Q, Qaux = solver.setup_simulation(mesh, nsm)
    t = 0.0
    for _ in range(n_steps):
        dt = float(solver.compute_timestep(Q, Qaux))
        assert dt > 0.0 and np.isfinite(dt), (
            f"non-positive/non-finite dt = {dt} at t = {t} — the march "
            f"stalled; this is a FAILURE, not a stopping condition")
        Qn = solver.step(dt, t, Q, Qaux)
        Q, Qaux = solver.post_step(t + dt, dt, Qn, Q, Qaux)
        t += dt
    return np.asarray(Q)[:, :n], np.asarray(Qaux)[:, :n]


def wet_dry_o2(nsm, mood=True, front_tol=None):
    """The ``march(**solver_kw)`` block that turns ON the order-2 wet/dry
    numerics for a free-surface model whose state is ``[b, h, q_0(, q_1)]``.

    Order 2 on a DRY front is only positivity-preserving with these three
    switched on together (``solver_jax.py:469-520, 909-925``):

    * ``reconstruction_variables="eta"`` — Audusse-Bouchut 2005 /
      Kurganov-Petrova 2007 well-balanced reconstruction on
      ``(b, η = h+b, hu)`` with the ``h_f = max(η_f − b_f, 0)`` face clip.
      Also the ONLY reconstruction that exposes ``supports_force_o1``;
    * ``free_surface_h_index`` / ``_b_index`` / ``_momentum_indices`` — the
      solver does NOT auto-detect these from state names (its own docstring
      says "caller-driven"), and ``reconstruction_variables='eta'`` asserts
      on a missing ``h_index``;
    * ``mood=True`` — the a-posteriori corrector.  Without it nothing at all
      enforces ``h ≥ 0`` on the accepted cell mean at order 2 unless
      ``positivity_method="zhang_shu"`` is used instead (which would need
      CFL ≤ 1/3 and so is barred by the CFL law).

    Indices are resolved BY NAME through ``models.state_index`` — never
    positionally.
    """
    import models
    names = [str(s) for s in nsm.state]
    mom = [i for i, s in enumerate(names)
           if s.startswith("q_") or s.startswith("hu")]
    assert mom, f"no momentum rows found in state {names}"
    return dict(
        reconstruction_variables="eta",
        free_surface_b_index=models.state_index(nsm, "b"),
        free_surface_h_index=models.state_index(nsm, "h"),
        free_surface_momentum_indices=mom,
        front_theta_tol=front_tol,
        mood=mood,
    )


def march_sharded(nsm, mesh, cfl, n_devices, t_end=None, n_steps=None,
                  **solver_kw):
    """The SPMD twin of :func:`march`: the REAL ``solver.step`` inside
    ``jax.shard_map`` over a contiguous 1-D partition with a ring halo
    (``zoomy_jax.fvm.spmd_jax.run_solver_sharded``).

    Two structural constraints of that path, both honoured here:

    * **fixed dt** — a global adaptive dt would need a ``lax.pmin``
      collective, so the step size is computed ONCE on the unsharded mesh at
      the SAME CFL law and then held.  The 1-device comparison run uses the
      identical fixed dt, so the twin compares SHARDING, not time-stepping.
    * **periodic halo** — ``halo_exchange_periodic`` wraps the ring, so the
      model must carry periodic BCs for the sharded and unsharded runs to be
      the same problem.

    Returns the gathered owned cells as ``(Q, Qaux)``.
    """
    import jax.numpy as jnp
    from zoomy_jax.fvm.solver_jax import HyperbolicSolver
    from zoomy_jax.fvm.spmd_jax import (gather_owned, run_solver_sharded,
                                        shard_global_state)
    from zoomy_jax.mesh import partition_1d_contiguous

    assert n_steps is not None, "the sharded path marches a fixed step count"
    halo = 2                     # >= 2 keeps MUSCL bit-identical at seams
    n = mesh.n_inner_cells

    # dt from the unsharded mesh at the SAME CFL law, then frozen.
    ref = HyperbolicSolver(time_end=np.inf, compute_dt=_adaptive(cfl, mesh),
                           **solver_kw)
    Q0, Qaux0 = ref.setup_simulation(mesh, nsm)
    dt = float(ref.compute_timestep(Q0, Qaux0))

    parts = partition_1d_contiguous(mesh, n_parts=n_devices, halo=halo)
    solver = HyperbolicSolver(time_end=np.inf, compute_dt=_adaptive(cfl, mesh),
                              **solver_kw)
    solver.setup_simulation(parts[0], nsm)

    Q_pad, n_local = shard_global_state(np.asarray(Q0)[:, :n], n_devices, halo)
    Qaux_pad, _ = shard_global_state(np.asarray(Qaux0)[:, :n], n_devices, halo)
    run = run_solver_sharded(solver, n_devices, halo, n_steps, dt)
    Q_out, Qaux_out = run(jnp.asarray(Q_pad), jnp.asarray(Qaux_pad))
    # Record how many devices actually held a shard, so the test can assert
    # the run did not silently collapse onto one device (see
    # ``cases.used_devices``).  Read off the real output sharding, not the
    # requested count.
    sharding = getattr(Q_out, "sharding", None)
    devs = getattr(sharding, "device_set", None)
    LAST_SHARD["devices"] = len(devs) if devs else 1
    return (gather_owned(np.asarray(Q_out), n_devices, n_local, halo),
            gather_owned(np.asarray(Qaux_out), n_devices, n_local, halo))


# How many devices the last :func:`march_sharded` actually sharded across.
LAST_SHARD: dict = {"devices": 1}


def _adaptive(cfl, mesh):
    """``timestepping.adaptive`` with the MESH's spatial dimension.

    ``dimension`` is NOT optional here, on purpose.  It defaults to 2 in core,
    so every 1-D march that omitted it silently ran at HALF the law (the
    ``1/d`` factor applied to a mesh that has no second direction).  Reading it
    off ``mesh.dimension`` is what makes the single ``CFL`` constant above
    correct in every dimension.
    """
    import zoomy_core.fvm.timestepping as timestepping
    return timestepping.adaptive(CFL=cfl, dimension=int(mesh.dimension))


def fit_order(sizes, errors):
    return float(-np.polyfit(np.log(sizes), np.log(errors), 1)[0])


def restrict(fine):      # conservative fine -> coarse, exact for cell averages
    return 0.5 * (fine[:, 0::2] + fine[:, 1::2])
