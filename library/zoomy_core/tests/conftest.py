"""Test-tiering policy (USER 2026-07-17).

Two opt-in tiers keep the DEFAULT run — the pre-publish gate — down to
seconds-to-a-couple-of-minutes on a warm derivation cache:

* ``@pytest.mark.large``    — real time-march / simulation tests (the VAM/MLVAM
  dam-break DAE Chorin solves, multi-step MOOD/positivity marches, any test
  with ~>=10 s of solver time).  Every large march has a 1-step "small twin"
  in the default tier that asserts finiteness + cheap invariants (bounded mass
  change, shapes, ``h >= 0``) — a real regression canary at ~seconds cost.
* ``@pytest.mark.rederive`` — tests that deliberately clear / bypass the
  derivation cache or force a fresh derivation of a heavy family.  Excluded by
  default so routine runs always hit the warm cache.

Both tiers are DESELECTED by default and re-enabled per tier:

    pytest                       # default (small) tier — the gate
    pytest --run-large           # add the time-march tier
    pytest --run-rederive        # add the cold-cache / fresh-derivation tier
    pytest --run-large --run-rederive   # everything

An explicit ``-m`` expression (e.g. ``-m large``) takes over selection
entirely and disables this auto-deselection, so the tiers stay directly
addressable.

Additionally (approved test-refactor spec 2026-07-19, final v3):

* ``@pytest.mark.gate``  — T1 pre-publish gate membership; ``scripts/verify.py``
  runs ``-m gate`` scoped to the touched areas.
* area tags ``model`` / ``systemmodel`` / ``nsm`` / ``printer`` / ``solver``.
* ``@pytest.mark.study`` — parked study scaffolding (REQ-194 depth-law study);
  excluded from EVERY tier, runs only via an explicit ``-m study``.
* ``tests/goldens/`` holds the checked-in golden snapshots + ``goldenlib``;
  its path is injected below so every test file can ``import goldenlib``.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "goldens"))


def pytest_addoption(parser):
    group = parser.getgroup("zoomy test tiers")
    group.addoption(
        "--run-large", action="store_true", default=False,
        help="run @pytest.mark.large tests (real time-march / simulation).",
    )
    group.addoption(
        "--run-rederive", action="store_true", default=False,
        help="run @pytest.mark.rederive tests (cold-cache / fresh heavy "
             "derivations).",
    )


@pytest.fixture
def one_hyperbolic_step():
    """Advance a ``HyperbolicSolver`` (or subclass) exactly ONE adaptive step.

    The small-twin idiom for the ``large`` marches that drive the numpy FVM via
    ``solver.solve(...)``: build the solver + mesh + NSM the SAME way the large
    march does, then take a single real timestep and return ``Q``.  Cheap
    regression canary at ~seconds cost without the full time integration.
    """
    import numpy as np

    def _run(solver, mesh, nsm):
        solver.setup_simulation(mesh, nsm, write_output=False)
        dt = solver.compute_dt(
            solver._sim_Q, solver._sim_Qaux, solver._sim_parameters,
            solver._sim_face_inradius,
            solver._sim_compute_max_abs_eigenvalue,
        )
        solver.step(float(dt))
        return np.asarray(solver._sim_Q, float)

    return _run


def pytest_collection_modifyitems(config, items):
    # An explicit -m expression owns selection; don't second-guess it.
    if config.option.markexpr:
        return
    run_large = config.getoption("--run-large")
    run_rederive = config.getoption("--run-rederive")

    selected, deselected = [], []
    for item in items:
        # ``study`` is parked: excluded from EVERY tier (only -m study runs it).
        drop = ("study" in item.keywords
                or ("large" in item.keywords and not run_large)
                or ("rederive" in item.keywords and not run_rederive))
        (deselected if drop else selected).append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
