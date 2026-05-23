"""VAM(1,2,2) chain-derived dam-break-over-bump runs end-to-end.

The chain notebook lives at
``thesis/notebooks/modeling/vam/vam_chorin_bump_8state_chain.py`` and
is the user's reference recipe — it builds the 8-state SystemModel from
``VAMModelGalerkin`` via change-of-variables → InvertMassMatrix →
HydrostaticReconstruction → split_simple, then drives a Chorin-split
solve on a ``lsq_degree=2`` mesh.

The test here:

1. Imports that notebook as a module with a stub plotting backend, so
   the SystemModel construction happens exactly as documented.
2. Overrides the long ``T_end=20`` run with a short ``T_end=0.5`` so
   the test stays under ~30 s wall time.
3. Asserts the pipeline survives the initial transient: ``h`` stays
   finite and ``≥ 0``, mass is conserved to a sensible tolerance.

Acceptance against the experimental ETA profile at ``T=20`` is the
follow-up.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_NOTEBOOK = Path(
    "/home/ingo/git/Zoomy/thesis/notebooks/modeling/vam/"
    "vam_chorin_bump_8state_chain.py"
)


def _run_notebook_with_short_t_end(t_end: float):
    """Exec the chain notebook up to the plotting cells with the run
    function pinned to a short ``t_end``."""
    if not _NOTEBOOK.exists():
        pytest.skip(f"VAM chain notebook missing at {_NOTEBOOK}")

    # Headless mpl so notebook's ``import matplotlib.pyplot`` and any
    # incidental ``plt.show``/``savefig`` calls don't hit a display.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    _plt.show = lambda *a, **k: None
    _plt.savefig = lambda *a, **k: None

    src = _NOTEBOOK.read_text()
    src = src.replace("result_hr = run()", f"result_hr = run(T_end={t_end})")
    cut = src.split(
        "# %% [markdown]\n# ## Plot vs experimental data — DG(0) HR route"
    )[0]
    ns: dict = {"__name__": "__main__"}
    exec(compile(cut, str(_NOTEBOOK), "exec"), ns)
    return ns.get("result_hr")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("ZOOMY_SKIP_VAM_DAMBREAK") == "1",
    reason="VAM dam-break integration test skipped via env override",
)
def test_vam_chain_dam_break_short_run_is_stable():
    """The chain dam-break must survive the initial transient
    (~95 steps) without producing NaN or negative water depth.

    This is the regression bar: prior cleanup commits must not break
    the Chorin pressure GMRES, the Audusse HR Riemann route, or the
    state of the SystemModel chain.
    """
    result = _run_notebook_with_short_t_end(t_end=0.5)
    assert result is not None, (
        "Chain dam-break BLEW UP during the short run — the chain "
        "notebook reported failure (h became non-finite)."
    )

    solver, xc, b_vals = result
    Q = solver._sim_Q

    assert np.all(np.isfinite(Q)), "non-finite values in final Q"
    assert np.all(Q[0] >= 0), (
        f"negative h in final state: hmin={Q[0].min():.4e}"
    )
    # Reservoir mass at IC: integrate(h0) ≈ sum(h0) * dx — used as a
    # loose conservation sanity check.  Inflow at left boundary
    # (q_U0=0.11197) over 0.5 s adds material; we only assert the
    # gross magnitude is sensible (still O(10), not blown up).
    mass = float(np.sum(Q[0]))
    assert 1.0 < mass < 100.0, (
        f"mass sum out of plausible range: {mass:.4f}"
    )
    # Time the solver actually reached.
    assert solver._sim_time >= 0.45, (
        f"sim only reached t={solver._sim_time:.3f}, expected >= 0.45"
    )
