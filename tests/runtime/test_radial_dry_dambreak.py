"""Radial dry dam break — the guards that positivity and mass CANNOT provide.

cid 210.  A wet/dry scheme can be badly broken while every existing assertion
stays green: during that investigation a variant ran with ``h >= 0`` everywhere
and mass conserved to the last printed digit, yet it took 5.75x the timesteps at
46x smaller dt and pushed trace water into 96% of the previously-dry domain
(fully-dry cells 3408 -> 216 of 9216 at nx=96).  Both of the usual invariants
held throughout.  Nothing in the suite noticed.

So this test asserts the two things that DID move:

* **step count** — a collapsed dt is not just a cost.  At a lower CFL number an
  explicit Rusanov scheme is MORE numerically dissipative for the same physical
  time, so a dt collapse smears the front.  It is a solution-quality defect that
  positivity and mass are structurally blind to.
* **fully-dry cell count** — the dry region must stay dry.  Water arriving in a
  cell as a 1e-11 m film is not physics; it is a velocity computed as a 0/0
  ratio being propelled across the mesh.

Both bounds are deliberately LOOSE (2x the reference step count, half the
reference dry count).  They are tripwires for a structural regression, not
golden values — a scheme change that legitimately moves these numbers a little
should not fail, while the failure mode above (5.75x steps, 94% of the dry
region lost) trips them by a wide margin.
"""
import numpy as np
import pytest

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.boundary_conditions import BoundaryConditions, FromModel
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import (
    ManningFriction, Newtonian, StressFree)
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_jax.fvm.solver_jax import HyperbolicSolver

from conftest import _adaptive, wet_dry_o2

NX, H0, RADIUS, CFL, TEND = 48, 2.0, 0.15, 0.45, 0.02

# Measured on the current scheme (see the module docstring for the failure this
# guards against).  nx=48: 22 steps, 544 of 2304 cells fully dry, mass exact.
REF_STEPS = 22
REF_FULLY_DRY = 544
MAX_STEPS = 2 * REF_STEPS            # a dt collapse trips this
MIN_FULLY_DRY = REF_FULLY_DRY // 2   # losing the dry region trips this


def _march():
    model = SME(level=0, dimension=3,
                closures=[Newtonian(), ManningFriction(), StressFree()],
                boundary_conditions=BoundaryConditions(
                    [FromModel(tag=t, definition="wall")
                     for t in ("left", "right", "bottom", "top")]))
    sm = SystemModel.from_model(model)
    names = [str(s) for s in sm.state]
    h_i = names.index("h")

    def _ic(x):
        q = np.zeros(len(names))
        q[h_i] = H0 if np.hypot(x[0] - 0.5, x[1] - 0.5) < RADIUS else 0.0
        return q

    sm.initial_conditions = IC.UserFunction(function=_ic)
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=2, limiter="minmod"))
    mesh = LSQMesh.create_2d(domain=(0.0, 1.0, 0.0, 1.0), nx=NX, ny=NX)
    solver = HyperbolicSolver(time_end=np.inf, compute_dt=_adaptive(CFL, mesh),
                              **wet_dry_o2(nsm))
    Q, Qaux = solver.setup_simulation(mesh, nsm)
    nc = mesh.n_inner_cells
    mass0 = float(np.asarray(Q)[h_i, :nc].sum())
    t, n, hmin = 0.0, 0, np.inf
    while t < TEND and n < 50 * MAX_STEPS:
        dt = float(solver.compute_timestep(Q, Qaux))
        assert np.isfinite(dt) and dt > 0.0, (
            f"dt died at step {n}: {dt} — the march stalled; that is a FAILURE, "
            "not a stopping condition")
        dt = min(dt, TEND - t)
        Qn = solver.step(dt, t, Q, Qaux)
        Q, Qaux = solver.post_step(t + dt, dt, Qn, Q, Qaux)
        hmin = min(hmin, float(np.asarray(Q)[h_i, :nc].min()))
        t += dt
        n += 1
    h = np.asarray(Q)[h_i, :nc]
    return dict(steps=n, hmin=hmin, h=h, nc=nc,
                mass0=mass0, mass1=float(h.sum()))


@pytest.fixture(scope="module")
def march():
    """ONE march shared by all three assertions — they interrogate the same
    run from different angles, so re-marching per test would triple a
    small-tier cost for nothing."""
    return _march()


@pytest.mark.small
@pytest.mark.jax
def test_radial_dry_dambreak_stays_positive_and_conservative(march):
    """The two invariants that already held even when the scheme was broken.

    Kept because they are cheap and because their PASSING is what makes the
    two guards below necessary — see the module docstring."""
    r = march
    assert r["hmin"] >= 0.0, f"negative depth {r['hmin']} — no floor is permitted"
    drift = abs(r["mass1"] - r["mass0"]) / abs(r["mass0"])
    assert drift < 1e-12, f"mass drifted {drift:.2e}"


@pytest.mark.small
@pytest.mark.jax
def test_radial_dry_dambreak_does_not_collapse_dt(march):
    """Step count stays near the reference — a dt collapse means a lower CFL
    number, hence MORE numerical dissipation and a smeared front, which
    positivity and mass cannot see."""
    r = march
    assert r["steps"] <= MAX_STEPS, (
        f"{r['steps']} steps to t={TEND} against a reference of {REF_STEPS} "
        f"(limit {MAX_STEPS}).  dt has collapsed: the scheme is running at a "
        "much lower CFL number and is therefore more dissipative, even though "
        "h >= 0 and mass are still exact.")


@pytest.mark.small
@pytest.mark.jax
def test_radial_dry_dambreak_keeps_the_dry_region_dry(march):
    """Cells outside the front stay EXACTLY dry.

    Trace water arriving as a ~1e-11 m film is not physics — it is a velocity
    computed as a 0/0 ratio being propelled across the mesh, and the
    Kurganov-Petrova reciprocal exists precisely to stop it."""
    r = march
    fully_dry = int((r["h"] == 0.0).sum())
    assert fully_dry >= MIN_FULLY_DRY, (
        f"only {fully_dry} of {r['nc']} cells are exactly dry, against a "
        f"reference of {REF_FULLY_DRY} (limit {MIN_FULLY_DRY}).  The dry region "
        "is being wetted by a numerical film; mass and positivity are blind to "
        "this.")
