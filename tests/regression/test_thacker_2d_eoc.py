"""2-D Thacker (SWASHES) EOC guard: order, mass conservation and positivity
on a GENUINE two-horizontal-direction mesh (``LSQMesh.create_2d``).

Companion FULL cases (both orders, 3-grid ladder n=40,80,160, cross-section +
convergence figures, deliverable write-ups): ``thesis/cases/verification/
2d_thacker_planar`` and ``2d_thacker_radial`` (SWASHES, Delestre et al. 2013,
2D type 1 domain 1 choice 2 / choice 1 -- tilted-plane / axisymmetric
paraboloid bowl). That is a SEPARATE, private thesis repo and not a
dependency of this suite, so the setup below MIRRORS (does not import) its
``run.py``: identical bed/exact formulas, identical BC/model/solver knobs
(``SME(level=0)``, Wall BCs, ``reconstruction_variables='eta'``,
``positivity_method='zhang_shu'`` at order >= 2, ``minmod``, CFL=1/6 at order
2, ``gate_eigenvalues_dry()`` in ``extra_operations``).

Before this file, NOTHING in the pytest suite referenced Thacker or asserted
a genuinely 2-D (nx * ny) order -- confirmed 2026-08-09 by grepping every
``test_*.py`` and every ``tests/`` tree for "thacker" (case-insensitive,
zero hits) before writing this. The other 2-D-*named* order tests
(``test_swashes_smooth_order2.py`` etc.) run on 1-D meshes carrying
``dimension=2`` model metadata; ``LSQMesh.create_2d`` here is the real thing.

LADDER: only ``nx = [40, 80]`` -- the two COARSEST levels of the case's own
``[40, 80, 160]`` -- to keep a real 2-D time-march affordable even in the
``--run-large`` tier. That pair is also exactly what the FIRST reported
pairwise EOC below measures. Order 2 ONLY: order 1 is not what regressed
(see the module note below) and is not guarded here.

FLOORS come from this test's OWN measured numbers at ``wet_dry_eps=1e-5``
(the value the case now uses -- see ``WET_DRY_EPS`` below), cross-checked
against the case's fresh 2026-08-09 re-run (``outputs/*_convergence.csv``
under ``thesis/cases/verification/2d_thacker_{planar,radial}``). This
supersedes BOTH the original stale ``run.py`` docstring numbers (dated
2026-07-27) AND this test's own first-cut floors (derived earlier the same
day at the case's then-default ``wet_dry_eps=1e-3``: planar 0.60, radial
0.90 -- since superseded when the case moved to ``1e-5``). Do NOT raise these
back to a hopeful 2.0: both cases are honestly front-limited (cid 152,
O1-at-front accepted -- the moving wet/dry shoreline is only
first-order-in-position, and via mass balance that error smears into a
spatially-uniform depth offset that floors the interior EOC) and genuinely
land below clean second order at n=40->80.

WHY THE PLANAR FLOOR MOVED (context, not asserted here):
``gate_eigenvalues_dry`` (zoomy_core, commit 1749ac2, 2026-08-07, cid 209)
stopped a pre-existing bug where the dry-cell gate OVERWROTE the shared
``sm.eigenvalues`` -- the array the Rusanov face dissipation reads --
zeroing it near the wet/dry front and running that face as a pure central
scheme. The fix publishes the gated copy to a separate ``eigenvalues_cfl``
slot and leaves ``eigenvalues`` physical. That repaired positivity (this
case's hmin: ~-3.5e-3..-4.4e-3 pre-fix -> ~+1e-8 now) but restored full
numerical dissipation at the moving shoreline, growing the front error that
dominates this case's L1 budget. At the case's THEN-default
``wet_dry_eps=1e-3`` this dropped the measured coarse-pair EOC to 0.76
(floor 0.60). A same-day wet_dry_eps sweep (1e-2/1e-3/1e-4/1e-5 ->
coarse-pair O2 core EOC 0.345/0.764/1.238/1.288; hmin and mass unaffected)
showed the rate recovering as eps shrinks, so the case adopted
``wet_dry_eps=1e-5`` -- see ``thesis/cases/verification/2d_thacker_planar/
run.py`` for the full A/B (limiter, wet_dry_eps, positivity_method, mood,
gate on/off/old-style). The floor below is re-based on that configuration.

RADIAL DOES NOT FOLLOW PLANAR (see the module note in
``2d_thacker_radial/run.py``): under the SAME cid-209 fix, radial's
coarse-pair O2 core EOC ROSE, not fell (pre-fix ~0.39-0.56 -> eps=1e-3 ~1.15
-> eps=1e-5 ~0.98-1.22). It also exposed a run-to-run noise source that
planar does not show: repeat GPU solves of radial's O2 nx=80 are NOT
bit-reproducible (a ~0.24 spread measured across two independent runs of
THIS test -- see the per-test docstring below). The radial floor is set with
a wider margin than planar's for that reason, not out of doubt about the
``wet_dry_eps=1e-5`` choice itself -- mass and positivity stay clean and
every sampled EOC is at or above the eps=1e-3 baseline.
"""
import numpy as np
import pytest

import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import (
    ManningFriction, Newtonian, StressFree)
from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.systemmodel.operations import gate_eigenvalues_dry
from zoomy_core.systemmodel.system_model import SystemModel

from conftest import march

G = 9.81
H0 = 0.1
L = 4.0
XC = YC = 0.0
WET_DRY_EPS = 1e-5
DRY = 1e-8
CORE_FLOOR = 0.05          # always-wet-core depth threshold [m] (case's own)
NLEVELS = [40, 80]         # coarsest 2 levels of the case's own [40, 80, 160]
ORDER = 2                  # the order under guard (cid 209 regression)
CFL_O2 = 1.0 / 6.0         # matches the case: positivity_method='zhang_shu' needs CFL <= 1/(2k+1)


# ── planar (tilted-plane) Thacker -- SWASHES 2D type 1 domain 1 choice 2 ────
_ETA_PLANAR = 0.25
_A_PLANAR = 1.0
_OMEGA_PLANAR = np.sqrt(2.0 * G * H0) / _A_PLANAR
T_PERIOD_PLANAR = 2.0 * np.pi / _OMEGA_PLANAR


def _bed_planar(x, y):
    return H0 * (((x - XC) ** 2 + (y - YC) ** 2) / _A_PLANAR ** 2 - 1.0)


def _exact_planar(x, y, t):
    X, Y = x - XC, y - YC
    z_surf = (_ETA_PLANAR * H0 / _A_PLANAR ** 2) * (
        2.0 * X * np.cos(_OMEGA_PLANAR * t) + 2.0 * Y * np.sin(_OMEGA_PLANAR * t)
        - _ETA_PLANAR)
    h = np.maximum(z_surf - _bed_planar(x, y), 0.0)
    u = np.full_like(h, -_ETA_PLANAR * _OMEGA_PLANAR * np.sin(_OMEGA_PLANAR * t))
    v = np.full_like(h, _ETA_PLANAR * _OMEGA_PLANAR * np.cos(_OMEGA_PLANAR * t))
    wet = h > 0.0
    return h, np.where(wet, u, 0.0), np.where(wet, v, 0.0)


# ── radial (axisymmetric paraboloid) Thacker -- SWASHES ...choice 1 ────────
_A_RADIAL = 1.0
_R0_RADIAL = 0.8
_OMEGA_RADIAL = np.sqrt(8.0 * G * H0) / _A_RADIAL
_AT_RADIAL = ((_A_RADIAL ** 2 - _R0_RADIAL ** 2)
             / (_A_RADIAL ** 2 + _R0_RADIAL ** 2))
T_PERIOD_RADIAL = 2.0 * np.pi / _OMEGA_RADIAL


def _bed_radial(x, y):
    return H0 * (((x - XC) ** 2 + (y - YC) ** 2) / _A_RADIAL ** 2 - 1.0)


def _exact_radial(x, y, t):
    X, Y = x - XC, y - YC
    r2 = X ** 2 + Y ** 2
    P = 1.0 - _AT_RADIAL * np.cos(_OMEGA_RADIAL * t)
    h = (H0 / P) * (np.sqrt(1.0 - _AT_RADIAL ** 2)
                    - (r2 / _A_RADIAL ** 2) * (1.0 - _AT_RADIAL ** 2) / P)
    h = np.maximum(h, 0.0)
    u = 0.5 * _OMEGA_RADIAL * _AT_RADIAL * X * np.sin(_OMEGA_RADIAL * t) / P
    v = 0.5 * _OMEGA_RADIAL * _AT_RADIAL * Y * np.sin(_OMEGA_RADIAL * t) / P
    wet = h > 0.0
    return h, np.where(wet, u, 0.0), np.where(wet, v, 0.0)


def _always_wet_core(exact_fn, x, y, t_period, floor=CORE_FLOOR, n_phase=64):
    """Boolean mask of cells wet at EVERY phase of the oscillation (the
    case's own always-wet-core mask, fixed physical floor on min_t h)."""
    hmin_t = None
    for k in range(n_phase):
        hk, _, _ = exact_fn(x, y, k / n_phase * t_period)
        hmin_t = hk if hmin_t is None else np.minimum(hmin_t, hk)
    return hmin_t > floor


def _run(bed_fn, exact_fn, t_period, nx):
    """One (case, nx) solve -> L1 core error, mass drift, hmin. Mirrors the
    case's ``build_system_model`` / ``build_solver`` / ``build_nsm`` /
    ``domain_metrics`` at order 2 exactly (see module docstring)."""
    bcs = BC.BoundaryConditions(
        [BC.Wall(tag=t) for t in ("left", "right", "top", "bottom")])
    model = SME(level=0, dimension=3,
                closures=[Newtonian(), ManningFriction(h_floor=WET_DRY_EPS),
                         StressFree()],
                parameters={"wet_dry_eps": WET_DRY_EPS}, boundary_conditions=bcs)
    sm = SystemModel.from_model(model)
    names = [str(s) for s in sm.state]
    idx = {n: i for i, n in enumerate(names)}

    def ic(p):
        x, y = float(p[0]), float(p[1])
        h, u, v = exact_fn(np.array(x), np.array(y), 0.0)
        h = max(float(h), DRY)
        return np.array([bed_fn(x, y), h, h * float(u), h * float(v)])

    sm.initial_conditions = IC.UserFunction(function=ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda k: np.zeros(k))
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=ORDER, limiter="minmod"),
        extra_operations=[gate_eigenvalues_dry()])
    mesh = LSQMesh.create_2d(domain=(-L / 2, L / 2, -L / 2, L / 2), nx=nx, ny=nx)
    ni = mesh.n_inner_cells
    Q, _ = march(nsm, mesh, cfl=CFL_O2, t_end=t_period,
                 free_surface_h_index=idx["h"], free_surface_b_index=idx["b"],
                 free_surface_momentum_indices=[idx["q_x_0"], idx["q_y_0"]],
                 reconstruction_variables="eta", positivity_method="zhang_shu",
                 mood=True)
    x = np.asarray(mesh.cell_centers[0, :ni])
    y = np.asarray(mesh.cell_centers[1, :ni])
    area = np.asarray(mesh.cell_volumes[:ni])
    h = np.asarray(Q[idx["h"], :ni])

    he, _, _ = exact_fn(x, y, t_period)
    abserr = np.abs(h - he) * area
    core = _always_wet_core(exact_fn, x, y, t_period)
    h0 = np.maximum(exact_fn(x, y, 0.0)[0], DRY)
    m0 = float((h0 * area).sum())
    mt = float((h * area).sum())
    return {
        "l1_core": float(abserr[core].sum()), "n_core": int(core.sum()),
        "mass_rel_err": (mt - m0) / m0, "hmin": float(h.min()),
    }


def _eoc_case(bed_fn, exact_fn, t_period):
    errs, hmins, masses = [], [], []
    for nx in NLEVELS:
        m = _run(bed_fn, exact_fn, t_period, nx)
        assert m["n_core"] > 0, (
            f"nx={nx}: always-wet core is EMPTY -- CORE_FLOOR/mesh mismatch, "
            "not a numerics result")
        errs.append(m["l1_core"])
        hmins.append(m["hmin"])
        masses.append(m["mass_rel_err"])
    eoc = float(np.log2(errs[0] / errs[1]))
    return eoc, errs, hmins, masses


@pytest.mark.large
@pytest.mark.jax
def test_thacker_planar_o2_eoc_mass_positivity():
    """2-D planar Thacker, order 2, nx 40 -> 80: EOC floor, mass, hmin.

    EOC floor 1.00 (was 0.60 at the case's superseded wet_dry_eps=1e-3): the
    case adopted wet_dry_eps=1e-5 on 2026-08-09 (user decision -- see
    ``2d_thacker_planar/run.py`` for the full sweep). At 1e-5 the fresh
    case re-run (``thesis/cases/verification/2d_thacker_planar/outputs/
    thacker_planar_convergence.csv``) measures core pairwise EOC 1.29
    (n=40->80) and 0.73 (n=80->160), LS slope 1.01 over n=40,80,160. This
    test's OWN mirror reproduces the 40->80 pair to 3 decimal places across
    two independent runs (1.288, 1.288) -- planar is NOT noise-sensitive
    (contrast the radial test below). 1.00 sits ~22% below the
    measured/reproduced 1.288, the same relative margin the old 0.60 floor
    kept below its (lower) 0.76.

    Mass floor 1e-10: the same csv measures |mass_rel_err| ~ 1e-16 (machine
    zero) at every level.

    hmin floor -1e-6: the same csv measures hmin ~ +2e-9..+1.0e-8 (POSITIVE)
    at every level -- a real improvement on the pre-cid-209 docstring's
    hmin ~ -3.5e-3..-4.4e-3. -1e-6 leaves ~2 orders of run-to-run float
    margin around the measured +1e-8 without re-accepting that old
    violation.
    """
    eoc, errs, hmins, masses = _eoc_case(_bed_planar, _exact_planar,
                                         T_PERIOD_PLANAR)
    print(f"planar O2 core: EOC[{NLEVELS[0]}->{NLEVELS[1]}]={eoc:.3f} "
          f"L1={errs} hmin={hmins} mass={masses}")
    assert eoc >= 1.00, (
        f"planar core O2 EOC dropped below the measured floor: {eoc:.3f}")
    for nx, mr, hm in zip(NLEVELS, masses, hmins):
        assert abs(mr) < 1e-10, f"nx={nx}: mass not conserved: {mr:.3e}"
        assert hm > -1e-6, (
            f"nx={nx}: hmin={hm:.3e} -- positivity regressed toward the "
            "pre-cid-209 -3.5e-3..-4.4e-3 violation")


@pytest.mark.large
@pytest.mark.jax
def test_thacker_radial_o2_eoc_mass_positivity():
    """2-D radial Thacker, order 2, nx 40 -> 80: EOC floor, mass, hmin.

    EOC floor 0.80 (was 0.90 at the case's superseded wet_dry_eps=1e-3): the
    case adopted wet_dry_eps=1e-5 on 2026-08-09 (user decision -- see
    ``2d_thacker_radial/run.py`` for the full story). Unlike planar, radial
    is NOT reproducible run-to-run on this GPU: two independent runs of
    THIS test measured core pairwise EOC[40->80] = 1.221 and 0.983 (a ~0.24
    spread); two back-to-back direct solves at nx=80 alone (same process,
    same code path) differ by ~3% in L1_core -- see ``2d_thacker_radial/
    run.py``'s NOISE note for the full A/B (snapshot-timing and ``mood``
    default both ruled out as the cause). The case's own fresh re-run
    (``thacker_radial_convergence.csv``) landed at 1.00 (n=40->80) and 1.49
    (n=80->160) -- inside the same band. 0.80 sits below EVERY sample
    observed so far (min 0.983) with real margin: it prioritizes a
    non-flaky floor over squeezing the tightest number out of a single run,
    while remaining far above the pre-cid-209 values (0.39, 0.56) so it
    still catches a real regression back to the old broken (h<0) behaviour.

    Mass floor 1e-10 / hmin floor -1e-6: same justification as the planar
    test above -- |mass_rel_err| ~ 1e-16 and hmin ~ +1.0e-8 at every level,
    in every run collected (the noise above affects L1_core, not mass or
    positivity).
    """
    eoc, errs, hmins, masses = _eoc_case(_bed_radial, _exact_radial,
                                         T_PERIOD_RADIAL)
    print(f"radial O2 core: EOC[{NLEVELS[0]}->{NLEVELS[1]}]={eoc:.3f} "
          f"L1={errs} hmin={hmins} mass={masses}")
    assert eoc >= 0.80, (
        f"radial core O2 EOC dropped below the measured floor: {eoc:.3f}")
    for nx, mr, hm in zip(NLEVELS, masses, hmins):
        assert abs(mr) < 1e-10, f"nx={nx}: mass not conserved: {mr:.3e}"
        assert hm > -1e-6, (
            f"nx={nx}: hmin={hm:.3e} -- positivity regressed toward the "
            "pre-cid-209 -2.3e-3..-3.2e-3 violation")
