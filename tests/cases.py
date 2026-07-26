"""Case content for the jax test suite: ICs, BCs, analytic comparisons.

The module the code proposal (``2026-07-20-jax-test-suite-code.md``) names but
does not write.  It owns everything case-shaped so the test files stay short:
initial conditions, boundary conditions, the SWASHES analytic comparison, the
long-march history helper, the Escalante bump comparison and the AHS26 helper.

**Boundary conditions are declared ON THE MODEL, before ``from_model``.**
``SystemModel.boundary_conditions`` is a COMPILED symbolic kernel built at
``SystemModel.from_model`` time, not a declaration slot — assigning a
``BC.BoundaryConditions`` onto a built SystemModel would clobber that kernel.
``models.py`` therefore takes a hashable ``bc`` STRING (so its ``lru_cache``
still keys correctly) and calls :func:`bcs_for` inside to build the live
objects.  Only ``initial_conditions`` is assigned post-build, which is what
the proposal does.

Import style: bare (``from cases import ...``) — see ``conftest.py`` for why
``tests.cases`` cannot be used here.
"""
from __future__ import annotations

import csv
import pathlib

import numpy as np

G = 9.81

# SWASHES depths — the exact configuration the hand-built SWE's
# wet_dry_eps=1e-2 cap silently zeroed (cid=54).
ETA_L, ETA_R, DAM_X = 0.005, 0.001, 5.0
SWASHES_DOMAIN = (0.0, 10.0)
SWASHES_T_END = 6.0

# ── SWASHES 3.1.1, subcritical flow over a bump ────────────────────────────
# The one SMOOTH, wet/wet SWASHES case: no shock, no dry front, so it is the
# only one that can carry a genuine second-order floor.  Constants read off
# reference/bump_sub.meta.txt and the table's own inlet row.
BUMP_SUB_DOMAIN = (0.0, 25.0)
BUMP_SUB_Q = 4.42          # m^2/s, imposed upstream (table q, constant)
BUMP_SUB_H = 2.0           # m, imposed downstream (h + z = 2, z = 0 there)
BUMP_SUB_T_END = 200.0
#: Extra march used to VERIFY steadiness: the state must not move over it.
BUMP_SUB_T_SETTLE = 25.0


def bump_sub_bed(x):
    """z(x) = max(0, 0.2 - 0.05 (x - 10)^2) — the SWASHES bump."""
    return np.maximum(0.0, 0.2 - 0.05 * (np.asarray(x) - 10.0) ** 2)


def bump_sub_ic(xv):
    """Rest state h + z = 2, q = 0 — the SWASHES initial condition.

    The steady profile is REACHED by the march; it is not prescribed, so this
    genuinely exercises the scheme rather than seeding the answer."""
    b = bump_sub_bed(xv[0])
    return _pad_state(np.array([b, 2.0 - b, 0.0]))

AHS26_DOMAIN, AHS26_T_END = (0.0, 1.0), 0.5

ESC_DOMAIN, ESC_NCELLS = (-1.5, 1.5), 60
ESC_H_RES, ESC_H_DRY = 0.34, 0.015

# Pressure tolerance for the SMOOTH periodic VAM runs (the Richardson order
# study and its small twin).  The spec's 1e-13 is KEPT DELIBERATELY, as a
# TRUTHFUL request that the solver currently cannot meet — do not loosen it to
# silence the warning, because the warning is the finding.
#
# MEASURED: the matrix-free GMRES reaches 1e-13 at n = 64, but STAGNATES on
# refinement — rel residual 1.4e-04 at n = 128 and ~6e-03 at n = 256.  It is
# stagnation, not an exhausted budget: quadrupling ``pressure_maxit``
# 100 -> 400 (2000 -> 8000 matvecs) leaves the n = 128 residual at 1.432e-04,
# bit-identical.  The elliptic block is solved UNPRECONDITIONED, so its
# condition number grows like O(1/h^2) and the attainable residual degrades
# with every refinement.
#
# The GMRES stagnation above is REAL and still worth fixing with a
# preconditioner.  But the paragraph that used to sit here — claiming the
# stagnation is "why ``test_vam_second_order`` fails", because the pressure
# error is set by the solver residual rather than the discretization — is
# WRONG, and it misled a later reader into recommending the preconditioner as
# the top VAM priority.  RETRACTED, with the measurement that killed it:
#
#   config                       aggregate   P_0     P_1
#   defaults (order=1!)            -1.456   -1.61   -1.86
#   ReconstructionSpec(order=2)    +1.183   +0.93   +1.00
#
# Same solver, same tolerance, same unpreconditioned GMRES — only the
# reconstruction order changed, and the pressure rows go from diverging to
# converging.  The test was failing because it never requested second order:
# ``ReconstructionSpec`` defaults to ``order=1`` and ``time_order`` defaults
# to 1, and the test passed neither.
#
# What IS still open at order 2: ``h`` converges at ~1.07, not 2.  That is the
# real remaining defect, and it is a spatial-convergence question about the
# depth row, not an elliptic-solver question.
VAM_PRESSURE_TOL = 1e-13

# The cached SWASHES analytic tables live in the thesis case; zoomy_jax always
# sits inside the superrepo, so this relative hop is stable.
SWASHES_REF_DIR = (pathlib.Path(__file__).resolve().parents[3]
                   / "thesis" / "cases" / "verification" / "swe_swashes"
                   / "reference")


# ── boundary conditions ─────────────────────────────────────────────────────
def bcs_for(kind: str, dimension: int):
    """Live ``BoundaryConditions`` for a named kind.

    ``dimension`` is the MODEL dimension: 2 -> 1-D horizontal (tags
    left/right), 3 -> 2-D horizontal (left/right/top/bottom).
    """
    import zoomy_core.model.boundary_conditions as BC

    tags = (("left", "right") if dimension == 2
            else ("left", "right", "top", "bottom"))

    if kind in ("extrapolation", "swashes", "bump"):
        # SWASHES and the Escalante bump both run open (extrapolation) ends.
        return BC.BoundaryConditions([BC.Extrapolation(tag=t) for t in tags])
    if kind == "bump_sub":
        # SWASHES 3.1.1 subcritical bump: DISCHARGE imposed upstream, DEPTH
        # imposed downstream — the standard subcritical pair (one characteristic
        # enters at each end).  This is the only SWASHES case whose solution is
        # SMOOTH and wet everywhere, so it is the only one entitled to a real
        # second-order floor; stoker carries a shock and ritter a dry front,
        # and both are pinned rather than floored for that reason.
        # CHARACTERISTIC, not Lambda/Dirichlet.  A hard Dirichlet on h at the
        # outlet REFLECTS: measured, the state never settles — spatially frozen
        # (q spread 5.63e-03 at t=200 vs 5.72e-03 at t=600) yet still moving in
        # time (1.06e-03 over a further 25 s), i.e. a standing wave, not a
        # transient.  ``Characteristic`` imposes the target only on the INCOMING
        # characteristics, ``Q_ghost = Q + P⁻(Q_target − Q)`` with
        # ``P⁻ = R·1_{λ<0}·L``, leaving the outgoing content as the interior's —
        # so the prescribed value is honoured without reflecting what leaves.
        # The unprescribed slots extrapolate by default, which is exactly the
        # "fix one field, extrapolate the rest, then project" construction.
        # SWE state is [b, h, q_0]: discharge is index 2, depth index 1.
        left = [BC.Characteristic(tag="left",
                                  prescribe_fields={2: lambda *a: BUMP_SUB_Q})]
        right = [BC.Characteristic(tag="right",
                                   prescribe_fields={1: lambda *a: BUMP_SUB_H})]
        rest = [BC.Extrapolation(tag=t) for t in tags[2:]]
        return BC.BoundaryConditions(left + right + rest)
    if kind == "wall":
        # NO ``momentum_field_indices``: it defaults to ``None`` = DERIVE from
        # the model's own ``interpolate_to_3d`` rows at resolve time
        # (``zoomy_core/model/boundary_conditions.py``
        # ``derive_momentum_field_indices`` / ``_DerivedMomentumIndices``).
        # It USED to be pinned here to ``[[2]]`` / ``[[2, 3]]`` because the
        # old default was a hard-coded ``[[1, 2]]`` that reflected (h, q_0) as
        # a 2-vector and raised a sympy ShapeError against a 1-D normal.  The
        # derived default now yields exactly those same rows for the DERIVED
        # SME state ``[b, h, q_0(, q_1)]``, and deriving beats pinning: a
        # pinned index silently goes stale the moment the state layout moves.
        return BC.BoundaryConditions([BC.Wall(tag=t) for t in tags])
    if kind == "periodic":
        # Periodic BCs come in PAIRS: each carries the tag it maps onto.
        # Leaving ``periodic_to_physical_tag`` at its "" default makes the
        # mesh raise KeyError('') when it resolves the partner.
        pairs = [("left", "right"), ("top", "bottom")][:len(tags) // 2]
        return BC.BoundaryConditions(
            [BC.Periodic(tag=a, periodic_to_physical_tag=b)
             for a, b in pairs]
            + [BC.Periodic(tag=b, periodic_to_physical_tag=a)
               for a, b in pairs])
    if kind == "inflow":
        # Dirichlet on h AND q_0 at both ends, at the SMOOTH IC's own boundary
        # values.  The prescribed value differs from the boundary CELL AVERAGE
        # (the profile has NONZERO slope there), so the LSQ boundary row
        # carries a nonzero delta — this is the ONLY BC family that can see
        # the halved boundary gradient (REQ-46).  Wall / Extrapolation /
        # periodic all produce a zero or reflected delta and are structurally
        # blind to it.
        hl, ql = smooth_state(0.0)
        hr, qr = smooth_state(1.0)
        return BC.BoundaryConditions([
            BC.Dirichlet(tag="left", on="h", value=hl),
            BC.Dirichlet(tag="left", on="q_0", value=ql),
            BC.Dirichlet(tag="right", on="h", value=hr),
            BC.Dirichlet(tag="right", on="q_0", value=qr),
        ])
    raise KeyError(f"unknown BC kind {kind!r}")


# ── initial conditions ──────────────────────────────────────────────────────
def stoker_ic(x):
    """Wet dam break (Stoker): both states wet, flat bed."""
    return np.array([0.0, ETA_L if float(x[0]) < DAM_X else ETA_R, 0.0])


def ritter_ic(x):
    """Dry dam break (Ritter): capless dry front, flat bed.  The dry side is
    EXACTLY zero — no floor anywhere (user law: never clip h)."""
    return np.array([0.0, ETA_L if float(x[0]) < DAM_X else 0.0, 0.0])


def lake_at_rest_ic(x):
    """Flat surface over a Gaussian bump — the topography gate.  Mass
    conservation is BLIND to well-balancing; this is not."""
    X = float(x[0])
    b = 0.1 * np.exp(-((X - 5.0) ** 2) / 0.5)
    return np.array([b, 0.3 - b, 0.0])


def tilted_ic(x):
    """A tilted free surface on the unit domain: whatever the BC does, the
    interior will move, so the BC kernel is actually exercised."""
    X = float(x[0])
    return np.array([0.0, 0.5 + 0.05 * (X - 0.5), 0.0])


def gaussian_pulse_2d(x):
    """Radial pulse in a closed basin — exercises both horizontal dims."""
    r2 = float(x[0]) ** 2 + float(x[1]) ** 2
    return np.array([0.0, 0.5 + 0.05 * np.exp(-r2 / 0.2), 0.0, 0.0])


def smooth_state(X: float):
    """The smooth (h, q) profile of the boundary-order study.  Slope is
    NONZERO at both x = 0 and x = 1 — that is the whole point."""
    h = 1.0 + 0.1 * np.sin(2.0 * np.pi * X + 0.7)
    q = 0.1 * np.cos(2.0 * np.pi * X)
    return float(h), float(q)


def smooth_dirichlet_ic(x):
    h, q = smooth_state(float(x[0]))
    return np.array([0.0, h, q])


def bump_ic(x):
    """Escalante dam break over a Gaussian bump (thesis/cases/
    escalante_vam_bump).  The dry side sits at the PHYSICAL still depth of the
    experiment — an initial condition, not a floor: nothing clips h after."""
    X = float(x[0])
    b = 0.20 * np.exp(-(X ** 2) / (2 * 0.20 ** 2))
    h = (ESC_H_RES - b) if X < 1.0 else ESC_H_DRY
    return _pad_state(np.array([b, h]))


def smooth_vam_ic(x):
    """A smooth periodic VAM state — the Richardson order study needs
    smoothness (a discontinuity caps the observable rate at 1)."""
    X = float(x[0])
    h = 1.0 + 0.1 * np.sin(2.0 * np.pi * X)
    q = 0.1 * np.cos(2.0 * np.pi * X)
    return _pad_state(np.array([0.0, h, q]))


def ahs26_ic(x):
    """AHS26 (Sec. 3.1) lake-at-rest multilayer gate: flat surface over a
    smooth bed, both layers quiescent."""
    X = float(x[0])
    b = 0.1 * np.exp(-((X - 0.5) ** 2) / 0.02)
    return _pad_state(np.array([b, 0.5 - b]))


_STATE_WIDTH = {}


def _pad_state(head):
    """Zero-pad an IC head (b, h, ...) to the model's state width.

    The VAM / ML-SME state widths are model-derived, so the test must not
    hard-code them: ``models.state_width`` records the true width when the NSM
    is built and this pads to it.  A missing width RAISES rather than guessing
    — a silently short IC would make the extra rows read whatever the solver
    allocated (user law: no silent defaults for state rows).
    """
    n = _STATE_WIDTH.get("n")
    if n is None:
        raise AssertionError(
            "state width unknown — call cases.set_state_width(nsm) after "
            "building the NSM and before marching")
    if len(head) > n:
        raise AssertionError(f"IC head of {len(head)} rows exceeds state {n}")
    out = np.zeros(n)
    out[:len(head)] = head
    return out


def set_state_width(nsm):
    """Record the model's state width so the padded ICs above know it."""
    _STATE_WIDTH["n"] = len(list(nsm.state))


def ic_for(case: str):
    return {"stoker_wet": stoker_ic, "ritter_dry": ritter_ic}[case]


# ── the MANDATED pre-march operator print ───────────────────────────────────
def describe(nsm) -> str:
    """Render the NSM operator matrices — the MANDATED pre-march print."""
    return "\n".join([
        "── NSM operator matrices (pre-march sanity print) ──",
        f"state: {list(nsm.state)}",
        f"aux_state: {list(nsm.aux_state)}",
        f"parameter_values: {getattr(nsm, 'parameter_values', None)}",
        f"flux:\n{nsm.flux}",
        f"hydrostatic_pressure:\n{getattr(nsm, 'hydrostatic_pressure', None)}",
        f"nonconservative_matrix:\n{nsm.nonconservative_matrix}",
        f"source:\n{nsm.source}",
        f"eigenvalues:\n{getattr(nsm, 'eigenvalues', None)}",
        f"update_variables (must be None — cap-free): "
        f"{getattr(nsm, 'update_variables', None)}",
        f"riemann: {getattr(nsm, 'riemann', None)}",
        f"dt_max: {getattr(nsm, 'dt_max', None)}",
    ])


# ── SWASHES analytic comparison ─────────────────────────────────────────────
def swashes_table(case: str) -> dict:
    """The cached SWASHES analytic table (the library's own output).

    Generated by the ``swashes`` binary at t = 6 s over (0, 10) and cached in
    the thesis case.  We read the cache rather than shelling out so the suite
    is reproducible without the binary.
    """
    path = SWASHES_REF_DIR / f"{case}.csv"
    if not path.exists():
        raise AssertionError(
            f"missing SWASHES analytic table {path} — regenerate it with "
            f"`python run_verification.py` in thesis/cases/"
            f"verification/swe_swashes (needs the `swashes` binary).")
    cols: dict = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return {k: np.asarray(v) for k, v in cols.items()}


def l1_vs_analytic(Q, mesh, case: str, t: float) -> float:
    """Mesh-normalized L1 error of h against the SWASHES analytic solution.

    ``t`` is ASSERTED against the table's time rather than used: the cached
    dam-break tables are t = 6 s only, and silently comparing a t = 1 s run
    against a t = 6 s table would manufacture a meaningless "error" that still
    converges.

    ``bump_sub`` is exempt from that assertion because its solution is STEADY —
    the table is the t → ∞ limit, so any march time long enough to reach
    steady state is the right comparison.  The march time is instead validated
    by the steady-state residual check at the call site.
    """
    if case != "bump_sub":
        assert abs(t - SWASHES_T_END) < 1e-12, (
            f"the cached SWASHES tables are t = {SWASHES_T_END} s only; got "
            f"t = {t}. Generate a new table before comparing at another time.")
    ref = swashes_table(case)
    n = mesh.n_inner_cells
    x = np.asarray(mesh.cell_centers[0, :n], float)
    dx = np.asarray(mesh.cell_volumes[:n], float)
    h_exact = np.interp(x, ref["x"], ref["h"])
    h = np.asarray(Q[1], float)
    return float(np.sum(np.abs(h - h_exact) * dx) / np.sum(dx))


# ── boundary-order probe: ACOUSTIC standing wave in a CLOSED BOX ────────────
# RETRIEVED case, not a constructed one.  Source:
# ``tests/scripts/zoomy_core/swe/run_acoustic_wall_convergence.py`` at superrepo
# sha d6cafaab ("Standing wave convergence test: O2 rate 2.25 at walls",
# 2026-04-11), deleted at 2cc22d51.
#
# WHY THIS REPLACED THE CONSTRUCTED SMOOTH-DIRICHLET CASE: that one had NO
# closed form, so it measured the error against a 1600-cell run of the SAME
# code conservatively averaged down.  A self-reference cannot separate "the
# scheme is 2nd order" from "the scheme and its own reference share a defect",
# and it sat at rate 0.83 with no way to tell which.  The standing wave has an
# EXACT solution and the wall BCs are exactly the BCs the boundary treatment
# has to get right.
#
# SWE linearised about h = H0 in a box with a WALL at each end:
#     h(x, t) = H0 + A cos(m π x / L) cos(m π c t / L)
#     u(x, t) = (A c / H0) sin(m π x / L) sin(m π c t / L),   c = sqrt(g H0)
# After one full period the exact solution is the IC again, so the error needs
# no quadrature of an analytic profile at an awkward time — it is |h − h(t=0)|.
ACOUSTIC_DOMAIN = (0.0, 1.0)
ACOUSTIC_H0 = 1.0
ACOUSTIC_A = 1e-4          # linear amplitude: 1e-4 keeps the wave acoustic,
                           # so the closed form is the solution and not just
                           # its leading order
ACOUSTIC_MODE = 1
ACOUSTIC_PERIODS = 1


def acoustic_t_end() -> float:
    """One full period of mode ``ACOUSTIC_MODE``."""
    L = ACOUSTIC_DOMAIN[1] - ACOUSTIC_DOMAIN[0]
    c = np.sqrt(G * ACOUSTIC_H0)
    return ACOUSTIC_PERIODS * 2.0 * L / (ACOUSTIC_MODE * c)


def acoustic_h_exact(x) -> np.ndarray:
    """``h`` of the standing wave at ``t = 0`` — and, after a whole number of
    periods, at ``t = acoustic_t_end()`` as well."""
    L = ACOUSTIC_DOMAIN[1] - ACOUSTIC_DOMAIN[0]
    return ACOUSTIC_H0 + ACOUSTIC_A * np.cos(
        ACOUSTIC_MODE * np.pi * np.asarray(x, float) / L)


def acoustic_ic(x):
    """``Q = [0, H0 + A cos(m π x / L), 0]`` — flat bed, fluid at rest."""
    return _pad_state(np.array([0.0, float(acoustic_h_exact(float(x[0]))),
                                0.0]))


def acoustic_l2_by_window(Q, mesh) -> dict:
    """L2 error of ``h`` against the CLOSED FORM, decomposed into
    full / interior / left-strip / right-strip windows.

    The windowed decomposition is kept from the constructed case it replaces:
    the boundary defect it was built to expose showed up as an ASYMMETRY
    between the left and right strips, which a full-domain norm dilutes below
    visibility.  Strips are the outer 5 % (the historical case's ``N // 20``).
    """
    n = mesh.n_inner_cells
    x = np.asarray(mesh.cell_centers[0, :n], float)
    h = np.asarray(Q[1, :n], float)
    assert np.all(np.isfinite(h)), "non-finite h — the march diverged"
    assert h.min() > 0.5 * ACOUSTIC_H0, (
        f"unphysical h.min() = {h.min()} — an acoustic wave of amplitude "
        f"{ACOUSTIC_A} cannot legitimately move h by more than that")
    d = h - acoustic_h_exact(x)

    k = max(2, n // 20)
    masks = {"full": slice(None), "interior": slice(k, n - k),
             "left": slice(0, k), "right": slice(n - k, n)}
    return {w: float(np.sqrt(np.mean(d[m] ** 2))) for w, m in masks.items()}


# ── long-march history (well-balancing drift over time) ─────────────────────
def march_with_history(nsm, mesh, t_end, cfl, n_samples: int = 50):
    """March ONCE to ``t_end``, sampling the lake-at-rest drift on the way.

    Returns ``(Q, Qaux, t, drift, umax)``: the final inner state, then sample
    times, max|eta - eta0| and max|u| at each sample.  Sampling (rather than a
    single end state) is what makes the DRIFT HISTORY observable — an end
    state alone cannot distinguish "never drifted" from "drifted and came
    back".

    Implemented on the step API, advancing ONE continuous trajectory.  The
    obvious alternative — one fresh ``solve`` per sample time — is O(n^2) in
    the sample count and cannot fit the regression budget.
    """
    from zoomy_jax.fvm.solver_jax import HyperbolicSolver
    from conftest import _adaptive

    n = mesh.n_inner_cells
    solver = HyperbolicSolver(time_end=t_end, compute_dt=_adaptive(cfl, mesh))
    Q, Qaux = solver.setup_simulation(mesh, nsm)
    Qi = np.asarray(Q)[:, :n]
    eta0 = float((Qi[0] + Qi[1])[0])
    targets = [t_end * k / n_samples for k in range(1, n_samples + 1)]

    ts, drifts, umaxs = [], [], []
    t = 0.0
    for t_target in targets:
        while t < t_target - 1e-15:
            dt = float(solver.compute_timestep(Q, Qaux))
            assert dt > 0.0 and np.isfinite(dt), (
                f"non-positive/non-finite dt = {dt} at t = {t} — the march "
                f"stalled; this is a FAILURE, not a stopping condition")
            dt = min(dt, t_target - t)
            Qn = solver.step(dt, t, Q, Qaux)
            Q, Qaux = solver.post_step(t + dt, dt, Qn, Q, Qaux)
            t += dt
        Qi = np.asarray(Q)[:, :n]
        eta = Qi[0] + Qi[1]
        ts.append(t)
        drifts.append(float(np.abs(eta - eta0).max()))
        umaxs.append(float(np.abs(Qi[2] / Qi[1]).max()))
    return (np.asarray(Q)[:, :n], np.asarray(Qaux)[:, :n],
            np.asarray(ts), np.asarray(drifts), np.asarray(umaxs))


# ── the Chorin split-solver march ───────────────────────────────────────────
def chorin_split_for(model, sm):
    """``model.chorin_split`` -> the ``(SM_pred, SM_press, SM_corr)`` triple.

    DEVIATION FROM THE PROPOSAL (API, forced): the proposal writes
    ``solver.solve(mesh, split)``, but ``ChorinSplitVAMSolverJax`` has NO
    ``solve`` method — its API is ``__init__(sm_pred, sm_press, sm_corr)`` +
    ``setup_simulation(mesh)`` + ``step(dt, t, Q, Qaux)`` (verified in
    ``zoomy_jax/fvm/solver_chorin_vam_jax.py:87,158,411``).  ``chorin_split``
    returns a ``SplitForPressureResult`` dataclass (``.SM_pred`` / ``.SM_press``
    / ``.SM_corr``), NOT a subscriptable tuple.  So the stages are unpacked
    here and :func:`chorin_march` drives the step loop.
    """
    import sympy as sp
    r = model.chorin_split(sp.Symbol("dt"), system_model=sm)
    return (r.SM_pred, r.SM_press, r.SM_corr)


def chorin_march(triple, mesh, cfl, ic, t_end=None, n_steps=None,
                 h_scale=1.0, **solver_kw):
    """March the jax Chorin split VAM solver; return the inner (Q, Qaux).

    Fixed dt at the 1-D CFL law from the still depth ``h_scale`` — the split
    solver has no adaptive controller.  Exactly one of ``t_end`` / ``n_steps``.

    ``ic`` is the initial-condition CALLABLE and is MANDATORY.  It is applied
    to the solver state HERE, explicitly, because
    ``ChorinSplitVAMSolverJax.setup_simulation`` **ignores initial conditions
    entirely** — the string ``initial_conditions`` does not occur anywhere in
    ``zoomy_jax/fvm/solver_chorin_vam_jax.py``.  Assigning
    ``sm.initial_conditions`` (what the first cut of these tests did) is a
    silent no-op: the march then ran from h == 0 in every cell, the elliptic
    pressure block's bare 1/h went singular, GMRES burned its whole restart
    budget every step, and the "reference" was a null state.  The thesis case
    ``thesis/cases/escalante_vam_bump/run_jax.py:39-46`` injects the state by
    hand for exactly this reason; this mirrors that proven pattern.

    Time loop: ``run_jit_steps`` (jit + ``lax.scan``), NOT the ``step()``
    wrapper.  ``step()`` drives ``chorin_cycle`` EAGERLY — measured 2.66 s per
    step against 3.4 ms through the scan, bit-identical output (max|diff| = 0).
    """
    import jax.numpy as jnp
    from zoomy_jax.fvm.solver_chorin_vam_jax import ChorinSplitVAMSolverJax

    if (t_end is None) == (n_steps is None):
        raise TypeError("chorin_march() takes exactly one of t_end / n_steps")
    sm_pred, sm_press, sm_corr = triple
    solver = ChorinSplitVAMSolverJax(sm_pred, sm_press, sm_corr, **solver_kw)
    Q0 = np.array(solver.setup_simulation(mesh))     # writable host copy
    nc = solver.nc

    # ── explicit IC injection (see docstring) ──
    xc = np.asarray(solver._rt_mesh.cell_centers)[:, :nc]
    for j in range(nc):
        Q0[:, j] = ic(xc[:, j])
    # HARD guard: a dry-everywhere state is the exact failure this function
    # exists to prevent, and it is otherwise invisible (it "runs", just ~780x
    # slower and against a meaningless reference).  Fail, never warn.
    assert Q0[1, :nc].max() > 0.0, (
        "chorin_march: initial depth is zero in EVERY cell — the IC did not "
        "reach the solver state. ChorinSplitVAMSolverJax ignores "
        "sm.initial_conditions; the IC must be injected here.")
    solver._sim_Q = jnp.asarray(Q0)
    solver.update_aux_variables()

    dx = float(np.asarray(solver._rt_mesh.cell_volumes)[0])
    dt = cfl * dx / (np.sqrt(G * h_scale) + 1.0)
    if n_steps is None:
        # Land EXACTLY on t_end: take the CFL dt as an UPPER bound, then
        # shrink it to divide t_end evenly.  dt only ever DECREASES here (by
        # < one step's worth), so this respects the CFL law rather than
        # relaxing it — the old ``ceil`` overshot t_end by up to a full step
        # (measured t = 0.5095 for a requested 0.5).
        n_steps = max(1, int(np.ceil(t_end / dt)))
        dt = t_end / n_steps
    Q, Qaux, _, _, _ = solver.run_jit_steps(
        jnp.asarray(dt), n_steps, solver._sim_Q, solver._sim_Qaux,
        solver.Qaux_press, solver.Qaux_corr, t_start=0.0)
    return np.asarray(Q)[:, :nc], np.asarray(Qaux)[:, :nc]


# ── Escalante dam-break over a bump ─────────────────────────────────────────
def escalante_experiment():
    """The digitized Escalante (2024) experiment arrays shipped by the case."""
    case = (pathlib.Path(__file__).resolve().parents[3]
            / "thesis" / "cases" / "escalante_vam_bump" / "run.py")
    if not case.exists():
        raise AssertionError(f"missing Escalante case {case}")
    # ``run.py`` executes a full case at import; read the literals instead.
    src = case.read_text()
    ns: dict = {"np": np}
    for name in ("ETA_EXP_X", "ETA_EXP_Y"):
        line = next(l for l in src.splitlines() if l.startswith(f"{name} ="))
        exec(line, ns)  # noqa: S102 — a single np.array literal from our repo
    return ns["ETA_EXP_X"], ns["ETA_EXP_Y"]


def rms_vs_experiment(Q, mesh) -> float:
    """RMS of the computed free surface against the digitized experiment,
    sampled at the EXPERIMENT's own x stations."""
    xe, ye = escalante_experiment()
    n = mesh.n_inner_cells
    x = np.asarray(mesh.cell_centers[0, :n], float)
    eta = np.asarray(Q[0], float) + np.asarray(Q[1], float)
    order = np.argsort(x)
    eta_at = np.interp(xe, x[order], eta[order])
    return float(np.sqrt(np.mean((eta_at - ye) ** 2)))


# ── AHS26 multilayer ────────────────────────────────────────────────────────
def ahs26_l1_vs_reference(Q, mesh) -> float:
    """L1 departure of the free surface from the flat equilibrium.

    The thesis case (``thesis/cases/hoern``) refines MLSME(level=0) over
    layers; core golden ``m10`` pins MLSME(n_layers=2, level=1, dimension=2).
    Every model the jax suite runs must be golden-covered, so this uses the
    GOLDEN model and measures what the AHS26 protocol measures on it: the L1
    departure from the initial well-balanced equilibrium.  It is a
    well-balancing drift norm, NOT a reproduction of the published Table 1
    numbers (those are a different model and a different refinement path).
    """
    n = mesh.n_inner_cells
    dx = np.asarray(mesh.cell_volumes[:n], float)
    eta = np.asarray(Q[0], float) + np.asarray(Q[1], float)
    return float(np.sum(np.abs(eta - 0.5) * dx) / np.sum(dx))


# ── SPMD helper ─────────────────────────────────────────────────────────────
def used_devices(arr=None) -> int:
    """How many devices actually held a shard of the last sharded march.

    A 2-device run that silently fell back to one device would otherwise pass
    the state comparison trivially, so the parallel test asserts on this.
    ``march_sharded`` gathers its result to host before returning (the caller
    needs a plain array to compare), so the sharding is read off the on-device
    output THERE and recorded in ``conftest.LAST_SHARD``; ``arr`` is accepted
    for call-site symmetry with the proposal and used only if it still carries
    a sharding of its own.
    """
    sharding = getattr(arr, "sharding", None)
    devs = getattr(sharding, "device_set", None)
    if devs:
        return len(devs)
    from conftest import LAST_SHARD
    return LAST_SHARD["devices"]
