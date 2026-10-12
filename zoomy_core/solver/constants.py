"""EMITTED march constants (design v7 / mandate 6a).

The user law is blunt: **no numerical constant is hard-coded in a backend**.
Every number the march compares against — the MOOD detector bound, the KP
wet/dry ``eps``, the eigenvalue-slot wave-speed floor ``eps_h``, the CFL
safety factor and its dimension/degree denominators, the write-cadence
tolerance and the (default-OFF) dt floor — is therefore RESOLVED FROM THE NSM
here and EMITTED as a named value at the head of the emitted procedure.

Consequence for the emitted code: a comparison site reads
``lam < c_eps_h``, never ``lam < 1e-14``.  The only place a float literal
survives is the ``Assign`` that DEFINES the named value, which is exactly what
"the constants are emitted" means.

Provenance, per constant
------------------------
``c_mood_h_bound``
    Strict ``0``.  The PAD predicate is ``h < 0`` — *detection*, never repair.
    The user law forbids flooring/clipping ``h``, so this bound is exactly
    zero and is NOT tunable; it is emitted so a backend cannot quietly widen
    it to ``-1e-12``.
``c_mood_require_finite``
    ``1`` when the CAD (non-finite) half of the troubled predicate is active.
    Emitted as a value rather than baked in so a backend that genuinely cannot
    test finiteness must turn it OFF explicitly and visibly.
``c_kp_eps``
    The Kurganov–Petrova desingularisation ``eps`` of ``hinv``.  Read from the
    model's REQ-48 ``wet_dry_eps`` parameter when it declares one, else
    :data:`zoomy_core.systemmodel.operations._DEFAULT_WET_DRY_EPS`.  Same
    resolution order ``desingularize_hinv`` itself uses — read, never
    re-decided.
``c_eps_h``
    The EIGENVALUE-SLOT wave-speed floor (REQ-82, user ruling cid=5).
    Recovered BY INSPECTION from ``nsm.eigenvalues``: ``regularize_pow``
    writes ``1/Max(Float(eps_h), h)`` into that slot, so the number is read
    back out of the derived operator rather than restated here.  If the slot
    carries no such floor (no ``h`` state, or no regularisation applied) the
    constant is emitted as ``None`` and no floor comparison is emitted at all.
``c_cfl``
    The CFL SAFETY FACTOR — the user law (0.9), a pure safety factor.  It is
    an input to the emit, not an NSM field, because it is a user ruling; it is
    never adjusted by the march and there is no dt-halving retry.
``c_cfl_dimension``
    The ``d`` of ``CFL*2r/(d*(2k+1)*|lam|)``.  The MESH dimension (user law:
    "pass dimension = the MESH dimension").
``c_cfl_degree_factor``
    ``2k+1`` for reconstruction degree ``k``.
``c_dt_max``
    ``nsm.dt_max`` (REQ-190): the cap every backend must read from the same
    place.  A wave-free domain steps at exactly this value.
``c_write_eps``
    The drift-free write-gate tolerance of ``should_write``
    (``time + c_write_eps >= i_snapshot*write_interval``).
``c_dt_floor``
    DEFAULT OFF (``None``).  When OFF, the ``dt_floor`` build flag is False
    and the emitted march contains NO floor comparison — the honesty guard is
    a FATAL abort on ``dt <= 0``, not a silent floor.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

import sympy as sp

#: Documented fallback for :func:`eigen_wave_speed_floor` when the NSM's
#: eigenvalue slot carries no recoverable floor.  Mirrors the internal value
#: ``zoomy_core.systemmodel.operations.regularize_pow`` writes; it is used
#: ONLY when inspection finds nothing, and the resolved provenance string says
#: so, so a drift between the two is visible rather than silent.
EPS_H_FALLBACK = 1e-14

#: Write-gate tolerance (``state.should_write``).  A pure floating-point
#: comparison guard on the snapshot stamp, not a physical scale.
WRITE_EPS = 1e-14

#: Two-point diffusion fluxes (``DenseDiffusionOperator``, the symbolic
#: solvers' ``diffusion`` block): the face normal's cosine against the unit
#: vector between the two cell centres, |n . e|, is raised to this floor, so a
#: badly non-orthogonal face cannot blow the flux coefficient up.
TPFA_COS_FLOOR = 0.1

# ── elliptic (projection / pressure) solve constants ───────────────────────
#
# Same law as the march constants above: a backend must not carry its own
# float beside the elliptic solve.  Both numbers below were forced by a
# MEASUREMENT on the VAM order-2 Richardson triple (n = 64/128/256):
#
#   * ``pressure_maxit`` counts RESTARTS of a Krylov space of size ``restart``,
#     so raising it does NOTHING once the restarted iteration stagnates — the
#     n=128 elliptic residual was identical to seven digits at 1x / 4x / 16x
#     the iteration budget.
#   * the residual of the FIXED restart=20 space grows under refinement,
#     9.8e-14 -> 1.4e-4 -> 9.6e-3 across the triple, so the pressure error is
#     set by the SOLVER, not by the discretisation.  The observed order of the
#     P rows was negative (P_0 -1.604, P_1 -1.867) while every conservative row
#     converged (h +0.900, q_0 +0.270, q_1 +1.187, r_0 +1.063, r_1 +0.338).
#
# A fixed restart is therefore not a tuning choice but a silent degradation:
# the elliptic operator's condition number is O(1/h^2), and restarted GMRES
# has no convergence theory on a non-symmetric, UNPRECONDITIONED operator of
# growing condition number.  Full (non-restarted) GMRES does: it terminates in
# at most ``n`` steps.  So the default restart SCALES WITH THE PROBLEM — it is
# the full space, capped only by the Krylov-basis memory ceiling.

#: Default relative/absolute tolerance of the elliptic (pressure) solve.
ELLIPTIC_RTOL = 1e-10

#: Floor of the emitted restart.  Below this the Krylov space is too small to
#: carry even a smooth mode; it is a floor, never a default.
ELLIPTIC_RESTART_MIN = 20

#: Krylov-basis memory ceiling, in vectors.  ``restart`` vectors of length
#: ``n`` cost ``restart * n`` doubles, so the cap bounds the basis at
#: ``ELLIPTIC_RESTART_CAP * n``.  Above the cap the solve is restarted again —
#: and a system that large needs a PRECONDITIONER, not a bigger basis; that is
#: a reported limitation, not something the default may hide.
ELLIPTIC_RESTART_CAP = 1000


def elliptic_restart(n_unknowns: int) -> int:
    """The GMRES restart for an elliptic block with ``n_unknowns`` unknowns.

    ``min(n, CAP)`` bounded below by ``MIN``: the FULL Krylov space (no
    restart, hence guaranteed termination in <= n steps) until the basis
    reaches the memory ceiling.  The point is that it MOVES with ``n`` — a
    constant here is the defect this replaces.
    """
    n = int(n_unknowns)
    if n < 1:
        raise ConstantResolutionError(
            f"elliptic block size must be >= 1, got {n_unknowns!r}")
    return max(ELLIPTIC_RESTART_MIN, min(n, ELLIPTIC_RESTART_CAP))


class ConstantResolutionError(ValueError):
    """A march constant could not be resolved unambiguously from the NSM."""


def _depth_symbol(nsm):
    return next((s for s in getattr(nsm, "state", ()) if str(s) == "h"), None)


def eigen_wave_speed_floor(nsm):
    """Recover ``eps_h`` from ``nsm.eigenvalues`` — the honest read.

    ``regularize_pow`` rewrites the eigenvalue slot's ``1/h`` to
    ``1/Max(Float(eps_h), h)``.  Scan the derived operator for exactly that
    shape and return ``(value, provenance)``.  Several DIFFERENT floors in one
    slot is a contradiction and RAISES — there is no "pick the smallest".
    """
    ev = getattr(nsm, "eigenvalues", None)
    h = _depth_symbol(nsm)
    if ev is None or h is None:
        return None, "absent (no eigenvalues slot / no depth state 'h')"
    found = set()
    for e in sp.flatten(ev):
        try:
            e = sp.sympify(e)
        except (sp.SympifyError, TypeError):
            continue
        for m in e.atoms(sp.Max):
            if h not in m.args:
                continue
            for a in m.args:
                if a.is_Number and a.is_positive:
                    found.add(float(a))
    if not found:
        return None, "absent (eigenvalue slot carries no Max(eps, h) floor)"
    if len(found) > 1:
        raise ConstantResolutionError(
            f"nsm.eigenvalues carries {len(found)} different wave-speed "
            f"floors {sorted(found)!r} — the emitted march needs ONE named "
            "eps_h; a slot with contradictory floors is a model defect")
    return found.pop(), "nsm.eigenvalues (regularize_pow Max(eps_h, h))"


def kp_eps(nsm):
    """The KP desingularisation ``eps``, resolved exactly as
    ``desingularize_hinv`` resolves it.  Returns ``(value, provenance)``."""
    params = getattr(nsm, "parameters", None)
    if params is not None and getattr(params, "contains", None) is not None:
        if params.contains("wet_dry_eps"):
            vals = getattr(nsm, "parameter_values", None) or {}
            raw = next((v for k, v in vals.items()
                        if str(k) == "wet_dry_eps"), None)
            if raw is not None:
                return float(raw), "nsm.parameter_values['wet_dry_eps']"
            return None, "nsm.parameters.wet_dry_eps (symbolic, unvalued)"
    from zoomy_core.systemmodel.operations import _DEFAULT_WET_DRY_EPS
    return (float(_DEFAULT_WET_DRY_EPS),
            "zoomy_core.systemmodel.operations._DEFAULT_WET_DRY_EPS")


@dataclasses.dataclass(frozen=True)
class MarchConstants:
    """The emitted named values + where each one came from.

    ``values`` maps EMITTED NAME -> number; ``provenance`` maps the same names
    to a human-readable source string.  A ``None`` value means the constant is
    inactive and the corresponding comparison is NOT emitted.
    """

    values: dict
    provenance: dict

    def __post_init__(self):
        object.__setattr__(self, "values", dict(self.values))
        object.__setattr__(self, "provenance", dict(self.provenance))
        missing = set(self.values) ^ set(self.provenance)
        if missing:
            raise ConstantResolutionError(
                f"every emitted constant needs a provenance string; "
                f"unpaired: {sorted(missing)}")

    def symbol(self, name: str) -> sp.Symbol:
        """The sympy Symbol a comparison site must use for ``name``."""
        if name not in self.values:
            raise ConstantResolutionError(
                f"{name!r} is not an emitted march constant "
                f"(have {sorted(self.values)})")
        if self.values[name] is None:
            raise ConstantResolutionError(
                f"{name!r} resolved to None (inactive: "
                f"{self.provenance[name]}) — its comparison must be guarded "
                "by a build flag, not emitted against a null value")
        return sp.Symbol(name)

    def active(self, name: str) -> bool:
        return self.values.get(name) is not None

    def assigns(self, *names) -> tuple:
        """``Assign`` statements DEFINING the named constants, in order.

        These are the only float literals in the emitted procedure; every
        later use is by name.
        """
        from zoomy_core.solver.ir import Assign
        out = []
        for n in (names or tuple(self.values)):
            v = self.values.get(n)
            if v is None:
                continue
            out.append(Assign(
                n, sp.Float(v) if n in _REAL_VALUED else sp.Integer(int(v)),
                ctype="" if n in _REAL_VALUED else "int"))
        return tuple(out)

    def report(self) -> str:
        """One line per emitted constant: name, value, provenance."""
        rows = []
        for n in sorted(self.values):
            v = self.values[n]
            rows.append(f"{n:24s} = {v!r:<12} <- {self.provenance[n]}")
        return "\n".join(rows)


#: Constants that stay REAL even at an integral value (``0.0`` must not be
#: emitted as the integer ``0`` where it is compared against a float depth).
_REAL_VALUED = frozenset({
    "c_mood_h_bound", "c_kp_eps", "c_eps_h", "c_cfl", "c_dt_max",
    "c_write_eps", "c_dt_floor", "c_elliptic_tol",
})


def march_constants(nsm, *, cfl: float, dimension: int, degree: int = 0,
                    dt_floor: Optional[float] = None,
                    require_finite: bool = True,
                    dt_max: Any = None) -> MarchConstants:
    """Resolve every emitted march constant from ``nsm`` + the user laws.

    ``cfl`` is the user's safety factor (law: 0.9) and ``dimension`` the MESH
    dimension — both are rulings, not NSM fields, so they are inputs.
    ``dt_floor`` defaults to ``None`` = OFF: there is no dt floor in the
    sanctioned march, only the FATAL guard.
    """
    if not (0.0 < float(cfl) <= 1.0):
        raise ConstantResolutionError(
            f"CFL safety factor {cfl!r} is out of (0, 1] — it is a pure "
            "safety factor and the march never adjusts it")
    if int(dimension) < 1:
        raise ConstantResolutionError(
            f"CFL dimension factor must be >= 1, got {dimension!r}")

    eps_h, eps_h_src = eigen_wave_speed_floor(nsm)
    if eps_h is None:
        eps_h, eps_h_src = EPS_H_FALLBACK, (
            f"{eps_h_src}; fell back to constants.EPS_H_FALLBACK")
    kp, kp_src = kp_eps(nsm)
    dtm = getattr(nsm, "dt_max", None) if dt_max is None else dt_max

    values = {
        "c_mood_h_bound": 0.0,
        "c_mood_require_finite": 1 if require_finite else 0,
        "c_kp_eps": kp,
        "c_eps_h": eps_h,
        "c_cfl": float(cfl),
        "c_cfl_dimension": int(dimension),
        "c_cfl_degree_factor": int(2 * int(degree) + 1),
        "c_dt_max": None if dtm is None else float(dtm),
        "c_write_eps": WRITE_EPS,
        "c_dt_floor": None if dt_floor is None else float(dt_floor),
    }
    provenance = {
        "c_mood_h_bound": "user law: PAD is h < 0, never floored/clipped",
        "c_mood_require_finite": "CAD half of the troubled predicate",
        "c_kp_eps": kp_src,
        "c_eps_h": eps_h_src,
        "c_cfl": "user law (pure safety factor, passed in, never adjusted)",
        "c_cfl_dimension": "MESH dimension (user law)",
        "c_cfl_degree_factor": f"2k+1 for reconstruction degree k={int(degree)}",
        "c_dt_max": "nsm.dt_max (REQ-190)" if dt_max is None else "caller override",
        "c_write_eps": "constants.WRITE_EPS (drift-free write gate)",
        "c_dt_floor": ("OFF (user law: no dt floor, the guard is FATAL)"
                       if dt_floor is None else "caller override"),
    }
    return MarchConstants(values=values, provenance=provenance)


def elliptic_constants(*, n_unknowns: int,
                       tol: Optional[float] = None,
                       restart: Optional[int] = None) -> MarchConstants:
    """Resolve the EMITTED constants of an elliptic (projection) solve.

    ``n_unknowns`` is the size of the elliptic block — for the VAM Chorin
    pressure stage, ``n_pressure_modes * n_cells``.  It is REQUIRED, because
    the whole point of this resolver is that the restart is a function of the
    problem size (see :func:`elliptic_restart`); a caller that cannot say how
    big its block is cannot get a default.

    ``tol`` / ``restart`` are caller overrides and are reported as such in the
    provenance, so an override is visible rather than indistinguishable from
    the resolved default.
    """
    if tol is not None and not (float(tol) > 0.0):
        raise ConstantResolutionError(
            f"elliptic tolerance must be > 0, got {tol!r}")
    if restart is not None and int(restart) < 1:
        raise ConstantResolutionError(
            f"elliptic restart must be >= 1, got {restart!r}")

    resolved_restart = (elliptic_restart(n_unknowns) if restart is None
                        else min(int(restart), int(n_unknowns)))
    values = {
        "c_elliptic_tol": float(ELLIPTIC_RTOL if tol is None else tol),
        "c_elliptic_restart": int(resolved_restart),
    }
    provenance = {
        "c_elliptic_tol": (
            "constants.ELLIPTIC_RTOL"
            if tol is None or float(tol) == ELLIPTIC_RTOL
            else "caller override"),
        "c_elliptic_restart": (
            f"constants.elliptic_restart(n={int(n_unknowns)}) "
            f"= max({ELLIPTIC_RESTART_MIN}, min(n, {ELLIPTIC_RESTART_CAP}))"
            if restart is None else
            f"caller override, clamped to n={int(n_unknowns)}"),
    }
    return MarchConstants(values=values, provenance=provenance)
