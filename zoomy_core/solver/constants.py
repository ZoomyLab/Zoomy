"""Named values the emitted march and the elliptic solve read instead of hard-coded
numbers; each is resolved from the NSM and carries a provenance string. A backend
never holds its own float beside them.
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
    """GMRES restart for an elliptic block of ``n_unknowns``: the full Krylov space
    ``min(n, CAP)``, bounded below by ``MIN``, so it moves with the problem size.

    References:
        Saad, Schultz 1986, https://doi.org/10.1137/0907058
    """
    n = int(n_unknowns)
    if n < 1:
        raise ConstantResolutionError(
            f"elliptic block size must be >= 1, got {n_unknowns!r}")
    return max(ELLIPTIC_RESTART_MIN, min(n, ELLIPTIC_RESTART_CAP))


class ConstantResolutionError(ValueError):
    """A march constant could not be resolved unambiguously from the NSM."""


def _depth_symbol(nsm):
    """The depth state symbol ``h`` of ``nsm``, or None."""
    return next((s for s in getattr(nsm, "state", ()) if str(s) == "h"), None)


def eigen_wave_speed_floor(nsm):
    """Read ``eps_h`` back from the ``Max(eps_h, h)`` floor that ``regularize_pow``
    wrote into ``nsm.eigenvalues``; returns ``(value, provenance)`` and raises on
    several different floors.
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
    """The Kurganov-Petrova desingularization ``eps``, resolved as
    ``desingularize_hinv`` resolves it; returns ``(value, provenance)``.

    References:
        Kurganov, Petrova 2007, https://doi.org/10.4310/CMS.2007.v5.n1.a6
    """
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
    """The emitted named values and where each came from: ``values`` maps name to number
    (``None``: inactive, its comparison is not emitted) and ``provenance`` maps the
    same names to a source string.
    """

    values: dict
    provenance: dict

    def __post_init__(self):
        """Freeze copies of both dicts and require a provenance for every value."""
        object.__setattr__(self, "values", dict(self.values))
        object.__setattr__(self, "provenance", dict(self.provenance))
        missing = set(self.values) ^ set(self.provenance)
        if missing:
            raise ConstantResolutionError(
                f"every emitted constant needs a provenance string; "
                f"unpaired: {sorted(missing)}")

    def symbol(self, name: str) -> sp.Symbol:
        """The sympy symbol a comparison site uses for ``name``; raises if the name is
        unknown or inactive.
        """
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
        """Whether ``name`` has a value."""
        return self.values.get(name) is not None

    def assigns(self, *names) -> tuple:
        """``Assign`` statements defining the named constants, in order; the only float
        literals in the emitted procedure.
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
        """One line per constant: name, value, provenance."""
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
    """Resolve every emitted march constant from ``nsm`` and the caller's settings.

    Args:
        cfl: safety factor in (0, 1], never adjusted by the march.
        dimension: mesh dimension of the CFL step.
        degree: reconstruction degree ``k`` of the factor ``2k + 1``.
        dt_floor: ``None``: off (the guard is a fatal abort on ``dt <= 0``, not a
            floor).
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
    """Resolve the emitted constants of an elliptic (projection) solve; the restart
    scales with the block size ``n_unknowns``, so it is required.

    Args:
        tol: caller override, reported as such in the provenance.
        restart: caller override, clamped to ``n_unknowns``.
    References:
        Saad, Schultz 1986, https://doi.org/10.1137/0907058
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
