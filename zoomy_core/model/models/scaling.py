"""The free-surface scaling of the thesis (`20_methodology.md`, section Dimensional analysis) on
the balance-law blueprints, and the ordered truncation the moment models run instead of their
hand rules when they are given a `ShallowScaling` (`scaling=` setting).

    eq. (scaling-variables)   x, y ~ L;  z, b, h ~ H;  t ~ L / U;  u, v ~ U;  w ~ eps U
    eq. (pressure-scaling)    p = rho g (s - z) + q,  q ~ rho U^2
    eq. (sme-stress-scaling)  tau_xz, tau_yz ~ S;  tau_xx, tau_xy, tau_yy, tau_zz ~ eps S
    eq. (fr-de)               eps = H / L,  Fr = U / sqrt(g H),  De = eps rho U^2 / S

`De` is written through the stress scale `S` (`1/De = S / (eps rho U^2)`, the coefficient of
`d_z tau_hz` in the horizontal row divided by `rho U^2 / L`); the closure fixes `S` (Newtonian
`S = rho nu U / H` gives `De = eps U H / nu`, eq. (fr-de)).
"""
from __future__ import annotations

import sympy as sp

from zoomy_core import coords as C
from zoomy_core.misc.options import Option
from zoomy_core.model.derivation.nondim import (
    NonDimensionalize, ShallowScaling, InsertClosure, Dimensionalize, Affine, Role,
    LENGTH, TIME, VELOCITY, STRESS, ScalingError)

#: reference scales and groups of the free-surface scaling
L, H, U, S = sp.symbols("L H U S", positive=True)
EPS, FR, DE = sp.symbols("epsilon Fr De", positive=True)


class Pressure(Option):
    """What the blueprint's pressure field is."""
    #: the total pressure `p` (`Momentum`): scaled about the hydrostatic part, `p = rho g (s - z)
    #: + q`
    total = "total"
    #: already the non-hydrostatic part (`MomentumNonHydrostatic`'s `p`, ML-SME's `Pi`): `~ rho U^2`
    nonhydrostatic = "nonhydrostatic"


class FreeSurfaceNonDim(NonDimensionalize):
    """`NonDimensionalize` with the thesis choices for a free-surface flow, read off a momentum
    blueprint (`Momentum`, `MomentumNonHydrostatic` or a stand-in with `uvel, w, p, tau, rows,
    vertical`).

    Rows: the horizontal momentum rows divided by `U^2 / L` (inertia), the vertical row by
    `U^2 / H` (the non-hydrostatic pressure gradient), the mass row by `U / L`.  Default regime:
    `Fr, De = O(1)`, the incline `e_x = O(eps)` (its body force then of the order of the surface
    gradient `g d_x eta`).

    Args:
        momentum: the blueprint.
        heights: the height fields (`h`, `b`, layer depths), scaled with `H`.
        free_surface: `s` of the hydrostatic part (needed for `Pressure.total`).
        pressure: the `Pressure` kind of `momentum.p`.
        mass: name of the mass row to scale with it (optional).
    """

    def __init__(self, momentum, *, heights, free_surface=None, pressure=Pressure.total,
                 mass=None):
        """Store the blueprint; the scales are built against the model (`apply_to_model`)."""
        super().__init__({}, {}, {}, {}, {}, EPS, name="free_surface_nondim")
        self.momentum, self.heights, self.free_surface = momentum, list(heights), free_surface
        self.pressure, self.mass = pressure, mass

    def _spec(self, m):
        """Scales, dimensions, groups and rows of the blueprint for the model `m`."""
        mom = self.momentum
        coords = list(mom.uvel[0].args)
        t, z = coords[0], coords[-1]
        horiz = coords[1:-1]
        rho, g = m.parameters.rho, m.parameters.g
        scales, dims = {t: L / U, z: H}, {t: TIME, z: LENGTH}
        for xd in horiz:
            scales[xd], dims[xd] = L, LENGTH
        for u in mom.uvel:
            scales[u], dims[u] = U, VELOCITY
        scales[mom.w], dims[mom.w] = EPS * U, VELOCITY
        for hh in self.heights:
            scales[hh], dims[hh] = H, LENGTH
        if self.pressure == Pressure.total:
            if self.free_surface is None:
                raise ScalingError("Pressure.total needs the free surface s")
            scales[mom.p] = Affine(rho * g * (self.free_surface - z), rho * U ** 2, name="q")
        else:
            scales[mom.p] = rho * U ** 2
        dims[mom.p] = STRESS
        for key, tau in mom.tau.items():
            shear = len(key) == 2 and key[1] == "z" and key[0] != "z"
            scales[tau], dims[tau] = (S if shear else EPS * S), STRESS
        groups = {EPS: H / L, FR: U / sp.sqrt(g * H), DE: EPS * rho * U ** 2 / S}
        rows = {eq.name: (Role.horizontal, U ** 2 / L) for eq in mom.rows}
        if getattr(mom, "vertical", None) is not None:
            rows[mom.vertical.name] = (Role.vertical, U ** 2 / H)
        if self.mass is not None:
            rows[self.mass] = (Role.mass, U / L)
        orders = {"Fr": 0, "De": 0, "e_x": 1}
        if "e_x" in m.parameters.keys():
            e_x = m.parameters.e_x
            if any(m._equations[n].expr.has(e_x) for n in rows):
                groups[e_x] = e_x                  # a dimensionless parameter is its own group
        self.scales, self.groups, self.dimensions, self.rows = scales, groups, dims, rows
        self.references = {L: LENGTH, H: LENGTH, U: VELOCITY, S: STRESS}
        self.orders, self.prefer = orders, (L, H, U, S)
        return horiz, z

    def apply_to_model(self, m):
        """Scale the rows and record the blueprint fields `InsertClosure` reads."""
        horiz, z = self._spec(m)
        super().apply_to_model(m)
        mom = self.momentum
        depth = next((hh for hh in self.heights if hh.func.__name__.startswith("h")),
                     self.heights[0])
        base = next((hh for hh in self.heights if hh.func.__name__ == "b"), None)
        m.nondim.blueprint = {"uvel": list(mom.uvel), "w": mom.w, "horiz": tuple(horiz),
                              "z": z, "stress": dict(mom.tau), "depth": depth, "base": base,
                              "stress_scale": S}
        return m


def ordered_truncation(m, momentum, closures, scaling, *, heights, free_surface=None,
                       pressure=Pressure.total, mass=None, requested=None):
    """The scaling pipeline on a model's blueprint rows, in place of the hand rules
    (`moment_scaling`, the hydrostatic inertia drop, `apply_full_tensor`):

        FreeSurfaceNonDim -> ShallowScaling -> InsertClosure -> ShallowScaling -> Dimensionalize

    Sets `m.stress` from the outcome (`Stress.full` when a horizontal row keeps the in-plane
    stress, which the closure step and the in-plane traction read) and returns the
    `ClosureRecord`.

    Args:
        requested: the model's `stress` setting; `Stress.full` there must agree with the
            outcome.
    Raises:
        ScalingError: the kept bulk shear is neither the vertical shear alone nor the whole
            `tau_dz` with the in-plane stress (no `Stress` option represents it); a closure
            contributes nothing at the order (remove it); `requested` disagrees.
    """
    from zoomy_core.model.models.equations import Stress
    m.apply(FreeSurfaceNonDim(momentum, heights=heights, free_surface=free_surface,
                              pressure=pressure, mass=mass))
    m.apply(ShallowScaling(horizontal=scaling.horizontal, vertical=scaling.vertical,
                           regime=scaling.regime))
    m.apply(InsertClosure(closures))
    m.apply(ShallowScaling())
    st = m.nondim
    m.apply(Dimensionalize())
    rec = st.closure
    bulk = [c for c in closures if getattr(c, "closes", None) == "bulk"]
    full_shear = bool(rec.shear) and all(
        sp.expand(rec.shear[k] - rec.tensor[k]) == 0 for k in rec.shear)
    m.stress = Stress.full if rec.inplane else Stress.shallow
    if bulk and rec.shear and full_shear != rec.inplane:
        raise ScalingError(
            "the ordered truncation keeps "
            + ("the in-plane stress but not the d_d w part of tau_dz" if rec.inplane
               else "the d_d w part of tau_dz but no in-plane stress")
            + f" ({rec.shear}); no Stress option represents that")
    if rec.dropped:
        raise ScalingError(f"{scaling!r}: the closures {rec.dropped} contribute only beyond the "
                           "order; remove them from the closure list")
    if requested is not None and Stress(requested) == Stress.full and m.stress != Stress.full:
        raise ScalingError(f"stress=Stress.full, but {scaling!r} drops the in-plane stress")
    m._history("ordered_truncation", "momentum", description=(
        f"{scaling!r}: stress {m.stress}, vertical stress kept: {rec.vertical_stress}; "
        + " | ".join(f"{n}: {d}" for n, d in st.log)))
    return rec


__all__ = ["Pressure", "FreeSurfaceNonDim", "ordered_truncation", "L", "H", "U", "S", "EPS",
           "FR", "DE"]
