"""``ClosureState``, the single full-access state object ``s`` handed to every composable stress
closure ``f(s)`` (see ``closures.py``).
It gives the closure sigma-aware derivatives, model parameters and the fields at the bulk or at
a boundary trace.
"""
from __future__ import annotations

import sympy as sp


class ClosureState:
    """Full-access state ``s`` for a closure ``f(s)``: fields by name (bulk expression or trace
    at an interface), sigma-aware ``dz``/``dx``, and parameters ``par``.

    Built from ``m.functions`` or an explicit ``{name: FunctionFamily}`` map; used by
    ``Closure`` subclasses during the Model -> SystemModel lift.
    """

    def __init__(self, fields, *, params=None, h=None, x=None, zeta=None,
                 at=None, alias=None, boundary_tag=None, horiz=None):
        """Store the field map, parameters, sigma frame and (optionally) the boundary frame.

        Args:
            fields: ``m.functions`` namespace or ``{name: FunctionFamily}`` map.
            at: ``None`` for the bulk, ``0`` for the bed trace, ``1`` for the surface trace.
            alias: field-name rebind (e.g. ``{"u": "v"}``) so one closure serves every
                horizontal row.
            boundary_tag: ``"b"`` or ``"eta"``, selects the opaque slope symbols of the boundary
                frame.
            horiz: horizontal coordinates of the model.
        """
        # ``object.__setattr__`` so __getattr__ does not recurse on these.
        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "_params", params)
        object.__setattr__(self, "_h", h)
        object.__setattr__(self, "_x", x)
        object.__setattr__(self, "_zeta", zeta)
        object.__setattr__(self, "_at", at)
        # Boundary frame: ``boundary_tag`` ∈ {"b","eta"} identifies which
        # interface this closure sits on (bed / free surface) and ``horiz`` the
        # horizontal coords — together they pin the OPAQUE slope symbols the
        # local frame {n, t_α} is built from (see ``s.normal`` / ``s.tangents``).
        object.__setattr__(self, "_btag", boundary_tag)
        object.__setattr__(self, "_horiz", list(horiz) if horiz else [])
        # Field-NAME aliasing for dimension-agnostic closures: e.g. when
        # closing the y-momentum row, ``alias={"u": "v"}`` makes the generic
        # ``s.u`` a closure writes resolve to the y-velocity ``v`` — so a single
        # direction-agnostic closure (``ρν ∂_z u``) closes every horizontal
        # component.  1-D / x-row pass ``{"u": "u"}`` (a no-op).
        object.__setattr__(self, "_alias", alias or {})

    @property
    def par(self):
        """Model parameter namespace (``s.par.rho``, ...)."""
        return self._params

    @property
    def zeta(self):
        """Sigma coordinate zeta in [0, 1], for position-dependent closures."""
        return self._zeta

    @property
    def depth(self):
        """Layer depth h (the sigma-map scale)."""
        return self._h

    def dz(self, e):
        """Physical vertical derivative ``d_z e = (1/h) d_zeta e`` of the expression ``e``."""
        return sp.Derivative(e, self._zeta) / self._h

    def dx(self, e):
        """Horizontal derivative ``d_x e`` of ``e`` at fixed zeta."""
        return sp.Derivative(e, self._x)

    # ── opaque boundary frame {n, t_α} (projected-traction closures) ──────────
    @property
    def _slopes(self):
        """Opaque geometric slopes of this boundary's interface, one per horizontal direction."""
        from zoomy_core.model.models.equations import frame_slope
        from zoomy_core import coords as _C
        cn = {_C.x: "x", _C.y: "y"}
        return [frame_slope(self._btag, cn[c]) for c in self._horiz]

    @property
    def normal(self):
        """Outward unit boundary normal ``[n_x, (n_y,) n_z]`` in the opaque slopes; reduces to
        ``z_hat`` under ``small_slope_scaling``.
        """
        sl = self._slopes
        comps = [-s for s in sl] + [sp.S.One]
        norm = sp.sqrt(sum(c ** 2 for c in comps))
        return sp.Matrix([c / norm for c in comps])

    @property
    def tangents(self):
        """Axis-fixed unit tangent basis ``[t_x, (t_y,)]`` of the boundary,
        ``t_a = (e_a + sigma_a z_hat)/sqrt(1 + sigma_a^2)``.
        """
        sl = self._slopes
        out = []
        for a, s in enumerate(sl):
            vec = [sp.S.Zero] * len(sl) + [s]
            vec[a] = sp.S.One
            out.append(sp.Matrix(vec) / sp.sqrt(1 + s ** 2))
        return out

    @property
    def _vel3(self):
        """Full velocity vector ``[u, (v,) w]`` at this boundary trace."""
        from zoomy_core import coords as _C
        names = ([{_C.x: "u", _C.y: "v"}[c] for c in self._horiz] + ["w"])
        return [getattr(self, n) for n in names]

    def get_normal_tangential(self, vec):
        """Project a 3-D vector onto the boundary frame ``{n, t_a}``.

        Returns:
            ``(f_n, [f_t1, (f_t2)])``, the normal and per-axis tangential components.
        """
        v = sp.Matrix(vec)
        f_n = (v.T * self.normal)[0]
        f_t = [(v.T * T)[0] for T in self.tangents]
        return f_n, f_t

    @property
    def u_tangent(self):
        """Tangential velocity per axis, ``[u.t_x, (u.t_y,)]``."""
        return self.get_normal_tangential(self._vel3)[1]

    def velocity_at(self, zeta_value):
        """Full velocity vector ``[u, (v,) w]`` evaluated at sigma-height ``zeta_value`` instead
        of the boundary trace (for wall functions reading ``u(zeta_p)``).
        """
        from zoomy_core import coords as _C
        names = [{_C.x: "u", _C.y: "v"}[c] for c in self._horiz] + ["w"]
        return [self._family(n).at(zeta_value) for n in names]

    def u_tangent_at(self, zeta_value):
        """Tangential velocity per axis with the velocity read at sigma-height ``zeta_value``."""
        return self.get_normal_tangential(self.velocity_at(zeta_value))[1]

    @property
    def u_normal(self):
        """Wall-normal velocity ``u.n``."""
        return self.get_normal_tangential(self._vel3)[0]

    def _family(self, name):
        """Resolve a field name (through the alias map) to its ``FunctionFamily``."""
        name = self._alias.get(name, name)        # dimension-agnostic rebind
        f = self._fields
        if isinstance(f, dict):
            if name not in f:
                raise AttributeError(
                    f"closure asked for field {name!r}; this model exposes "
                    f"{sorted(f)} to the closure")
            return f[name]
        return getattr(f, name)

    def __getattr__(self, name):
        """Return the bulk expression (``at=None``) or the trace of the field ``name``."""
        if name.startswith("_"):
            raise AttributeError(name)
        fam = self._family(name)
        return fam.expr if self._at is None else fam.at(self._at)


# MaterialModel and its factories (newtonian_navier_slip, bingham_navier_slip,
# rough_wall, kepsilon_eddy_viscosity) were REMOVED in the clean cut to the
# composable closure list — there is no legacy stress-closure path.  Their
# physics now lives as composable Closure pieces in closures.py:
#   newtonian_navier_slip  →  [Newtonian(), NavierSlip(), StressFree()]
#   bingham_navier_slip    →  [Bingham(), NavierSlip(), StressFree()]  (+ quadrature_order>0)
#   rough_wall             →  [RoughWall()]
#   kepsilon_eddy_viscosity→  [KEpsilonViscosity()]   (+ quadrature_order>0)
# Only ClosureState (above) remains here — the full-access state object f(s).
