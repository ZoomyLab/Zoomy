"""Reusable `SystemModel.apply` operations (`op(sm)` callables) that rewrite operator slots:
aux registration, 1/h regularization, eigenvalue guards.
They run on the way from `SystemModel` to `NumericalSystemModel`; sweeping operations cover
every slot of `OPERATOR_SLOTS` unless a slot is explicitly excluded.
"""
from __future__ import annotations

from typing import Union

import sympy as sp

from zoomy_core.misc.misc import ZArray

__all__ = [
    "OPERATOR_SLOTS",
    "map_operator_slots",
    "normalize_face_normal",
    "register_aux",
    "regularize_pow",
    "regularize_depth_direct",
    "regularize_depth_aux",
    "kp_hinv",
    "desingularize_hinv",
    "symbolic_spectrum",
    "desingularize_positivity",
    "equilibrate_constraint_rows",
    "guard_eigenvalue_powers",
    "gate_eigenvalues_dry",
    "regularize_fractional_powers",
]


# Default wet/dry threshold when the model declares no ``wet_dry_eps``
# parameter (REQ-48).  Small enough that the desingularized ``hinv`` matches
# ``1/h`` everywhere a model without a wetting/drying threshold would actually
# run, but non-zero so the desingularized inverse depth (and the dry-cell
# eigenvalue gate) never divide by / compare against an exactly-dry cell.
_DEFAULT_WET_DRY_EPS = 1e-8


# Plain floating-point floor used for the ``1/h`` of the EIGENVALUE / wave-speed
# velocity (REQ-82).  The wave speed must NOT use the KP-desingularized ``hinv``:
# when ``h* < eps`` the ``max(h*, eps)⁴`` term in ``hinv`` suppresses the
# advective (and gravity) speed by ``(h*/eps)²``, starving the Rusanov
# dissipation at a steep wet/dry front so the centred Audusse mass flux drains a
# dry cell (``h<0``).  Instead the eigenvalue velocity uses the same classical
# floor ``max(1e-14, h)`` the flux velocity already uses, so ``λ = |u·n| +
# √(g·h*)·|n|`` with the TRUE velocity ``u = q/max(1e-14, h)``.


# REGULARIZATION epsilon (REQ-194) — a DIFFERENT QUANTITY at a DIFFERENT SCALE
# from the wet/dry threshold above, and deliberately its OWN knob.  It is the
# depth below which the desingularized reciprocal is allowed to depart from the
# exact ``1/h``.  It must never be conflated with ``wet_dry_eps``: borrowing the
# wet/dry threshold as a regularization scale is what put ``h/(h + 1e-2)`` on
# the celerity in the first attempt at this REQ.
#
# The default is the user's instruction for the parameter study — start near the
# old working state (``wet_dry_eps = 1e-2`` for SWE / Malpasset) so the first
# sweep point reproduces roughly what shipped, then vary it.  It is a STUDY
# default on the new study ops only; the legacy ``desingularize_hinv`` path
# still reads ``wet_dry_eps`` and is untouched.
_DEFAULT_REGULARIZATION_EPS = 1e-2


# Transcendental nodes whose ARGUMENT must never be touched by the
# ``regularize_pow`` ``1/h -> hinv`` sweep — an ``h`` inside ``log(z/h)`` is
# not a multiplicative factor of a denominator and must stay ``h``.
_TRANSCENDENTAL = (
    sp.log, sp.exp,
    sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc,
    sp.asin, sp.acos, sp.atan, sp.acot, sp.atan2,
    sp.sinh, sp.cosh, sp.tanh, sp.coth,
    sp.asinh, sp.acosh, sp.atanh,
    sp.erf, sp.erfc,
)


# ── The operator-slot registry ─────────────────────────────────────────────
#
# EVERY symbolic slot an NSM operation may rewrite, in lowering order.  This is
# the DEFAULT scope of the swept operations (see the module docstring).
#
# NOT in the registry, and why:
#   * ``initial_conditions`` / ``aux_initial_conditions`` — initial DATA
#     (``Constant`` objects / user callables), not operators.  There is no
#     symbolic tree to rewrite, and rewriting initial data would change the
#     problem rather than its discretisation.
#   * ``state_update`` — a scheme NAME (``str``), not an expression.
#   * ``position`` / ``normal`` / ``parameters`` / ``state`` — symbol
#     containers, not operators.
OPERATOR_SLOTS = (
    # primaries (the PDE operators)
    "flux",
    "hydrostatic_pressure",
    "nonconservative_matrix",
    "source",
    "source_explicit",
    "mass_matrix",
    "diffusion_matrix",
    "diffusion_matrix_explicit",
    # derived operators (lazily materialized; see ``map_operator_slots``)
    "quasilinear_matrix",
    "eigenvalues",
    "eigenvalues_cfl",
    "source_jacobian_wrt_variables",
    "source_jacobian_wrt_aux_variables",
    # per-cell maps
    "update_variables",
    "update_aux_variables",
    # well-balanced reconstruction pair
    "reconstruction_variables",
    "state_from_reconstruction",
    # 3-D lift / projection pair
    "interpolate_to_3d",
    "project_from_3d",
    # boundary kernels (``basefunction.Function``; definition may be Piecewise)
    "boundary_conditions",
    "aux_boundary_conditions",
    "boundary_gradients",
    # admissibility list (expressions that must stay >= 0)
    "admissible",
)


#: Slots of :data:`OPERATOR_SLOTS` that carry a ``Function`` KERNEL — a
#: ``(definition, args)`` pair whose ``args`` Zstruct is the lowering signature.
KERNEL_SLOTS = (
    "boundary_conditions",
    "aux_boundary_conditions",
    "boundary_gradients",
)


def _is_kernel(obj) -> bool:
    """True for a `basefunction.Function` kernel (a named `(definition, args)` pair) rather
    than a bare operator array.
    """
    return (hasattr(obj, "definition") and hasattr(obj, "args")
            and not isinstance(obj, sp.Basic))


def _sync_kernel_aux_signature(sm) -> None:
    """Re-bind the `aux_variables` group of every boundary-kernel signature to the current
    `sm.aux_state`, so kernels see aux added after lowering.
    """
    from zoomy_core.misc.misc import Zstruct
    for slot in KERNEL_SLOTS:
        kernel = getattr(sm, slot, None)
        if kernel is None:
            continue
        args = getattr(kernel, "args", None)
        if (args is None or not hasattr(args, "contains")
                or not args.contains("aux_variables")):
            continue
        aux_z = Zstruct()
        for s in sm.aux_state:
            setattr(aux_z, str(s), s)
        aux_z._symbolic_name = getattr(
            args.aux_variables, "_symbolic_name", None)
        new_args = Zstruct()
        for key in args.keys():
            setattr(new_args, key,
                    aux_z if key == "aux_variables" else getattr(args, key))
        new_args._symbolic_name = getattr(args, "_symbolic_name", None)
        setattr(sm, slot, type(kernel)(
            name=kernel.name, definition=kernel.definition, args=new_args))


def _rebuild_like(M, flat):
    """Rebuild the array `M` from its flat scalar entries."""
    return type(M)(flat).reshape(*M.shape)


def _row_of(shape, idx: int) -> int:
    """Row index of flat position `idx` in a row-major array of `shape`."""
    stride = 1
    for s in shape[1:]:
        stride *= int(s)
    return idx // max(stride, 1)


def _rewrite_container(obj, rewrite, skip_rows=()):
    """Apply `rewrite` to every scalar of an operator container (array, list/tuple or bare
    expression), leaving `skip_rows` untouched.

    Returns:
        `(changed, new_obj)`.
    """
    if obj is None:
        return False, obj
    if isinstance(obj, (sp.NDimArray, sp.MatrixBase)):
        flat = list(sp.flatten(obj))
        skip = set(skip_rows)
        out = [
            e if _row_of(obj.shape, i) in skip else rewrite(sp.sympify(e))
            for i, e in enumerate(flat)
        ]
        changed = any(a != b for a, b in zip(flat, out))
        return changed, (_rebuild_like(obj, out) if changed else obj)
    if isinstance(obj, (list, tuple)):
        flat = list(sp.flatten(obj))
        out = [rewrite(sp.sympify(e)) for e in flat]
        changed = any(a != b for a, b in zip(flat, out))
        return changed, (type(obj)(out) if changed else obj)
    if isinstance(obj, sp.Basic):
        out = rewrite(obj)
        return out != obj, out
    return False, obj


def map_operator_slots(sm, rewrite, *, exclude=(), skip_rows=None) -> dict:
    """Apply `rewrite(expr) -> expr` to every slot of `OPERATOR_SLOTS` except `exclude`,
    freezing the lazy `quasilinear_matrix` from the clean primaries first.

    Args:
        exclude: slot names to leave alone; a name outside `OPERATOR_SLOTS` raises `KeyError`.
        skip_rows: `{slot: rows}` left untouched inside a swept slot.

    Returns:
        `{slot: "rewritten" | "unchanged" | "absent" | "excluded"}`.
    """
    exclude = tuple(exclude)
    unknown = [s for s in exclude if s not in OPERATOR_SLOTS]
    if unknown:
        raise KeyError(
            f"map_operator_slots: unknown slot(s) in exclude={exclude}: "
            f"{unknown}.  Every exclusion must name a member of "
            f"OPERATOR_SLOTS ({list(OPERATOR_SLOTS)}).")
    excluded = set(exclude)
    skip_rows = skip_rows or {}

    # Freeze the lazy derived operator from the clean primaries first.
    _ = sm.quasilinear_matrix

    report = {}
    for slot in OPERATOR_SLOTS:
        if slot in excluded:
            report[slot] = "excluded"
            continue
        obj = getattr(sm, slot, None)
        if obj is None:
            report[slot] = "absent"
            continue
        rows = skip_rows.get(slot, ())
        if _is_kernel(obj):
            changed, new_def = _rewrite_container(obj.definition, rewrite, rows)
            if changed:
                # Build a NEW kernel: the ``Function`` object is SHARED with the
                # source Model (``sm.boundary_conditions is
                # model._boundary_conditions``), so mutating it in place would
                # rewrite the upstream (possibly cached) Model too.
                setattr(sm, slot, type(obj)(
                    name=obj.name, definition=new_def, args=obj.args))
        else:
            changed, new_obj = _rewrite_container(obj, rewrite, rows)
            if changed:
                setattr(sm, slot, new_obj)
        report[slot] = "rewritten" if changed else "unchanged"
    return report


# ── unit face normal ───────────────────────────────────────────────────────

def _unit_normal_rewrite(normal_syms):
    """Return `rewrite(expr)` imposing `|n| = 1` on the face normal (`(sum n_d^2)^k -> 1`;
    in 1-D `n0^k -> n0^(k mod 2)` and `|n0| -> 1`).
    """
    n = list(normal_syms)
    nsq = sp.Add(*[nd ** 2 for nd in n])
    one_d = len(n) == 1

    def _is_unit_pow(x):
        """Whether `x` is a power of `|n|^2` (or of `n0` in 1-D) that collapses under ``|n|
        = 1``.
        """
        if not isinstance(x, sp.Pow):
            return False
        if not one_d and x.base == nsq:
            return True
        return (one_d and x.base == n[0]
                and x.exp.is_Integer and int(x.exp) not in (0, 1))

    def _unit_pow_value(x):
        """Value of such a power under `|n| = 1`."""
        if one_d and x.base == n[0]:
            return n[0] if int(x.exp) % 2 else sp.S.One
        return sp.S.One

    def _is_unit_abs(x):
        """Whether `x` is `|n0|` in 1-D."""
        return one_d and isinstance(x, sp.Abs) and x.args[0] == n[0]

    def _rewrite(e):
        """Impose `|n| = 1` on one expression and expand if it changed."""
        e = sp.sympify(e)
        if not e.has(*n):
            return e
        out = e.replace(_is_unit_pow, _unit_pow_value)
        if one_d:
            out = out.replace(_is_unit_abs, lambda x: sp.S.One)
        else:
            out = out.xreplace({nsq: sp.S.One})
        return sp.expand(out) if out != e else e

    return _rewrite


def normalize_face_normal(*, exclude=()):
    """Operation: impose `|n| = 1` on every slot (default scope `OPERATOR_SLOTS`), which
    separates the SWE celerity from the `1/h` of the momentum terms.
    It is a `NumericalSystemModel` default (`normalize_normal=True`).
    """
    def _op(sm):
        """Apply the operation to `sm` in place."""
        n = list(sm.normal.values()) if sm.normal is not None else []
        if not n:
            return
        _op.coverage = map_operator_slots(
            sm, _unit_normal_rewrite(n), exclude=exclude)

    _op.name = "normalize_face_normal"
    _op.description = "impose |n| = 1 on the face normal"
    _op.coverage = {}
    return _op


# ── KP-desingularized inverse depth ────────────────────────────────────────

def kp_hinv(h, eps):
    """Kurganov-Petrova desingularized inverse depth `sqrt(2) h / sqrt(h^4 + max(h, eps)^4)`,
    equal to `1/h` for `h >= eps` and tending to 0 as `h -> 0`.
    The numerator is bare `h` (no clamp), so a negative depth gives a negative `hinv` and the
    failure stays visible.

    Args:
        h: depth, sympy expression or numpy/jax array.
        eps: regularization scale.

    References:
        Kurganov, Petrova 2007, https://doi.org/10.4310/CMS.2007.v5.n1.a6
    """
    if isinstance(h, sp.Basic) or isinstance(eps, sp.Basic):
        Max, sqrt = sp.Max, sp.sqrt
    else:
        Max, sqrt = max, (lambda x: x ** 0.5)
    return sqrt(2) * h / sqrt(h ** 4 + Max(h, eps) ** 4)


# ── PHYSICAL operator slots a 1/h→hinv substitution must sweep ─────────────
# The flux/source/diffusion/NCP/pressure/mass operators are where a bare
# ``1/h`` corrupts the wet/dry front, so those are swept.  ``update_variables``
# (the per-cell reconstruction / corrector projection map) and the
# reconstruction maps are DELIBERATELY excluded: they are not wavespeed- or
# flux-bearing, and rewriting them changes the projection update (and, for a
# split corrector, would pull ``hinv`` into the dt-bearing update kernel).
# The derived operators (``quasilinear_matrix`` / source jacobians) are NOT
# listed either — they are recomputed from these primaries by
# ``refresh_derived_operators``.
_PRIMARY_OPERATORS = (
    "flux",
    "hydrostatic_pressure",
    "source",
    "source_explicit",
    "mass_matrix",
    "nonconservative_matrix",
    "diffusion_matrix",
    "diffusion_matrix_explicit",
)


def _pow_rewrite(hs, inv):
    """Return `rewrite(expr)` replacing every `hs**(-n)` by `inv**n`, also through factored
    and expanded bases; transcendental nodes are not descended.
    Shared by `regularize_pow` and the depth-law study operations; `inv` may be an aux symbol
    or an inline expression.
    """
    def _h_multiplicity(base):
        """Multiplicity of `hs` as a multiplicative factor of `base` (0 if it only sits in a
        transcendental argument or a sum).
        """
        base = sp.sympify(base)
        if not base.has(hs):
            return 0
        e = sp.factor(base).as_powers_dict().get(hs, sp.S.Zero)
        return int(e) if e.is_Integer and e > 0 else 0

    def _rw(e):
        """Recursively rewrite the negative powers of `hs` in `e`."""
        if isinstance(e, _TRANSCENDENTAL):
            return e
        if isinstance(e, sp.Pow) and e.exp.is_number and e.exp.is_negative:
            p = _h_multiplicity(e.base)
            if p > 0:
                # base = q·hs**p (q hs-free) ⇒ base**e = q**e·inv**(-p·e).
                # Clean ``hs**(-n)`` ⇒ q == 1 ⇒ byte-identical to a plain
                # ``inv**(-exp)`` rewrite.
                q = sp.cancel(e.base / hs ** p)
                return _rw(q) ** e.exp * inv ** (-p * e.exp)
            return _rw(e.base) ** e.exp
        if e.args:
            return e.func(*[_rw(a) for a in e.args])
        return e

    return lambda e: _rw(sp.sympify(e))


def _resolve_state_symbol(sm, field: Union[str, sp.Symbol]) -> sp.Symbol:
    """The state Symbol named `field` (the Symbol itself is accepted); raises `KeyError` if
    absent.
    """
    name = str(field)
    for s in sm.state:
        if str(s) == name:
            return s
    raise KeyError(
        f"regularize_pow: '{name}' is not a state variable of this "
        f"SystemModel (state = {[str(s) for s in sm.state]}).")


def _resolve_aux_symbol(sm, name: str) -> sp.Symbol:
    """The aux Symbol named `name`; raises `KeyError` if it is not registered."""
    for s in sm.aux_state:
        if str(s) == str(name):
            return s
    raise KeyError(
        f"regularize_pow: aux '{name}' is not in aux_state "
        f"({[str(s) for s in sm.aux_state]}); register it first with "
        f"register_aux('{name}', …).")


# ── register_aux ───────────────────────────────────────────────────────────

def register_aux(name: str, formula, **assumptions):
    """Operation: add the algebraic aux variable `name = formula` to `aux_state` and
    `update_aux_variables` (identity on existing rows) and refresh the derived operators.
    `**assumptions` go to the new Symbol (default `real=True`).
    """
    if not assumptions:
        assumptions = {"real": True}

    def _op(sm):
        """Apply the operation to `sm` in place."""
        sym = sp.Symbol(name, **assumptions)
        if any(str(s) == name for s in sm.aux_state):
            raise ValueError(
                f"register_aux: aux '{name}' is already in aux_state.")
        expr = sp.sympify(formula)

        # Preserve any existing per-row aux formulas (rows a model already
        # declared); identity-passthrough for the rest.  ``update_aux_variables``
        # is the ``(n_aux, 1)`` column convention, so read one scalar per row.
        prev = sm.update_aux_variables
        existing = ([prev[i, 0] for i in range(prev.shape[0])]
                    if prev is not None else [])

        sm.aux_state = list(sm.aux_state) + [sym]

        rows = []
        for i, s in enumerate(sm.aux_state):
            if s is sym:
                rows.append(expr)
            elif i < len(existing):
                rows.append(existing[i])           # keep declared formula
            else:
                rows.append(s)                     # identity passthrough
        sm.update_aux_variables = ZArray([[r] for r in rows])

        # Resize / recompute the aux-dependent derived operators (the source
        # jacobian wrt aux now has one more column); flux/source unchanged here.
        sm.refresh_derived_operators(eigenvalues=False)

    _op.name = "register_aux"
    _op.description = f"add aux '{name}' = {formula}"
    return _op


# ── regularize_pow ─────────────────────────────────────────────────────────

# Derived operators that carry an exact ``∂(…/h)/∂h`` term and therefore must
# be FROZEN-THEN-SUBSTITUTED (not recomputed) by ``regularize_pow`` — see its
# docstring.
_DERIVED_OPERATORS = (
    "quasilinear_matrix",
    "eigenvalues",
    "eigenvalues_cfl",
    "source_jacobian_wrt_variables",
    "source_jacobian_wrt_aux_variables",
)

# The order-2 primitive reconstruction map ``state → primitive`` (e.g. SWE's
# ``[b, b+h, hu/h, hv/h]``) divides the momentum by ``h``.  On a dry bed that
# raw ``1/h`` is a division by zero — a SIGFPE on step 1 under OpenFOAM's FP
# trapping, an inf/NaN elsewhere (REQ-156).  Sweep it through the SAME
# desingularized ``hinv`` the flux/source already use, so ``hu/h → hu·hinv`` is
# dry-bed-safe on every backend by construction.  ``state_from_reconstruction``
# (the WB inverse) carries no ``1/h`` today, but include it so any future
# reciprocal there is desingularized too.
_RECONSTRUCTION_OPERATORS = (
    "reconstruction_variables",
    "state_from_reconstruction",
)


def wave_speed_inverse_depth(h):
    """The `1/h` of every wave speed, `1/max(1e-14, h)`, never the KP `hinv`: large but finite
    at a wet/dry front (REQ-82). Used by `regularize_pow` and the numerical spectral radius.
    """
    # eps_h — the wave-speed floor.  Deliberately an INTERNAL constant (user
    # ruling 2026-07-19, task cid=5): not a parameter, not exposed, not
    # tunable.  Every PHYSICAL operator uses only the wet/dry KP reciprocal
    # (u -> 0 as h -> 0); the spectrum needs the OPPOSITE guarantee — large-
    # but-finite at a wet/dry front — because a KP-damped celerity starves the
    # Riemann dissipation and drains cells negative (measured: lake h_min
    # -2.5e-15 with this floor vs -1.6e-2 with KP on the eigenvalues).  A
    # fully-dry domain's dt is governed by NSM dt_max (REQ-190), so this floor
    # never sets a global timestep.
    eps_h = 1e-14
    return sp.S.One / sp.Max(sp.Float(eps_h), h)


def regularize_pow(field: Union[str, sp.Symbol], aux_name: str):
    """Operation: replace every `field**(-n)` in the physical, derived and reconstruction
    operators by `aux_name**n`; eigenvalues get a floored plain `1/max(1e-14, h)` instead.
    The lazy derived operators are frozen from the clean primaries first, so the exact
    `d(q^2/h)/dh` term survives the substitution.
    """
    def _op(sm):
        """Apply the operation to `sm` in place."""
        hs = _resolve_state_symbol(sm, field)
        aux = _resolve_aux_symbol(sm, aux_name)

        # ``inv`` is the replacement for ``1/h``: the desingularized ``aux``
        # (``hinv``) for the flux/source/NCP operators, but the plain floored
        # inverse ``1/Max(1e-14, h)`` for the EIGENVALUE / wave-speed (REQ-82 —
        # see `wave_speed_inverse_depth`).  See :func:`_pow_rewrite` for the
        # rewrite rules themselves.
        def _rw(e, inv):
            """Rewrite `e` with `inv` as the replacement for `1/h`."""
            return _pow_rewrite(hs, inv)(e)

        # 1. Force-materialize the lazy derived operators from the CLEAN
        #    primaries (``quasilinear_matrix`` is a lazy property; touching it
        #    computes ``−q²/h²`` from the still-``1/h`` flux).  ``eigenvalues``
        #    and the source jacobians are already materialized.
        _ = sm.quasilinear_matrix

        # 2. Substitute 1/h**n -> hinv**n in the physical operators AND the
        #    frozen derived operators (so the exact h-derivative terms become
        #    hinv-powers instead of being recomputed away).  EXCEPTION: the
        #    ``eigenvalues`` (the wave-speed / Rusanov-dissipation source) get
        #    the plain ``1/Max(1e-14, h)`` floor, NOT ``hinv`` — the KP
        #    regulariser suppresses the speed by ``(h*/eps)²`` at a wet/dry
        #    front and starves the dissipation, draining the cell negative
        #    (REQ-82).  The flux velocity already uses this same FP floor.
        inv_floor = wave_speed_inverse_depth(hs)
        for nm in (_PRIMARY_OPERATORS + _DERIVED_OPERATORS
                   + _RECONSTRUCTION_OPERATORS):
            M = getattr(sm, nm, None)
            if M is not None:
                inv = (inv_floor if nm in ("eigenvalues", "eigenvalues_cfl")
                       else aux)
                flat = [_rw(sp.sympify(e), inv) for e in sp.flatten(M)]
                setattr(sm, nm, type(M)(flat).reshape(*M.shape))

    _op.name = "regularize_pow"
    _op.description = f"{field}**(-n) -> {aux_name}**n"
    return _op


# ── desingularize_hinv ─────────────────────────────────────────────────────

def _wet_dry_eps(sm, override=None):
    """Wet/dry threshold: `override`, else the model's `wet_dry_eps` parameter, else
    `_DEFAULT_WET_DRY_EPS`.
    """
    if override is not None:
        return sp.sympify(override)
    params = getattr(sm, "parameters", None)
    if params is not None and params.contains("wet_dry_eps"):
        return params.wet_dry_eps
    return sp.Float(_DEFAULT_WET_DRY_EPS)


def _depth_state(sm, what):
    """The depth state Symbol (the declared `depth` role; legacy fallback: the state named
    `"h"` for systems that declared no role); raises `ValueError` if none.
    """
    roles = getattr(sm, "roles", None)
    if roles is not None and roles.depth is not None:
        return sm.state[roles.depth]
    h = None
    if roles is None or not roles.depth_declared:
        h = next((s for s in sm.state if str(s) == "h"), None)
    if h is None:
        raise ValueError(
            f"{what}: the system declares no depth role "
            f"(state = {[str(s) for s in sm.state]}).")
    return h


def _min_h_power(expr, h):
    """Lowest power of `h` in `expr` as a rational function (negative means a `1/h**n`), or
    `None` if undecidable.
    """
    e = sp.sympify(expr)
    if e == 0 or h not in e.free_symbols:
        return 0
    try:
        num, den = sp.fraction(sp.cancel(sp.together(e)))

        def _mindeg(p):
            """Lowest monomial degree in `h` of polynomial `p`."""
            p = sp.expand(p)
            if p == 0:
                return 0
            return min(m[0] for m in sp.Poly(p, h).monoms())

        return _mindeg(num) - _mindeg(den)
    except (sp.PolynomialError, TypeError, ValueError):
        return None


def equilibrate_constraint_rows(mode=True):
    """Operation: multiply each algebraic (zero mass-matrix) row by the minimal power of `h`
    that clears its negative powers of `h`.
    This changes the conditioning of the elliptic block, not the constraint manifold, wherever ``h
    > 0``.
    """
    def _op(sm):
        """Apply the operation to `sm` in place."""
        if not mode:
            return
        h = _depth_state(sm, "equilibrate_constraint_rows")
        M = sm.mass_matrix
        if M is None:
            return
        n_eq = M.shape[0]
        scaled = []
        for i in range(n_eq):
            if any(sp.sympify(M[i, j]) != 0 for j in range(M.shape[1])):
                continue                      # differential row — not a constraint
            entries = []
            for slot in ("flux", "hydrostatic_pressure", "source",
                         "source_explicit"):
                op = getattr(sm, slot, None)
                if op is None:
                    continue
                arr = sp.Array(op)
                if arr.rank() == 1:
                    entries.append((slot, (i,), arr[i]))
                else:
                    for d in range(arr.shape[1]):
                        entries.append((slot, (i, d), arr[i, d]))
            ncp = getattr(sm, "nonconservative_matrix", None)
            if ncp is not None:
                arr = sp.Array(ncp)
                for a in range(arr.shape[1]):
                    for d in range(arr.shape[2]):
                        entries.append(("nonconservative_matrix", (i, a, d),
                                        arr[i, a, d]))
            powers = [_min_h_power(e, h) for _, _, e in entries]
            if any(p is None for p in powers):
                continue                      # undecidable — leave the row alone
            n = -min(powers)
            if n <= 0:
                continue                      # already free of 1/h
            for slot, idx, e in entries:
                op = getattr(sm, slot)
                op[idx] = sp.expand(sp.cancel(sp.sympify(e) * h ** n))
            scaled.append((i, n))
        if scaled:
            sm.refresh_derived_operators(eigenvalues=False)
        _op.scaled_rows = scaled

    _op.name = "equilibrate_constraint_rows"
    _op.description = "minimal-power h row-equilibration of DAE constraint rows"
    return _op


def desingularize_hinv(mode="kp"):
    """Operation: register the `hinv` aux with the Kurganov-Petrova inverse depth (`eps` =
    `wet_dry_eps`) and rewrite every `h**(-n)` to `hinv**n`; `mode` is `True` or
    `"kp"`, falsy is a no-op.
    Legacy NSM depth default; the depth-law study uses `regularize_depth_direct` /
    `regularize_depth_aux`.

    References:
        Kurganov, Petrova 2007, https://doi.org/10.4310/CMS.2007.v5.n1.a6
    """
    def _op(sm):
        """Apply the operation to `sm` in place."""
        if not mode:
            return
        if isinstance(mode, str) and mode.lower() != "kp":
            raise ValueError(
                f"desingularize_hinv: unknown mode {mode!r}; "
                "expected True or 'kp'.")
        h = _depth_state(sm, "desingularize_hinv")
        eps = _wet_dry_eps(sm)
        if not any(str(s) == "hinv" for s in sm.aux_state):
            # NO ``positive=True``.  It used to be declared here and it is now
            # FALSE: the ``Max(h, 0)`` numerator clamp was removed from
            # ``kp_hinv`` (positivity must come from the scheme, not a clamp),
            # so ``hinv = sqrt(2)*h/sqrt(...)`` is NEGATIVE wherever h < 0.
            # Declaring it positive let sympy simplify against a fact that does
            # not hold, AND left an assumption-carrying symbol on the NUMERICAL
            # side of the boundary -- 7 sites in ML-SME, 68 in VAM dim=3, 113 in
            # ML-VAM, none of which the gate could see.
            sm.apply(register_aux("hinv", kp_hinv(h, eps)))
        sm.apply(regularize_pow(h, "hinv"))

    _op.name = "desingularize_hinv"
    _op.description = f"KP 1/h desingularization (mode={mode!r})"
    return _op


def symbolic_spectrum():
    """Operation: derive the closed-form spectrum of the normal-projected quasilinear matrix into
    `sm.eigenvalues`; opt-in because it is expensive for high moment levels.
    """
    def _op(sm):
        """Apply the operation to `sm` in place."""
        sm.eigenvalues = ZArray(sm._compute_eigenvalues())
    _op.name = "symbolic_spectrum"
    _op.description = "closed-form spectrum of the quasilinear matrix"
    return _op


# ── REQ-194 depth-law study: two regularization PATHS, one KP definition ───
#
# Both paths desingularize the SAME reciprocal with the SAME Kurganov–Petrova
# form :func:`kp_hinv` — there is exactly ONE definition of the regularized
# ``1/h`` in this module, and neither ``1/(h + eps)`` nor ``1/max(eps, h)``
# exists as an alternative.  They differ ONLY in where that form lands:
#
#   * DIRECT — the KP expression is substituted inline wherever ``h**(-n)``
#     appears.  No aux row, no extra state; the regularization is visible in
#     every operator and its ``h``-derivative is exact wherever a Jacobian was
#     frozen before the substitution.
#   * AUX    — the reciprocal is promoted to an ``hinv`` aux row whose update
#     rule IS the KP expression, and the operators then carry ``hinv**n``.  One
#     named place holds the regularization; the operators stay small and
#     ``hinv`` is an INDEPENDENT symbol to the CAS.
#
# They are MUTUALLY EXCLUSIVE.  Applying both is an error, not a
# double-regularization: the aux path would rewrite ``h**(-n)`` that the direct
# path has already replaced by a KP expression (leaving a half-converted system
# whose ``hinv`` row and inline forms disagree), and nothing downstream could
# tell which reciprocal a given operator is using.
#
# NEITHER is a default.  ``default_operations()`` still returns the legacy
# ``desingularize_hinv()``; these are selected per construction so a sweep is a
# pure loop over the axes (see ``zoomy_core.numerics.depth_law_study``).

#: Attribute stamped on a SystemModel by whichever depth-regularization path ran.
_DEPTH_REGULARIZATION_ATTR = "_depth_regularization_applied"

#: Legal values of the ``eigenvalues=`` axis shared by both paths.
_EIGENVALUE_TREATMENTS = ("regularize", "exclude")


def _regularization_eps(sm, override=None):
    """Regularization epsilon: `override`, else the model's `regularization_eps` parameter,
    else `_DEFAULT_REGULARIZATION_EPS`; never `wet_dry_eps`.
    """
    if override is not None:
        return sp.sympify(override)
    params = getattr(sm, "parameters", None)
    if params is not None and params.contains("regularization_eps"):
        return params.regularization_eps
    return sp.Float(_DEFAULT_REGULARIZATION_EPS)


def _claim_depth_regularization(sm, tag: str) -> None:
    """Record that `tag` regularized the depth on `sm`; raises if another path (or the same
    one twice) already did.
    """
    prev = getattr(sm, _DEPTH_REGULARIZATION_ATTR, None)
    if prev is not None:
        raise ValueError(
            f"depth regularization already applied to this system by "
            f"{prev!r}; refusing to also apply {tag!r}.  The DIRECT and AUX "
            f"paths are mutually exclusive — pick ONE per system (they "
            f"regularize the same reciprocal, so applying both would leave "
            f"the operators disagreeing about what 1/h means).  Build a "
            f"fresh SystemModel to try the other path.")
    setattr(sm, _DEPTH_REGULARIZATION_ATTR, tag)


def _eigenvalue_exclusion(eigenvalues: str) -> tuple:
    """Translate the `eigenvalues=` axis (`"regularize"` | `"exclude"`) into a
    `map_operator_slots` exclusion tuple.
    """
    if eigenvalues not in _EIGENVALUE_TREATMENTS:
        raise ValueError(
            f"eigenvalues={eigenvalues!r} is not a legal treatment; expected "
            f"one of {list(_EIGENVALUE_TREATMENTS)}.")
    return (("eigenvalues", "eigenvalues_cfl")
            if eigenvalues == "exclude" else ())


def _assert_no_forward_aux_reference(sm, aux, aux_row: int, name: str) -> None:
    """Raise if an aux row evaluated before `aux` now reads it, since it would see a stale value
    each step.
    """
    uav = getattr(sm, "update_aux_variables", None)
    if uav is None:
        return
    offenders = [
        str(s) for i, s in enumerate(sm.aux_state[:uav.shape[0]])
        if i < aux_row and sp.sympify(uav[i, 0]).has(aux)
    ]
    if offenders:
        raise ValueError(
            f"regularize_depth_aux: aux row(s) {offenders} are evaluated "
            f"BEFORE '{name}' (row {aux_row}) but now reference it, so they "
            f"would read a stale value each step.  Order the aux vector so "
            f"'{name}' precedes its consumers.")


def regularize_depth_direct(eps=None, *, field: str = "h",
                            eigenvalues: str = "regularize", exclude=()):
    """Operation: substitute the Kurganov-Petrova reciprocal inline for every `field**(-n)` in
    every slot (the DIRECT path); no aux row is added.
    Mutually exclusive with `regularize_depth_aux`.

    Args:
        eps: regularization scale (not `wet_dry_eps`).
        eigenvalues: `"regularize"` sweeps the spectrum, `"exclude"` keeps the exact `1/h`.
        exclude: further slots to leave alone (opt out, never opt in).

    References:
        Kurganov, Petrova 2007, https://doi.org/10.4310/CMS.2007.v5.n1.a6
    """
    # Validated HERE, not inside ``_op``: the axes are a construction-time
    # choice, so a bad value must fail when the op is built, before it has
    # claimed the depth regularization on a system it then fails to rewrite.
    excl = _eigenvalue_exclusion(eigenvalues) + tuple(exclude)

    def _op(sm):
        """Apply the operation to `sm` in place."""
        h = _resolve_state_symbol(sm, field)
        e = _regularization_eps(sm, eps)
        _claim_depth_regularization(sm, "direct")
        inv = kp_hinv(h, e)
        _op.eps = e
        _op.coverage = map_operator_slots(sm, _pow_rewrite(h, inv),
                                          exclude=excl)

    _op.name = "regularize_depth_direct"
    _op.description = (
        f"inline KP 1/{field} (eps={'default' if eps is None else eps}, "
        f"eigenvalues={eigenvalues!r})")
    _op.coverage = {}
    _op.eps = None
    return _op


def regularize_depth_aux(eps=None, *, name: str = "hinv", field: str = "h",
                         eigenvalues: str = "regularize", exclude=()):
    """Operation: register the aux `name` whose update row is the Kurganov-Petrova reciprocal
    and rewrite every `field**(-n)` to `name**n` in every slot (the AUX path).
    Mutually exclusive with `regularize_depth_direct`; takes the same `eps` / `eigenvalues`
    / `exclude` arguments, and the row defining `name` is not rewritten.

    References:
        Kurganov, Petrova 2007, https://doi.org/10.4310/CMS.2007.v5.n1.a6
    """
    # Validated at construction time — see :func:`regularize_depth_direct`.
    excl = _eigenvalue_exclusion(eigenvalues) + tuple(exclude)

    def _op(sm):
        """Apply the operation to `sm` in place."""
        h = _resolve_state_symbol(sm, field)
        e = _regularization_eps(sm, eps)
        _claim_depth_regularization(sm, "aux")
        _op.eps = e

        if not any(str(s) == name for s in sm.aux_state):
            sm.apply(register_aux(name, kp_hinv(h, e), positive=True))
            # The aux VECTOR grew, so every kernel lowered against it must see
            # the new layout — otherwise the sweep below (which DOES reach the
            # boundary kernels) could write the new aux into a kernel whose
            # frozen signature never binds it.
            _sync_kernel_aux_signature(sm)
        else:
            i = next(i for i, s in enumerate(sm.aux_state) if str(s) == name)
            uav = sm.update_aux_variables
            if (uav is None or i >= uav.shape[0]
                    or sp.sympify(uav[i, 0]) == sm.aux_state[i]):
                raise ValueError(
                    f"regularize_depth_aux: aux '{name}' is declared in "
                    f"aux_state but ``update_aux_variables`` gives it no "
                    f"formula (identity passthrough).  The operators are about "
                    f"to read it as 1/{field}; give the model an "
                    f"``update_aux_variables`` row for '{name}', or drop the "
                    f"declaration and let this operation register it.")

        aux = _resolve_aux_symbol(sm, name)
        aux_row = [i for i, s in enumerate(sm.aux_state) if s == aux]
        _op.coverage = map_operator_slots(
            sm, _pow_rewrite(h, aux), exclude=excl,
            skip_rows={"update_aux_variables": aux_row})
        _assert_no_forward_aux_reference(sm, aux, aux_row[0], name)

    _op.name = "regularize_depth_aux"
    _op.description = (
        f"1/{field}**n -> aux '{name}'**n, KP row "
        f"(eps={'default' if eps is None else eps}, "
        f"eigenvalues={eigenvalues!r})")
    _op.coverage = {}
    _op.eps = None
    return _op


# ── desingularize_positivity ───────────────────────────────────────────────

def desingularize_positivity(floor):
    """Operation: replace the base of each negative or fractional power in the source that
    involves a `positive_state` symbol by `sqrt(base**2 + floor**2)` and re-derive the
    source Jacobians.
    A no-op if the model declares no `positive_state`.
    """
    def _op(sm):
        """Apply the operation to `sm` in place."""
        protected = set(getattr(sm, "positive_state", []) or [])
        if not protected:
            return
        f2 = sp.Float(floor) ** 2

        def safe(expr):
            """Floor the singular positive-state powers in one expression."""
            e = sp.sympify(expr)
            repl = {}
            for p in e.atoms(sp.Pow):
                if (p.exp.is_number and (p.exp < 0 or not p.exp.is_integer)
                        and (p.base.free_symbols & protected)):
                    repl[p] = sp.Pow(sp.sqrt(p.base ** 2 + f2), p.exp)
            return e.xreplace(repl) if repl else e

        n = sm.n_equations
        state = list(sm.state)
        new_src = sp.zeros(n, 1)
        for i in range(n):
            new_src[i, 0] = safe(sm.source[i, 0])
        sm.source = ZArray(new_src)
        # The regularised source invalidates the channeled jacobians —
        # re-derive both from the new source (the in-place-mutation fallback,
        # mirroring ``SystemModel.refresh_derived_operators``).
        aux = list(sm.aux_state)
        J = sp.zeros(n, len(state))
        Ja = sp.zeros(n, len(aux))
        for i in range(n):
            si = sp.sympify(sm.source[i, 0])
            for j, s in enumerate(state):
                J[i, j] = sp.diff(si, s)
            for k, a in enumerate(aux):
                Ja[i, k] = sp.diff(si, a)
        sm.source_jacobian_wrt_variables = ZArray(J)
        sm.source_jacobian_wrt_aux_variables = ZArray(Ja)

    _op.name = "desingularize_positivity"
    _op.description = f"floor positive-state singularities (floor={floor})"
    return _op


# ── eigenvalue power guard + dry gate ───────────────────────────────────────
#
# Two composable ops (REQ-181, "operations, not options" — neither a default):
#
# * ``guard_eigenvalue_powers()`` — A, the ALWAYS-SAFE half: wrap every
#   fractional power of ``h`` in ``Max(., 0)`` so the wet-branch expression
#   stays real at a transient ``h < 0``.  A no-op for physical ``h >= 0``.
# * ``gate_eigenvalues_dry()`` — B, the dry zeroing: wrap each eigenvalue in
#   ``conditional(h > eps, e, 0)``.  Because a backend ``conditional`` lowered
#   branchless (``mask*a + (1-mask)*b``) computes BOTH arms and ``NaN*0 = NaN``
#   would leak a ``sqrt(negative)`` past the gate (REQ-74), the dry gate MUST
#   carry the power guard — so ``gate_eigenvalues_dry`` applies A internally
#   before the ``conditional``.  It therefore reproduces the pre-split combined
#   op BYTE-FOR-BYTE as a single op, and the explicit composition
#   ``[guard_eigenvalue_powers(), gate_eigenvalues_dry()]`` reproduces it too
#   (the guard is idempotent — ``Max(., 0)`` is never re-wrapped).


def _guard_frac_pow_of_h(h):
    """Predicate for `expr.replace`: a fractional power of `h` not already floored by ``Max(.,
    0)`` (which makes the guard idempotent).
    """
    def _q(x):
        """Whether `x` is such a power."""
        return (isinstance(x, sp.Pow) and x.exp.is_number
                and not x.exp.is_integer and x.base.has(h)
                and not (isinstance(x.base, sp.Max)
                         and sp.S.Zero in x.base.args))
    return _q


def _guard_eigenvalue_expr(e, h):
    """`e` with every fractional power of `h` wrapped in `Max(., 0)`; idempotent and a no-op
    for `h >= 0`.
    """
    return sp.sympify(e).replace(
        _guard_frac_pow_of_h(h),
        lambda x: sp.Pow(sp.Max(x.base, sp.S.Zero), x.exp))


def _flat_eigenvalues(sm, what):
    """`(ev, scalar entries)` of `sm.eigenvalues`, or `(None, None)` when the model emits
    numerical wave speeds.
    """
    ev = getattr(sm, "eigenvalues", None)
    if ev is None:
        return None, None
    h = _depth_state(sm, what)
    return ev, [sp.sympify(e) for e in sp.flatten(ev)]


def guard_eigenvalue_powers():
    """Operation: wrap every fractional power of the depth in `sm.eigenvalues` in `Max(., 0)`
    so wave speeds stay real at a transient `h < 0` (no dry zeroing).
    Half A of the guard/gate split; `gate_eigenvalues_dry` is half B.
    """
    def _op(sm):
        """Apply the operation to `sm` in place."""
        ev, flat = _flat_eigenvalues(sm, "guard_eigenvalue_powers")
        if ev is None:
            return
        h = _depth_state(sm, "guard_eigenvalue_powers")
        guarded = [_guard_eigenvalue_expr(e, h) for e in flat]
        sm.eigenvalues = ZArray(guarded).reshape(*ev.shape)

    _op.name = "guard_eigenvalue_powers"
    _op.description = "guard fractional powers of h in eigenvalues with Max(.,0)"
    return _op


def _is_dry_gated(e, h, eps) -> bool:
    """True when `e` is already `conditional(h > eps, ., 0)`, so re-gating is a no-op."""
    return (isinstance(e, sp.Function) and e.func.__name__ == "conditional"
            and len(e.args) == 3 and e.args[0] == (h > eps)
            and e.args[2] == sp.S.Zero)


def gate_eigenvalues_dry(eps=None):
    """Operation: publish a timestep-only spectrum `conditional(h > eps, e, 0)` into
    `sm.eigenvalues_cfl`, leaving `sm.eigenvalues` physical (with its power guard) for the
    Riemann dissipation.
    The dry gate carries the power guard because a branchless backend evaluates both arms; without
    a closed form it gates `numerical_eigenvalues` (or the model's published
    `eigenvalues_cfl`) and is idempotent.
    """
    def _op(sm):
        """Apply the operation to `sm` in place."""
        ev, flat = _flat_eigenvalues(sm, "gate_eigenvalues_dry")
        h = _depth_state(sm, "gate_eigenvalues_dry")
        if ev is None:
            # NO CLOSED FORM (SME / VAM): the model computes its wave speeds by
            # a runtime eigensolve.  EMBED that computation inside the gate
            # rather than skipping — wrap the opaque
            # ``numerical_eigenvalues`` kernel (itself derived from the CURRENT
            # ``quasilinear_matrix``) in the same dry conditional.  The slot
            # then carries "if wet, solve for the spectrum; if dry, 0", so a
            # backend still just evaluates one operator and never needs a
            # wet/dry rule of its own.
            #
            # Before this, the op returned early for these models, which is why
            # the numerical-spectrum path was the one place a backend still had
            # to carry its own dry threshold.
            e_eps = _wet_dry_eps(sm, eps)
            cond = sp.Function("conditional")
            # The MODEL may already have published a dt spectrum of its own —
            # SME(level>=2) puts its closed-form ``spectral_radius_bound`` cone
            # here precisely because it has no valid closed-form ``eigenvalues``
            # (``SME._wave_speed_cone``).  Gate THAT; only synthesise one from
            # the opaque eigensolve when the model published nothing.  Before
            # this the branch overwrote it unconditionally, which put a per-face
            # numerical eigen-decomposition back on the dt path of exactly the
            # models that had gone to the trouble of avoiding one.
            published = getattr(sm, "eigenvalues_cfl", None)
            num = published if published is not None else sm.numerical_eigenvalues
            # IDEMPOTENT.  The closed-form branch below rebuilds the gated copy
            # from ``sm.eigenvalues`` (which this op never zeroes), so applying
            # it twice is harmless there.  This branch reads the slot it also
            # WRITES — and the op is genuinely applied twice on the common path
            # (``NumericalSystemModel.eigenvalue_guard`` defaults to ``"gate"``
            # AND cases pass it in ``extra_operations``), so without this the
            # cone comes out wrapped in ``conditional(h>eps, conditional(h>eps,
            # …), 0)`` — same value, twice the tree.
            gated = [e if _is_dry_gated(e, h, e_eps)
                     else cond(h > e_eps, e, sp.S.Zero)
                     for e in sp.flatten(num)]
            # ``numerical_eigenvalues`` is RANK-1 (length n), but the
            # eigenvalue slot contract is ``(n_eq, 1)`` — the same column shape
            # the closed-form branch writes.  Reshaping to the rank-1 source
            # shape instead made every downstream 2-D index raise
            # "Dimension of index greater than rank of array".
            sm.eigenvalues_cfl = ZArray(gated).reshape(len(gated), 1)
            return
        e_eps = _wet_dry_eps(sm, eps)
        cond = sp.Function("conditional")
        guarded = [_guard_eigenvalue_expr(e, h) for e in flat]
        gated = [cond(h > e_eps, e, sp.S.Zero) for e in guarded]
        # cid=209: only the GATED copy goes to the dt-only slot.  The ZEROING is
        # what must not reach ``sm.eigenvalues`` (it is the Rusanov dissipation).
        # The power guard is a different thing and DOES belong there: it is
        # ``Max(., 0)`` under a fractional power of h, a no-op for the physical
        # h >= 0 and the only reason a transient h < 0 does not turn the
        # dissipation into NaN (REQ-74).  Dropping it here would have handed the
        # dissipation an unguarded sqrt(h) that the old in-place gate protected.
        sm.eigenvalues = ZArray(guarded).reshape(*ev.shape)
        sm.eigenvalues_cfl = ZArray(gated).reshape(*ev.shape)

    _op.name = "gate_eigenvalues_dry"
    _op.description = ("publish dt-only eigenvalues_cfl gated to 0 for dry "
                       "cells (h < eps); guard sqrt(h) args")
    return _op


# ── AD-safe fractional powers ───────────────────────────────────────────────

def regularize_fractional_powers(eps, *, exclude=()):
    """Operation: shift the base of every fractional power by `eps**2` (``b**r -> (b +
    eps**2)**r``) so the operators stay differentiable at `b = 0` under AD.
    Opt-in and idempotent; `eps` must be positive, and the value changes by `O(eps**2)` away
    from `b = 0`.
    """
    if not eps > 0:
        raise ValueError(
            f"regularize_fractional_powers: eps must be > 0, got {eps!r}")
    e2 = sp.Float(eps) ** 2

    def _is_fractional_pow(x):
        """Whether `x` is a fractional power with a non-numeric base not yet shifted by
        `eps**2`.
        """
        return (isinstance(x, sp.Pow) and x.exp.is_Rational
                and not x.exp.is_integer and not x.base.is_number
                and not (isinstance(x.base, sp.Add) and e2 in x.base.args))

    def _rewrite(expr):
        """Shift the fractional-power bases of one expression."""
        return sp.sympify(expr).replace(
            _is_fractional_pow, lambda x: sp.Pow(x.base + e2, x.exp))

    def _op(sm):
        """Apply the operation to `sm` in place."""
        _op.coverage = map_operator_slots(sm, _rewrite, exclude=exclude)

    _op.name = "regularize_fractional_powers"
    _op.description = f"b**r -> (b + eps**2)**r for fractional r (eps={eps})"
    _op.coverage = {}
    return _op
