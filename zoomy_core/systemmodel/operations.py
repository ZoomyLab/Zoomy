"""Reusable :meth:`SystemModel.apply` operations.

These are plain callables ``op(system_model)`` consumed through the generic
:meth:`zoomy_core.systemmodel.SystemModel.apply` hook::

    sm.apply(register_aux("hinv", kp_hinv(h, eps)))
    sm.apply(regularize_pow("h", "hinv"))

They replace the per-case "system surgery" a model wrapper used to hand-roll
(walk every operator, substitute, append to ``aux_state``, hand-build a
full-length ``update_aux_variables``, call ``refresh_derived_operators``).  Each
operation owns its own refresh, so callers chain them and never touch the
derived-operator slots by hand.

SCOPING RULE (REQ-194).  The operations added for the depth-law study route
their rewrite through :func:`map_operator_slots`, which applies to **ALL**
operator slots (:data:`OPERATOR_SLOTS`) by DEFAULT.  An operation that must not
touch a slot declares an explicit ``exclude=(...)`` with a stated reason, and a
name that is not a member of the registry RAISES.  Opt OUT, never opt in —
dropping a slot from coverage is then a deliberate, visible decision rather than
an omission.  (The predecessor was a closed hard-coded list of 14 slot names
iterated with ``getattr(sm, nm, None)``, so ``interpolate_to_3d`` kept a bare
``hu/h`` and the boundary-condition kernels were never reached at all —
silently.)

The LEGACY pair :func:`register_aux` + :func:`regularize_pow` (and their
composition :func:`desingularize_hinv`, which is still the NSM depth default) is
deliberately left on its old closed list so every backend that already opted
into it keeps emitting byte-identical code.  The study paths
(:func:`regularize_depth_direct` / :func:`regularize_depth_aux`) are the swept
ones.
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
)


#: Slots of :data:`OPERATOR_SLOTS` that carry a ``Function`` KERNEL — a
#: ``(definition, args)`` pair whose ``args`` Zstruct is the lowering signature.
KERNEL_SLOTS = (
    "boundary_conditions",
    "aux_boundary_conditions",
    "boundary_gradients",
)


def _is_kernel(obj) -> bool:
    """True for a :class:`zoomy_core.model.basefunction.Function` kernel — a
    named ``(definition, args)`` pair rather than a bare operator array."""
    return (hasattr(obj, "definition") and hasattr(obj, "args")
            and not isinstance(obj, sp.Basic))


def _sync_kernel_aux_signature(sm) -> None:
    """Re-bind the ``aux_variables`` group of every boundary-kernel signature to
    the CURRENT ``sm.aux_state``.

    A boundary kernel is lowered against a FROZEN signature: the runtime
    flattener reads ``Qaux`` positionally using the symbols listed in
    ``args.aux_variables``, and the compiled lambda's parameters come from the
    same list.  Registering a new aux grows the aux vector, so without this
    re-bind a kernel that afterwards mentions the new aux (which it now can —
    the study sweep reaches the boundary kernels) would carry an UNBOUND symbol:
    the flattener never extracts the new ``Qaux`` slot and the generated code
    closes over a bare sympy ``Symbol``.

    Names are attached with ``setattr`` rather than ``Zstruct(**names)`` so aux
    whose names are not Python identifiers (VAM's ``\\hat{\\sigma}_1``) survive;
    only ORDER and COUNT matter to the flattener."""
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
    """Rebuild the array ``M`` from its flat scalar entries."""
    return type(M)(flat).reshape(*M.shape)


def _row_of(shape, idx: int) -> int:
    """Row index of flat position ``idx`` in a row-major array of ``shape``."""
    stride = 1
    for s in shape[1:]:
        stride *= int(s)
    return idx // max(stride, 1)


def _rewrite_container(obj, rewrite, skip_rows=()):
    """Apply ``rewrite`` to every scalar of an operator container.

    Handles the three shapes a slot can take: an array (``ZArray`` /
    ``MatrixBase`` / ``NDimArray``), a plain ``list``/``tuple``, or a bare sympy
    expression (e.g. the ``Piecewise`` a boundary-condition kernel carries).
    ``skip_rows`` leaves the listed ROWS untouched (used for the self-reference
    guard on ``update_aux_variables``).

    Returns ``(changed, new_obj)``.
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
    """Apply ``rewrite(expr) -> expr`` to EVERY slot in :data:`OPERATOR_SLOTS`.

    This is the single sweep the study operations route through, so coverage is
    a property of the REGISTRY, not of each operation's private wish-list.

    ``exclude`` — slot names this operation must not touch.  Every name must be
    a member of :data:`OPERATOR_SLOTS`; a typo RAISES rather than silently
    widening coverage, which is exactly the failure mode the old closed list
    had.  ``skip_rows`` — ``{slot: (row, …)}`` rows left untouched inside an
    otherwise-swept slot.

    The lazy ``quasilinear_matrix`` is force-materialized FIRST, before any
    primary is written, so the derived operator is frozen from the CLEAN
    primaries and then substituted (never recomputed from an already-rewritten
    flux — that is what would drop the exact ``∂(q²/h)/∂h = −q²/h²`` term).

    Returns a ``{slot: "rewritten" | "unchanged" | "absent" | "excluded"}``
    coverage report.
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
    """Return ``rewrite(expr)`` imposing ``|n| = 1`` on the face normal.

    The face normal is a unit vector BY DEFINITION — every mesh in the project
    normalises it (``mesh_util._face_normals_*`` divide by ``np.linalg.norm``;
    in 1-D ``n0 = ±1``).  sympy has no way to know that, so it carries
    ``√(n0² + n1²)`` symbolically, which holds the 2-D spectrum over a common
    denominator and blocks every subsequent simplification::

        (√g·h^{3/2}·√(n0²+n1²) + hu·n0 + hv·n1) / h

    Imposing the relation and re-expanding splits it into the physical form::

        √(g·h) + hu·n0/h + hv·n1/h

    — after which ``1/h`` sits ONLY on the momentum terms and the celerity
    ``√(g·h)`` is structurally separate, so a later ``1/h`` regularization can
    never reach (and never damp) the wave speed.

    NB this separation is a property of the SWE spectrum, NOT a general fact.
    The SME spectrum is ``hinv·(n0·q_0 ± n0·√(g·h³ + q_1²))``: there the
    reciprocal multiplies the celerity and ``√(g·h³ + q_1²)`` does not factor,
    so normalising the normal does NOT take the regularization off the wave
    speed.  Which of the two behaviours the eigenvalues should get is the open
    question this machinery exists to measure — see the ``eigenvalues=`` axis of
    :func:`regularize_depth_direct` / :func:`regularize_depth_aux`.

    Rewrite rules, valid for ANY dimension:

    * ``(Σ_d n_d²)**k → 1``  for every exponent ``k`` (covers ``√(n0²+n1²)``,
      ``(n0²+n1²)^{3/2}``, ``1/(n0²+n1²)``, …);
    * the bare node ``Σ_d n_d² → 1``;
    * 1-D, where ``Σ n_d² = n0²`` collapses under sympy's own power algebra:
      ``n0**k → n0**(k mod 2)`` for integer ``k ∉ {0, 1}``, and ``|n0| → 1``.

    ``sp.expand`` afterwards is what makes the simplification FIRE (and is the
    only post-processing needed — ``powsimp`` / ``cancel`` are no-ops here: with
    ``g`` positive and ``h`` nonnegative, ``√g·√h`` IS sympy's canonical form of
    ``√(g·h)``, same ``srepr``).  It runs only on entries the relation actually
    changed, so a normal-free operator costs nothing.
    """
    n = list(normal_syms)
    nsq = sp.Add(*[nd ** 2 for nd in n])
    one_d = len(n) == 1

    def _is_unit_pow(x):
        if not isinstance(x, sp.Pow):
            return False
        if not one_d and x.base == nsq:
            return True
        return (one_d and x.base == n[0]
                and x.exp.is_Integer and int(x.exp) not in (0, 1))

    def _unit_pow_value(x):
        if one_d and x.base == n[0]:
            return n[0] if int(x.exp) % 2 else sp.S.One
        return sp.S.One

    def _is_unit_abs(x):
        return one_d and isinstance(x, sp.Abs) and x.args[0] == n[0]

    def _rewrite(e):
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
    """Operation: impose ``|n| = 1`` as a system-level fact (REQ-194).

    See :func:`_unit_normal_rewrite` for the rewrite rules and why this must run
    BEFORE any ``1/h`` regularization: for SWE it is what separates the celerity
    ``√(g·h)`` from the ``1/h`` of the momentum terms, and therefore what keeps
    the celerity exact once the reciprocal is desingularized.

    Default scope is every slot in :data:`OPERATOR_SLOTS` — the relation is a
    fact about the face normal, so it holds wherever the normal appears
    (spectrum, Riemann-facing operators, wall-mirror boundary kernels …).

    An NSM DEFAULT since REQ-208 item (2) (``normalize_normal=True``).  It
    changes the emitted expression tree of every model carrying a normal,
    which is why it started opt-in — but the relation is a fact about the
    mesh, not a modelling choice (every ``_face_normals_*`` builder divides
    by ``np.linalg.norm``), and being opt-in made it unreachable: the flag is
    set at NSM construction inside each backend's case code, so no case ever
    passed it and all three face kernels kept recomputing ``√g·|n|``.
    Construct with ``normalize_normal=False`` to opt back out.
    """
    def _op(sm):
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
    """Kurganov–Petrova desingularized inverse depth.

        hinv = √2·h / √(h⁴ + max(h, eps)⁴)

    Equals ``1/h`` for ``h ≥ eps`` but → 0 (not ``1/eps``) as ``h → 0``, so the
    reconstructed velocity ``u = q·hinv → 0`` at a dry front instead of blowing
    up.  This is the reusable building block behind the ``hinv`` aux: feed it to
    :func:`register_aux` and then sweep the operators with
    :func:`regularize_pow`::

        sm.apply(register_aux("hinv", kp_hinv(h, eps), positive=True))
        sm.apply(regularize_pow(h, "hinv"))

    Works symbolically (``h`` a sympy expression) or numerically (``h`` a
    numpy/jax array).

    THE NUMERATOR IS BARE ``h`` — no ``Max(h, 0)`` (user ruling, 2026-07-22).
    It used to carry that clamp so a transient negative-h cell could not flip
    ``hinv`` negative and wrong-sign ``u = q·hinv``.  That treats a symptom: the
    scheme is required to be positive BY CONSTRUCTION, and "by construction"
    excludes clamping and excludes reshaping denominators.  A cell reaching
    h < 0 means the scheme is not positivity-preserving, and THAT is the bug —
    so ``hinv`` now goes negative and the failure is loud instead of masked.
    Only KP's DENOMINATOR scale ``Max(h, eps)`` remains: the single sanctioned
    exception, whose numerator still vanishes at h = 0.
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
    """Return ``rewrite(expr)`` replacing every ``hs**(-n)`` (``n > 0``) by
    ``inv**n``, where ``inv`` is the replacement for ``1/hs``.

    ``inv`` may be a bare Symbol (the ``hinv`` AUX path) or a whole expression
    (the DIRECT path, where the Kurganov–Petrova form is substituted inline).
    Factored out of :func:`regularize_pow` so the legacy op and the study ops
    (:func:`regularize_depth_direct` / :func:`regularize_depth_aux`) share ONE
    definition of what "rewrite the reciprocal" means and can only differ in
    WHERE they apply it.

    The rewrite factors the maximal multiplicative power of ``hs`` out of a
    negative-power base — ``hs**(-n)``, ``(hs**p·f)**(-n)`` AND the expanded
    ``(Σ hs·…)**(-n)`` Add.  Transcendental nodes are returned WHOLE (never
    descended), so an ``hs`` that lives only inside a transcendental argument
    (``log(z/hs)``) is never rewritten — only a genuine multiplicative ``hs``
    factor becomes ``inv``.

    ``inv`` is NOT re-scanned: the KP form itself contains the negative power
    ``(h⁴ + max(h, eps)⁴)**(-1/2)``, whose base carries no multiplicative ``h``,
    so descending into it would be a no-op anyway — but not descending makes
    that a structural guarantee rather than a coincidence.
    """
    def _h_multiplicity(base):
        """Multiplicity of ``hs`` as a genuine MULTIPLICATIVE factor of
        ``base`` — ``p`` such that ``base / hs**p`` is ``hs``-free.  Returns 0
        when ``hs`` is not a factor, e.g. it only appears inside a
        transcendental arg (``log(z/hs)``) or in a sum with no common ``hs``
        (``hs + 1``): those must be left untouched.  ``sp.factor`` pulls the
        common ``hs`` out of an EXPANDED denominator (the RoughWall
        ``-h·(…)²`` form) so its multiplicity is readable from
        ``as_powers_dict``."""
        base = sp.sympify(base)
        if not base.has(hs):
            return 0
        e = sp.factor(base).as_powers_dict().get(hs, sp.S.Zero)
        return int(e) if e.is_Integer and e > 0 else 0

    def _rw(e):
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
    """Return the state Symbol named ``field`` (accepts the Symbol itself)."""
    name = str(field)
    for s in sm.state:
        if str(s) == name:
            return s
    raise KeyError(
        f"regularize_pow: '{name}' is not a state variable of this "
        f"SystemModel (state = {[str(s) for s in sm.state]}).")


def _resolve_aux_symbol(sm, name: str) -> sp.Symbol:
    for s in sm.aux_state:
        if str(s) == str(name):
            return s
    raise KeyError(
        f"regularize_pow: aux '{name}' is not in aux_state "
        f"({[str(s) for s in sm.aux_state]}); register it first with "
        f"register_aux('{name}', …).")


# ── register_aux ───────────────────────────────────────────────────────────

def register_aux(name: str, formula, **assumptions):
    """Operation: add an algebraic auxiliary variable ``name = formula``.

    Adds the Symbol ``name`` to :attr:`SystemModel.aux_state` and AUTO-AUGMENTS
    :attr:`SystemModel.update_aux_variables` with the row ``name = formula``
    (``formula`` a sympy expression in the state / parameters / existing aux,
    e.g. a KP-desingularized ``1/h``).  The resulting ``update_aux_variables``
    is the FULL-length aux vector — identity (passthrough) on every pre-existing
    row, ``formula`` on the new row — so the per-cell solver leg (which writes
    the lowered update as a prefix of ``Qaux``) populates ``name`` at its true
    row each step without clobbering, or being clobbered by, the other aux
    (the derivative-aux LSQ walk re-fills its own rows afterwards).

    Derived operators are refreshed so the freshly sized aux vector is reflected
    in ``source_jacobian_wrt_aux_variables``.

    ``**assumptions`` are forwarded to the ``Symbol`` (default ``real=True``);
    e.g. ``register_aux("hinv", expr, positive=True)``.
    """
    if not assumptions:
        assumptions = {"real": True}

    def _op(sm):
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


def regularize_pow(field: Union[str, sp.Symbol], aux_name: str):
    """Operation: replace every ``field**(-n)`` (``n > 0``) by ``aux_name**n``.

    The conservative change of variables ``u = q/h`` leaves raw ``h**(-1)`` /
    ``h**(-2)`` in the flux / source / NCP; near a dry front those blow up.
    This sweeps every (negative-integer) power of the state ``field`` in every
    PHYSICAL operator and rewrites it in terms of a desingularized auxiliary
    ``aux_name`` (typically registered via :func:`register_aux`), so e.g.
    ``q/h`` becomes ``q·hinv``.

    FREEZE-THEN-SUBSTITUTE — why the derived operators are NOT recomputed.
    ``aux_name`` (``hinv``) is an INDEPENDENT symbol, so once the flux holds
    ``q²·hinv`` a fresh ``∂/∂h`` is ``0`` — the ``−u²`` wavespeed term is lost
    and every flux-Jacobian wavespeed is corrupted (the wet 40% error).  And
    chain-ruling it back via ``∂hinv/∂h = ∂kp_hinv/∂h`` injects a
    ``Heaviside``/``DiracDelta`` mess into every Jacobian entry — the opposite
    of factored codegen.  So instead we:

    1. FORCE-MATERIALIZE the lazy derived operators from the CLEAN
       (pre-substitution) primaries — ``quasilinear_matrix`` and (when present)
       ``eigenvalues`` — so they hold the exact ``∂(q²/h)/∂h = −q²/h²``.  The
       source jacobians are already eager (channeled / refreshed by
       :func:`register_aux`).
    2. SUBSTITUTE ``field**(-n) → aux_name**n`` into the physical operators
       AND those now-frozen derived operators, so ``−q²/h² → −q²·hinv²`` (a few
       ops, no Heaviside) while ``hinv`` stays symbolic.

    No ``refresh_derived_operators`` is called (that would recompute from the
    substituted primaries and re-drop the ``−u²`` term).
    """
    def _op(sm):
        hs = _resolve_state_symbol(sm, field)
        aux = _resolve_aux_symbol(sm, aux_name)

        # ``inv`` is the replacement for ``1/h``: the desingularized ``aux``
        # (``hinv``) for the flux/source/NCP operators, but the plain floored
        # inverse ``1/Max(1e-14, h)`` for the EIGENVALUE / wave-speed (REQ-82 —
        # see the internal ``eps_h`` at the use site).  See :func:`_pow_rewrite` for the
        # rewrite rules themselves.
        def _rw(e, inv):
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
        # eps_h — the EIGENVALUE-SLOT wave-speed floor.  Deliberately an
        # INTERNAL constant of this operator (user ruling 2026-07-19, task
        # cid=5): not a parameter, not exposed, not tunable.  Every PHYSICAL
        # operator uses only the wet/dry KP reciprocal (u -> 0 as h -> 0);
        # the spectrum needs the OPPOSITE guarantee — large-but-finite at a
        # wet/dry front — because a KP-damped celerity starves the Riemann
        # dissipation and drains cells negative (measured: lake h_min
        # -2.5e-15 with this floor vs -1.6e-2 with KP on the eigenvalues).
        # A fully-dry domain's dt is governed by NSM dt_max (REQ-190), so
        # this floor never sets a global timestep.
        eps_h = 1e-14
        inv_floor = sp.S.One / sp.Max(sp.Float(eps_h), hs)
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
    """Resolve the wet/dry threshold ``eps`` for an operation.

    ``override`` (when not ``None``) wins; otherwise the model's REQ-48
    ``wet_dry_eps`` parameter is used when present, falling back to
    :data:`_DEFAULT_WET_DRY_EPS`."""
    if override is not None:
        return sp.sympify(override)
    params = getattr(sm, "parameters", None)
    if params is not None and params.contains("wet_dry_eps"):
        return params.wet_dry_eps
    return sp.Float(_DEFAULT_WET_DRY_EPS)


def _depth_state(sm, what):
    """Return the state Symbol named ``"h"`` or raise a descriptive error."""
    h = next((s for s in sm.state if str(s) == "h"), None)
    if h is None:
        raise ValueError(
            f"{what}: no depth state 'h' found "
            f"(state = {[str(s) for s in sm.state]}).")
    return h


def _min_h_power(expr, h):
    """Lowest power of ``h`` in ``expr`` as a rational function of ``h``.

    Returns ``mindeg(numerator) - mindeg(denominator)``; negative means the
    entry carries a ``1/h**n``.  ``None`` when it cannot be decided (the caller
    then leaves the row alone rather than guessing)."""
    e = sp.sympify(expr)
    if e == 0 or h not in e.free_symbols:
        return 0
    try:
        num, den = sp.fraction(sp.cancel(sp.together(e)))

        def _mindeg(p):
            p = sp.expand(p)
            if p == 0:
                return 0
            return min(m[0] for m in sp.Poly(p, h).monoms())

        return _mindeg(num) - _mindeg(den)
    except (sp.PolynomialError, TypeError, ValueError):
        return None


def equilibrate_constraint_rows(mode=True):
    """Operation: ROW-EQUILIBRATE the DAE constraint rows in ``h``.

    Multiplies each ALGEBRAIC row (zero ``mass_matrix`` row — the Lagrange-
    multiplier constraints of a DAE, e.g. the divergence constraints whose
    multipliers are the non-hydrostatic pressure modes) by the **minimal**
    power of ``h`` that clears every negative power of ``h`` from that row.

    Why this is safe: scaling a CONSTRAINT row by a positive factor leaves the
    constraint manifold — and hence the solution — unchanged wherever ``h > 0``.
    It changes only the CONDITIONING of the elliptic block.  The layer-averaged
    non-hydrostatic literature does the same thing with a blanket ``h**2``
    (Escalante–Fernández-Nieto–Morales de Luna–Castro 2019, their Eq. 19: the
    incompressibility constraints are "multiplied by ``h**2`` … in terms of
    discharges", so no ``1/h`` remains and the pressure matrix stays invertible
    as ``h -> 0``).  Taking the MINIMAL per-row power instead avoids the
    over-scaling a blanket factor introduces on rows that only needed ``h**1``.

    Applied at the SystemModel level ON PURPOSE: by this stage ``d_x(q/h)`` has
    already been resolved into aux symbols plus an EXPLICIT ``h**-n`` factor, so
    a row multiply genuinely reaches it.  At derivation time the ``1/h`` is
    still buried inside an unevaluated ``Derivative`` and an external multiply
    is a no-op (measured).

    ``h`` is found generically as the state named ``"h"``; rows are identified
    structurally by a zero mass-matrix row — no per-model knowledge.
    """
    def _op(sm):
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
    """Operation: apply the Kurganov–Petrova ``1/h`` desingularization (REQ-67).

    Registers a named ``hinv`` aux carrying the KP inverse depth
    ``√2·h/√(h⁴ + max(h, eps)⁴)`` and rewrites every ``h**(-n)`` in the
    operators to ``hinv**n`` so the flux/source use the desingularized inverse
    depth instead of a bare ``1/h`` — the reusable core form of the per-case
    ``_register_hinv_aux`` the Malpasset SME model used to hand-roll.

    ``mode`` is ``True`` or ``"kp"`` (only the KP variant exists today);
    anything falsy makes the op a no-op.  ``h`` is found generically as the
    state named ``"h"``; ``eps`` is the model's ``wet_dry_eps`` parameter when
    present, else :data:`_DEFAULT_WET_DRY_EPS`."""
    def _op(sm):
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
    """Operation: derive the closed-form spectrum of the normal-projected
    quasilinear matrix and carry it as ``sm.eigenvalues`` (``eigenvalue_mode``
    becomes ``"symbolic"``).

    A system model built from a derivation carries no spectrum: every
    in-process solver takes the numerical eigenvalues of the quasilinear
    matrix in that case, but a printer that emits a wave-speed kernel needs
    the closed form and refuses otherwise. This is where a caller pays for
    the symbolic derivation — cheap for a shallow water system, expensive
    for a high moment level, which is why it is opt-in."""
    def _op(sm):
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
    """Resolve the REGULARIZATION epsilon (never the wet/dry threshold).

    ``override`` wins; else a model-declared ``regularization_eps`` parameter
    (so a case can keep it run-time tunable); else
    :data:`_DEFAULT_REGULARIZATION_EPS`.

    ``wet_dry_eps`` is deliberately NOT consulted.  It is a wet/dry
    state-CLASSIFICATION threshold at a different scale (1e-2 m for SWE), and
    reusing it as a regularization scale is what put ``h/(h + 1e-2)`` on the
    celerity — a 50% loss of ``√(g·h)`` at a 1 cm depth.  Keeping the two
    quantities separate is the whole point of this knob; see
    :func:`_wet_dry_eps` for the other one."""
    if override is not None:
        return sp.sympify(override)
    params = getattr(sm, "parameters", None)
    if params is not None and params.contains("regularization_eps"):
        return params.regularization_eps
    return sp.Float(_DEFAULT_REGULARIZATION_EPS)


def _claim_depth_regularization(sm, tag: str) -> None:
    """Record that ``tag`` regularized the depth on ``sm``; raise if another
    path (or the same one twice) already did.

    Mutual exclusion is enforced on the SYSTEM, not by convention in the
    caller, so composing the two ops by hand — ``sm.apply(direct);
    sm.apply(aux)`` — fails loudly instead of silently producing a system whose
    operators disagree about what ``1/h`` means."""
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
    """Translate the ``eigenvalues=`` axis into a ``map_operator_slots``
    exclusion tuple.

    ``"regularize"`` — the spectrum is swept like every other slot, so the
    reciprocal in ``λ`` is the desingularized one.  For SWE (with the face
    normal normalised) the celerity ``√(g·h)`` carries no reciprocal at all, so
    only the advective ``u·n`` is affected.  For SME it is NOT separable —
    ``λ = hinv·(n0·q_0 ± n0·√(g·h³ + q_1²))`` — so the regularization multiplies
    the celerity too.

    ``"exclude"`` — the spectrum keeps the EXACT ``1/h``.  This is the honest
    form of the historical carve-out.  The carve-out as it shipped used a
    floored inverse ``1/max(1e-14, h)``; that is a floor on the depth, which is
    forbidden, and it is also the worst of the options at a draining face
    (``hu = 1e-3`` at ``h → 0`` gives ``|u| = 1e11``).  Excluding the slot keeps
    the wave speed exact where the model is wet and leaves the dry limit to the
    (independent) eigenvalue guard axis, instead of trading a singularity for a
    floor.

    Which of the two is right is exactly what the study measures; neither is
    baked in here."""
    if eigenvalues not in _EIGENVALUE_TREATMENTS:
        raise ValueError(
            f"eigenvalues={eigenvalues!r} is not a legal treatment; expected "
            f"one of {list(_EIGENVALUE_TREATMENTS)}.")
    return (("eigenvalues", "eigenvalues_cfl")
            if eigenvalues == "exclude" else ())


def _assert_no_forward_aux_reference(sm, aux, aux_row: int, name: str) -> None:
    """Fail loudly if an aux row EVALUATED BEFORE ``aux`` now reads it.

    ``register_aux`` APPENDS the reciprocal, so its row is last and the per-cell
    solver — which writes ``update_aux_variables`` top-down as a prefix of
    ``Qaux`` — computes every other row first.  That is fine as long as no
    earlier row references it.  But the sweep rewrites ``update_aux_variables``
    like every other slot, so a model whose aux row genuinely carries a
    ``1/h`` BEFORE the reciprocal's row would come out reading a stale
    (previous-step) value: numerically plausible, silently wrong, and
    indistinguishable from a physics result in a parameter study.

    No model in the tree does this today (the derivative aux are identity
    passthroughs filled by the LSQ walk), so this never fires — it exists so
    that if one ever does, the study stops instead of quietly measuring
    garbage.  The fix would be to order the aux vector, not to drop the sweep.
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
    """Operation: substitute the KP reciprocal INLINE wherever ``1/field**n``
    appears — the DIRECT path of the REQ-194 depth-law study.

    Every ``field**(-n)`` in every slot of :data:`OPERATOR_SLOTS` becomes
    ``kp_hinv(field, eps)**n``.  No aux row is registered and no state is added:
    the regularized reciprocal is written out in place.

    ``eps`` is the REGULARIZATION scale (see :func:`_regularization_eps`),
    default :data:`_DEFAULT_REGULARIZATION_EPS`.  It is NOT the wet/dry
    threshold — do not pass ``wet_dry_eps`` here.

    ``eigenvalues`` selects the spectrum's treatment: ``"regularize"`` (swept
    like every other slot) or ``"exclude"`` (keeps the exact ``1/h``).  See
    :func:`_eigenvalue_exclusion`.

    ``exclude`` drops further slots from the sweep; the default is empty on
    purpose (opt OUT, never opt in).  A name that is not in
    :data:`OPERATOR_SLOTS` raises.

    FREEZE-THEN-SUBSTITUTE.  :func:`map_operator_slots` materializes the lazy
    ``quasilinear_matrix`` from the CLEAN primaries before the first write, so
    the flux Jacobian holds the exact ``∂(q²/h)/∂h = −q²/h²`` and that term
    becomes ``−q²·kp²`` rather than being recomputed (and lost) from an
    already-substituted flux.

    Mutually exclusive with :func:`regularize_depth_aux` — see
    :func:`_claim_depth_regularization`.
    """
    # Validated HERE, not inside ``_op``: the axes are a construction-time
    # choice, so a bad value must fail when the op is built, before it has
    # claimed the depth regularization on a system it then fails to rewrite.
    excl = _eigenvalue_exclusion(eigenvalues) + tuple(exclude)

    def _op(sm):
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
    """Operation: promote ``1/field**n`` to an aux row carrying the KP
    reciprocal — the AUX path of the REQ-194 depth-law study.

    Registers the aux ``name`` whose ``update_aux_variables`` row IS
    ``kp_hinv(field, eps)``, then sweeps every slot of :data:`OPERATOR_SLOTS`
    rewriting ``field**(-n) → name**n``.  The operators therefore carry a single
    opaque symbol and the regularization lives in exactly one row.

    Same ``eps`` / ``eigenvalues`` / ``exclude`` axes as
    :func:`regularize_depth_direct`, and the same freeze-then-substitute
    discipline — which matters MORE here: ``name`` is an INDEPENDENT symbol, so
    a recomputed ``∂/∂h`` of a flux already holding ``q²·hinv`` would be ``0``
    and the ``−u²`` wave-speed term would vanish outright.  Chain-ruling it back
    through ``∂hinv/∂h`` would instead inject ``Heaviside``/``DiracDelta`` into
    every Jacobian entry.  So the derived operators are frozen from the clean
    primaries and substituted, never refreshed.

    The only carve-out is a ROW-level self-reference guard rather than a slot
    exclusion: the ``update_aux_variables`` row that DEFINES ``name`` is left
    alone, since rewriting it would produce the tautology ``hinv = hinv`` and
    lose the formula entirely.

    When the model already declares the aux (e.g. MalpassetSWE), it must carry a
    REAL formula — a bare identity passthrough raises rather than letting the
    swept operators read an uncomputed slot.

    Mutually exclusive with :func:`regularize_depth_direct`.
    """
    # Validated at construction time — see :func:`regularize_depth_direct`.
    excl = _eigenvalue_exclusion(eigenvalues) + tuple(exclude)

    def _op(sm):
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
    """Operation: floor the source's singular dependence on positive state.

    For each ``base**p`` in a source row where ``base`` involves a symbol in
    ``sm.positive_state`` and ``p`` is negative (denominator) or non-integer
    (root), replace ``base`` by ``√(base² + floor²)``.  Recomputes the source
    jacobians from the regularised source.  No-op if the model declares no
    ``positive_state`` (this keeps ν_t=C_μk²/ε finite and the wall-function
    √k(0) real WITHOUT touching the symbolic derivation)."""
    def _op(sm):
        protected = set(getattr(sm, "positive_state", []) or [])
        if not protected:
            return
        f2 = sp.Float(floor) ** 2

        def safe(expr):
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
    """Predicate for ``expr.replace``: a fractional power of ``h`` whose base
    can go negative and is NOT already floored by ``Max(., 0)``.

    The ``Max(., 0)`` exclusion makes the guard idempotent, so applying it
    twice (``[guard, gate]``) equals applying it once (``gate`` alone)."""
    def _q(x):
        return (isinstance(x, sp.Pow) and x.exp.is_number
                and not x.exp.is_integer and x.base.has(h)
                and not (isinstance(x.base, sp.Max)
                         and sp.S.Zero in x.base.args))
    return _q


def _guard_eigenvalue_expr(e, h):
    """Return ``e`` with every fractional power of ``h`` (e.g. ``sqrt(h**5)``
    from the desingularized wave speed) wrapped in ``Max(., 0)``.  Idempotent;
    a no-op for the physical ``h >= 0`` case, so wet wave speeds are unchanged."""
    return sp.sympify(e).replace(
        _guard_frac_pow_of_h(h),
        lambda x: sp.Pow(sp.Max(x.base, sp.S.Zero), x.exp))


def _flat_eigenvalues(sm, what):
    """Return ``(ev, [scalar entries])`` for ``sm.eigenvalues`` or ``(None,
    None)`` when the model emits numerical wave speeds.

    ``sp.flatten`` yields SCALARS for both the ZArray and the sympy-Matrix forms
    ``sm.eigenvalues`` can take (the WB/audusse path stores a Matrix).  Operating
    on scalars is essential: a ``(n, 1)`` column iterates row-wise into 1-element
    sub-arrays, which the numpy backend would lower to inhomogeneous-rank
    branches (``[0]`` shape ``(1,)`` vs ``[e(h)]`` shape ``(1, n)``) and raise."""
    ev = getattr(sm, "eigenvalues", None)
    if ev is None:
        return None, None
    h = _depth_state(sm, what)
    return ev, [sp.sympify(e) for e in sp.flatten(ev)]


def guard_eigenvalue_powers():
    """Operation: guard every fractional power of the depth ``h`` in
    ``sm.eigenvalues`` with ``Max(., 0)`` — the ALWAYS-SAFE half of the wet/dry
    eigenvalue treatment, with NO dry zeroing.

    ``sqrt(h**5)``/``h**(5/2)`` → ``sqrt(Max(h**5, 0))``/``Max(h, 0)**(5/2)`` so
    the wave-speed expression stays real at a transient ``h < 0`` (roundoff at
    the wet/dry front).  ``Max(., 0)`` is a no-op for the physical ``h >= 0``
    case, so wet wave speeds are unchanged, and the op is idempotent.

    ``h`` is the state named ``"h"``.  No-op when ``sm.eigenvalues is None`` (the
    model emits numerical wave speeds, gated by the solver instead).

    This is A of the (A, B) split (REQ-181); B is
    :func:`gate_eigenvalues_dry`.  Neither is an NSM default — opt in via
    ``extra_operations``."""
    def _op(sm):
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
    """``True`` when ``e`` is ALREADY ``conditional(h > eps, ·, 0)`` — the exact
    wrapper :func:`gate_eigenvalues_dry` applies, so re-gating it is a no-op."""
    return (isinstance(e, sp.Function) and e.func.__name__ == "conditional"
            and len(e.args) == 3 and e.args[0] == (h > eps)
            and e.args[2] == sp.S.Zero)


def gate_eigenvalues_dry(eps=None):
    """Operation: publish a TIMESTEP-ONLY spectrum gated to 0 in dry cells
    (``h < eps``) into ``sm.eigenvalues_cfl``, leaving ``sm.eigenvalues``
    physical.

    Wraps every entry ``e`` of ``sm.eigenvalues`` in ``conditional(h > eps, e,
    0)`` so the dt/CFL reduction sees no spurious dry-cell wave speed — the
    reusable core form of the Malpasset SME ``ev_gate``.  ``h`` is the state
    named ``"h"``; ``eps`` defaults to the model's ``wet_dry_eps`` parameter
    when present, else :data:`_DEFAULT_WET_DRY_EPS` (pass ``eps=`` to
    override).

    **It writes a SEPARATE slot, and that split is the whole point** (cid=209).
    Until 2026-08-07 the gate overwrote ``sm.eigenvalues`` in place, and the
    spectrum has TWO consumers that want opposite things at a dry cell:

    * the dt/CFL estimate (``solver_numpy.get_compute_max_abs_eigenvalue``) —
      wants dry cells suppressed, which is the single job this op was added for;
    * the Rusanov face dissipation
      (``numerics.riemann_solvers.local_max_abs_eigenvalue``) — needs the TRUE
      ``sqrt(g h)``, because ``s_max = max(|λ(qL)|, |λ(qR)|)`` IS the artificial
      viscosity coefficient of the first-order positivity argument.

    Zeroing the shared array ran every face whose two cells sit below
    ``wet_dry_eps`` as a pure CENTRAL scheme.  MEASURED on a 16×16 radial dry
    dam break (H=2, CFL 0.45, order 2, positivity=mood, only this op differing):
    gated-in-place, the MOOD cascade EXHAUSTED at step 1 with four cells still
    at ``h = -2.1e-07`` and every troubled stencil already demoted to order 1;
    ungated, the same run marched 21 steps to ``t_end`` with worst
    ``h = 0.0`` exactly.  Sweeping CFL 0.45/0.30/0.25/0.167/0.10 did not move
    it, so it was never a timestep problem.  Order 1 was clean either way (no
    slopes to demote, both face states cell averages).

    So ``eigenvalues`` stays the physical spectrum every dissipation, Roe and
    hyperbolicity consumer reads, and ``eigenvalues_cfl`` carries the gated
    copy for the timestep alone.  A consumer that finds ``eigenvalues_cfl is
    None`` falls back to ``eigenvalues`` — i.e. no gate opted in.

    The dry gate MUST carry the power guard (:func:`guard_eigenvalue_powers`):
    a backend ``conditional`` lowered branchless (``mask*a + (1-mask)*b``)
    computes BOTH arms, and ``NaN*0 = NaN`` would leak a ``sqrt(negative)`` past
    the ``h > eps`` gate (REQ-74).  So this op applies the guard to each wet
    branch before the ``conditional``.  The guard is idempotent, so the explicit
    composition ``[guard_eigenvalue_powers(), gate_eigenvalues_dry()]`` produces
    the same gated expression — with the difference that the explicit ``power``
    half acts on ``sm.eigenvalues`` (where it belongs: guarding a fractional
    power of ``h`` is always safe and changes nothing for ``h >= 0``) while the
    ``gate`` half only ever writes ``sm.eigenvalues_cfl``.

    This is B of the (A, B) split (REQ-181); A is
    :func:`guard_eigenvalue_powers`.  Neither is an NSM default — opt in via
    ``extra_operations``.  No-op when ``sm.eigenvalues is None`` (the model
    emits numerical wave speeds, gated by the solver instead)."""
    def _op(sm):
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
    """Operation: shift the base of every fractional power by ``eps**2`` —
    ``b**r -> (b + eps**2)**r`` for every ``sp.Pow(b, r)`` with ``r`` a
    non-integer Rational and ``b`` non-numeric — so the operators stay
    differentiable at ``b = 0``.

    WHY.  Forward and reverse AD through the time loop differentiate every
    operator.  ``d/dx x**r = r*x**(r-1)`` is ``inf`` at ``x = 0`` for ``r < 1``,
    and the chain rule multiplies it by the incoming tangent; where that tangent
    is exactly zero the product is ``0*inf = NaN``, which then poisons the whole
    gradient.  The standard trigger is a velocity norm
    ``sqrt(q_x**2 + q_y**2)`` in a friction source (Manning, Chezy) evaluated
    at rest: the value is a harmless 0, the derivative is NaN.  With the shift,
    ``d/dx (x + eps**2)**r`` is finite at ``x = 0`` and the NaN never forms.

    VALUE CHANGE.  ``(b + eps**2)**r - b**r = O(eps**2)`` wherever ``b`` is
    bounded away from 0 (relative change ``r*eps**2/b``); at ``b = 0`` the value
    becomes ``eps**(2r)`` (e.g. ``eps`` for a ``sqrt``).  Pick ``eps`` far below
    the scales of the problem (``1e-6`` is a reasonable default for velocities
    and depths in SI units).

    USAGE.  Opt-in, never a default — the model stays analytic and the
    regularization lives only in the numerical system::

        NumericalSystemModel.from_system_model(
            sm, extra_operations=[regularize_fractional_powers(1e-6)])

    Coverage is the :data:`OPERATOR_SLOTS` registry via
    :func:`map_operator_slots`; ``exclude`` names slots to leave untouched.

    IDEMPOTENT: a power whose base is an ``Add`` already carrying the exact
    ``Float(eps)**2`` term is skipped, so applying the op twice equals applying
    it once.  (A base whose own numeric constant sympy merges with ``eps**2``
    on construction loses that marker — such bases do not occur in the shipped
    models, whose fractional-power bases are sums of squares and depths.)

    ``eps`` is required and must be ``> 0`` (``ValueError`` otherwise).
    """
    if not eps > 0:
        raise ValueError(
            f"regularize_fractional_powers: eps must be > 0, got {eps!r}")
    e2 = sp.Float(eps) ** 2

    def _is_fractional_pow(x):
        return (isinstance(x, sp.Pow) and x.exp.is_Rational
                and not x.exp.is_integer and not x.base.is_number
                and not (isinstance(x.base, sp.Add) and e2 in x.base.args))

    def _rewrite(expr):
        return sp.sympify(expr).replace(
            _is_fractional_pow, lambda x: sp.Pow(x.base + e2, x.exp))

    def _op(sm):
        _op.coverage = map_operator_slots(sm, _rewrite, exclude=exclude)

    _op.name = "regularize_fractional_powers"
    _op.description = f"b**r -> (b + eps**2)**r for fractional r (eps={eps})"
    _op.coverage = {}
    return _op
