"""Per-model SystemModel builders (REQ-143).

A :class:`~zoomy_core.model.basemodel.Model` must NOT know that
:class:`~zoomy_core.systemmodel.system_model.SystemModel` exists.  The single
public entry point for turning a model instance into its runtime system is
:meth:`SystemModel.from_model`; when handed a production model it delegates
HERE, dispatching on the model's plain-string ``_system_model_kind`` class
attribute to the matching builder below.

Each builder holds the EXACT body that the model's old ``system_model``
@property used to run (``self`` → ``model``): Q selection, the manual
hydrostatic-pressure tag/un-tag, ``canonical_source=model``, the β-HSWME
spectrum registration, the positivity flags, and the boundary-condition
resolution.  The builders call the LOW-LEVEL
``SystemModel.from_model(model.derivation, Q=...)`` path (the declarative
model + Q signature), which is untouched.
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.coords import t, x, y
from zoomy_core.systemmodel.system_model import SystemModel
from zoomy_core.model.derivation.system_extract import HydrostaticPressure
from zoomy_core.model.boundary_conditions import resolve_and_attach, Extrapolation
from zoomy_core.systemmodel.roles import declare_free_surface_roles


def _free_surface(sm, depth, bed, kind):
    """Every builder below is a FREE-SURFACE family: declare its roles from
    the fields the builder itself put into Q (depth / bed Functions, the
    model's gravity parameter ``g``), momentum derived from the model's own
    ``interpolate_to_3d`` lift.  Generic code reads these roles only."""
    def _name(f):
        return getattr(getattr(f, "func", None), "__name__", None) or str(f)
    declare_free_surface_roles(
        sm, depth=_name(depth), bed=None if bed is None else _name(bed),
        gravity="g" if sm.parameters.contains("g") else None,
        description=f"{kind} free-surface roles")
    return sm


# ── SWE ─────────────────────────────────────────────────────────────────────
def build_swe(model) -> SystemModel:
    # ``SystemModel.from_model`` parses the model's function groups AND attaches
    # its ``_coupling_bcs`` on raw promotion (REQ-87), so every backend adapter /
    # FVM solver that builds from the model inherits it.
    # Roles: ``SWE.declare_system_roles`` runs inside ``_from_model_impl``.
    return SystemModel._from_model_impl(model)


# ── SME ─────────────────────────────────────────────────────────────────────
def build_sme(model) -> SystemModel:
    m = model.derivation
    dim = int(model.dimension)
    horiz = (x,) if dim == 2 else (x, y)
    qs = list(m.explicit_state())
    # b evolves via the (trivial) bottom equation ∂_t b = 0, so it is
    # already an explicit unknown; prepend only if absent.
    bed = sp.Function("b", real=True)(t, *horiz)
    if bed not in qs:
        qs = [bed, *qs]
    # Manual hydrostatic-pressure tag (one-liner): mark g·h²/2 so the
    # structural extractor routes it to hydrostatic_pressure (well-balanced
    # reconstruction) instead of the conservative flux.  Recomputed from m.
    h = sp.Function("h", positive=True)(t, *horiz)
    pf = m.parameters.g * h ** 2 / 2
    m.apply({pf: HydrostaticPressure(pf)})
    sm = SystemModel.from_model(m, Q=qs, canonical_source=model)
    m.apply({HydrostaticPressure(pf): pf})   # un-tag: leave derivation clean
    # Closed-form spectrum only on request (``symbolic_spectrum``).  It is a
    # symbolic eigen-decomposition of the truncated quasilinear matrix, and it
    # dominated the build: 67 s of a 123 s SME(level=2, dimension=3), 61 s of
    # that inside sympy's heuristic integer polynomial GCD from 16 calls, and
    # ~9x worse per level -- so level 4 in 2-D ran for hours and blocked
    # postprocessing that never looks at ``eigenvalues`` at all.  Left off, the
    # attribute stays None and the runtime takes NUMERICAL eigenvalues from the
    # quasilinear matrix, which riemann_solvers / solver_numpy already treat as
    # a first-class path (it is what hand-built models like VAM always used).
    if getattr(model, "symbolic_spectrum", False):
        model._register_hswme_spectrum(sm)
    _free_surface(sm, h, bed, "SME")
    resolve_and_attach(sm, model.boundary_conditions,
                       aux_bcs=model.aux_boundary_conditions)
    return sm


# ── VAM ─────────────────────────────────────────────────────────────────────
def build_vam(model) -> SystemModel:
    m = model.derivation
    Nu = int(model.level)
    # Pressure modes carry the SAME horizontal dependence as the rest of the
    # state: dim=2 → P(j,t,x); dim=3 → P(j,t,x,y).
    horiz = (x,) if int(model.dimension) == 2 else (x, y)
    P_modes = [m.functions.P.head(j, t, *horiz) for j in range(Nu + 1)]
    qs = list(m.explicit_state())
    bed = sp.Function("b", real=True)(t, *horiz)
    if bed not in qs:
        qs = [bed, *qs]
    # Manual hydrostatic-pressure tag (one-liner): mark g·h²/2 → pressure.
    h = sp.Function("h", positive=True)(t, *horiz)
    pf = m.parameters.g * h ** 2 / 2
    m.apply({pf: HydrostaticPressure(pf)})
    sm = SystemModel.from_model(m, Q=[*qs, *P_modes], canonical_source=model)
    m.apply({HydrostaticPressure(pf): pf})   # un-tag: leave derivation clean
    _free_surface(sm, h, bed, "VAM")
    resolve_and_attach(sm, model.boundary_conditions,
                       aux_bcs=model.aux_boundary_conditions)
    return sm


# ── MLSWE ────────────────────────────────────────────────────────────────────
def build_mlswe(model) -> SystemModel:
    m = model.derivation
    sm = SystemModel.from_model(
        m, Q=[m.bed, m.ht, *m.q_l])
    _free_surface(sm, m.ht, m.bed, "MLSWE")
    resolve_and_attach(sm, model.boundary_conditions,
                       aux_bcs=model.aux_boundary_conditions)
    return sm


# ── MLSME ────────────────────────────────────────────────────────────────────
def build_mlsme(model) -> SystemModel:
    # Per-layer hydrostatic ``g·h_ℓ²/2`` is marked HydrostaticPressure at its
    # fold site in ``MLSME.derive_model`` (each h_ℓ is a DISTINCT function there,
    # so the mark is unambiguous — a builder-side ``.subs`` cannot separate the
    # bottom layer's g·l₁²h²/2 from the same monomial inside the top layer's
    # expanded g·(1−l₁)²h²/2).  Nothing to tag here.
    m = model.derivation
    sm = SystemModel.from_model(
        m, Q=[m.bed, m.ht, *m.q_flat], canonical_source=model)
    _free_surface(sm, m.ht, m.bed, "MLSME")
    resolve_and_attach(sm, model.boundary_conditions,
                       aux_bcs=model.aux_boundary_conditions)
    return sm


# ── MLVAM ────────────────────────────────────────────────────────────────────
def build_mlvam(model) -> SystemModel:
    # Per-layer hydrostatic marked at its fold site in ``MLVAM.derive_model``
    # (see ``build_mlsme`` — distinct h_ℓ makes the mark unambiguous).
    m = model.derivation
    sm = SystemModel.from_model(
        m, Q=[m.bed, m.ht, *m.q_flat, *m.r_flat, *m.P_flat],
        canonical_source=model)
    _free_surface(sm, m.ht, m.bed, "MLVAM")
    resolve_and_attach(sm, model.boundary_conditions,
                       aux_bcs=model.aux_boundary_conditions)
    return sm


# ── Sigma3D ──────────────────────────────────────────────────────────────────
def build_sigma3d(model) -> SystemModel:
    m = model.derivation
    qs = list(m.explicit_state())
    bed = sp.Function("b", real=True)(t, x)
    if bed not in qs:
        qs = [bed, *qs]
    # Manual hydrostatic-pressure tag (one-liner): route g·h²/2 → pressure.
    h = sp.Function("h", positive=True)(t, x)
    pf = m.parameters.g * h ** 2 / 2
    m.apply({pf: HydrostaticPressure(pf)})
    sm = SystemModel.from_model(m, Q=qs, canonical_source=model)
    m.apply({HydrostaticPressure(pf): pf})   # un-tag: leave derivation clean
    _free_surface(sm, h, bed, "Sigma3D")
    # CLOSURE → ζ-FACE-BC REDUCTION: the user states the bed/surface conditions
    # as physical closures; here the reduction emits them as ordinary solver BCs
    # on the ζ-faces (combined with the user's horizontal BCs).
    user = model.boundary_conditions
    if user is None:
        horiz_bcs = [Extrapolation(tag="left"), Extrapolation(tag="right")]
    elif isinstance(user, list):
        horiz_bcs = list(user)
    else:
        horiz_bcs = list(user.boundary_conditions_list)
    all_bcs = horiz_bcs + model._vertical_face_bcs(sm)
    resolve_and_attach(sm, all_bcs, aux_bcs=model.aux_boundary_conditions)
    return sm


# ── KESME ────────────────────────────────────────────────────────────────────
def build_kesme(model) -> SystemModel:
    # SME build + flag the k, ε moments as positivity-constrained so
    # ``NumericalSystemModel(regularization.positivity_floor>0)`` floors their
    # singular source dependence (ν_t=C_μk²/ε, wall √k).
    sm = build_sme(model)
    sm.positive_state = [s for s in sm.state
                         if str(s).startswith(("K_", "E_"))]
    return sm


# ── QRKESME ──────────────────────────────────────────────────────────────────
def build_qrkesme(model) -> SystemModel:
    # SME build + flag the sk, se moments as positivity-constrained so the floor
    # guards the 1/sk, 1/se, 1/sk² source prefactors as se (=√ε) passes near zero.
    sm = build_sme(model)
    sm.positive_state = [s for s in sm.state
                         if str(s).startswith(("SK_", "SE_"))]
    return sm


_BUILDERS = {
    "swe": build_swe,
    "sme": build_sme,
    "vam": build_vam,
    "mlswe": build_mlswe,
    "mlsme": build_mlsme,
    "mlvam": build_mlvam,
    "sigma3d": build_sigma3d,
    "kesme": build_kesme,
    "qrkesme": build_qrkesme,
}


def build_system_model(model) -> SystemModel:
    """Dispatch on ``model._system_model_kind`` to the matching builder.

    Cached BY DEFAULT (REQ-163): the built SystemModel (final lowered operator
    matrices) is stored across memory / user-dir / shipped-prebuilt tiers via
    :mod:`zoomy_core.systemmodel.sm_cache`; a hit skips the derivation AND the
    build entirely and returns a fresh, safely-mutable object.  Parameter
    VALUES are not part of the identity (free symbols end-to-end).  Disable
    with ``ZOOMY_DERIVATION_CACHE=0``; force ``ZOOMY_DERIVATION_REBUILD=1``.
    """
    kind = model._system_model_kind
    try:
        builder = _BUILDERS[kind]
    except KeyError:
        raise ValueError(
            f"unknown _system_model_kind {kind!r} on {type(model).__name__}")
    from zoomy_core.systemmodel import sm_cache
    key = sm_cache.cache_key(model, builder)
    cached = sm_cache.fetch(key)
    if cached is not None:
        _attach_runtime_data(model, cached)
        return cached
    sm = builder(model)
    sm_cache.store(key, sm)
    return sm


def _attach_runtime_data(model, sm) -> None:
    """Re-attach the INSTANCE's runtime data to a cache-fetched SystemModel.

    BCs/ICs embed parameter values and case choices — they are deliberately
    NOT part of the cache identity.  ``resolve_and_attach`` replaces any
    BCs the entry was built with; ICs are plain field assignments."""
    from zoomy_core.model.boundary_conditions import resolve_and_attach
    # Two BC homes (REQ-87): production models POP the constructor
    # ``boundary_conditions=`` into ``_coupling_bcs`` (their param is then a
    # fresh EMPTY default — do not let it shadow the real one); declarative
    # models keep theirs on the param.  Mirror from_model's promotion check:
    # only attach a container that actually HAS entries.
    def _nonempty(bc):
        if bc is None:
            return None
        entries = (list(bc) if isinstance(bc, list)
                   else getattr(bc, "boundary_conditions_list", None))
        return bc if entries else None

    bcs = (_nonempty(getattr(model, "_coupling_bcs", None))
           or _nonempty(getattr(model, "boundary_conditions", None)))
    if bcs is not None:
        resolve_and_attach(
            sm, bcs, aux_bcs=getattr(model, "aux_boundary_conditions", None))
    for field in ("initial_conditions", "aux_initial_conditions"):
        val = getattr(model, field, None)
        if val is not None:
            setattr(sm, field, val)
    # Parameter VALUES are runtime solver inputs (the whole point of keeping
    # them out of the cache identity) — the fetched entry carries the
    # BUILD-TIME numbers.  Update BY NAME inside the sm's own container: the
    # lambdified kernels index the parameter vector positionally, so the sm's
    # ordering/length must be preserved (never replace the container).
    pv = getattr(model, "parameter_values", None)
    tgt = getattr(sm, "parameter_values", None)
    if pv is not None and tgt is not None:
        names = pv.keys() if hasattr(pv, "keys") else ()
        for k in names:
            val = pv[k] if hasattr(pv, "__getitem__") else getattr(pv, k)
            try:
                if hasattr(tgt, "__setitem__"):
                    if not hasattr(tgt, "keys") or k in tgt.keys():
                        tgt[k] = val
                elif hasattr(tgt, k):
                    setattr(tgt, k, val)
            except (KeyError, TypeError):
                pass
