"""`NumericalSystemModel`: a `SystemModel` plus the numerics (Riemann solver, reconstruction,
diffusion scheme, regularization, kernels table) that the solver templates consume.
Pipeline: Model -> SystemModel -> NumericalSystemModel -> solver template -> backend runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Type

from zoomy_core.misc.options import Option
from zoomy_core.systemmodel.system_model import SystemModel


class Limiter(Option):
    """Slope limiter of the order-2 reconstruction."""
    #: unlimited LSQ gradients
    none = "none"
    minmod = "minmod"
    venkatakrishnan = "venkatakrishnan"
    barth_jespersen = "barth_jespersen"


class ReconstructedVariables(Option):
    """What the symbolic solvers reconstruct."""
    #: the model's reconstruction variables W(Q) (e.g. SWE / SME: eta = b + h, u = hinv q)
    model = "model"
    #: Q itself
    conservative = "conservative"


class BedReconstruction(Option):
    """The bed's face value with the model's free-surface map (bed b, eta = b + h)."""
    #: reconstruct h and eta, b_f = eta_f - h_f (Audusse et al. 2004: keeps the lake at rest)
    derived = "derived"
    #: reconstruct b and eta
    reconstructed = "reconstructed"

# Operator arrays a code printer lowers — the complete set of symbolic-PDE
# operators carried on a (Numerical)SystemModel.  ``quasilinear_matrix`` is a
# lazy cached property derived from ``flux``/``nonconservative_matrix``; it is
# guarded (read) but never assigned back to.
_OPERATOR_ATTRS = (
    "flux", "hydrostatic_pressure", "nonconservative_matrix", "source",
    "source_explicit", "mass_matrix", "eigenvalues", "eigenvalues_cfl",
    "source_jacobian_wrt_variables", "source_jacobian_wrt_aux_variables",
    "diffusion_matrix", "update_variables", "update_aux_variables",
    "state_update",
)


#: Sign predicates a state symbol may carry into the seam.  Mirrors
#: ``model.derivation.system_extract._SIGN_PREDICATES``.
_SIGN_PREDICATES = ("zero", "positive", "negative",
                    "nonnegative", "nonpositive", "nonzero")


def _carries_sign_assumption(sym) -> bool:
    """Whether the Symbol carries a declared sign predicate."""
    return any(getattr(sym, f"is_{k}", None) is True for k in _SIGN_PREDICATES)


def _numerical_symbol_map(sm) -> dict:
    """Map every state or aux symbol carrying an assumption to its bare twin; parameters are
    exempt.
    """
    import sympy as sp
    pool = list(getattr(sm, "state", []) or []) + \
        list(getattr(sm, "aux_state", []) or [])
    return {s: sp.Symbol(str(s), real=True)
            for s in pool
            if isinstance(s, sp.Symbol) and _carries_sign_assumption(s)}


def _rename(obj, mapping):
    """Substitute `mapping` through sympy content and plain containers."""
    import sympy as sp
    if obj is None or isinstance(obj, (str, bytes, bool, int, float, complex)):
        return obj
    if isinstance(obj, sp.Symbol):
        return mapping.get(obj, obj)
    if hasattr(obj, "xreplace"):
        try:
            return obj.xreplace(mapping)
        except Exception:
            pass
    if hasattr(obj, "subs"):
        try:
            return obj.subs(mapping)
        except Exception:
            return obj
    if isinstance(obj, dict):
        # KEYS TOO.  ``state_function_map`` is keyed BY the state symbols, so
        # renaming only values leaves the old symbol live as a key and a lookup
        # with the new symbol misses.
        return {_rename(k, mapping): _rename(v, mapping) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [_rename(v, mapping) for v in obj]
        return tuple(out) if isinstance(obj, tuple) else out
    return obj


def _rename_kernel(kernel, mapping) -> None:
    """Move a BC/IC kernel into the new symbol space, definition and `args.variables` binding
    together.
    """
    if kernel is None:
        return
    args = getattr(kernel, "args", None)
    if args is not None and hasattr(args, "contains"):
        for group in ("variables", "aux_variables"):
            if not args.contains(group):
                continue
            z = getattr(args, group)
            for key in list(z.keys()):
                val = getattr(z, key, None)
                if val in mapping:
                    setattr(z, key, mapping[val])
    definition = getattr(kernel, "definition", None)
    if definition is not None:
        try:
            kernel.definition = _rename(definition, mapping)
        except Exception:
            pass


def construct_numerical(sm):
    """Build the numerical twin of `sm`: a deep copy whose state and aux symbols carry no sign
    assumptions, so sympy cannot delete a numerical guard as a tautology.
    This is the SystemModel -> NumericalSystemModel boundary; the caller's `sm` keeps its
    assumptions.
    """
    import copy
    from zoomy_core.systemmodel.operations import OPERATOR_SLOTS, KERNEL_SLOTS

    nsm = copy.deepcopy(sm)          # measured: 11 ms SME, 59 ms VAM(1, dim=3)
    mapping = _numerical_symbol_map(nsm)
    if not mapping:
        return nsm

    for name in ("state", "aux_state", "state_function_map", "aux_registry") \
            + tuple(OPERATOR_SLOTS):
        if name in KERNEL_SLOTS or name not in nsm.__dict__:
            continue
        nsm.__dict__[name] = _rename(nsm.__dict__[name], mapping)

    for name in KERNEL_SLOTS:
        _rename_kernel(getattr(nsm, name, None), mapping)

    # Lazy cache derived from flux/NCP.  Set to None, never delete: the property
    # READS the attribute and a missing one raises AttributeError.
    if "_quasilinear_matrix" in nsm.__dict__:
        nsm.__dict__["_quasilinear_matrix"] = None
    return nsm


def assert_no_assumptions(sm) -> None:
    """Raise unless no state or aux symbol of the built object carries a sign predicate."""
    import sympy as sp
    from zoomy_core.systemmodel.operations import OPERATOR_SLOTS
    names = {str(s) for s in list(getattr(sm, "state", []) or [])
             + list(getattr(sm, "aux_state", []) or [])}
    bad = set()
    for attr in ("state", "aux_state", "state_function_map", "aux_registry") \
            + tuple(OPERATOR_SLOTS):
        val = getattr(sm, attr, None)
        if val is None:
            continue
        val = getattr(val, "definition", val)
        for s in _free_symbols_deep(val):
            if (isinstance(s, sp.Symbol) and str(s) in names
                    and _carries_sign_assumption(s)):
                bad.add((attr, str(s)))
    if bad:
        raise AssertionError(
            f"NSM invariant violated: {sorted(bad)} still carry assumptions. "
            "Such a symbol prints identically to its bare twin and compares "
            "unequal, so printers silently miss it (cid=87).")


def _free_symbols_deep(obj):
    """Free symbols of a container that sympy cannot introspect directly."""
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out |= _free_symbols_deep(k) | _free_symbols_deep(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= _free_symbols_deep(v)
    else:
        fs = getattr(obj, "free_symbols", None)
        if fs:
            out |= set(fs)
    return out



def numerics_fluctuations_are_zero(numerics) -> bool:
    """True only if the built `numerical_fluctuations` of `numerics` is proven structurally
    zero; unknown counts as nonzero.
    It reads the built face kernel, not the model's `nonconservative_matrix`.
    """
    import sympy as sp
    definition = getattr(
        getattr(getattr(numerics, "functions", None),
                "numerical_fluctuations", None),
        "definition", None)
    if definition is None:
        return False
    try:
        entries = list(sp.flatten(definition))
    except TypeError:
        entries = [definition]
    if not entries:
        return False
    return all(sp.sympify(e).is_zero is True for e in entries)


# ── implicit-stage mode identification ────────────────────────────────────
#
# The user's ruling: ``implicit_source`` and ``implicit_diffusion`` collapse
# into ONE implicit stage solve, so the scheme is a genuine IMEX-ARK rather
# than a Lie split — WITH a fast path for the case where the implicit operator
# is purely a LOCAL SOURCE (cell-local, no neighbour coupling), which is then
# N independent small nonlinear solves instead of one global system.  And:
# "the identification of which path we take should be happening in the Zoomy
# core, hence the numerical system model stage."
#
# So the identification lives here, next to ``fluctuations_are_zero``, and
# follows exactly that precedent: derived from the BUILT operators, never
# declared by hand, re-read rather than cached, and emitted to the printers.

#: No implicit stage at all — the implicit slots are structurally empty.
IMPLICIT_MODE_NONE = "none"
#: FAST PATH.  The implicit operator is a cell-local source: no entry reaches
#: a neighbour cell, so the stage is N independent small nonlinear solves.
IMPLICIT_MODE_LOCAL_SOURCE = "local_source"
#: ONE global implicit stage solve — an implicit diffusion / elliptic operator
#: couples neighbours, or the implicit source itself reads a non-local aux.
IMPLICIT_MODE_COUPLED = "coupled"

#: ``aux_registry`` kinds whose value is computed from a STENCIL over
#: neighbouring cells (LSQ gradients, limited gradients).  An implicit
#: operator that reads one of these is not cell-local, however "source"-shaped
#: it looks — this is the same distinction ``_linearize_source`` already makes
#: when it drops the ∂aux/∂Q channel for derivative aux ("a non-local
#: derivative cannot be inverted per cell").  Every other kind ("function", a
#: closure evaluated pointwise from the local state) IS local.
_NONLOCAL_AUX_KINDS = frozenset({"derivative", "limited_derivative"})


def _operator_is_live(arr) -> bool:
    """Whether operator `arr` has any entry not proven zero (unknown counts as live)."""
    import sympy as sp
    for e in _op_flat(arr):
        try:
            if sp.sympify(e).is_zero is not True:
                return True
        except (sp.SympifyError, TypeError):
            return True                       # undecidable ⇒ assume live
    return False


# ── per-operator structural-zero booleans ─────────────────────────────────
#
# ``_operator_is_live`` already answers "is this slot anything but a proven
# zero?" for the implicit-mode decision.  The same question is worth asking of
# EVERY operator slot, because a backend that knows an operator is identically
# zero can skip building the machinery that evaluates it — and that machinery
# is not free: the JAX dense-diffusion operator costs 2287 s of a 2539 s
# LowerTriangle run assembling per-face geometry for a diffusion tensor that is
# structurally zero.  Shape alone cannot see this: SWE's ``diffusion_matrix``
# is a well-shaped tensor of literal zeros.
#
# Decided LATE (a property, re-read on every access, never cached), exactly
# like ``fluctuations_are_zero`` and ``implicit_mode``: ``apply()`` rewrites
# operators long after construction, so a boolean frozen at construction time
# goes stale in both directions.

#: Operator slot -> the name of its structural-zero boolean.  Ordered; a
#: printer LOOPS this instead of hardcoding names, so adding a slot here
#: reaches every C-family header without touching a printer.
OPERATOR_ZERO_FLAGS = {
    "flux": "flux_is_zero",
    "hydrostatic_pressure": "hydrostatic_pressure_is_zero",
    "nonconservative_matrix": "nonconservative_matrix_is_zero",
    "eigenvalues": "eigenvalues_are_zero",
    "source": "source_is_zero",
    "source_explicit": "source_explicit_is_zero",
    "diffusion_matrix": "diffusion_matrix_is_zero",
    "diffusion_matrix_explicit": "diffusion_matrix_explicit_is_zero",
}


def operator_is_zero(arr) -> bool:
    """True if operator `arr` is structurally zero (absent, empty or every entry proven zero),
    so a backend may skip evaluating it.
    """
    return not _operator_is_live(arr)


def operator_zero_flags(sm) -> dict:
    """`{flag_name: bool}` for every slot of `OPERATOR_ZERO_FLAGS`, read off `sm` with
    `operator_is_zero` so printers can loop.
    """
    return {flag: operator_is_zero(getattr(sm, attr, None))
            for attr, flag in OPERATOR_ZERO_FLAGS.items()}


def _nonlocal_aux_symbols(sm) -> set:
    """Aux symbols that depend on neighbour cells, read off `sm.aux_registry`; all aux if the
    registry was never derived.
    """
    import sympy as sp
    registry = getattr(sm, "aux_registry", None)
    aux_state = list(getattr(sm, "aux_state", ()) or ())
    if registry is None:
        return {sp.sympify(a) for a in aux_state}
    out = set()
    for entry in registry:
        if entry.get("kind") not in _NONLOCAL_AUX_KINDS:
            continue
        symbol = entry.get("aux_symbol")
        if symbol is None:
            name = entry.get("name")
            symbol = sp.Symbol(name, real=True) if name else None
        if symbol is not None:
            out.add(sp.sympify(symbol))
    return out


def implicit_stage_mode(nsm) -> str:
    """Which implicit-stage path `nsm` needs: `"coupled"` (live implicit diffusion, or an
    implicit source reading a non-local aux), `"local_source"` (cell-local fast path) or
    `"none"`.
    Derived from the operators, so a model with both implicit diffusion and source gives one
    coupled stage.

    References:
        Kennedy, Carpenter 2003, https://doi.org/10.1016/s0168-9274(02)00138-1
    """
    if _operator_is_live(getattr(nsm, "diffusion_matrix", None)):
        diffusion = getattr(nsm, "diffusion", None)
        if diffusion is None or bool(getattr(diffusion, "enabled", True)):
            return IMPLICIT_MODE_COUPLED

    source = getattr(nsm, "source", None)
    if not _operator_is_live(source):
        return IMPLICIT_MODE_NONE

    import sympy as sp
    nonlocal_aux = _nonlocal_aux_symbols(nsm)
    if nonlocal_aux:
        for e in _op_flat(source):
            try:
                expr = sp.sympify(e)
            except (sp.SympifyError, TypeError):
                return IMPLICIT_MODE_COUPLED      # undecidable ⇒ assume coupled
            if expr.free_symbols & nonlocal_aux:
                return IMPLICIT_MODE_COUPLED
    return IMPLICIT_MODE_LOCAL_SOURCE


def _op_flat(arr):
    """Flat list of the scalar entries of an operator array, `[]` for non-arrays."""
    import sympy as sp
    if arr is None:
        return []
    if hasattr(arr, "_array"):                 # ZArray — flat backing store
        return list(arr._array)
    if isinstance(arr, sp.Basic):
        try:
            return list(sp.flatten(arr))
        except Exception:
            return [arr]
    if isinstance(arr, (list, tuple)):
        return list(sp.flatten(arr))
    return []


def _resolve_boundary_traces(nsm) -> None:
    """Collapse every bed/surface-trace `Subs(f(zeta), zeta, 0|1)` node left after symbolic
    lowering into its concrete value on all operator arrays, and raise if any remain.
    """
    import sympy as sp
    is_subs = lambda n: isinstance(n, sp.Subs)
    resolve = lambda n: n.doit()

    for attr in _OPERATOR_ATTRS:
        arr = getattr(nsm, attr, None)
        if arr is None or not hasattr(arr, "replace"):
            continue
        if any(sp.sympify(e).has(sp.Subs) for e in _op_flat(arr)):
            setattr(nsm, attr, arr.replace(is_subs, resolve))

    # Guard (defense-in-depth, mirrors system_extract's unresolved-integral
    # raise): after the resolve NO ``Subs`` atom and NO free vertical ζ symbol
    # may remain in ANY lowered operator (including the derived
    # ``quasilinear_matrix``).  A leftover means a boundary trace was genuinely
    # unresolvable (e.g. free ζ after ``.doit()``) — no symbolic-tree printer
    # can lower it, so fail here rather than emit uncompilable code.
    # The stray-vertical rule below only MEANS anything if the vertical was
    # eliminated in the first place.  A projected model (SME, VAM, ...) ends on
    # a Galerkin step ∫₀¹(·)φ_i dζ, so ζ is integrated out and a surviving ζ can
    # only be an unresolved integral or boundary trace.  A resolved-vertical
    # system (Sigma3D) never projects: it keeps ζ as a genuine mesh COORDINATE,
    # its operators depend on ζ exactly as they depend on x, and an Elder eddy
    # viscosity ν_t = κ u_⋆ h ζ(1−ζ) legitimately leaves ζ in the diffusion
    # matrix.  Running the check there rejects a correct system.
    #
    # The test is the system's OWN declared vertical against its OWN space --
    # not the literal name "zeta" -- so a model that names its vertical
    # differently cannot silently re-arm the check.
    _sm = getattr(nsm, "system_model", None) or getattr(nsm, "sm", None)
    _space = (getattr(nsm, "space", None) or getattr(_sm, "space", None) or [])
    _vert = getattr(nsm, "vertical", None) or getattr(_sm, "vertical", None)
    # vertical projected out  <=>  it is not one of the system's coordinates
    vertical_is_projected = _vert is not None and not any(
        str(s) == str(_vert) for s in _space)

    for attr in _OPERATOR_ATTRS + ("quasilinear_matrix",):
        arr = getattr(nsm, attr, None)
        for e in _op_flat(arr):
            e = sp.sympify(e)
            if e.atoms(sp.Subs):
                raise ValueError(
                    f"REQ-130: unresolved Subs boundary trace survives in "
                    f"operator {attr!r}: {e} — a bed/surface trace could not be "
                    "collapsed by .doit() (free-ζ profile?).  A symbolic-tree "
                    "printer (generic_c / amrex / ufl) cannot lower Subs; fix "
                    "the trace at its birthplace in the model derivation.")
            stray_zeta = {s for s in e.free_symbols
                          if _vert is not None and str(s) == str(_vert)}
            if stray_zeta and vertical_is_projected:
                raise ValueError(
                    f"REQ-130: free vertical symbol {stray_zeta} survives in "
                    f"operator {attr!r}: {e} — the σ-reference coordinate ζ must "
                    "not reach the lowered operators (it is an integration / "
                    "evaluation dummy).  Fix at the model-derivation birthplace.")


# ── Slot dataclasses ────────────────────────────────────────────────


@dataclass
class ReconstructionSpec:
    """Face-state reconstruction configuration: `order` 1 (piecewise constant) or 2 (LSQ-MUSCL
    with `limiter`), `free_surface_aware` wet/dry handling, and `positivity` (`""`,
    `"zhang_shu"` a-priori cap, `"mood"` a-posteriori order-1 re-step).

    References:
        van Leer 1979, https://doi.org/10.1016/0021-9991(79)90145-1
        Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
        Clain, Diot, Loubere 2011, https://doi.org/10.1016/j.jcp.2011.02.026
        Venkatakrishnan 1995, https://doi.org/10.1006/jcph.1995.1084
        Xing, Zhang, Shu 2010, https://doi.org/10.1016/j.advwatres.2010.08.005
    """
    order: int = 1
    limiter: Limiter = Limiter.venkatakrishnan
    free_surface_aware: bool = False
    positivity: str = ""
    # ── the symbolic solvers (zoomy_core.symsolver) ──
    # ``variables``: what is reconstructed -- "model": the model's
    # ``reconstruction_variables`` W(Q) and its inverse (identity unless the
    # model declares a map, e.g. SWE/SME: eta = b + h, u = hinv q);
    # "conservative": Q itself.
    # ``limiters``: per-row override of ``limiter`` by state name (the row of
    # W in that state's slot), e.g. {"b": "none"}.
    # ``ties``: {row: lead} -- the row's limiter factor never exceeds the
    # lead row's, phi_row <- min(phi_row, phi_lead).  Default: none, every
    # row limited on its own.
    variables: ReconstructedVariables = ReconstructedVariables.model
    # ``bed``: with the model's free-surface map (bed b, eta = b + h):
    # "derived" -- reconstruct h and eta, b_f = eta_f - h_f at the faces
    # (Audusse et al. 2004, Sec. 3 / Fig. 1: the only pair that keeps the lake
    # at rest and h_f >= 0 at a dry shore); "reconstructed" -- b and eta.
    bed: BedReconstruction = BedReconstruction.derived
    limiters: dict = field(default_factory=dict)
    ties: dict = field(default_factory=dict)

    def __post_init__(self):
        self.limiter = Limiter(self.limiter)
        self.variables = ReconstructedVariables(self.variables)
        self.bed = BedReconstruction(self.bed)
        self.limiters = {k: Limiter(v) for k, v in self.limiters.items()}


@dataclass
class DiffusionSpec:
    """Diffusion configuration: `enabled` (auto-False when `diffusion_matrix` is zero), time
    `scheme` and `nu` overriding `sm.parameter_values['nu']`.
    """
    enabled: bool = True
    scheme: str = "crank_nicolson"
    nu: Optional[float] = None


def _linearize_source(sm):
    """Rewrite `sm.source` into the point-implicit update `S_lin = (I - dt J)^-1 S` with the
    consistent jacobian `J = dS/dQ + dS/daux * daux/dQ`, as one opaque per-row `solve`
    call.
    """
    import sympy as sp
    from zoomy_core.misc.misc import ZArray
    from zoomy_core.model.kernel_functions import solve as _solve, pick as _pick
    from zoomy_core.transformation.to_numpy import DT_SYMBOL

    n = sm.n_equations
    if sm.n_state != n:
        raise ValueError(
            "source_treatment='linearized' requires a square system "
            f"(n_equations == n_state); got n_equations={n}, "
            f"n_state={sm.n_state}.")
    if sm.source is None:
        return
    state = list(sm.state)
    aux = list(sm.aux_state)

    Jvar = sp.Matrix(
        n, n, lambda i, j: sp.sympify(sm.source_jacobian_wrt_variables[i, j]))
    # ∂aux/∂Q from the local aux-update map; None / no aux ⇒ aux is independent
    # of Q in the point-implicit step ⇒ the channel contributes nothing.
    uav = getattr(sm, "update_aux_variables", None)
    if uav is not None and aux:
        Jaux = sp.Matrix(
            n, len(aux),
            lambda i, j: sp.sympify(sm.source_jacobian_wrt_aux_variables[i, j]))
        Jauxupd = sp.zeros(len(aux), n)
        nrows = uav.shape[0]
        for k in range(min(nrows, len(aux))):
            fk = sp.sympify(uav[k, 0])
            for j in range(n):
                Jauxupd[k, j] = sp.diff(fk, state[j])
        J = Jvar + Jaux * Jauxupd
    else:
        J = Jvar

    A = sp.eye(n) - DT_SYMBOL * J
    A_flat = [A[i, j] for i in range(n) for j in range(n)]
    b_flat = [sp.sympify(sm.source[i, 0]) for i in range(n)]
    sol = _solve(*A_flat, *b_flat)
    rows = [[_pick(sol, sp.Integer(i))] for i in range(n)]
    sm.source = ZArray(rows)


# ── The NSM itself ───────────────────────────────────────────────────


@dataclass
class NumericalSystemModel(SystemModel):
    """`SystemModel` subclass that adds the numerics the solver templates consume: Riemann
    class, reconstruction, diffusion, regularization and `extra_operations`.
    Build it with `from_system_model`, which works on a numerical twin of the system; the
    `BoundaryConditions` container stays at `_bc_source`.

    References:
        Rusanov 1962, https://doi.org/10.1016/0041-5553(62)90062-9
        Pares 2006, https://doi.org/10.1137/050628052
    """

    riemann: Optional[Type[Any]] = None
    # True iff the caller explicitly chose ``riemann`` (vs the Rusanov
    # default). Solvers with their own default numerics defer to the NSM's
    # choice only when this is set (REQ-157).
    riemann_explicit: bool = False
    reconstruction: ReconstructionSpec = field(default_factory=ReconstructionSpec)
    diffusion: DiffusionSpec = field(default_factory=DiffusionSpec)
    # Numerical eigenvalue regularization — added to the diagonal of the local
    # quasi-linear matrix before the LAPACK eigensolve in the numerical-wave-
    # speed path (without it, dry/near-dry SWE cells yield ``A·n`` matrices with
    # repeated zero eigenvalues and the eigensolve can spike spurious large
    # modes).  This is solver config, not a system operation, so it lives as a
    # slim NSM attribute rather than driving a ``.apply`` op.
    eigenvalue_eps: float = 1e-8
    # Standard numerical timestep CAP (seconds) carried on the NSM so EVERY
    # backend reads the SAME value instead of a per-solver magic knob (REQ-190).
    # Default 5.0 s = amrex's historical ``dtmax``.  Rationale (wave-free):
    # a fully-dry / sub-``wet_dry_eps`` domain has ``max|λ| = 0`` everywhere,
    # so the gated eigenvalues are 0 and the CFL imposes NO constraint — the
    # local dt limits are ``+inf`` and the step is bounded only by this
    # ``dt_max`` (and by output/end-time clipping), never by a hardcoded floor.
    # Every printer emits it alongside the other numerical parameters; the numpy
    # timestepper reads it off the NSM when the caller passed no explicit cap.
    dt_max: float = 5.0
    # Round-off of the declared depth (symbolic solvers): a depth in
    # [-depth_round_off, 0) after an update is round-off and is set to 0, so
    # MOOD and the next stage see h >= 0.  Absolute; None disables it.
    depth_round_off: Optional[float] = 1e-12
    # Opt-in system operations run on this NSM by :meth:`derive`, in addition to
    # :meth:`default_operations`.  Each entry is a reusable ``op(sm)`` callable
    # from :mod:`zoomy_core.systemmodel.operations` (e.g.
    # ``desingularize_hinv()``); the old ``desingularize=True`` regularization
    # opt-in is now ``extra_operations=[desingularize_hinv()]``.
    extra_operations: list = field(default_factory=list)
    # Captured at promotion time so SDM-style derivative declarations
    # survive Model → SystemModel.  SystemModel itself has no back-
    # reference to its source Model and ``derivative_specs`` is a
    # StructuredDerivativeModel attribute — without this snapshot the
    # NSM would lose the lift-to-degree-2 signal for SDM models whose
    # ``D.dxx(Q.h)`` calls are substituted by Symbols before
    # ``SystemModel.from_model`` runs (so the aux_registry's
    # ``kind=="derivative"`` scan misses them).
    source_derivative_specs: Optional[list] = None
    # Co-running SystemModels whose derivative requirements must be
    # max'd into the LSQ stencil sizing.  For Chorin VAM the predictor
    # alone declares only first-order spatial derivatives, but the
    # pressure block's elliptic operator carries ``∂_xx P`` — the LSQ
    # stencil at every cell needs degree 2 to fit them.  Without
    # ``additional_systems`` the predictor-only NSM would silently
    # pick degree 1 and the pressure GMRES residual would zero the
    # curvature contribution (yesterday's root-cause for the dam-break
    # blow-up).  Solvers that compose sub-systems set this slot;
    # single-system solvers leave it empty.
    additional_systems: list = field(default_factory=list)
    # Indices (into ``sm.state``) of state rows that the Audusse HR /
    # ``PositiveRusanov``-style Riemann should rescale by ``h*/h`` at
    # face states.  Default (``None``) lets the Riemann class pick —
    # which excludes only ``h`` and ``b``.  For VAM-Chorin we must
    # exclude pressure modes ``P_k`` too (they are amplitudes, not
    # depth-scaled momenta).  ``ChorinSplitVAMSolverJax`` computes the
    # right list at setup time and passes it here.
    scaled_q_indices: Optional[list] = None
    # How the NSM supplies the source to every backend solver:
    #   "explicit"   — raw ``S`` (default, unchanged behaviour);
    #   "linearized" — the point-implicit update ``S_lin = (I − dt·J)⁻¹ S`` with
    #                  the consistent ``J = J_var + J_aux · (∂aux/∂Q)``.
    # The transform happens ONCE at construction (``_linearize_source``); every
    # backend then consumes ``source`` as an ordinary source, the only new atom
    # being the per-cell opaque ``solve``.  This is an NSM-level knob, NOT a
    # per-backend solver mode.
    source_treatment: str = "explicit"

    # ── REQ-194 depth-law axes (all default to TODAY'S behaviour) ──
    #
    # Four independent construction-time knobs so a parameter study over the
    # depth law is a pure loop with no code edit per run (see
    # ``zoomy_core.numerics.depth_law_study.build_nsm``).  Every default below
    # reproduces what shipped before this REQ: ``depth_regularization=None``
    # keeps the legacy ``desingularize_hinv()`` as the sole depth default, and
    # the other three are off.  Nothing here changes an existing consumer's
    # emitted code unless that consumer opts in.
    #
    # Which reciprocal path regularizes the depth:
    #   None       — LEGACY ``desingularize_hinv()`` (hinv aux at wet_dry_eps);
    #   "aux"      — ``regularize_depth_aux()``    (hinv aux at regularization_eps);
    #   "direct"   — ``regularize_depth_direct()`` (KP inlined, no aux row);
    #   "none"     — no depth regularization at all (bare 1/h everywhere).
    # "aux" and "direct" are mutually exclusive BY CONSTRUCTION here (one
    # string, one path); the operations enforce it on the system too.
    depth_regularization: Optional[str] = None
    # The REGULARIZATION scale of the selected path — its OWN quantity, never
    # ``wet_dry_eps`` (a wet/dry state-classification threshold at a different
    # scale).  Conflating the two is what put ``h/(h + 1e-2)`` on the celerity.
    # Read only when ``depth_regularization`` is "aux"/"direct".
    regularization_eps: float = 1e-2
    # How the SPECTRUM is treated by that path: "regularize" (swept like every
    # other operator slot) or "exclude" (keeps the exact 1/h).  Separable for
    # SWE — with a normalized normal the celerity carries no reciprocal — but
    # NOT for SME, whose spectrum is ``hinv·(n0·q_0 ± n0·√(g·h³ + q_1²))``.
    # Which is right is what the study measures; neither is baked in.
    eigenvalue_treatment: str = "regularize"
    # Wet/dry eigenvalue treatment, independent of the three knobs above:
    #   None      — neither op;
    #   "power"   — ``guard_eigenvalue_powers()`` (Max(.,0) under fractional
    #               powers of h; always-safe, idempotent, no dry zeroing);
    #   "gate"    — ``gate_eigenvalues_dry()`` (dry conditional; carries the
    #               power guard internally per REQ-74/REQ-181);
    #   "both"    — the explicit composition of the two, which REQ-181 pins as
    #               byte-identical to "gate" alone (the guard is idempotent).
    #
    # DEFAULT IS NOW "gate" for depth-carrying transport systems (cid 210).
    # REQ-181 removed the gate from the defaults for ONE reason: it overwrote
    # ``sm.eigenvalues``, so gating for the timestep also gutted the Rusanov
    # dissipation.  cid 209 removed that reason — the gate now publishes to the
    # separate ``eigenvalues_cfl`` slot and leaves ``eigenvalues`` physical — so
    # the objection that made it opt-in no longer exists.
    #
    # It has to be a DEFAULT, not an opt-in, because a dt guard is a SCHEME
    # decision and scheme decisions live here, never in a backend.  While it was
    # opt-in (and almost nobody opted in) each backend grew its own substitute:
    # numpy hand-rolled a face-level dry-cell skip keyed on a DIFFERENT
    # parameter (``eps_wet``) with a hard-coded 1e-3 default in backend source,
    # while jax grew nothing at all and let a dry cell's u = q/h drive dt.  Two
    # thresholds five orders of magnitude apart plus a backend with none is
    # exactly the drift the "no numerical constant in a backend" rule exists to
    # prevent.  Emitting ``eigenvalues_cfl`` from here makes every backend agree
    # by construction, because they all just evaluate the operator they are
    # given.
    eigenvalue_guard: Optional[str] = "gate"
    # Impose |n| = 1 on the face normal before anything else runs.
    #
    # REQ-208 item (2) flipped this ON by default.  It is a FACT, not a
    # modelling choice: every face normal in the project is normalised at
    # construction — ``mesh_util._face_normals_2d`` :112,
    # ``_face_normals_3d`` :138, ``_face_normals_wface`` :155 and
    # ``base_mesh`` :321 all divide by ``np.linalg.norm``, and in 1-D
    # ``n0 = ±1``.  Carrying ``√(n0²+n1²)`` symbolically was therefore pure
    # overhead AND actively harmful: it holds the 2-D spectrum over a common
    # denominator, so a later ``1/h`` regularization reaches (and damps) the
    # celerity that ought to be structurally separate from it.
    #
    # It was opt-in only because it rewrites the emitted expression tree.
    # That made the flag unreachable in practice: it is set at NSM
    # construction, which lives in each backend's case code, so no case ever
    # passed it and the face kernels kept recomputing ``√g·|n|`` in all three
    # of them.  ``depth_law_study.build_nsm`` already defaulted it True, so
    # the two entry points disagreed.  Pass ``normalize_normal=False`` to
    # recover the old expression trees.
    normalize_normal: bool = True

    # ── SystemModel identity bridge ───────────────────────────────
    @property
    def sm(self) -> "SystemModel":
        """Return `self`, keeping the historical `nsm.sm.<x>` access working."""
        return self

    # ── Derivation pipeline ───────────────────────────────────────

    def _is_transport_system(self) -> bool:
        """True iff the system has a structurally nonzero `flux` or `nonconservative_matrix`;
        the Chorin pressure and corrector stages are not transport.
        """
        import sympy as sp
        for attr in ("flux", "nonconservative_matrix"):
            M = getattr(self, attr, None)
            if M is None:
                continue
            if any(sp.sympify(e) != 0 for e in sp.flatten(M)):
                return True
        return False

    # ── emitted-numerics predicate (REQ-209) ──────────────────────

    def _probe_numerics(self):
        """A `Numerics` instance over this NSM for interrogating the built face kernels, reusing
        a live one or building and keeping one.
        """
        if self.riemann is None:
            return None
        for ref in (getattr(self, "_built_numerics", None) or []):
            numerics = ref()
            if numerics is not None and type(numerics) is self.riemann:
                return numerics
        probe = getattr(self, "_fluctuation_probe", None)
        if probe is None or type(probe) is not self.riemann:
            probe = self._fluctuation_probe = self.build_numerics()
        return probe

    @property
    def fluctuations_are_zero(self) -> bool:
        """True iff the built `numerical_fluctuations` is structurally zero, i.e. no fluctuation
        term is emitted.
        It depends on the (model x Riemann solver x current representation) triple and is re-read
        on every access.
        """
        return numerics_fluctuations_are_zero(self._probe_numerics())

    @property
    def implicit_mode(self) -> str:
        """Implicit-stage path this system needs (`"none"`, `"local_source"`, `"coupled"`),
        re-read from the operators on every access; see `implicit_stage_mode`.

        References:
            Kennedy, Carpenter 2003, https://doi.org/10.1016/s0168-9274(02)00138-1
        """
        return implicit_stage_mode(self)

    # ── per-operator structural-zero booleans ─────────────────────

    @property
    def operators_are_zero(self) -> dict:
        """`{flag_name: bool}` of the structural-zero flag of every operator slot (see
        `operator_is_zero`), re-read on every access.
        """
        return operator_zero_flags(self)

    @property
    def flux_is_zero(self) -> bool:
        """True iff `flux` is structurally zero."""
        return operator_is_zero(getattr(self, "flux", None))

    @property
    def hydrostatic_pressure_is_zero(self) -> bool:
        """True iff `hydrostatic_pressure` is structurally zero (hand-built SWE keeps ``g
        h^2/2`` in `flux` instead).
        """
        return operator_is_zero(getattr(self, "hydrostatic_pressure", None))

    @property
    def nonconservative_matrix_is_zero(self) -> bool:
        """True iff the model's `nonconservative_matrix` is structurally zero; not the same as
        `fluctuations_are_zero`, which reads the built face kernel.
        """
        return operator_is_zero(getattr(self, "nonconservative_matrix", None))

    @property
    def eigenvalues_are_zero(self) -> bool:
        """True iff `eigenvalues` is structurally zero, i.e. no hyperbolic transport."""
        return operator_is_zero(getattr(self, "eigenvalues", None))

    @property
    def source_is_zero(self) -> bool:
        """True iff the implicit `source` is structurally zero."""
        return operator_is_zero(getattr(self, "source", None))

    @property
    def source_explicit_is_zero(self) -> bool:
        """True iff `source_explicit` is structurally zero."""
        return operator_is_zero(getattr(self, "source_explicit", None))

    @property
    def diffusion_matrix_is_zero(self) -> bool:
        """True iff the implicit `diffusion_matrix` is structurally zero, so a backend may skip
        building its diffusion operator.
        """
        return operator_is_zero(getattr(self, "diffusion_matrix", None))

    @property
    def diffusion_matrix_explicit_is_zero(self) -> bool:
        """True iff `diffusion_matrix_explicit` is structurally zero."""
        return operator_is_zero(
            getattr(self, "diffusion_matrix_explicit", None))

    def default_operations(self) -> list:
        """Operations `derive` applies to every NSM before `extra_operations`, in load-bearing
        order: face-normal normalization, the selected depth regularization, then the
        eigenvalue guard.
        Only shallow-water transport systems (depth state `h`, flux or NCP) get the
        Kurganov-Petrova desingularization.

        References:
            Kurganov, Petrova 2007, https://doi.org/10.4310/CMS.2007.v5.n1.a6
        """
        from zoomy_core.systemmodel.operations import (
            desingularize_hinv,
            gate_eigenvalues_dry,
            guard_eigenvalue_powers,
            normalize_face_normal,
            regularize_depth_aux,
            regularize_depth_direct,
        )

        ops = []
        if self.normalize_normal:
            ops.append(normalize_face_normal())

        # The depth ops are gated on:
        #  * ``_is_transport_system`` — the Chorin pressure/corrector
        #    sub-systems share the same ``h``-bearing state but are NOT
        #    transport (no flux/NCP); they must stay clean, and
        #  * the presence of a depth state ``h`` — non-shallow-water systems
        #    (no ``h``) get nothing.
        # ROLE-gated (never name-gated): only a system that DECLARES a depth
        # role is free-surface.  A tracer that happens to be called ``h`` gets
        # neither the KP reciprocal nor the dry CFL gate.
        depth = (self._is_transport_system()
                 and self.roles.depth is not None)
        if depth:
            ops.extend(self._depth_regularization_operations(
                desingularize_hinv, regularize_depth_aux,
                regularize_depth_direct))
            ops.extend(self._eigenvalue_guard_operations(
                guard_eigenvalue_powers, gate_eigenvalues_dry))
        return ops

    def _depth_regularization_operations(self, legacy, aux, direct) -> list:
        """The operation(s) implementing `depth_regularization` (legacy, `"aux"`, `"direct"`
        or `"none"`).
        """
        choice = self.depth_regularization
        if choice is None:
            return [legacy()]
        if choice == "none":
            return []
        depth_name = str(self.state[self.roles.depth])
        if choice == "aux":
            return [aux(self.regularization_eps,
                        eigenvalues=self.eigenvalue_treatment,
                        field=depth_name)]
        if choice == "direct":
            return [direct(self.regularization_eps,
                           eigenvalues=self.eigenvalue_treatment,
                           field=depth_name)]
        raise ValueError(
            f"depth_regularization={choice!r} is not a legal path; expected "
            "None (legacy desingularize_hinv), 'aux', 'direct' or 'none'.")

    def _eigenvalue_guard_operations(self, power, gate) -> list:
        """The operation(s) implementing `eigenvalue_guard` (`None`, `"power"`, `"gate"`
        or `"both"`).
        """
        choice = self.eigenvalue_guard
        if choice is None:
            return []
        if choice == "power":
            return [power()]
        if choice == "gate":
            return [gate()]
        if choice == "both":
            return [power(), gate()]
        raise ValueError(
            f"eigenvalue_guard={choice!r} is not a legal guard; expected "
            "None, 'power', 'gate' or 'both'.")

    def derive(self) -> "NumericalSystemModel":
        """Run `default_operations() + extra_operations` on this NSM in place via
        `SystemModel.apply`; returns `self`.
        """
        for op in self.default_operations() + list(self.extra_operations):
            self.apply(op)
        return self

    # ── operations reach the already-built numerics too ───────────

    def apply(self, operation, *, name=None, description=None):
        """Apply a system operation, then refresh every `Numerics` already built from this NSM
        so its registered functions see the new operators.
        """
        out = super().apply(operation, name=name, description=description)
        self._refresh_built_numerics()
        return out

    def _refresh_built_numerics(self) -> None:
        """Re-register the symbolic functions of every live numerics built from this NSM (no-op if
        none).
        """
        refs = getattr(self, "_built_numerics", None)
        if not refs:
            return
        alive = []
        for ref in refs:
            numerics = ref()
            if numerics is None:
                continue
            alive.append(ref)
            numerics._initialize_functions()
        self._built_numerics = alive

    # ── Constructors ──────────────────────────────────────────────

    @classmethod
    def from_model(cls, model, **kwargs) -> "NumericalSystemModel":
        """Shortcut for `from_system_model(SystemModel.from_model(model), **kwargs)`."""
        return cls.from_system_model(SystemModel.from_model(model), **kwargs)

    @classmethod
    def from_system_model(
        cls,
        sm,
        *,
        riemann=None,
        reconstruction: Optional[ReconstructionSpec] = None,
        diffusion: Optional[DiffusionSpec] = None,
        eigenvalue_eps: float = 1e-8,
        dt_max: float = 5.0,
        depth_round_off: Optional[float] = 1e-12,
        tag_rules: Optional[dict] = None,
        extra_operations: Optional[list] = None,
        additional_systems: Optional[list] = None,
        scaled_q_indices: Optional[list] = None,
        source_treatment: str = "explicit",
        depth_regularization: Optional[str] = None,
        regularization_eps: float = 1e-2,
        eigenvalue_treatment: str = "regularize",
        # cid 210: "gate" by default — the dt spectrum is emitted from HERE, so
        # no backend has to (or may) invent a dry-cell CFL rule of its own.
        # Safe as a default only since cid 209 made the gate write the separate
        # ``eigenvalues_cfl`` slot instead of clobbering ``eigenvalues``.
        # Pass ``eigenvalue_guard=None`` to opt out explicitly.
        eigenvalue_guard: Optional[str] = "gate",
        normalize_normal: bool = True,
    ) -> "NumericalSystemModel":
        """Build an NSM from a `SystemModel` (or a `Model`, auto-promoted) on its numerical
        twin, resolving the numerics defaults and running `derive`.

        Args:
            riemann: Riemann class; defaults to the tag rule, else `NonconservativeRusanov`.
            reconstruction: `ReconstructionSpec`; defaults to first order.
            diffusion: `DiffusionSpec`; enabled iff `diffusion_matrix` is nonzero.
            eigenvalue_eps: regularization of the numerical eigensolve.
            dt_max: timestep cap shared by all backends.
            tag_rules: per-call override of the tag -> numerics defaults.
            extra_operations: opt-in operations run by `derive`.
            additional_systems: co-running sub-systems whose derivatives size the LSQ stencil.
            scaled_q_indices: rows a hydrostatic reconstruction rescales by `h*/h`.
            source_treatment: `"explicit"` or `"linearized"` (point-implicit).
            depth_regularization, regularization_eps, eigenvalue_treatment, eigenvalue_guard, normalize_normal: depth-law axes, see `depth_law_study`.

        Returns:
            The derived NSM; the LSQ degree is always auto-derived.
        """
        source_specs = getattr(sm, "derivative_specs", None)
        if not isinstance(sm, SystemModel):
            sm = SystemModel.from_model(sm)
            source_specs = getattr(sm, "derivative_specs", source_specs)
        # THE BOUNDARY.  Build the numerical twin instead of relabelling the
        # symbolic one: from here down we work on ``sm``, which is a COPY whose
        # state/aux symbols carry no assumptions.  The caller's SystemModel is
        # left untouched and still carries its declared facts, which is what the
        # analytic layer needs.  See :func:`construct_numerical`.
        from zoomy_core.systemmodel.roles import (
            declared_role_record, assumption_positive_names, declare_roles)
        # ROLES: the analytic sign assumptions seed the ``positive`` role HERE,
        # on the SystemModel side of the boundary — the numerical twin below is
        # assumption-free by construction and could no longer tell.
        seed_positive = (None if "positive" in declared_role_record(sm)
                         else assumption_positive_names(sm))
        sm = construct_numerical(sm)
        if seed_positive:
            declare_roles(sm, positive=seed_positive,
                          description="positive role seeded from the "
                                      "analytic sign assumptions")
        additional_systems = [construct_numerical(sub)
                              if isinstance(sub, SystemModel) else sub
                              for sub in (additional_systems or [])] or None
        _assert_bc_kernels_match_state(sm)
        for sub in (additional_systems or []):
            if isinstance(sub, SystemModel):
                _assert_bc_kernels_match_state(sub)
        # Whether the caller EXPLICITLY chose a Riemann solver.  The NSM
        # defaults ``riemann`` to :class:`NonconservativeRusanov` below, so
        # ``riemann is not None`` post-default cannot distinguish "the case
        # asked for Rusanov" from "nobody asked" — solvers with their own
        # sensible default (e.g. the free-surface positive Rusanov) must only
        # be overridden when the choice was explicit (REQ-157).
        riemann_explicit = riemann is not None
        if riemann is None and sm.is_square:   # the model's tags may imply one
            from zoomy_core.systemmodel.tags import tag_defaults
            riemann = tag_defaults(sm, tag_rules).get("riemann")
        if riemann is None:
            # Imported lazily — fvm/riemann_solvers.py imports from
            # zoomy_core.transformation.to_numpy which transitively
            # imports zoomy_core.model; doing it at module level here
            # creates a cycle on first import of the numerics package.
            from zoomy_core.fvm.riemann_solvers import NonconservativeRusanov
            riemann = NonconservativeRusanov
        if reconstruction is None:
            reconstruction = ReconstructionSpec()
        if diffusion is None:
            diffusion = DiffusionSpec(enabled=_diffusion_auto_enabled(sm))
        if source_treatment not in ("explicit", "linearized"):
            raise ValueError(
                "source_treatment must be 'explicit' or 'linearized'; "
                f"got {source_treatment!r}.")
        if source_treatment == "linearized":
            _linearize_source(sm)

        # The NSM IS-A SystemModel, so PROMOTE the frozen ``sm`` instance
        # in place — re-class it to the NSM and attach the numerics
        # fields — rather than copy its field values into a separate
        # object.  Both classes are plain (``__dict__``-backed) dataclasses
        # so the re-class is layout-compatible, and the promotion keeps the
        # SystemModel's full state (every dataclass field PLUS the dynamic
        # attributes ``_bc_source`` / ``_aux_bc_source`` (the
        # BoundaryConditions CONTAINER — REQ-16, reached as
        # ``nsm._bc_source``), ``aux_registry`` (LSQ-degree sizing),
        # ``positive_state``, ``derivative_specs``, the lazy
        # ``_quasilinear_matrix`` cache, …) intact.
        #
        # READ THIS BEFORE RELYING ON IDENTITY.  The re-class below promotes
        # the LOCAL ``sm``, which ``construct_numerical`` rebound to a COPY at
        # THE BOUNDARY above.  So ``nsm is sm`` holds only for that local:
        # **the CALLER's SystemModel is a different object and is left
        # untouched.**  This comment used to claim the opposite — that a caller
        # could keep mutating its SystemModel (``sm.initial_conditions = …``
        # between successive solves) and see the change through the NSM.  That
        # is FALSE and has silently broken real work twice:
        #
        #   * ``model/splitter.py`` called ``to_numerical_system_model`` on the
        #     three chorin stages and discarded the return, so no stage was
        #     ever promoted and the cid=50 ``Pow(h, -2)`` regularization was
        #     dead code for as long as the copy has existed.
        #   * ~30 case scripts under ``thesis/{cases,notebooks}`` assign
        #     ``sm.initial_conditions`` AFTER lowering, which is a no-op.  It
        #     froze ``bingham_inclined.py`` at 87.08 % with all 40 frames
        #     bit-identical — a wrong ANSWER, not a crash.
        #
        # Post-lowering mutation of a SystemModel does not reach the NSM.
        # Either mutate BEFORE lowering, or re-lower, or drive one continuous
        # solve.  Callers must BIND the return value: ``nsm = to_numerical_
        # system_model(sm)``.
        #
        sm.__class__ = cls
        sm.riemann = riemann
        sm.riemann_explicit = riemann_explicit
        sm.reconstruction = reconstruction
        sm.diffusion = diffusion
        sm.eigenvalue_eps = eigenvalue_eps
        sm.dt_max = dt_max
        sm.depth_round_off = depth_round_off
        sm.extra_operations = list(extra_operations or [])
        sm.source_derivative_specs = (
            list(source_specs) if source_specs else None)
        sm.additional_systems = list(additional_systems or [])
        sm.scaled_q_indices = (
            list(scaled_q_indices) if scaled_q_indices is not None
            else None)
        sm.source_treatment = source_treatment
        # REQ-194 depth-law axes — set BEFORE ``derive()`` so
        # ``default_operations()`` can read them.  Every default reproduces the
        # pre-REQ behaviour (legacy ``desingularize_hinv()``, no guard, no
        # normal normalization), so a caller that passes none of them sees no
        # change at all.
        sm.depth_regularization = depth_regularization
        sm.regularization_eps = regularization_eps
        sm.eigenvalue_treatment = eigenvalue_treatment
        sm.eigenvalue_guard = eigenvalue_guard
        sm.normalize_normal = normalize_normal
        # Run the operation pipeline (default_operations() + extra_operations)
        # on the now-promoted NSM.  Phase A1: default_operations() is empty, so
        # this is a no-op unless the caller opted in via ``extra_operations``.
        sm.derive()
        # REQ-130: collapse any surviving bed/surface-trace ``Subs(f(ζ),ζ,0|1)``
        # nodes on every operator array — the single backend-agnostic lowering
        # seam — and GUARD that none (nor a free ζ) remain, so no symbolic-tree
        # printer ever receives a node it cannot compile.
        _resolve_boundary_traces(sm)
        return sm

    # ── LSQ-degree resolution ─────────────────────────────────────

    def resolved_lsq_degree(self) -> int:
        """LSQ polynomial degree for the mesh: the maximum derivative order over the aux registry,
        `additional_systems` and `source_derivative_specs` (at least 1); never set by hand.

        References:
            Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
        """
        candidates = [1, _lsq_degree_from_aux_registry(self.sm)]
        candidates.append(
            _lsq_degree_from_derivative_specs(self.source_derivative_specs))
        for sm in self.additional_systems:
            candidates.append(_lsq_degree_from_aux_registry(sm))
            candidates.append(_lsq_degree_from_derivative_specs(
                getattr(sm, "derivative_specs", None)))
        return max(candidates)

    # ── Numerics + runtime construction ───────────────────────────

    def build_numerics(self):
        """Instantiate the symbolic Riemann numerics over this NSM (threading
        `scaled_q_indices`) and register it weakly so later operations refresh it.
        """
        import weakref
        if self.scaled_q_indices is not None:
            numerics = self.riemann(self.sm,
                                    scaled_q_indices=self.scaled_q_indices)
        else:
            numerics = self.riemann(self.sm)
        refs = getattr(self, "_built_numerics", None)
        if refs is None:
            refs = self._built_numerics = []
        refs.append(weakref.ref(numerics))
        return numerics

    @property
    def call(self):
        """The kernels table: one `KernelCall` per kernel the NSM carries (operators, boundary
        kernels, numerics functions), `None` where empty, printed as
        `<_symbolic_name>.<slot>`.
        Two boundary kernels exist: `boundary_conditions` (face value, for the Riemann problem)
        and `boundary_ghost_state` (mirror-image ghost, for gradients, limiters, derivative aux
        and diffusion).
        """
        owner = getattr(self, "_symbolic_name", None) or "nsm"
        if getattr(self, "_call", None) is None or self._call_owner != owner:
            from zoomy_core.misc.misc import Zstruct
            from zoomy_core.model.basefunction import KernelCall, KernelRef
            from zoomy_core.systemmodel.system_model import OPERATOR_ARG_SLOTS
            kernels = [self.operator(name) for name in OPERATOR_ARG_SLOTS]
            ghost = (self.boundary_ghost_kernel()[0]
                     if self.boundary_conditions is not None else None)
            kernels += [self.boundary_conditions, ghost,
                        *self.build_numerics().functions.values()]
            kernels = [f for f in kernels if f is not None]
            functions = Zstruct(**{f.name: f for f in kernels})
            call = {f.name: None if f.is_empty else KernelCall(owner, functions, f.name)
                    for f in kernels}
            call["aux_derivatives"] = (KernelRef(owner, "aux_derivatives")
                                       if self.aux_derivatives else None)
            self._call, self._call_owner = Zstruct(**call), owner
        return self._call

    @property
    def aux_derivatives(self):
        """Aux rows filled from mesh derivatives: `aux_registry` plus `aux_input_registry`."""
        return (list(getattr(self, "aux_registry", None) or [])
                + list(getattr(self, "aux_input_registry", None) or []))

    def build_runtime_numpy(self):
        """Lambdify this NSM into the runtime model the NumPy solvers consume."""
        from zoomy_core.transformation.to_numpy import NumpyRuntimeModel
        return NumpyRuntimeModel.from_system_model(self.sm)


# ── Canonical coercion (the printers' front door) ───────────────────


def to_numerical_system_model(obj) -> "NumericalSystemModel":
    """Coerce a `Model`, `SystemModel` or NSM to a `NumericalSystemModel`, the single front
    door of the core code printers; other types raise `TypeError`.
    """
    if isinstance(obj, NumericalSystemModel):
        return obj
    if isinstance(obj, SystemModel):
        return NumericalSystemModel.from_system_model(obj)
    if getattr(obj, "_system_model_kind", None) is not None:
        return NumericalSystemModel.from_system_model(
            SystemModel.from_model(obj))
    raise TypeError(
        f"to_numerical_system_model: cannot coerce {type(obj).__name__!r} "
        "to a NumericalSystemModel — expected a NumericalSystemModel, a "
        "SystemModel, or a Model built via `SystemModel.from_model`."
    )


# ── Helpers ─────────────────────────────────────────────────────────


def _diffusion_auto_enabled(sm) -> bool:
    """True iff the system carries a structurally nonzero `diffusion_matrix`."""
    import sympy as sp
    A = getattr(sm, "diffusion_matrix", None)
    if A is None:
        return False
    try:
        flat = list(sp.flatten(A))
    except Exception:
        return False
    return any(sp.simplify(e) != 0 for e in flat)


def _lsq_degree_from_aux_registry(sm) -> int:
    """Maximum derivative order over the derivative entries of `sm.aux_registry` (1 if none)."""
    registry = getattr(sm, "aux_registry", None) or []
    orders = [
        sum(int(k) for k in entry["multi_index"])
        for entry in registry
        if entry.get("kind") in ("derivative", "limited_derivative")
        and entry.get("multi_index") is not None
    ]
    return max(orders) if orders else 1


def _assert_bc_kernels_match_state(sm) -> None:
    """Raise if a BC kernel was built against state symbols that no longer match `sm.state`, as
    after `change_state_variables`.
    """
    expected = list(sm.state)
    for attr in ("boundary_conditions",
                 "boundary_gradients",
                 "aux_boundary_conditions"):
        kernel = getattr(sm, attr, None)
        if kernel is None:
            continue
        args = getattr(kernel, "args", None)
        if args is None or not hasattr(args, "contains") \
                or not args.contains("variables"):
            continue
        bound = list(args.variables.values())
        if bound != expected:
            raise ValueError(
                f"SystemModel.{attr} is stale: its state signature\n"
                f"  {[str(s) for s in bound]}\n"
                f"does not match sm.state\n"
                f"  {[str(s) for s in expected]}.\n"
                "This typically means sm.change_state_variables(...) "
                "remapped the state symbols in flux/source/NCP but did "
                "NOT rebuild the BC kernel.  The BC body still "
                "references the original state symbols and will read "
                "the wrong slot of Q at runtime — Lambda-inflow values "
                "land on the wrong (scaled) state, mass drifts.\n\n"
                "Either rebuild the BC kernel against the new state "
                "(see vam_chorin_bump_8state_chain.py for the canonical "
                "pattern) — or, once SystemModel.change_state_variables "
                "is fixed upstream, the BC remap will happen "
                "automatically and this check becomes a no-op."
            )


def _lsq_degree_from_derivative_specs(specs) -> int:
    """Maximum axes count over `StructuredDerivativeModel` derivative specs (1 if empty)."""
    if not specs:
        return 1
    orders = [sum(1 for a in spec.axes if a != "t") for spec in specs]
    return max(orders) if orders else 1
