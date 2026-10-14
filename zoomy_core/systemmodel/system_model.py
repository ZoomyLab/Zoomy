"""Symbolic operator-form PDE system (`SystemModel`) extracted from a `Model`: flux,
nonconservative matrix, source, diffusion, eigenvalues and boundary conditions in operator
slots.
Sits between `Model` and `NumericalSystemModel`; also holds the system-level operations that
repack operators in place.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import sympy as sp

from zoomy_core.misc.misc import Zstruct, ZArray
from zoomy_core.misc.options import Option


# ── SINGLE SOURCE OF TRUTH: operator argument signatures ─────────────────────
# Each operator kernel (``flux`` / ``source`` / ``update_aux_variables`` / …) is
# CARRIED on a frozen SystemModel as a :class:`zoomy_core.model.basefunction.
# Function` whose ``args`` declare its ordered argument groups ONCE — here.
# Every backend printer/runtime reads ``function.args`` (via :meth:`SystemModel.
# operator`) and only translates SYNTAX (how a signature/call is spelled in
# C / python / UFL); no printer re-declares or routes arguments.  This is the
# same representation the boundary-condition kernel already used end-to-end.
#
# Slot tokens are the numpy-runtime Zstruct field names (== the C-family
# ``GenericCppBase.ARG_MAPPING`` group keys), so both families derive from the
# same declaration:
#   variables      Q      the state vector
#   aux_variables  Qaux   the auxiliary vector
#   parameters     p      the parameter vector
#   normal         n      the interface normal
#   time           time   scalar simulation time ``t``
#   dt             dt     scalar time step
#   position       X      the length-3 cell/face position vector ``x``
#
# ``dt`` is declared on ``source`` / ``update_variables``.  The C-family emits
# it as a distinct by-value scalar arg unconditionally; the numpy/jax path DROPS
# it when a split sub-system baked ``dt`` in as a model PARAMETER (the dt symbol
# would then be a duplicate lambdify argument) — a lowering-SYNTAX constraint,
# not a signature difference (handled in ``NumpyRuntimeModel``).
def face_normal_symbols(n_dim):
    """Canonical outward face-normal symbols `n0 ... n_{n_dim-1}`, the one naming convention
    shared by `SystemModel.normal` and the derivations.
    """
    return [sp.Symbol(f"n{d}", real=True) for d in range(int(n_dim))]


_V, _A, _P, _N, _T, _DT, _X, _R = (
    "variables", "aux_variables", "parameters",
    "normal", "time", "dt", "position", "reconstruction")

OPERATOR_ARG_SLOTS = {
    "flux": (_V, _A, _P),
    "hydrostatic_pressure": (_V, _A, _P),
    "nonconservative_matrix": (_V, _A, _P),
    "quasilinear_matrix": (_V, _A, _P),
    "diffusion_matrix": (_V, _A, _P),
    "diffusion_matrix_explicit": (_V, _A, _P),
    "mass_matrix": (_V, _A, _P),
    "source_explicit": (_V, _A, _P),
    "source_jacobian_wrt_variables": (_V, _A, _P),
    "source_jacobian_wrt_aux_variables": (_V, _A, _P),
    "flux_jacobian_wrt_aux_variables": (_V, _A, _P),
    "eigenvalues": (_V, _A, _P, _N),
    "eigenvalues_cfl": (_V, _A, _P, _N),
    # The well-balanced reconstruction pair.  ``reconstruction_variables`` maps
    # the state to the variables the limiter works in (SWE: b, eta, u);
    # ``state_from_reconstruction`` is its inverse and is therefore expressed in
    # the ``WB_<state>`` symbols, NOT in the state — hence the ``reconstruction``
    # group.  Declared here so a backend READS the signature like any other
    # operator instead of hand-assembling it (cid 214: numpy could not register
    # them at all because operator_signature had no entry, and jax built its
    # own tuple locally).
    "reconstruction_variables": (_V, _A, _P),
    "state_from_reconstruction": (_R, _A, _P),
    "source": (_V, _A, _P, _T, _DT, _X),
    "update_variables": (_V, _A, _P, _DT),
    "update_aux_variables": (_V, _A, _P, _T, _X),
    "update_aux_variables_jacobian_wrt_variables": (_V, _A, _P, _T, _X),
    # The admissibility list: expressions that must stay >= 0 (free surface:
    # [depth]; Euler: [rho, p(Q)]).  Read by MOOD and positivity scaling.
    "admissible": (_V, _A, _P),
}

# The canonical time-step symbol (matches ``to_numpy.DT_SYMBOL`` and the
# ``sp.Symbol("dt", positive=True)`` the C printers bind).
_DT_SYMBOL = sp.Symbol("dt", positive=True)


def _to_zarray(obj):
    """Coerce a sympy Matrix, NDimArray or list to `ZArray` with its shape preserved; `None`
    passes through.
    """
    if obj is None or isinstance(obj, ZArray):
        return obj
    if isinstance(obj, sp.MatrixBase):
        return ZArray(obj.tolist(), shape=tuple(obj.shape))
    if isinstance(obj, sp.NDimArray):
        return ZArray(obj.tolist(), shape=tuple(obj.shape))
    return ZArray(obj)


def _channel_explicit_diffusion(model, canonical_source):
    """Read the `diffusion_matrix_explicit()` override off `canonical_source` or `model` on
    the declarative path.
    Returns `None` when the override is absent or all zero.
    """
    def _is_zero(e):
        """Whether the entry `e` is zero."""
        z = getattr(e, "is_zero", None)
        return z if z is not None else (e == 0)
    for src in (canonical_source, model):
        fn = getattr(src, "diffusion_matrix_explicit", None)
        if not callable(fn):
            continue
        A_z = fn()
        if A_z is None:
            continue
        if hasattr(A_z, "todense"):
            A = sp.MutableDenseNDimArray(A_z.todense())
        elif hasattr(A_z, "tolist"):
            A = sp.MutableDenseNDimArray(A_z.tolist())
        else:
            A = sp.MutableDenseNDimArray(A_z)
        flat = [A[idx] for idx in _iter_indices(tuple(A.shape))]
        if any(not _is_zero(e) for e in flat):
            return A
    return None


def _iter_indices(shape):
    """Iterate over all multi-indices of an array of `shape`."""
    if not shape:
        yield ()
        return
    for i in range(shape[0]):
        for rest in _iter_indices(shape[1:]):
            yield (i,) + rest


def _cov_diffusion_tensor(A_old, sub_map, J, *, n_state):
    """Apply a state change of variables to the diffusion tensor: ``A_new[i,k,d,e] = sum_j
    A_old[i,j,d,e](T) J[j,k]``; `None` stays `None`.
    """
    if A_old is None:
        return None
    n_eq, _, n_dim, _ = A_old.shape
    A_new = sp.MutableDenseNDimArray.zeros(n_eq, n_state, n_dim, n_dim)
    for i in range(n_eq):
        for d in range(n_dim):
            for e in range(n_dim):
                slab = sp.Matrix(
                    1, A_old.shape[1],
                    lambda _r, _j, _i=i, _d=d, _e=e: sp.sympify(
                        A_old[_i, _j, _d, _e]),
                )
                slab_sub = slab.xreplace(sub_map)
                slab_new = sp.expand(slab_sub * J)
                for k in range(n_state):
                    A_new[i, k, d, e] = slab_new[0, k]
    return _to_zarray(A_new)


def _cov_propagate_bc_function(bc_func, sub_map, new_state):
    """Apply a state change of variables to a symbolic BC kernel: substitute the old state symbols
    in its definition and rebuild `args.variables` over the new state.
    """
    if bc_func is None:
        return None
    from zoomy_core.model.basefunction import Function

    new_defn = bc_func.definition
    if hasattr(new_defn, "xreplace"):
        new_defn = new_defn.xreplace(sub_map)

    new_args = bc_func.args
    if (new_args is not None
            and hasattr(new_args, "keys")
            and "variables" in list(new_args.keys())):
        old_vars = new_args.variables
        sym_name = getattr(old_vars, "_symbolic_name", "Q")
        new_vars = Zstruct(**{str(s): s for s in new_state})
        new_vars._symbolic_name = sym_name
        new_args_kwargs = {
            k: (new_vars if k == "variables" else new_args[k])
            for k in list(new_args.keys())
        }
        new_args = Zstruct(**new_args_kwargs)

    return Function(
        name=bc_func.name,
        args=new_args,
        definition=new_defn,
    )


# ── function-group slot registry ──────────────────────────────────────────
# Maps a derivation function-group name (registered via
# ``Model.register_group``) to the ``SystemModel`` attribute it fills.  Open
# by design — ``register_function_slot`` lets user-defined derived operators
# ride the same rails as the built-in vertical reconstruction.
_FUNCTION_SLOTS: Dict[str, str] = {
    "interpolate": "interpolate_to_3d",
    "project": "project_from_3d",
    "reconstruction": "reconstruction_variables",
}


def register_function_slot(group_name: str, sm_attr: str) -> None:
    """Route the derivation function group `group_name` to the `SystemModel` attribute
    `sm_attr` (e.g. a derived wall BC to a boundary-condition slot).
    """
    _FUNCTION_SLOTS[group_name] = sm_attr


def _grad_aux_name(base, axis="x") -> str:
    """Name of the gradient aux of field `base` along `axis` (concatenated axis letters), e.g.
    `dhdx`, `dq0dy`, `dbdxx`.
    """
    if isinstance(base, sp.Function):
        head = base.func.__name__
        idx = "".join(str(int(a)) for a in base.args
                      if getattr(a, "is_Integer", False))
        return f"d{head}{idx}d{axis}"
    return f"d{base}d{axis}"


def _gradients_to_aux(expr, spaces):
    """Replace every purely spatial derivative in `expr` (any order, any axis of `spaces`)
    with a gradient-aux Symbol the solver fills from its LSQ stencil.

    Returns:
        `(new_expr, grads)` with `grads` the ordered unique ``(aux_symbol, field,
            multi_index)``.
    """
    space_index = {s: d for d, s in enumerate(spaces)}
    n_dim = len(spaces)
    grads: list = []
    seen: set = set()

    def _is_spatial_grad(e):
        """Whether `e` is a purely spatial derivative."""
        return (isinstance(e, sp.Derivative)
                and all(v in space_index for v, _ in e.variable_count))

    def _to_sym(e):
        """The gradient-aux Symbol of derivative `e`, recorded in `grads` on first use."""
        mi = [0] * n_dim
        for v, n in e.variable_count:
            mi[space_index[v]] += int(n)
        axis = "".join(str(spaces[d]) for d in range(n_dim) for _ in range(mi[d]))
        s = sp.Symbol(_grad_aux_name(e.expr, axis), real=True)
        if s.name not in seen:
            seen.add(s.name)
            grads.append((s, e.expr, tuple(mi)))
        return s

    return expr.replace(_is_spatial_grad, _to_sym), grads


def _fields_to_state_symbols(expr, x):
    """Collapse applied state/aux fields to bare symbols (`q(i, t, x) -> q_i`, ``h(t, x) ->
    h``).
    """
    from zoomy_core.model.derivation.system_extract import _state_symbol

    def _is_field(e):
        """Whether `e` is an applied field depending on `x`."""
        return (isinstance(e, sp.Function) and bool(e.args)
                and any(a == x for a in e.args))

    return expr.replace(_is_field, lambda e: _state_symbol(e))


# canonical operator slots: METHOD on the model class → group name.  The
# method returns the RAW definition rows ({slot_or_field: expr}, built and
# stashed by derive_model); the SAME parser below converts them — one source
# of truth, no hand-copied symbolic definitions.  register_group() remains
# for user-defined extras (boundary:<name>, custom slots).
_CANONICAL_METHODS = {
    "interpolate": "interpolate_to_3d",
    "project": "project_from_3d",
    "reconstruction": "reconstruction_variables",
}


def _attach_function_groups(sm, model, canonical_source=None) -> None:
    """Parse `model._function_groups` into the matching `SystemModel` slots, turning each
    `d_x(state)` into a gradient aux.
    Groups named `boundary:<name>` land in `sm.boundary_specs` for `FromModel` boundary
    conditions.
    """
    groups = dict(getattr(model, "_function_groups", None) or {})
    if canonical_source is not None:
        for slot, meth in _CANONICAL_METHODS.items():
            fn = getattr(canonical_source, meth, None)
            rows = fn() if callable(fn) else None
            if not isinstance(rows, dict) or not rows:
                continue            # legacy/default return — not a definition
            if slot in groups:
                raise ValueError(
                    f"operator {slot!r} is defined BOTH via "
                    f"register_group({slot!r}, …) and the canonical method "
                    f"{meth}() — exactly one definition is allowed.")
            groups[slot] = rows
    if not groups:
        return
    x = sm.space[0]
    zeta = sp.Symbol("zeta", real=True)
    # the runtime evaluates the vertical reconstruction at the 3-D position's
    # vertical component; bind the unit-interval ζ to a real ``z`` (= position[2])
    # so interpolate_to_3d lambdifies with a genuine argument (not a free ζ).
    # A truly z-dependent (un-σ-mapped) definition already reads in ``z`` and
    # passes through untouched; mixing BOTH coordinates in one expression is
    # ambiguous and refused.
    y_pos, z_pos = sp.Symbol("y", real=True), sp.Symbol("z", real=True)
    new_aux: list = []
    # Did a function group (re)register the WB reconstruction forward map?  If
    # so it is the AUTHORITATIVE runtime map (the limiter reconstructs these
    # primitives), and ``state_from_reconstruction`` must be re-derived as its
    # genuine inverse — a model-method inverse built from a stale/duplicate
    # ``reconstruction_variables()`` may disagree (e.g. SWE's method left the
    # velocity row identity while the group limits ``hu/h``).  REQ-180.
    recon_group_registered = False

    def _parse(rhs, group, idx):
        """Rewrite one group right-hand side into bare state symbols and gradient aux, mapping
        `zeta` onto the vertical position.
        """
        # ``.doit()`` expands derivatives of compounds (the CoV leaves
        # ``∂_x(q_i/h)``) into the bare-field gradients ``∂_x q_i``, ``∂_x h``
        # so the gradient aux are named cleanly (``dq0dx``, ``dhdx``).
        e, grads = _gradients_to_aux(sp.sympify(rhs).doit(), sm.space)
        new_aux.extend(grads)
        e = _fields_to_state_symbols(sp.expand(e), x)
        if e.has(zeta) and e.has(z_pos):
            raise ValueError(
                f"function group {group!r}[{idx}] mixes the σ-reference "
                "coordinate ζ ('zeta') and the physical vertical 'z' — "
                "ambiguous; σ-map the definition fully or write it in z.")
        return e.subs(zeta, z_pos)

    def _state_slot(key):
        """Resolve a boundary-group key (a state field) to its slot in `sm.state`."""
        if isinstance(key, int):
            return key
        sym = _fields_to_state_symbols(sp.sympify(key), x)
        state = list(sm.state)
        if sym not in state:
            raise KeyError(
                f"boundary group key {key} → {sym} is not a state entry "
                f"of {[str(s) for s in state]}")
        return state.index(sym)

    for group, comps in groups.items():
        if group.startswith("boundary:"):
            spec = {_state_slot(k): _parse(rhs, group, k)
                    for k, rhs in comps.items()}
            specs = getattr(sm, "boundary_specs", None) or {}
            specs[group[len("boundary:"):]] = spec
            sm.boundary_specs = specs
            continue
        attr = _FUNCTION_SLOTS.get(group)
        if attr is None:
            continue
        # Keys may be ints (slot positions — for ``interpolate`` these are
        # CANONICAL-PROFILE slots, not state slots) or state FIELDS, resolved
        # to their state slot — so registrations never hard-code the layout.
        comps = {_state_slot(k): rhs for k, rhs in comps.items()}
        if attr == "reconstruction_variables":
            # state-shaped map; unregistered slots default to the IDENTITY
            # (reconstruct the conservative entry itself), never to zero.
            vec = list(sm.state)
        else:
            vec = [sp.S.Zero] * (max(comps) + 1)
        for idx, rhs in comps.items():
            vec[idx] = _parse(rhs, group, idx)
        # reconstruction_variables is consumed entry-wise (invert, runtime
        # limiting) — rank-1; the 3-D maps stay column matrices.
        arr = (sp.Array(vec) if attr == "reconstruction_variables"
               else sp.Matrix(vec))
        setattr(sm, attr, _to_zarray(arr))
        if attr == "reconstruction_variables":
            recon_group_registered = True
    # A model-registered WB reconstruction implies its inverse (the runtime
    # feeds reconstructed primitives back through ``WB_<name>`` symbols).  Derive
    # it whenever a group set the forward map (that map is authoritative — its
    # inverse must ×-back every quotient row, e.g. hu = WB_hu·(η−b)), or when a
    # model supplied ``reconstruction_variables`` but no inverse yet.
    if getattr(sm, "reconstruction_variables", None) is not None and (
            recon_group_registered
            or getattr(sm, "state_from_reconstruction", None) is None):
        from zoomy_core.model.reconstruction_inverse import (
            invert_reconstruction,
        )
        sm.state_from_reconstruction = invert_reconstruction(
            sm.reconstruction_variables, list(sm.state))
    # Append the reconstruction gradient aux to aux_state AND register them in
    # aux_registry, so the solver's shared derivative-aux walk
    # (Solver._walk_derivative_aux / the jax canonical update_qaux) computes
    # them.  WITHOUT the registration the slot lowers but reads ZERO
    # derivatives (a wrong w); WITHOUT the dimension-complete _gradients_to_aux
    # above the 2-D w slot kept raw ``∂_y`` atoms and was dropped by the jax
    # printer.  Append to whatever operator-scan registry expose_aux_atoms
    # already built; de-dup by symbol name (a derivative may appear in both an
    # operator and a reconstruction slot).
    state_syms = {str(s): i for i, s in enumerate(sm.state)}
    registry = list(getattr(sm, "aux_registry", None) or [])
    registered = {e.get("name") for e in registry}
    for sym, target, mi in new_aux:
        if sym not in sm.aux_state:
            sm.aux_state = list(sm.aux_state) + [sym]
        if sym.name in registered:
            continue
        tname = str(_fields_to_state_symbols(sp.sympify(target), x))
        entry = {
            "kind": "derivative", "name": sym.name,
            "row": list(sm.aux_state).index(sym), "atom": None,
            "aux_symbol": sym, "target_name": tname, "multi_index": tuple(mi),
        }
        if tname in state_syms:
            entry["target_kind"] = "state"
            entry["state_index"] = state_syms[tname]
        else:
            frow = next((e["row"] for e in registry
                         if e.get("kind") == "function"
                         and e.get("name") == tname), None)
            if frow is not None:
                entry["target_kind"] = "function"
                entry["function_row"] = frow
            else:
                entry["target_kind"] = "unknown"
        registry.append(entry)
        registered.add(sym.name)
    sm.aux_registry = registry
    # Appending the reconstruction gradient aux changed ``aux_state``; resize
    # source_jacobian_wrt_aux_variables to the full aux vector (these gradient
    # aux do not enter the source, so the new columns are zero — but the field
    # must stay shape-consistent for the printers that read it).
    if new_aux:
        sm.source_jacobian_wrt_aux_variables = _to_zarray(
            sm._compute_source_jacobian_wrt_aux_variables())
        if getattr(sm, "derive_jacobians", False) and len(sm.aux_state or []):
            sm.flux_jacobian_wrt_aux_variables = _to_zarray(
                sm._compute_flux_jacobian_wrt_aux_variables())
    if sm.position is None:
        sm.position = Zstruct(X0=x, X1=y_pos, X2=z_pos)


# ── spectral-radius bound (no eigen-decomposition) ───────────────────────


def _perron_radius_bound(M, iterations):
    """`Min` over `k <= iterations` of `(max_i (M^k 1)_i)^(1/k)`, an upper bound on
    `rho(M)` for a nonnegative symbolic matrix `M` given as rows; see
    `SystemModel.spectral_radius_bound`.

    References:
        Golub, Van Loan 2013, https://doi.org/10.56021/9781421407944
    """
    n = len(M)
    v = [sp.Integer(1)] * n
    candidates = []
    for k in range(1, iterations + 1):
        v = [sum(M[i][j] * v[j] for j in range(n)) for i in range(n)]   # Mᵏ·1
        peak = sp.Max(*v, evaluate=False)
        # ``evaluate=False`` on the root as well: ``sqrt(a·b)`` otherwise
        # SPLITS into ``sqrt(a)·sqrt(b)`` (sympy can see the factors are
        # non-negative), and the C printers emit one ``pow`` call per factor.
        candidates.append(peak if k == 1 else
                          sp.Pow(peak, sp.Rational(1, k), evaluate=False))
    if len(candidates) == 1:
        return candidates[0]
    return sp.Min(*candidates, evaluate=False)


class SpectralRadius(Option):
    """How a Riemann solver bounds the wave speed rho(A_n) when no closed-form eigenvalues exist."""
    #: numerical eigenvalues of A_n
    eig = "eig"
    #: the upper bound `||A_n^m||^(1/m)`, `m = 2^spectral_squarings`, with a Collatz-Wielandt
    #: step (`kernel_functions.gelfand_bound`; zoomy_core.symsolver.WAVE_SPEED)
    gelfand = "gelfand"
    #: the closed-form scalar upper bound (cheaper, more diffusive)
    bound = "bound"


@dataclass
class SystemModel:
    """Operator-form PDE system `M dQ/dt + div(F + P) + sum_d B[:,:,d] d_d Q - S = 0` with
    operators indexed `(equation_row, state_col[, dim])`.
    It may be rectangular (`equation_to_state_index` maps equations to state entries); derived
    operators are cached or refreshed by `refresh_derived_operators`.
    """

    time: sp.Symbol
    space: List[sp.Symbol]
    state: List[Any]
    aux_state: List[Any]
    parameters: Zstruct                      # name -> Symbol
    # Every operator tensor is a :class:`ZArray` — the project's
    # unified symbolic tensor type.  ZArray inherits from
    # :class:`sp.MutableDenseNDimArray`, so it is a drop-in for any
    # sympy NDimArray consumer (lambdify, xreplace, atoms, etc.) while
    # carrying our matrix-algebra extensions (``@``, ``+``, ``-`` with
    # lists / Zstructs).  Eigenvalue computation (the one place that
    # genuinely needs ``sp.Matrix``) wraps via ``zarray.tomatrix()`` →
    # ``M.eigenvals()`` → ``ZArray(...)``.
    flux: ZArray                             # (n_eq, n_dim)
    hydrostatic_pressure: ZArray             # (n_eq, n_dim)
    nonconservative_matrix: ZArray           # (n_eq, n_state, n_dim)
    source: ZArray                           # (n_eq,)
    mass_matrix: ZArray                      # (n_eq, n_state)
    # Derived operators.  ``quasilinear_matrix`` (flux Jacobian) is LAZY:
    # exposed as a cached property that computes from the primaries on FIRST
    # access (and caches), not eagerly at construction.  The source jacobians
    # are real FIELDS (channeled from the Model — see below).
    # ``eigenvalues`` stays ``None`` when skipped.
    eigenvalues: Optional[ZArray] = None           # (n_eq,)
    # ``eigenvalues_cfl`` — the TIMESTEP-ONLY spectrum.  ``None`` (the normal
    # case) means "no separate estimate; the dt path reads ``eigenvalues``".
    # It exists because the two consumers of the spectrum want OPPOSITE things
    # at a dry cell (cid=209): the dt/CFL reduction wants spurious dry-cell
    # speeds suppressed so they do not drive dt to zero, while the Rusanov
    # face dissipation needs the TRUE ``sqrt(g h)`` — zeroing it there removes
    # the artificial viscosity the first-order positivity argument rests on,
    # and the MOOD cascade then exhausts at the wet/dry front.  So
    # :func:`~zoomy_core.systemmodel.operations.gate_eigenvalues_dry` publishes
    # its gated copy HERE and leaves ``eigenvalues`` physical.  Same shape and
    # same lowering signature as ``eigenvalues``.
    eigenvalues_cfl: Optional[ZArray] = None      # (n_eq,)
    # ∂S/∂Q and ∂S/∂Qaux — CHANNELED from the source Model in
    # :meth:`from_model` (the Model already derived & cached them — printers
    # READ them off the SystemModel and never re-derive with ``sp.diff``).  A
    # SystemModel built WITHOUT a Model (standalone analysis, or the structural
    # derivation path whose ``derivation.model.Model`` carries no precomputed
    # jacobian) leaves these ``None``; ``__post_init__`` then derives them once
    # from ``source`` via ``sp.diff`` as the FALLBACK only.  Canonical
    # orientation ``(n_eq, n_state)`` / ``(n_eq, n_aux)`` — row = equation,
    # column = state/aux field (the Model's ``derive_by_array`` output is the
    # transpose, reoriented at the channel).
    source_jacobian_wrt_variables: Optional[ZArray] = None      # (n_eq, n_state)
    source_jacobian_wrt_aux_variables: Optional[ZArray] = None  # (n_eq, n_aux)
    # dF/dQaux — (n_eq, n_aux, n_dim).  The FLUX twin of the source jacobian:
    # needed wherever a total x-derivative of the flux is taken and the flux
    # depends on aux, since d/dx F(Q,Qaux) = dF/dQ·dQ/dx + dF/dQaux·dQaux/dx.
    # Without it that second term is silently dropped.
    flux_jacobian_wrt_aux_variables: Optional[ZArray] = None    # (n_eq,n_aux,n_dim)
    #: Derive ``flux_jacobian_wrt_aux_variables`` (and any future aux-side
    #: jacobian) when a Model did not channel it?  OFF by default: it is
    #: ``n_eq x n_aux x n_dim`` sympy.diff calls, a real cost on a high-level
    #: moment model, and wasted whenever nothing downstream asks for it.
    #:
    #: Scope is deliberately narrow.  The LONG-STANDING source jacobians
    #: (``source_jacobian_wrt_variables`` / ``_wrt_aux_variables``) are NOT
    #: gated — IMEX and friends have always relied on them being derived, and
    #: making them optional would break consumers that predate this flag.
    #: Consumers wanting the new aux-side jacobian call
    #: :meth:`ensure_jacobians` before the runtime is lowered.
    derive_jacobians: bool = False
    #: Derive ``source_jacobian_wrt_{variables,aux_variables}`` symbolically.
    #: MEASURED on QRKESME N_k=1: this is 600 s of a 661 s lowering stage, i.e.
    #: 90 per cent of codegen and 71 per cent of the whole build, because it is
    #: an (n_eq x n_state) matrix of ``sp.diff`` over a quadrature-unrolled
    #: source, each entry then lambdified.  It is REQUIRED by the analytic
    #: implicit path; it is DEAD WEIGHT for solvers that form the
    #: Jacobian-vector product by AD or FD (``IMEXSourceSolverJax`` defaults to
    #: ``jv_backend="ad"``).  Default stays True so existing consumers are
    #: unaffected; set False when the solver does not read it.
    #: Env override ZOOMY_SOURCE_JACOBIANS=0 turns it off process-wide, which is
    #: how a case opts out without threading the flag through every builder.
    derive_source_jacobians: bool = field(
        default_factory=lambda: os.environ.get("ZOOMY_SOURCE_JACOBIANS", "1") != "0")
    normal: Optional[Zstruct] = None
    parameter_values: Optional[Zstruct] = None   # name -> numeric default
    equation_to_state_index: Optional[List[int]] = None
    # ``boundary_conditions`` / ``aux_boundary_conditions`` are the
    # *indexed symbolic BC functions* (``Function`` objects with
    # ``Piecewise(…, Eq(idx,i))`` definition).  The runtime
    # (``from_system_model``) lambdifies them; solvers call
    # ``runtime.boundary_conditions(bc_idx, time, position, distance,
    # Q, Qaux, p, normal)`` per face.
    # ``boundary_gradients`` is the parallel kernel for the
    # face-normal gradient ``∂Q/∂n`` — consumed by the diffusion path.
    boundary_conditions: Optional[Any] = None
    aux_boundary_conditions: Optional[Any] = None
    boundary_gradients: Optional[Any] = None
    # Application-logic carried over from the source Model so the
    # SystemModel is the self-contained numerical model a solver
    # consumes — no Model needed alongside it.
    initial_conditions: Optional[Any] = None
    aux_initial_conditions: Optional[Any] = None
    # ``update_variables``: per-cell state transform (h-clamp etc.) — an
    # ``(n_eq, 1)`` expression in ``(Q, Qaux, p)``.
    #
    # Two-slot IMEX operator design — solvers route by treatment:
    #
    # - ``source``                — implicit (default).  Added to the
    #   source-step Newton residual at ``Qnp1``; appropriate for stiff
    #   friction / reaction.
    # - ``source_explicit``       — explicit.  Added to the convective
    #   step at ``Qn``; appropriate for body forces / gravity / cheap
    #   non-stiff RHS.
    # - ``diffusion_matrix``      — implicit (default).  IP-DG/TPFA
    #   evaluated at ``Qnp1`` inside the source step.  No parabolic CFL.
    # - ``diffusion_matrix_explicit`` — explicit.  IP-DG/TPFA evaluated
    #   at ``Qn`` inside the convective step.  Subject to
    #   ``dt ≤ h²/(2ν)`` on top of the hyperbolic CFL.
    #
    # The constitutive tensor ``A(Q, Qaux, p)`` has shape
    # ``(n_eq, n_state, n_dim, n_dim)``; the diffusive flux is recovered
    # downstream as ``F_diff[i, d] = Σ_{j, e} A[i, j, d, e] · ∂_e Q[j]``,
    # and the residual contributes ``-∇·(A:∇Q)``.  Derivatives enter
    # ``A`` only via ``Qaux`` (auto-derived as needed) so ``A`` itself
    # is a pure ``(Q, Qaux, p)`` expression — same call convention as
    # every other operator.
    #
    # Backends that support only one treatment (e.g. explicit-only FV)
    # may *compound* the two: ``F_total = F_explicit + F_implicit``.
    # ``update_variables``: per-cell explicit state remap lowered as
    # ``update_variables(Q, Qaux, p, dt)``.  For a full model it returns the
    # whole state (h-clamp, momentum ramp; dt typically unused).  For a Chorin
    # corrector sub-system it returns one value per ``equation_to_state_index``
    # row — the closed-form projection ``Q[e2s] ← U_k − (dt/h)·T_u_k(P)`` — so
    # the dt argument is load-bearing there.  ``None`` ⇒ identity.
    update_variables: Optional[ZArray] = None        # (n_eq, 1)
    # Per-cell LOCAL aux formula (e.g. KP-desingularized hinv); symmetric to
    # update_variables but for the aux vector, lowered as
    # ``update_aux_variables(Q, Qaux, p, dt)``.  ``None`` ⇒ identity.  The
    # runtime lowers + applies it (the aux leg of post_step).  Derivative-aux
    # (LSQ gradients) is a SEPARATE, non-local leg (aux_registry) — not this.
    update_aux_variables: Optional[ZArray] = None     # (n_aux_decl, 1)
    diffusion_matrix: Optional[ZArray] = None         # (n_eq, n_state, n_dim, n_dim) — implicit
    diffusion_matrix_explicit: Optional[ZArray] = None  # (n_eq, n_state, n_dim, n_dim) — explicit
    source_explicit: Optional[ZArray] = None          # (n_eq,) — explicit
    # ``reconstruction_variables``: symbolic forward map state → primitive
    # well-balanced variables used by MUSCL-style reconstruction
    # (η = h+b, u = q_U/h, …).  ``state_from_reconstruction`` is the
    # symbolic inverse in terms of fresh ``WB_<state_name>`` symbols,
    # auto-derived from the forward via
    # :func:`zoomy_core.model.reconstruction_inverse.invert_reconstruction`.
    # See ``thesis/chapters/30_numerics.md`` "Primitive-variable MUSCL
    # reconstruction".  The pair is refreshed whenever the state vector
    # changes (CoV via :meth:`change_state_variables`).
    reconstruction_variables: Optional[ZArray] = None      # (n_state,)
    state_from_reconstruction: Optional[ZArray] = None     # (n_state,) in WB_<name> symbols
    # ``interpolate_to_3d``: depth-averaged → 3D reconstruction.  Vector of
    # 3D physical quantities (typically ``[u, v, w, p, ...]``) evaluated at
    # ``(x, y, z)`` given the cell-mean state.  The symbolic expression
    # references ``self.position`` (x, y, z) so the runtime evaluates it
    # at a caller-supplied ``z`` per cell.  Used by post-processing /
    # sensitivity analyses that need to lift the 2D solution back to a
    # 3D velocity profile (Manning friction sensitivity at the bed, max
    # velocity over depth, etc.).  Default ``None`` ⇒ no projection
    # defined; the runtime exposes ``None`` and callers handle absence.
    interpolate_to_3d: Optional[ZArray] = None        # (n_3d_components,)
    # ``project_from_3d``: the inverse of ``interpolate_to_3d`` — a 3-D
    # field column (canonical ``[b,h,u,v,w,p]`` profile sampled over the
    # vertical) → the depth-averaged state.  Each row is a Galerkin
    # projection ``∫₀¹ g(ζ) dζ`` left OPAQUE as a column-quadrature
    # accumulator the backend fills (the ``I[j]`` convention emitted by the
    # code printers), so the symbolic operator carries the basis weights and
    # the backend supplies the quadrature.  Declared symmetrically with
    # ``interpolate_to_3d`` and filled by the SAME function-group registry
    # (``register_group("project", …)`` / the canonical ``project()``
    # method).  Default ``None`` ⇒ no projection defined (a depth-collapsed
    # model with no 3-D maps); the runtime exposes ``None``.
    project_from_3d: Optional[ZArray] = None          # (n_state,)
    # 3D position symbols used by ``interpolate_to_3d`` (and any future
    # 3D-aware operator).  Default ``None`` → SystemModel constructed
    # without a model context; the interpolate_to_3d slot then carries
    # no position dependence (and runtime treats it as a pure cell-
    # local expression).  When set from ``Model.position`` it is a
    # Zstruct ``Zstruct(X0=…, X1=…, X2=…)`` even for 2D models.
    position: Optional[Any] = None
    # Well-balanced reconstruction strategy carried from the declarative Model
    # (Model.equilibrium_reconstruction): 'none' | 'audusse' | 'bernoulli'.
    # The numpy solver reads it in ``_build_numerics`` to select the WB numerics.
    equilibrium_reconstruction: str = "none"
    # Per-state-field applied-Function map — ``h → h(t,x)``, ``ũ → ũ(t,x,ζ)`` —
    # carrying each field's FULL coordinate signature so per-field dimensionality
    # survives the ``_state_symbol`` name-collapse (the state twin of
    # ``aux_function_map``).  Read it via :meth:`is_vertical_dependent`.  Default
    # ``None`` ⇒ no signatures recorded (e.g. a SystemModel assembled directly).
    state_function_map: Optional[Dict] = None
    # The vertical coordinate Symbol (``model.coords[-1]`` / the σ ``ζ`` after the
    # σ-map), or ``None`` for a depth-collapsed model with no vertical direction.
    vertical: Optional[Any] = None
    # Terms the structural extractor matched NO classification branch for — a
    # per-row ``(row_index, remainder)`` list produced by the term-conservation
    # guard in ``system_extract._classification_remainder``.  EMPTY for a
    # correctly-routed model (every additive term reached a slot); a nonzero
    # entry is a SILENTLY-DROPPED term (the class of bug that hid the VAM
    # bed-slope diffusion).  Surfaced by ``describe(full=True)`` and enforced by
    # :meth:`assert_all_classified`.  Only the declarative extraction path sets
    # it; hand-built production models leave it empty.
    unclassified_terms: List = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    # ADMISSIBILITY: expressions in (Q, Qaux, p) that must stay >= 0 for the
    # state to be physical (free surface: [depth]; Euler: [rho, p(Q)]).  The
    # model DECLARES it (``declare_admissible``); ``None`` means "not declared",
    # and :meth:`admissible_constraints` then derives the default from the
    # declared roles ([depth] for a free-surface model, [] otherwise).
    admissible: Optional[ZArray] = None

    def __post_init__(self):
        """Normalise all operator fields to `ZArray` and derive the source jacobians when no
        Model channeled them.
        """
        # Lazy derived-operator cache — filled on first ``quasilinear_matrix``
        # access.  The source jacobians are real FIELDS (channeled in
        # ``from_model``); the fallback below derives them once when absent.
        self._quasilinear_matrix = None
        # Normalise every operator tensor field to ZArray.  Construction
        # sites that still produce sympy types are wrapped transparently.
        self.flux                   = _to_zarray(self.flux)
        self.hydrostatic_pressure   = _to_zarray(self.hydrostatic_pressure)
        self.nonconservative_matrix = _to_zarray(self.nonconservative_matrix)
        self.source                 = _to_zarray(self.source)
        self.mass_matrix            = _to_zarray(self.mass_matrix)
        self.eigenvalues            = _to_zarray(self.eigenvalues)
        self.eigenvalues_cfl        = _to_zarray(self.eigenvalues_cfl)
        self.update_variables           = _to_zarray(self.update_variables)
        self.diffusion_matrix           = _to_zarray(self.diffusion_matrix)
        self.diffusion_matrix_explicit  = _to_zarray(self.diffusion_matrix_explicit)
        self.source_explicit            = _to_zarray(self.source_explicit)
        self.update_aux_variables       = _to_zarray(self.update_aux_variables)
        self.reconstruction_variables   = _to_zarray(self.reconstruction_variables)
        self.state_from_reconstruction  = _to_zarray(self.state_from_reconstruction)
        self.interpolate_to_3d           = _to_zarray(self.interpolate_to_3d)
        self.project_from_3d             = _to_zarray(self.project_from_3d)
        self.source_jacobian_wrt_variables = _to_zarray(
            self.source_jacobian_wrt_variables)
        self.source_jacobian_wrt_aux_variables = _to_zarray(
            self.source_jacobian_wrt_aux_variables)
        self.flux_jacobian_wrt_aux_variables = _to_zarray(
            self.flux_jacobian_wrt_aux_variables)
        self.admissible = _to_zarray(self.admissible)

        # FALLBACK only: a SystemModel built without a Model channel (standalone
        # analysis, or the structural derivation path) arrives here with the
        # source jacobians unset — derive them once from ``source`` via
        # ``sp.diff``.  When ``from_model`` channeled the Model's matrices these
        # are already populated and we leave them untouched.
        if (self.derive_source_jacobians and self.source is not None
                and self.source_jacobian_wrt_variables is None):
            self.source_jacobian_wrt_variables = _to_zarray(
                self._compute_source_jacobian_wrt_variables())
        if (self.derive_source_jacobians and self.source is not None
                and self.source_jacobian_wrt_aux_variables is None):
            self.source_jacobian_wrt_aux_variables = _to_zarray(
                self._compute_source_jacobian_wrt_aux_variables())
        if (self.derive_jacobians and self.flux is not None
                and self.flux_jacobian_wrt_aux_variables is None
                and len(self.aux_state or [])):
            self.flux_jacobian_wrt_aux_variables = _to_zarray(
                self._compute_flux_jacobian_wrt_aux_variables())

        if self.equation_to_state_index is None:
            self.equation_to_state_index = list(range(self.n_equations))
        if self.normal is None:
            self.normal = Zstruct(
                **{str(n): n for n in face_normal_symbols(self.n_dim)}
            )
            self.normal._symbolic_name = "n"
        if self.parameter_values is None:
            self.parameter_values = Zstruct()

    # ── Shape accessors ────────────────────────────────────────────────

    @property
    def n_equations(self) -> int:
        """Number of equations (rows of the operators)."""
        return self.flux.shape[0]

    @property
    def n_state(self) -> int:
        """Number of state entries."""
        return len(self.state)

    @property
    def n_dim(self) -> int:
        """Number of space dimensions."""
        return len(self.space)

    # ── Free-surface ROW MAP ───────────────────────────────────────────
    # The ONE answer-holder for "which state row is what" in a free-surface
    # system.  Every consumer READS these — the Audusse hydrostatic
    # reconstruction (``fvm.riemann_solvers.PositiveRusanov``), the HLLC
    # contact wave, and the emitted C headers (``transformation.generic_c``
    # publishes them so a hand-written driver loop cannot guess).
    #
    # WHY THEY EXIST.  A consumer that infers the layout from the state ORDER
    # is right only for the four-row SWE state ``[b, h, q_x_0, q_y_0]``, where
    # the two momentum rows happen to sit contiguously after ``h``.  Every
    # wider free-surface system breaks that: ``SME(level=N, dimension=3)`` lays
    # out ``[b, h, q_x_0..q_x_N, q_y_0..q_y_N]``, so ``idx_h + 2`` is
    # ``q_x_1``, NOT ``q_y_0``.  MEASURED consequence of the guess in the amrex
    # driver: the Audusse rescale skipped ``q_y_0``, a face whose reconstructed
    # depth was 0 still exported y-momentum, and the LowerTriangle catchment
    # reached ``h = -0.170 m`` at level 2 (thesis/cases/river/serghei/run/
    # matrix_sme/README.md, "Positivity at order 1").

    @property
    def depth_scaled_state_indices(self) -> list:
        """State rows carrying one factor of the depth, i.e. every row except depth and bed, which
        a hydrostatic reconstruction must rescale by `h*/h`.

        References:
            Audusse et al. 2004, https://doi.org/10.1137/S1064827503431090
        """
        roles = self.roles
        if roles.depth_declared:
            skip = {roles.depth, roles.bed}
            return [i for i in range(len(self.state)) if i not in skip]
        # LEGACY (undeclared hand-built systems only): the conventional names.
        return [i for i, s in enumerate(self.state)
                if str(s) not in ("h", "b")]

    @property
    def discharge_state_indices(self) -> list:
        """Per horizontal direction, the state row that is the depth-integrated mass flux (`hu`,
        `q_x_0`), read off the mass balance; `-1` where no unique row exists.
        """
        roles = self.roles
        if roles.depth_declared:
            i_h = roles.depth
        else:
            # LEGACY (undeclared hand-built systems only): the conventional name.
            names = [str(s) for s in self.state]
            i_h = names.index("h") if "h" in names else None
        if i_h is None:
            return [-1] * self.n_dim
        e2s = list(self.equation_to_state_index)
        if i_h not in e2s:            # h is carried but not evolved here
            return [-1] * self.n_dim
        mass_row = e2s.index(i_h)
        out = []
        for d in range(self.n_dim):
            expr = sp.sympify(self.flux[mass_row, d])
            if self.hydrostatic_pressure is not None:
                expr = expr + sp.sympify(self.hydrostatic_pressure[mass_row, d])
            hits = []
            for k, s in enumerate(self.state):
                dv = sp.diff(expr, s)
                if dv != 0 and sp.simplify(dv) != 0:
                    hits.append(k)
            out.append(hits[0] if len(hits) == 1 else -1)
        return out

    # ── Model-compatible facade ───────────────────────────────────────
    # So a solver can consume a SystemModel exactly where it used to
    # consume a Model — same accessor surface, SystemModel internals.

    @property
    def n_variables(self) -> int:
        """Number of state variables (equal to `n_equations`)."""
        return self.n_equations

    @property
    def n_aux_variables(self) -> int:
        """Number of aux variables."""
        return len(self.aux_state)

    @property
    def n_parameters(self) -> int:
        """Number of parameters."""
        return self.parameters.length()

    @property
    def dimension(self) -> int:
        """Number of space dimensions (`Model` API alias of `n_dim`)."""
        return self.n_dim

    @property
    def variables(self) -> Zstruct:
        """Zstruct view `name -> state Symbol` (mirrors `Model.variables`)."""
        v = Zstruct(**{str(s): s for s in self.state})
        v._symbolic_name = "Q"
        return v

    def update_variables_or_identity(self) -> ZArray:
        """`update_variables` as an explicit `(n_eq, 1)` expression, the identity
        `state[equation_to_state_index[i]]` when the model has none; the C-family printers
        emit from this.
        """
        if (self.update_variables is not None
                and len(sp.flatten(self.update_variables)) > 0):
            return self.update_variables
        e2s = list(self.equation_to_state_index)
        return _to_zarray(
            sp.Array([self.state[e2s[i]] for i in range(self.n_equations)])
        ).reshape(self.n_equations, 1)

    @property
    def aux_variables(self) -> Zstruct:
        """Zstruct view `name -> aux Symbol` (mirrors `Model.aux_variables`)."""
        v = Zstruct(**{str(s): s for s in self.aux_state})
        v._symbolic_name = "Qaux"
        return v

    @property
    def _parameter_symbols(self) -> Zstruct:
        """Alias of `parameters` (name -> Symbol) for Model-API parity."""
        return self.parameters

    # ── Operators carried as declared Functions (single signature source) ──

    def position_struct(self) -> Zstruct:
        """The length-3 position group `(x, y, z)` that printers and analysis read; a bare
        system resolves the canonical symbols.
        """
        if self.position is not None:
            return self.position
        return Zstruct(X0=self.space[0],
                       X1=sp.Symbol("y", real=True),
                       X2=sp.Symbol("z", real=True))

    _position_struct = position_struct

    def operator_signature(self, name: str) -> Zstruct:
        """The declared `args` Zstruct of operator `name`, built from this system's symbols in
        `OPERATOR_ARG_SLOTS` order; the single source every backend reads.
        """
        group = {
            "variables": self.variables,
            "aux_variables": self.aux_variables,
            "parameters": self.parameters,
            "normal": self.normal,
            "time": self.time,
            "dt": _DT_SYMBOL,
            "position": self._position_struct(),
            "reconstruction": self._reconstruction_struct(),
        }
        return Zstruct(**{s: group[s] for s in OPERATOR_ARG_SLOTS[name]})

    def _reconstruction_struct(self) -> Zstruct:
        """The `WB_<state>` symbols in which the reconstruction inverse
        `state_from_reconstruction` is written, since it takes limiter face values rather
        than the cell state.
        """
        from zoomy_core.model.reconstruction_inverse import reconstruction_symbols
        syms = reconstruction_symbols(list(self.state))
        return Zstruct(**{s.name: s for s in syms})

    def operator(self, name: str) -> "Any":
        """Operator `name` as a `basefunction.Function` whose `args` is the declared
        signature and whose `definition` is the live array; backends only translate syntax.
        """
        from zoomy_core.model.basefunction import Function
        defn = getattr(self, name, None)
        if name == "admissible" and defn is None:      # declared, else [depth]
            rows = self.admissible_constraints()
            defn = ZArray(rows) if rows else None
        return Function(name=name, args=self.operator_signature(name),
                        definition=defn)

    @property
    def eigenvalue_mode(self) -> str:
        """`"symbolic"` when a symbolic spectrum is carried, `"numerical"` when it was skipped
        (`eigenvalues is None`).
        """
        return "numerical" if self.eigenvalues is None else "symbolic"

    def attach_boundary_conditions(self, bcs, aux_bcs=None):
        """Build the indexed symbolic BC kernels (`boundary_conditions`, `boundary_gradients`,
        `aux_boundary_conditions`) from existing `BoundaryConditions` objects.

        Args:
            bcs: a `BoundaryConditions` container (or list).
            aux_bcs: aux BCs; defaults to per-tag Extrapolation.
        """
        from zoomy_core.model.boundary_conditions import (
            BoundaryConditions, PerFieldBoundary, Periodic, Coupled,
            resolve_per_field)
        import zoomy_core.model.aux_boundary_conditions as AuxBC
        state_names = [str(s) for s in self.state]
        # The ``on=`` field mask is rooted on BoundaryCondition (each BC self-
        # masks its owned slots), so a single per-field BC per tag needs no
        # wrapper.  Only tags carrying MULTIPLE explicit BCs must be merged into
        # one PerFieldBoundary (the runtime dispatches ONE bc per tag).  Accept a
        # flat list or a BoundaryConditions container; whole-patch BCs pass
        # through.  Already-composed input (resolve_and_attach) is left as-is.
        raw = (list(bcs) if isinstance(bcs, list)
               else list(bcs.boundary_conditions_list))
        by_tag = {}
        for bc in raw:
            by_tag.setdefault(bc.tag, []).append(bc)
        merged = []
        for tag, group in by_tag.items():
            non_patch = [b for b in group
                         if not isinstance(b, (PerFieldBoundary, Periodic, Coupled))]
            if len(non_patch) > 1:               # multiple BCs at one tag → merge
                merged.extend(
                    resolve_per_field(group, state_names,
                                      aliases={"momentum": "q"}).boundary_conditions_list)
            else:
                merged.extend(group)
        bcs = BoundaryConditions(merged)
        # Bind each BC's ``on=`` to slot indices against this state (so the root
        # mask fires), and run the SystemModel-needing hooks (FromModel,
        # Characteristic).  Recurse into any PerFieldBoundary slot BCs and
        # face= / ghost= channel BCs.
        def _prepare(bc):
            """Resolve `bc` against this system and bind it to the state names, recursing into
            slot and channel BCs.
            """
            if hasattr(bc, "resolve"):
                bc.resolve(self)
            bc.bind_on(state_names)
            nested = [*getattr(bc, "_slot_bc", {}).values(),
                      *getattr(bc, "channels", dict)().values()]
            for inner in {id(v): v for v in nested}.values():
                _prepare(inner)
        for bc in bcs.boundary_conditions_list:
            _prepare(bc)
        if self.position is None:
            # the BC kernel signature needs a 3-component position; models
            # without registered function groups never set one.
            self.position = Zstruct(X0=self.space[0],
                                    X1=sp.Symbol("y", real=True),
                                    X2=sp.Symbol("z", real=True))
        dist = sp.Symbol("distance", real=True)
        sig = (self.time, self.position, dist, self.variables,
               self.aux_variables, self.parameters, self.normal)
        self.boundary_conditions = bcs.get_boundary_condition_function(
            *sig, function_name="boundary_conditions")
        self.boundary_gradients = bcs.get_boundary_gradient_function(
            *sig, function_name="boundary_gradients")
        if aux_bcs is None:
            tags = [bc.tag for bc in bcs.boundary_conditions_list]
            aux_bcs = BoundaryConditions([AuxBC.Extrapolation(tag=tg) for tg in tags])
        self.aux_boundary_conditions = aux_bcs.get_boundary_condition_function(
            *sig, function_name="aux_boundary_conditions")
        # Retain the source objects, mirroring the production ``from_model``
        # path — runtime consumers (face_value dispatch, codegen emitters)
        # read them from here instead of poking private attrs.
        self._bc_source = bcs
        self._aux_bc_source = aux_bcs
        return self

    # ── Declared roles + admissibility (general PDE core) ─────────────

    @property
    def roles(self):
        """The declared state roles resolved against the current state
        (`systemmodel.roles.Roles`).
        """
        from zoomy_core.systemmodel.roles import resolve_roles
        return resolve_roles(self)

    @property
    def tags(self):
        """The declared model tags (see `systemmodel/tags.py`)."""
        from zoomy_core.systemmodel.tags import declared_tags
        return declared_tags(self)

    def declare_tags(self, *tags):
        """Record model tags; returns `self`."""
        from zoomy_core.systemmodel.tags import declare_tags
        return declare_tags(self, tags)

    def declare_roles(self, **roles):
        """Record a role declaration (see `systemmodel.roles.declare_roles`); returns `self`."""
        from zoomy_core.systemmodel.roles import declare_roles
        return declare_roles(self, **roles)

    def declare_admissible(self, constraints):
        """Declare the admissibility list, expressions in `(Q, Qaux, p)` that must stay `>= 0`
        (empty: every state admissible); returns `self`.
        """
        exprs = [sp.sympify(e) for e in sp.flatten(constraints)] \
            if constraints is not None else None
        self.admissible = None if exprs is None else ZArray(exprs)
        self.history.append({"name": "declare_admissible",
                             "description": f"{len(exprs or [])} constraint(s)"})
        return self

    def boundary_ghost_kernel(self):
        """The ghost-state boundary kernel `(function, branches)`, built on demand (see
        `build_boundary_ghost_kernel`).
        """
        from zoomy_core.model.boundary_conditions import (
            build_boundary_ghost_kernel)
        return build_boundary_ghost_kernel(self)

    def admissible_constraints(self) -> list:
        """The admissibility list as sympy expressions: the declared one, else `[depth]` for a
        free-surface system, else `[]`.
        """
        if self.admissible is not None:
            return [sp.sympify(e) for e in sp.flatten(self.admissible)]
        depth = self.roles.depth
        return [] if depth is None else [self.state[depth]]

    @property
    def is_square(self) -> bool:
        """True when `n_equations == n_state` and equations map identity-style."""
        return (self.n_equations == self.n_state
                and self.equation_to_state_index == list(range(self.n_state)))

    @property
    def stationary_indices(self) -> frozenset:
        """State indices whose row vanishes in every transport, source and diffusion slot, i.e.
        fields stationary by construction (e.g. the bed).
        """
        n = self.n_variables
        slots = (
            self.flux, self.hydrostatic_pressure,
            self.nonconservative_matrix,
            self.source, self.source_explicit,
            self.diffusion_matrix, self.diffusion_matrix_explicit,
        )
        return frozenset(
            i for i in range(n)
            if all(self._row_is_identically_zero(t, i) for t in slots)
        )

    @staticmethod
    def _row_is_identically_zero(tensor, i) -> bool:
        """Whether `tensor[i, ...]` is identically zero (`None` counts as zero); checked with
        `== 0` only, never `simplify`.
        """
        if tensor is None:
            return True
        try:
            row = tensor[i]
        except (IndexError, TypeError):
            return True
        try:
            flat = np.asarray(row, dtype=object).ravel()
        except Exception:
            flat = [row]
        for entry in flat:
            if entry != 0:
                return False
        return True

    def field_index(self, field) -> int:
        """Integer state index of a field handle (Symbol or name matched by name, or an int
        bounds-checked); raises `ValueError` otherwise.
        """
        names = list(self.variables.keys())
        if isinstance(field, str):
            try:
                return names.index(field)
            except ValueError as e:
                raise ValueError(
                    f"Unknown state field {field!r}; available: {names}"
                ) from e
        if isinstance(field, (int, np.integer)):
            idx = int(field)
            if not 0 <= idx < len(names):
                raise ValueError(
                    f"State index {idx} out of range [0, {len(names)})"
                )
            return idx
        name = getattr(field, "name", None)
        if name is None:
            raise ValueError(
                f"Cannot resolve field handle {field!r} (type "
                f"{type(field).__name__}); expected sp.Symbol, str, "
                "or int."
            )
        try:
            return names.index(name)
        except ValueError as e:
            raise ValueError(
                f"Unknown state field {name!r}; available: {names}"
            ) from e

    # ── Derived-operator computation ────────────────────────────────────

    @property
    def quasilinear_matrix(self):
        """Flux Jacobian `dF/dQ + dP/dQ + B`, computed lazily from the primaries on first access
        and cached; assigning re-seeds the cache.
        """
        if self._quasilinear_matrix is None:
            self._quasilinear_matrix = _to_zarray(self._compute_quasilinear_matrix())
        return self._quasilinear_matrix

    @quasilinear_matrix.setter
    def quasilinear_matrix(self, value):
        """Re-seed the cached quasilinear matrix."""
        self._quasilinear_matrix = _to_zarray(value)

    def _compute_quasilinear_matrix(self):
        """`dF/dQ + dP/dQ + B` of shape `(n_eq, n_state, n_dim)`, from the clean primaries
        (bare `1/h`).
        `regularize_pow` freezes it before substituting `hinv`, so the `-u^2` wave-speed
        term survives.
        """
        n_eq = self.n_equations
        n_st = self.n_state
        d = self.n_dim
        Q = sp.MutableDenseNDimArray.zeros(n_eq, n_st, d)
        for i in range(n_eq):
            for j in range(n_st):
                for k in range(d):
                    djF = sp.diff(self.flux[i, k], self.state[j])
                    djP = sp.diff(self.hydrostatic_pressure[i, k],
                                  self.state[j])
                    Q[i, j, k] = djF + djP + self.nonconservative_matrix[i, j, k]
        return Q

    def _compute_source_jacobian_wrt_variables(self):
        """`dS/dQ` of shape `(n_eq, n_state)`; fallback derivation when no Model channeled it."""
        n_eq = self.n_equations
        n_st = self.n_state
        out = sp.zeros(n_eq, n_st)
        for i in range(n_eq):
            for j in range(n_st):
                out[i, j] = sp.diff(self.source[i, 0], self.state[j])
        return out

    def _compute_source_jacobian_wrt_aux_variables(self):
        """`dS/dQaux` of shape `(n_eq, n_aux)`; fallback derivation when no Model channeled
        it.
        """
        n_eq = self.n_equations
        n_aux = len(self.aux_state)
        out = sp.zeros(n_eq, n_aux)
        for i in range(n_eq):
            for k in range(n_aux):
                out[i, k] = sp.diff(self.source[i, 0], self.aux_state[k])
        return out

    def ensure_jacobians(self):
        """Derive the source and flux jacobians now if absent (off by default, see
        `derive_jacobians`); call before the runtime is lowered, since printers emit what is
        present.
        """
        self.derive_jacobians = True
        if self.source is not None and self.source_jacobian_wrt_variables is None:
            self.source_jacobian_wrt_variables = _to_zarray(
                self._compute_source_jacobian_wrt_variables())
        if (self.source is not None
                and self.source_jacobian_wrt_aux_variables is None
                and len(self.aux_state or [])):
            self.source_jacobian_wrt_aux_variables = _to_zarray(
                self._compute_source_jacobian_wrt_aux_variables())
        if (self.flux is not None
                and self.flux_jacobian_wrt_aux_variables is None
                and len(self.aux_state or [])):
            self.flux_jacobian_wrt_aux_variables = _to_zarray(
                self._compute_flux_jacobian_wrt_aux_variables())
        return self

    def _compute_flux_jacobian_wrt_aux_variables(self):
        """`dF/dQaux` of shape `(n_eq, n_aux, n_dim)`, the flux twin of the aux source
        jacobian needed for total flux derivatives.
        """
        n_eq = self.n_equations
        n_aux = len(self.aux_state)
        d = self.n_dim
        out = sp.MutableDenseNDimArray.zeros(n_eq, n_aux, d)
        for i in range(n_eq):
            for k in range(n_aux):
                for j in range(d):
                    out[i, k, j] = sp.diff(self.flux[i, j], self.aux_state[k])
        return out

    def _compute_eigenvalues(self):
        """Eigenvalues of the normal-projected quasilinear matrix, shape `(n_eq, 1)`; a
        potentially expensive symbolic derivation.
        """
        n_eq = self.n_equations
        n = list(self.normal.values())
        A = sp.zeros(n_eq, n_eq)
        for k in range(self.n_dim):
            for i in range(n_eq):
                for j in range(n_eq):
                    A[i, j] += n[k] * self.quasilinear_matrix[i, j, k]
        lam = sp.Symbol("lam")
        evs = sp.solve(A.charpoly(lam), lam)
        return sp.Matrix(n_eq, 1, lambda i, _j: sp.simplify(evs[i]))

    @property
    def eigensystem(self) -> ZArray:
        """Opaque eigendecomposition of the normal-projected quasilinear matrix ``A_n = sum_d n_d
        quasilinear_matrix[:, :, d]``.

        Returns:
            Flat row-major array `[lambda (n), R (n*n), L = R^-1 (n*n)]`.
        """
        from zoomy_core.model.kernel_functions import (
            eigensystem as _es, pick as _pick)
        n = self.n_equations
        qm = self.quasilinear_matrix
        normal = list(self.normal.values())
        A_flat = [
            sum(qm[i, j, d] * normal[d] for d in range(self.n_dim))
            for i in range(n) for j in range(n)
        ]
        stack = _es(*A_flat)
        return ZArray([_pick(stack, sp.Integer(idx))
                       for idx in range(n + 2 * n * n)])

    @property
    def numerical_eigenvalues(self) -> ZArray:
        """Eigenvalues of `A_n` as `n` calls to the opaque eigenvalue kernel, used for the
        wave speed when the model has no closed-form `eigenvalues` (SME, VAM).
        """
        from zoomy_core.model.kernel_functions import (
            eigenvalues as _ev, pick as _pick)
        n = self.n_equations
        qm = self.quasilinear_matrix
        normal = list(self.normal.values())
        A_flat = [
            sum(qm[i, j, d] * normal[d] for d in range(self.n_dim))
            for i in range(n) for j in range(n)
        ]
        stack = _ev(*A_flat)
        return ZArray([_pick(stack, sp.Integer(idx)) for idx in range(n)])

    # Power-norm steps in :attr:`spectral_radius_bound`.  A pure
    # cost/tightness dial — see that property for the measured table.
    # Deliberately NOT a dataclass field: it is a scheme constant, not
    # per-instance state, so it neither enters the SM cache key nor breaks
    # unpickling of older blobs.
    spectral_bound_iterations = 2

    # WHICH path ``Numerics.local_max_abs_eigenvalue`` takes when the model
    # carries no closed-form ``eigenvalues``.  DEFAULT FALSE: the exact
    # :attr:`numerical_eigenvalues` spectrum, not :attr:`spectral_radius_bound`.
    #
    # The bound is an UPPER bound, and ``max|lambda|`` is consumed TWICE: by the
    # CFL, where over-estimating is merely conservative, and by the Rusanov / LF
    # / HLL dissipation, where over-estimating is EXTRA NUMERICAL DIFFUSION.
    # ``spectral_radius_bound`` records its own tightness as a min ratio of
    # 1.224 over 5064 states, so the dissipation is at least ~22 % too large
    # wherever it is used, and looser near dry states.  MEASURED on the Ritter
    # dry dam break (thesis 1d_wetdry, order 2, n = 50…400): with the bound the
    # depth error is 1.44–1.55x the closed-form-spectrum result at every
    # resolution and the velocity error 2.7–3.5x, with the velocity EOC turning
    # erratic (0.56/1.96/1.37 against 0.93/1.09/0.95).  The convergence RATE
    # survives, which is why this hid: only the error CONSTANT moves.
    #
    # Set True to trade that accuracy back for speed.  The bound exists for a
    # real reason: it took the emitted foam ``SME(level=4, dimension=3)`` kernel
    # from 23.13 to 0.241 us/call, and a per-face eigen-decomposition is O(n^3)
    # per side per step.  So a large-level or coupled production run may well
    # want it; a verification or convergence study must not have it silently on.
    use_spectral_radius_bound = False
    # Spectral radius of a model without a closed-form spectrum: `eig` (the
    # numerical eigenvalues; CPU-only in JAX), `gelfand` (the upper bound
    # ||A^m||^(1/m), m = 2^spectral_squarings, any device; ANALYSIS 48),
    # `bound` (the closed-form |A| bound, = use_spectral_radius_bound).
    # The symbolic solvers default to `gelfand` on every backend
    # (zoomy_core.symsolver.WAVE_SPEED); this default serves the old solvers.
    spectral_radius_method = SpectralRadius.eig
    spectral_squarings = 7
    # The Collatz-Wielandt vector of `gelfand_bound` is the row sums of |A^m| plus this times
    # their max: positive, so the bound is continuous in A and a row of round-off (q_y ~ 1e-20
    # in a quasi-1-D run) does not switch it between backends (ANALYSIS 48).
    spectral_cw_offset = 2.0 ** -30

    @property
    def spectral_radius_bound(self):
        """Closed-form scalar upper bound on `rho(A_n)`, the `Min` over `k <= iterations` of
        `(max_i (|A_n|^k 1)_i)^(1/k)`, so the CFL needs no eigen-decomposition.
        Every candidate is an upper bound on its own, so `iterations` buys tightness only and
        the CFL cannot become unsafe.

        References:
            Golub, Van Loan 2013, https://doi.org/10.56021/9781421407944
        """
        n = self.n_equations
        qm = self.quasilinear_matrix
        normal = list(self.normal.values())
        M = [[sp.Abs(sum(qm[i, j, d] * normal[d] for d in range(self.n_dim)))
              for j in range(n)] for i in range(n)]
        return _perron_radius_bound(M, self.spectral_bound_iterations)

    def refresh_derived_operators(self, *, eigenvalues: bool = False):
        """Recompute `quasilinear_matrix` and the source jacobians from the current (possibly
        mutated) primaries; `eigenvalues=True` also re-derives the spectrum.
        """
        self.quasilinear_matrix = self._compute_quasilinear_matrix()
        # POINT-IMPLICIT SOURCE: once ``source_treatment=SourceTreatment.linearized`` has
        # rewritten ``source`` into ``(I − dt·J)⁻¹ S``, that expression is a
        # single OPAQUE per-cell ``solve`` kernel.  ``sp.diff`` through an
        # opaque Function does not evaluate — it yields
        # ``Subs(Derivative(solve(…), _xi), _xi, …)``, which no printer can
        # compile and which ``_resolve_boundary_traces`` then reports as an
        # "unresolved Subs boundary trace" (a misleading diagnosis: no ζ-trace
        # is involved).  In this mode the jacobians are stale BY DESIGN —
        # ``_linearize_source`` baked J into ``S_lin`` and consumers take the
        # source explicitly — so keep the pre-rewrite matrices rather than
        # re-derive nonsense from the opaque source.
        #
        # The one thing that must still track the mutation is the aux-jacobian
        # WIDTH: ``register_aux`` calls us precisely because ``aux_state`` grew,
        # and a stale matrix would be one column short for the printers.  That
        # new column is genuinely zero here — ``register_aux`` runs BEFORE
        # ``regularize_pow``, so the freshly added aux does not yet occur in the
        # source — so pad instead of recomputing.
        from zoomy_core.numerics.numerical_system_model import SourceTreatment
        if getattr(self, "source_treatment", None) is SourceTreatment.linearized:
            self.source_jacobian_wrt_aux_variables = _to_zarray(
                self._pad_aux_jacobian_columns(
                    self.source_jacobian_wrt_aux_variables))
        else:
            self.source_jacobian_wrt_variables = _to_zarray(
                self._compute_source_jacobian_wrt_variables())
            self.source_jacobian_wrt_aux_variables = _to_zarray(
                self._compute_source_jacobian_wrt_aux_variables())
        if eigenvalues and self.eigenvalues is not None:
            self.eigenvalues = self._compute_eigenvalues()

    def _pad_aux_jacobian_columns(self, J):
        """Widen `J` to `(n_equations, len(aux_state))` with zero columns (point-implicit
        path).
        """
        n_aux = len(self.aux_state or [])
        if J is None:
            return sp.zeros(self.n_equations, n_aux)
        have = J.shape[1] if len(J.shape) > 1 else 0
        if have >= n_aux:
            return J
        return sp.Matrix(
            self.n_equations, n_aux,
            lambda i, j: sp.sympify(J[i, j]) if j < have else sp.S.Zero)

    def assert_diagonal_mass_matrix(self):
        """Check that `mass_matrix` is diagonal and nonzero on every evolution row and zero on
        every algebraic row; raises otherwise.
        """
        n_eq = self.n_equations
        n_st = self.n_state
        e2s = list(self.equation_to_state_index)
        offenders = []
        for i in range(n_eq):
            diag_col = e2s[i]
            for k in range(n_st):
                if k == diag_col:
                    continue
                entry = sp.simplify(self.mass_matrix[i, k])
                if entry != 0:
                    offenders.append((i, k, entry))
        if offenders:
            msg_lines = [
                "Mass matrix is not diagonal on evolution rows:",
            ]
            for (i, k, entry) in offenders:
                msg_lines.append(
                    f"  M[row={i}, state_col={k}={self.state[k]}] = {entry}"
                )
            msg_lines.append(
                "Pick a state-variable transformation in "
                "``change_state_variables`` that diagonalises M, or "
                "address the system structurally."
            )
            raise ValueError("\n".join(msg_lines))

    # ── apply / row view ───────────────────────────────────────────────

    def apply(self, operation, *, name: Optional[str] = None,
              description: Optional[str] = None):
        """Apply a system-level operation in place; returns `self`."""
        op_name = name or getattr(operation, "name", None) or \
                  type(operation).__name__
        op_desc = description or getattr(operation, "description", op_name)
        operation(self)
        self.history.append({"name": op_name, "description": op_desc})
        return self

    def __getitem__(self, i: int) -> "SystemModelRow":
        """Row view `SystemModelRow` of equation `i`."""
        return SystemModelRow(self, i)

    # ── from_model factory ────────────────────────────────────────────

    @classmethod
    def _from_derivation_model(cls, model, Q=None, Qaux=None,
                               canonical_source=None,
                               dae=False) -> "SystemModel":
        """Build a `SystemModel` from a declarative derivation `Model` by structurally
        extracting the operators from its untagged residuals.

        Args:
            Q: one applied field per evolution row (required).
            Qaux: aux fields; auto-populated when `None`.
        """
        from zoomy_core.model.derivation.system_extract import (
            extract_system_operators,
        )
        if Q is None:
            # Derive the state vector from the model's own ∂_t evolution rows
            # (one field per primary row, families flattened) — no hand-passed Q.
            Q = model.explicit_state()
            if not Q:
                raise TypeError(
                    "SystemModel.from_model: could not derive Q from the model "
                    "(no ∂_t evolution rows found); pass Q= explicitly.")
        ops = extract_system_operators(model, Q, Qaux, dae=dae)
        # One normal component per horizontal direction — ``space`` length is
        # ``n_dim`` (1 for the usual SME, 2 for the dimension-agnostic
        # ``dimension=3`` map-view model).
        normal = Zstruct(**{str(n): n for n in
                            face_normal_symbols(len(ops["space"]))})
        normal._symbolic_name = "n"
        sm = cls(
            aux_state=ops["aux_state"],
            time=ops["time"], space=ops["space"], state=ops["state"],
            parameters=ops["parameters"],
            flux=ops["flux"], hydrostatic_pressure=ops["hydrostatic_pressure"],
            nonconservative_matrix=ops["nonconservative_matrix"],
            source=ops["source"], mass_matrix=ops["mass_matrix"],
            diffusion_matrix=ops.get("diffusion_matrix"),
            # Channel an EXPLICIT diffusion override (``∇·(ν_h h ∇u)``) off the
            # user-facing model — exactly as the basemodel ``from_model`` branch
            # does — instead of dropping it (REQ-49).  ``diffusion_matrix`` (the
            # extractor-derived implicit slot) is left untouched.
            diffusion_matrix_explicit=_channel_explicit_diffusion(
                model, canonical_source),
            eigenvalues=None, normal=normal,
            parameter_values=ops["parameter_values"],
            state_function_map=ops.get("state_function_map"),
            vertical=getattr(model, "vertical", None),
            # Carry the initial conditions onto the built SystemModel, exactly
            # as the basemodel ``from_model`` branch does — otherwise a
            # declarative model's ``initial_conditions`` (e.g. a dam-break IC set
            # on ``SME(...)``) is silently dropped and adapters can't run the
            # model-card default without re-attaching IC by hand (REQ-103).  The
            # IC lives on the USER-FACING model, passed here as
            # ``canonical_source`` (the ``model`` arg is the bare derivation
            # model, which has no IC); fall back to ``model`` when no canonical
            # source was supplied.
            initial_conditions=getattr(
                canonical_source if canonical_source is not None else model,
                "initial_conditions", None),
            aux_initial_conditions=getattr(
                canonical_source if canonical_source is not None else model,
                "aux_initial_conditions", None),
            unclassified_terms=ops.get("unclassified", []),
        )
        sm.history.append({
            "name": "from_model",
            "description": f"structural extraction from {type(model).__name__}",
        })
        # Classification-completeness guard: an additive term the extractor
        # routed to NO slot is a silent drop.  Warn by default (opt into a hard
        # failure with ``sm.assert_all_classified()``).
        if sm.unclassified_terms:
            import warnings
            rows = ", ".join(str(i) for i, _ in sm.unclassified_terms)
            warnings.warn(
                f"SystemModel.from_model({type(model).__name__}): "
                f"{len(sm.unclassified_terms)} equation row(s) [{rows}] carry "
                "term(s) the operator extractor classified into NO slot — they "
                "are silently dropped from the balance law.  Inspect "
                "describe(full=True) and call assert_all_classified() to fail "
                "hard.", stacklevel=2)
        # Auto-scan: route every non-state Function atom (e.g. the open
        # viscous-stress boundary trace ``tau_xz(t, x, 0)``) and every
        # Derivative atom into ``aux_state`` — the SAME aux-exposure the
        # production ``from_model`` path performs.  This collapses the open
        # stress boundary atoms (``tau_xz(t, x, 0)``, ``tau_xz(t, x, 1)``,
        # ``tau_xz(t, x, ẑ)``) onto a single ``tau_xz`` aux Symbol, so the
        # open-SME source vanishes (the stress is an auxiliary field, computed
        # separately) exactly as production reports.
        sm.expose_aux_atoms()
        _assert_declared_closures_closed(sm, model,
                                         canonical_source=canonical_source)
        # Parse any registered function groups (vertical reconstruction →
        # interpolate_to_3d, its inverse → project_from_3d, …) into their
        # SystemModel slots; runtime gradient aux (∂_x h, ∂_x q_i) land in Qaux.
        _attach_function_groups(sm, model,
                                canonical_source=canonical_source)
        # ROLES: a sigma-mapped (z = b + h*zeta) derivation IS a free-surface
        # model — its depth and bed are read off the coordinate relation, not
        # guessed from field names.  Builders refine (gravity, layers).
        from zoomy_core.systemmodel.roles import infer_sigma_map_roles
        infer_sigma_map_roles(sm, model)
        _declare_model_hooks(sm, canonical_source)
        # Carry the WB reconstruction choice from the declarative Model so the
        # solver can pick the numerics (e.g. SME(equilibrium_reconstruction=...)).
        sm.equilibrium_reconstruction = getattr(
            canonical_source, "equilibrium_reconstruction", "none")
        return sm

    def has_active_source(self) -> bool:
        """True if any source row is nonzero at the configured parameter values (matched by symbol
        name).
        """
        import sympy as _sp
        pv = dict(self.parameter_values or {})

        def _numify(e):
            """Substitute the configured parameter values into `e`."""
            e = _sp.sympify(e)
            return e.subs({s: pv[s.name] for s in e.free_symbols
                           if s.name in pv})
        try:
            return any(_sp.simplify(_numify(e)) != 0
                       for e in _sp.flatten(self.source))
        except Exception:                                # noqa: BLE001
            return True          # unknown ⇒ assume active, the safe branch

    def has_higher_moments(self) -> bool:
        """True if the system carries moment rows beyond the depth-averaged one."""
        names = [str(s) for s in self.state]
        return len([n for n in names if n not in ("h", "b")]) - 1 >= 1

    def resolve_equilibrium_kernel(self, *, force_steady_ode: bool = False) -> str:
        """Which concrete well-balanced kernel a backend must run: `none` / `audusse` pass
        through, the Bernoulli family resolves to `steady_ode` (friction, higher moments or
        `force_steady_ode`) or `bernoulli_closed_form`.

        References:
            Pimentel-Garcia et al. 2022, https://doi.org/10.1016/j.jcp.2022.111152
        """
        eqr = getattr(self, "equilibrium_reconstruction", "none") or "none"
        if eqr in ("none", "audusse"):
            return eqr
        if force_steady_ode or self.has_higher_moments() or self.has_active_source():
            return "steady_ode"
        return "bernoulli_closed_form"

    def is_vertical_dependent(self, symbol) -> bool:
        """True if state `symbol` is a vertical-dependent field such as `u(t,x,zeta)`, False
        for depth-collapsed fields or when no vertical is known.
        """
        if self.vertical is None or not self.state_function_map:
            return False
        if isinstance(symbol, str):
            symbol = sp.Symbol(symbol, real=True)
        f = self.state_function_map.get(symbol)
        return f is not None and self.vertical in getattr(f, "args", ())

    @classmethod
    def from_model(cls, model, Q=None, Qaux=None,
                   canonical_source=None, dae=False,
                   derive_source_jacobians=None) -> "SystemModel":
        """Build a `SystemModel` from a `Model`; the single entry point.
        Production models with a `_system_model_kind` dispatch to their builder in
        `model_builders`, others go through `_from_model_impl`.

        Args:
            Q: state fields of a declarative model (required there).
            Qaux: aux fields; auto-populated when `None`.
            canonical_source: user-facing model that owns the source and explicit-diffusion
                overrides.
            dae: retain a non-diagonal or singular mass matrix (index-1 DAE) instead of requiring
                unit diagonal.
            derive_source_jacobians: override of the field of the same name.
        """
        if getattr(model, "_system_model_kind", None) is not None:
            from zoomy_core.systemmodel.model_builders import build_system_model
            return build_system_model(model)
        return cls._from_model_impl(
            model, Q=Q, Qaux=Qaux, canonical_source=canonical_source, dae=dae)

    @classmethod
    def _from_model_impl(cls, model, Q=None, Qaux=None,
                         canonical_source=None, dae=False) -> "SystemModel":
        """Low-level build of a `SystemModel` from a declarative model plus `Q` or from a
        production model with a baked-in state; see `from_model`.
        """
        from zoomy_core.model.derivation.model import Model as _DerivationModel
        if isinstance(model, _DerivationModel):
            return cls._from_derivation_model(
                model, Q=Q, Qaux=Qaux, canonical_source=canonical_source,
                dae=dae)

        if hasattr(model, "_finalize_for_systemmodel") \
                and not getattr(model, "_finalized", True):
            model._finalize_for_systemmodel()
        time_sym = getattr(model, "time", sp.Symbol("t", real=True))
        dim = getattr(model, "dimension", 1)
        coord_names = ["x", "y", "z"]
        space = [sp.Symbol(coord_names[d], real=True) for d in range(dim)]

        state = [model.variables[k] for k in model.variables.keys()]
        aux_state = [model.aux_variables[k]
                     for k in model.aux_variables.keys()]
        # NEW naming convention (matches the Model side after the
        # ``_parameter_symbols → parameters`` / ``parameters → parameter_values``
        # rename): ``model.parameters`` carries sympy Symbols,
        # ``model.parameter_values`` carries numeric floats.
        parameters = Zstruct(
            **{k: model.parameters[k]
               for k in model.parameters.keys()}
        )
        parameters._symbolic_name = "p"
        parameter_values = Zstruct(
            **{k: getattr(model.parameter_values, k, 0.0)
               for k in model.parameters.keys()}
        )
        normal = Zstruct(
            **{k: model.normal[k] for k in model.normal.keys()}
        )
        normal._symbolic_name = "n"

        n_eq = model.n_variables

        def _to_matrix(z, n_rows, n_cols):
            """Coerce a model ZArray, `sp.Matrix` or 1-D ZArray to an `sp.Matrix` of the
            requested shape.
            """
            # A ``derive_by_array`` of a COLUMN source ``(n_eq, 1)`` (e.g. SWE's
            # friction-only source) yields a rank-3 ZArray ``(n_state, n_eq, 1)``
            # with a trailing singleton axis, whereas a rank-1 source ``(n_eq,)``
            # (SME/VAM) yields the rank-2 ``(n_state, n_eq)`` that ``tomatrix``
            # handles directly.  Squeeze any trailing singleton axes so both
            # shapes channel identically — preserves axis order, so the caller's
            # ``.T`` to the canonical ``(n_eq, n_state)`` is unchanged.
            shp = getattr(z, "shape", None)
            if (shp is not None and len(shp) > 2 and shp[-1] == 1
                    and hasattr(z, "reshape")):
                while len(shp) > 2 and shp[-1] == 1:
                    z = z.reshape(*shp[:-1])
                    shp = z.shape
            # Try direct tomatrix() first (rank-2 ZArrays).
            if hasattr(z, "tomatrix"):
                try:
                    m = z.tomatrix()
                except (ValueError, AttributeError):
                    m = None
            else:
                m = z
            if m is None:
                # Rank-1 ZArray: extract via tolist() / iteration.
                if hasattr(z, "tolist"):
                    items = list(z.tolist())
                else:
                    items = [z[i] for i in range(n_rows)]
                # Reshape to (n_rows, n_cols).
                if n_cols == 1:
                    m = sp.Matrix(items)
                else:
                    m = sp.Matrix(n_rows, n_cols,
                                  lambda i, j: items[i * n_cols + j])
                return m
            if not isinstance(m, sp.Matrix):
                m = sp.Matrix(m)
            if m.shape == (n_rows, n_cols):
                return m
            if m.shape == (n_rows * n_cols, 1) or m.shape == (1, n_rows * n_cols):
                return m.reshape(n_rows, n_cols)
            return m

        F = _to_matrix(model.flux(), n_eq, dim)
        P = _to_matrix(model.hydrostatic_pressure(), n_eq, dim)

        ncp_z = model.nonconservative_matrix()
        if hasattr(ncp_z, "todense"):
            B = sp.MutableDenseNDimArray(ncp_z.todense())
        elif hasattr(ncp_z, "tolist"):
            B = sp.MutableDenseNDimArray(ncp_z.tolist())
        else:
            B = sp.MutableDenseNDimArray(ncp_z)

        def _extract_A(name):
            """The rank-4 `(n_eq, n_state, n_dim, n_dim)` tensor from `model.<name>()`, or an
            explicit zero tensor if the model defines none.
            """
            if hasattr(model, name) and callable(getattr(model, name)):
                A_z = getattr(model, name)()
                if hasattr(A_z, "todense"):
                    return sp.MutableDenseNDimArray(A_z.todense())
                if hasattr(A_z, "tolist"):
                    return sp.MutableDenseNDimArray(A_z.tolist())
                return sp.MutableDenseNDimArray(A_z)
            return sp.MutableDenseNDimArray.zeros(n_eq, n_eq, dim, dim)

        A_diff_impl = _extract_A("diffusion_matrix")
        A_diff_expl = _extract_A("diffusion_matrix_explicit")

        S_z = model.source()
        S_mat = _to_matrix(S_z, n_eq, 1)
        # Explicit source slot — defaults to zero when the model does
        # not opt in by overriding ``source_explicit()``.
        if (hasattr(model, "source_explicit")
                and callable(model.source_explicit)):
            S_expl_mat = _to_matrix(model.source_explicit(), n_eq, 1)
        else:
            S_expl_mat = sp.zeros(n_eq, 1)

        # Mass matrix: ALWAYS extracted from the model via
        # ``model.mass_matrix()`` (which tag-extracts ``time_derivative``
        # contributions from the derivation).  No identity fallback —
        # if the derivation didn't produce ``∂_t`` terms, the resulting
        # row of M is genuinely zero (constraint equation), and the
        # solver must use a DAE-aware time-stepper.  This is the
        # honest formulation: the mass matrix arises NATURALLY from
        # the derivation, not from a default override.
        M_mat = _to_matrix(model.mass_matrix(), n_eq, n_eq)

        equation_names = getattr(model, "equation_names", None)
        # Carry the *indexed symbolic BC function* ``_boundary_conditions``
        # — a ``Function(name="boundary_conditions",
        #              args=(idx, time, position, distance, Q, Qaux, p, normal),
        #              definition=Piecewise((res_i, Eq(idx, i)), …))``.
        # This is the representation a SystemModel-driven runtime
        # lambdifies and the solver flux operators call per face.
        bcs = model._boundary_conditions
        aux_bcs = model._aux_boundary_conditions
        bgrads = model._boundary_gradients

        # Eigenvalues — carry over whatever ``Model.eigenvalues()``
        # produced.  In numerical-eigenvalue mode the model deliberately
        # skips the spectral derivation, so the SystemModel records
        # ``None`` ("skipped") rather than a zero placeholder.
        eigenvalues = None
        if getattr(model, "eigenvalue_mode", "symbolic") != "numerical":
            ev_def = model.functions["eigenvalues"].definition
            if ev_def is not None:
                eigenvalues = _to_matrix(ev_def, n_eq, 1)

        sm = cls(
            time=time_sym,
            space=space,
            state=state,
            aux_state=aux_state,
            parameters=parameters,
            flux=F,
            hydrostatic_pressure=P,
            nonconservative_matrix=B,
            source=S_mat,
            mass_matrix=M_mat,
            eigenvalues=eigenvalues,
            normal=normal,
            parameter_values=parameter_values,
            boundary_conditions=bcs,
            aux_boundary_conditions=aux_bcs,
            boundary_gradients=bgrads,
            initial_conditions=getattr(model, "initial_conditions", None),
            aux_initial_conditions=getattr(
                model, "aux_initial_conditions", None),
            update_variables=_to_matrix(model.update_variables(), n_eq, 1),
            # CHANNEL the source jacobians off the Model (it already derived &
            # cached them) instead of re-deriving on the SystemModel.  The
            # Model's ``derive_by_array`` output is ``(n_state|n_aux, n_eq)``;
            # transpose to the canonical ``(n_eq, n_state|n_aux)`` the printers
            # / solvers index.
            source_jacobian_wrt_variables=_to_matrix(
                model.source_jacobian_wrt_variables(), n_eq, n_eq).T,
            source_jacobian_wrt_aux_variables=(
                _to_matrix(model.source_jacobian_wrt_aux_variables(),
                           len(aux_state), n_eq).T
                if aux_state else sp.zeros(n_eq, 0)),
            # Carry the model's per-cell LOCAL aux formula (e.g. hinv) so it
            # reaches the runtime generically.  getattr-guarded: only the
            # basemodel branch reaches here (derivation models dispatch to
            # _from_derivation_model, which has no such method); skip when the
            # model declares no aux.
            update_aux_variables=(
                _to_matrix(model.update_aux_variables(), len(aux_state), 1)
                if (aux_state
                    and callable(getattr(model, "update_aux_variables", None))
                    # skip the trivial identity (ZArray of the aux symbols) so
                    # identity-aux models stay true no-ops (None) and don't pay
                    # a per-step apply; only a real formula (e.g. hinv) is carried.
                    and list(model.update_aux_variables())
                        != [model.aux_variables[k]
                            for k in model.aux_variables.keys()])
                else None),
            diffusion_matrix=A_diff_impl,
            diffusion_matrix_explicit=A_diff_expl,
            source_explicit=S_expl_mat,
            reconstruction_variables=_to_zarray(
                model.reconstruction_variables()),
            state_from_reconstruction=_to_zarray(
                model.state_from_reconstruction()),
            interpolate_to_3d=(
                _to_zarray(model.interpolate_to_3d())
                if (hasattr(model, "interpolate_to_3d")
                    and callable(model.interpolate_to_3d))
                else None),
            project_from_3d=(
                _to_zarray(model.project_from_3d())
                if (hasattr(model, "project_from_3d")
                    and callable(model.project_from_3d))
                else None),
            position=getattr(model, "position", None),
            vertical=getattr(model, "vertical", None),
        )
        if equation_names is not None:
            sm.equation_names = list(equation_names)
        # Stash the source ``BoundaryConditions`` instances so
        # :meth:`change_state_variables` can *rebuild* the BC / gradient
        # / aux-BC kernels against the new state Zstruct rather than
        # ``xreplace`` on the old Piecewise body.  ``xreplace`` corrupts
        # extrapolation slots — the ``ZArray(Q)`` body produced by
        # ``Extrapolation.compute_boundary_condition`` contains the OLD
        # state Symbols as *index placeholders*, and a CoV substitution
        # remaps them into the physical inverse transform (e.g.
        # ``U_0 → q_U0/h``).  The runtime then divides the inflow
        # momentum by the depth at every extrapolation face — h NaNs.
        # Rebuilding via the source object treats the BC bodies
        # semantically: extrapolation comes out as ``ZArray(new_state)``,
        # which is what the user wants.
        sm._bc_source = getattr(model, "boundary_conditions", None)
        sm._aux_bc_source = getattr(model, "aux_boundary_conditions", None)
        sm.history.append({"name": "from_model",
                            "description": f"extracted from {type(model).__name__}"})
        # Auto-scan: every non-state Function atom and every Derivative
        # atom in the operator matrices becomes an aux Symbol with a
        # structured registry entry.  Solvers walk ``sm.aux_registry``
        # to compute aux values per step.
        sm.expose_aux_atoms()
        _assert_declared_closures_closed(sm, model,
                                         canonical_source=canonical_source)

        # ── raw-Model promotion: function groups + coupling/boundary conditions
        # (REQ-87).  Production models (SWE and subclasses) declare their WB
        # reconstruction / interpolate-project / ``boundary:<name>`` specs as
        # ``_function_groups`` and POP the constructor ``boundary_conditions=``
        # into ``_coupling_bcs`` — both were only wired in the ``.system_model``
        # PROPERTY.  Backend adapters and the FVM solvers call ``from_model``
        # DIRECTLY, so do that wiring HERE: without it the promoted system has an
        # EMPTY BC kernel (walls don't reflect — the malpasset "16 m/s eruption")
        # and no model-declared reconstruction.  The declarative branch attaches
        # its own function groups (see ``_from_derivation_model``); this covers
        # the production ``Model`` branch only.
        if getattr(model, "_function_groups", None):
            _attach_function_groups(sm, model, canonical_source=canonical_source)
        coupling = getattr(model, "_coupling_bcs", None)
        has_coupling = coupling is not None and (
            (list(coupling) if isinstance(coupling, list)
             else getattr(coupling, "boundary_conditions_list", None)))
        if has_coupling:
            from zoomy_core.model.boundary_conditions import resolve_and_attach
            # ``boundary:<name>`` specs now live on ``sm.boundary_specs`` (from
            # the function groups above), so ``FromModel`` BCs resolve; aux BCs
            # default to per-tag extrapolation (mirrors the old property).
            resolve_and_attach(sm, coupling)
        _declare_model_hooks(sm, model)
        return sm

    # ── reconstruct_residuals ─────────────────────────────────────────

    def reconstruct_residuals(self):
        """Re-assemble the row-wise residuals `M d_t Q + d_x F + d_x P + B d_x Q - S` as sympy
        expressions, with aux symbols back-substituted to their original atoms.
        """
        n_eq = self.n_equations
        n_st = self.n_state
        n_dim = self.n_dim
        t = self.time
        coords = self.space

        # Build per-state time- and spatial-derivative Functions if the
        # state is held as Symbols.  We need a coordinate-dependent
        # representation to take symbolic derivatives.
        def _as_function(sym):
            """The applied Function of `sym` over time and coordinates (`sym` itself if
            already one).
            """
            if isinstance(sym, sp.Function):
                return sym
            args = [t] + list(coords)
            return sp.Function(str(sym), real=True)(*args)

        state_funcs = [_as_function(s) for s in self.state]
        sym_to_func = dict(zip(self.state, state_funcs))
        aux_reverse = self._aux_reverse_map()

        def _restore(expr):
            """Put back the original aux atoms, then upgrade state Symbols to Functions."""
            return (sp.sympify(expr)
                    .xreplace(aux_reverse)
                    .xreplace(sym_to_func))

        residuals = []
        for i in range(n_eq):
            res = sp.S.Zero
            # M · ∂_t Q
            for j in range(n_st):
                m_ij = _restore(self.mass_matrix[i, j])
                if m_ij != 0:
                    res = res + m_ij * sp.Derivative(state_funcs[j], t)
            # ∂_x F[i] (and ∂_y if 2D)
            for d in range(n_dim):
                f_id = _restore(self.flux[i, d])
                if f_id != 0:
                    res = res + sp.Derivative(f_id, coords[d])
                p_id = _restore(self.hydrostatic_pressure[i, d])
                if p_id != 0:
                    res = res + sp.Derivative(p_id, coords[d])
            # B · ∂_x Q
            for j in range(n_st):
                for d in range(n_dim):
                    b_ijd = _restore(self.nonconservative_matrix[i, j, d])
                    if b_ijd != 0:
                        res = res + b_ijd * sp.Derivative(state_funcs[j], coords[d])
            # + ∂_x(A · ∂_x Q)  (diffusion tensors, implicit AND explicit —
            # bathymetry-curvature fluxes like ∂_x(D(Q)·∂_x b) land here, so
            # omitting A makes reference comparisons silently lossy)
            for A in (self.diffusion_matrix, self.diffusion_matrix_explicit):
                if A is None:
                    continue
                for j in range(n_st):
                    for d in range(n_dim):
                        a_ijd = _restore(A[i, j, d, d])
                        if a_ijd != 0:
                            res = res + sp.Derivative(
                                a_ijd * sp.Derivative(state_funcs[j], coords[d]),
                                coords[d])
            # − S[i]
            s_i = _restore(self.source[i, 0])
            res = res - s_i
            residuals.append(sp.expand(res))
        return residuals

    # ── from_pdesystem factory (REMOVED) ──────────────────────────────


    # ── expose_aux_atoms (auto-scan) ──────────────────────────────────

    def expose_aux_atoms(self):
        """Route every non-state symbolic input (bare Functions such as the bed, and Derivative
        atoms) of the operators into `aux_state` and record each in `aux_registry`.
        """
        import sympy as sp
        from itertools import product as _product

        if getattr(self, "aux_registry", None) is not None:
            return

        state_names = {str(s) for s in self.state}
        space_names = {str(s): d for d, s in enumerate(self.space)}
        n_dim = len(self.space)

        def _iter_entries(M):
            """Iterate over the scalar entries of operator `M`."""
            if isinstance(M, sp.Matrix):
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        yield M[i, j]
            else:
                for idx in _product(*[range(s) for s in M.shape]):
                    yield M[idx]

        matrices = [self.flux, self.hydrostatic_pressure,
                    self.nonconservative_matrix, self.source,
                    self.mass_matrix]
        if self.update_variables is not None:
            matrices.append(self.update_variables)
        # State-dependent diffusion (e.g. MY-2.5 K_M(ℓ, q, G_H)) can
        # carry ``Derivative(state, x)`` atoms via the Galperin G_H
        # argument; auto-expose them so the rank-4 A tensor lambdifies
        # cleanly through the Firedrake-DG / NumpyRuntimeModel path.
        if self.diffusion_matrix is not None:
            matrices.append(self.diffusion_matrix)
        if self.diffusion_matrix_explicit is not None:
            matrices.append(self.diffusion_matrix_explicit)

        # ── Pass 1: collect *user-defined* Function atoms (excl. state).
        # ``atoms(sp.Function)`` matches every Function subclass — including
        # built-ins like ``Piecewise``, ``Min``, ``Max``, ``conditional``.
        # Those should stay inline; only undefined / user-named function
        # atoms (e.g. topography ``b(t, x, y)`` declared via
        # ``sp.Function("b")``) are candidates for promotion to aux state.
        # Filter via ``sp.AppliedUndef`` which is precisely "user-defined
        # function application".
        # NOTE: ``.atoms()`` returns a SET whose iteration order is str-hash
        # (PYTHONHASHSEED) dependent — it decided the aux row order and made
        # the derived layout (and every emitted backend arg list) vary from
        # process to process.  Iterate in ``default_sort_key`` order so the
        # aux ordering is canonical and the golden snapshots are stable.
        function_atoms = {}     # name → [atoms]
        for M in matrices:
            for entry in _iter_entries(M):
                for atom in sorted(
                        sp.sympify(entry).atoms(sp.core.function.AppliedUndef),
                        key=sp.default_sort_key):
                    name = atom.func.__name__
                    if name in state_names:
                        continue
                    function_atoms.setdefault(name, []).append(atom)

        # ── Pass 1.5: collect ``limit(Derivative(...), scheme)`` atoms.
        # These flag a derivative for runtime TVD limiting via the
        # numerics layer (see ``zoomy_core.model.numerics.limit``).  We
        # substitute each with a dedicated aux Symbol BEFORE the plain
        # Derivative scan below — so the limited derivative gets a
        # ``limited_derivative`` aux entry with a ``limiter_scheme``
        # field, and the un-wrapped Derivative atom is no longer present
        # to be picked up by Pass 2.
        from zoomy_core.model.numerics import limit as _limit_fn
        limit_atoms = {}   # (target_name, multi_index_tuple, scheme) → [atoms]
        for M in matrices:
            for ent in _iter_entries(M):
                for atom in sorted(sp.sympify(ent).atoms(_limit_fn),
                                   key=sp.default_sort_key):
                    inner = atom.args[0]
                    scheme = str(atom.args[1])
                    if not isinstance(inner, sp.Derivative):
                        continue
                    target = inner.args[0]
                    target_name = (target.func.__name__
                                   if isinstance(target, sp.Function)
                                   else str(target))
                    mi = [0] * n_dim
                    has_time = False
                    for var, n in inner.variable_count:
                        vn = str(var)
                        if vn in space_names:
                            mi[space_names[vn]] += int(n)
                        else:
                            has_time = True
                            break
                    if has_time or all(o == 0 for o in mi):
                        continue
                    key = (target_name, tuple(mi), scheme)
                    limit_atoms.setdefault(key, []).append(atom)

        # ── Pass 2: collect Derivative atoms (skip time-derivs). ────
        # Note: any Derivative that was inside a ``limit(...)`` wrapper
        # is collected in Pass 1.5 above and skipped here (the .atoms
        # scan walks the tree, so it WOULD pick up the inner derivative
        # too — we filter by checking whether the parent expression
        # contains a wrapping ``limit`` atom).
        deriv_atoms = {}    # (target_name, multi_index_tuple) → [atoms]
        limited_inner_atoms = set()
        for lst in limit_atoms.values():
            for a in lst:
                limited_inner_atoms.add(a.args[0])
        for M in matrices:
            for entry in _iter_entries(M):
                for d in sorted(sp.sympify(entry).atoms(sp.Derivative),
                                key=sp.default_sort_key):
                    if d in limited_inner_atoms:
                        continue
                    target = d.args[0]
                    target_name = (target.func.__name__
                                   if isinstance(target, sp.Function)
                                   else str(target))
                    mi = [0] * n_dim
                    has_time = False
                    for var, n in d.variable_count:
                        vn = str(var)
                        if vn in space_names:
                            mi[space_names[vn]] += int(n)
                        else:
                            has_time = True
                            break
                    if has_time or all(o == 0 for o in mi):
                        continue
                    key = (target_name, tuple(mi))
                    deriv_atoms.setdefault(key, []).append(d)

        if not function_atoms and not deriv_atoms:
            self.aux_registry = []
            return

        registry = []
        sub_dict = {}
        n_aux_before = len(self.aux_state)
        new_syms = []
        # Rows a derivative entry may target.  Seeded with the aux that is
        # ALREADY declared on the model (prescribed bathymetry / geometry
        # supplied via ``add_equation(..., group='aux')``), then extended
        # below by the newly auto-promoted function atoms.  Without the seed
        # a ``Derivative(b, x)`` whose ``b`` was declared aux resolved to
        # ``target_kind='unknown'``, and ``Solver._walk_derivative_aux``
        # SKIPS unknown targets — the LSQ gradient was never computed and the
        # row read whatever was in the buffer (zero), i.e. a prescribed bed
        # slope silently vanished from the source term.
        function_row_of_name = {str(a): i for i, a in enumerate(self.aux_state)}

        # ── Function entries first (so derivative entries can ─────
        # reference them by row in the registry). ──────────────────
        for name, atoms in function_atoms.items():
            # Mints from a real field HEAD, so it inherits the declared
            # assumptions — it has to agree with what ``system_extract`` minted
            # for the same field, or the two compare unequal while printing
            # identically.
            from zoomy_core.model.derivation.system_extract import (
                head_assumptions)
            _probe = next(iter(atoms), None)
            sym = sp.Symbol(name, **(head_assumptions(_probe)
                                     if _probe is not None else {"real": True}))
            for a in atoms:
                sub_dict[a] = sym
            row = n_aux_before + len(new_syms)
            new_syms.append(sym)
            function_row_of_name[name] = row
            registry.append({
                "kind": "function",
                "name": name,
                "row": row,
                "atom": atoms[0],
                "aux_symbol": sym,
            })

        # ── Limited-derivative entries (from Pass 1.5). ──────────
        # Substitute the ``limit(D, scheme)`` atom with a dedicated aux
        # Symbol, and register the limiter scheme so the solver knows to
        # apply it after the LSQ gradient computation.
        for (target_name, mi, scheme), atoms in limit_atoms.items():
            suffix = "_".join(
                str(self.space[d])
                for d in range(n_dim) for _ in range(mi[d])
            )
            aux_name = f"{target_name}_{suffix}__{scheme}"
            sym = sp.Symbol(aux_name, real=True)
            for a in atoms:
                sub_dict[a] = sym
            row = n_aux_before + len(new_syms)
            new_syms.append(sym)
            entry = {
                "kind": "limited_derivative",
                "name": aux_name,
                "row": row,
                "atom": atoms[0],
                "aux_symbol": sym,
                "target_name": target_name,
                "multi_index": tuple(mi),
                "limiter_scheme": scheme,
            }
            if target_name in state_names:
                entry["target_kind"] = "state"
                entry["state_index"] = next(
                    i for i, s in enumerate(self.state)
                    if str(s) == target_name
                )
            elif target_name in function_row_of_name:
                entry["target_kind"] = "function"
                entry["function_row"] = function_row_of_name[target_name]
            else:
                entry["target_kind"] = "unknown"
            registry.append(entry)

        # ── Derivative entries. ───────────────────────────────────
        for (target_name, mi), atoms in deriv_atoms.items():
            suffix = "_".join(
                str(self.space[d])
                for d in range(n_dim) for _ in range(mi[d])
            )
            aux_name = f"{target_name}_{suffix}"
            sym = sp.Symbol(aux_name, real=True)
            for d in atoms:
                sub_dict[d] = sym
            row = n_aux_before + len(new_syms)
            new_syms.append(sym)
            entry = {
                "kind": "derivative",
                "name": aux_name,
                "row": row,
                "atom": atoms[0],
                "aux_symbol": sym,
                "target_name": target_name,
                "multi_index": tuple(mi),
            }
            if target_name in state_names:
                entry["target_kind"] = "state"
                entry["state_index"] = next(
                    i for i, s in enumerate(self.state)
                    if str(s) == target_name
                )
            elif target_name in function_row_of_name:
                entry["target_kind"] = "function"
                entry["function_row"] = function_row_of_name[target_name]
            else:
                entry["target_kind"] = "unknown"
            registry.append(entry)

        # ── Apply substitutions to every primary matrix. ──────────
        # All operator fields are ZArray (or None); ZArray.xreplace
        # is element-wise and handles non-sympy entries (e.g. Python
        # int 0 from a zero-initialised array) gracefully.
        self.flux = self.flux.xreplace(sub_dict)
        self.hydrostatic_pressure = self.hydrostatic_pressure.xreplace(
            sub_dict)
        self.source = self.source.xreplace(sub_dict)
        self.mass_matrix = self.mass_matrix.xreplace(sub_dict)
        self.nonconservative_matrix = self.nonconservative_matrix.xreplace(
            sub_dict)
        if self.update_variables is not None:
            self.update_variables = self.update_variables.xreplace(sub_dict)
        if self.diffusion_matrix is not None:
            self.diffusion_matrix = self.diffusion_matrix.xreplace(sub_dict)
        if self.diffusion_matrix_explicit is not None:
            self.diffusion_matrix_explicit = (
                self.diffusion_matrix_explicit.xreplace(sub_dict))

        # The freshly exposed Symbols join the aux vector FIRST, so the
        # source-jacobian-wrt-aux refresh below sees the full ``aux_state``
        # (the substituted source now references these new aux Symbols).
        self.aux_state = list(self.aux_state) + new_syms
        self.aux_registry = registry

        # Keep the derived operators consistent: recompute
        # quasilinear_matrix / source jacobians from the substituted
        # primaries, and push the substitution through eigenvalues
        # (cheap — avoids re-paying the spectral derivation).
        self.refresh_derived_operators(eigenvalues=False)
        if self.eigenvalues is not None:
            self.eigenvalues = self.eigenvalues.xreplace(sub_dict)
        if self.eigenvalues_cfl is not None:
            self.eigenvalues_cfl = self.eigenvalues_cfl.xreplace(sub_dict)

        self.history.append({
            "name": "expose_aux_atoms",
            "description": (
                f"auto-scan: {sum(1 for r in registry if r['kind']=='function')} "
                f"function-aux, "
                f"{sum(1 for r in registry if r['kind']=='derivative')} "
                f"derivative-aux"
            ),
        })

    # ── reverse-aux substitution helper (for tests / inspection) ──

    def _aux_reverse_map(self):
        """`{aux_Symbol: original_atom}` over `aux_registry` and `aux_input_registry`, used
        to display reconstructed residuals in `Derivative` form.
        """
        registry = ((getattr(self, "aux_registry", None) or [])
                    + (getattr(self, "aux_input_registry", None) or []))
        return {entry["aux_symbol"]: entry["atom"] for entry in registry}

    # ── (legacy) expose_functions_as_aux / expose_derivatives_as_aux ─

    def expose_functions_as_aux(self, function_names=None):
        """Replace bare Function atoms (those named in `function_names`, else all) by aux
        Symbols of the same name appended to `aux_state`.
        """
        import sympy as sp
        from itertools import product as _product

        names = (set(function_names)
                 if function_names is not None else None)

        def _matches(a):
            """Whether `a` is a Function selected by `function_names`."""
            if not isinstance(a, sp.Function):
                return False
            n = a.func.__name__
            return names is None or n in names

        atoms = set()
        for M in (self.flux, self.hydrostatic_pressure,
                  self.nonconservative_matrix, self.source,
                  self.mass_matrix):
            if isinstance(M, sp.Matrix):
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        for a in M[i, j].atoms(sp.Function):
                            if _matches(a):
                                atoms.add(a)
            else:
                for idx in _product(*[range(s) for s in M.shape]):
                    for a in M[idx].atoms(sp.Function):
                        if _matches(a):
                            atoms.add(a)

        if not atoms:
            return

        # One aux Symbol per distinct function name (atoms with
        # different dummy args still map to the same conceptual field).
        sub_dict = {}
        name_to_sym = {}
        aux_function_map = getattr(self, "aux_function_map", {})
        for a in atoms:
            name = a.func.__name__
            if name not in name_to_sym:
                # from a field HEAD → inherit its declared assumptions, so this
                # agrees with the symbol ``system_extract`` minted for it.
                from zoomy_core.model.derivation.system_extract import (
                    head_assumptions)
                name_to_sym[name] = sp.Symbol(name, **head_assumptions(a))
            sub_dict[a] = name_to_sym[name]
            aux_function_map[name_to_sym[name]] = a

        # Apply substitution.
        self.flux = self.flux.xreplace(sub_dict)
        self.hydrostatic_pressure = self.hydrostatic_pressure.xreplace(
            sub_dict)
        self.source = self.source.xreplace(sub_dict)
        self.mass_matrix = self.mass_matrix.xreplace(sub_dict)
        B = self.nonconservative_matrix
        new_B = sp.MutableDenseNDimArray.zeros(*B.shape)
        for idx in _product(*[range(s) for s in B.shape]):
            new_B[idx] = B[idx].xreplace(sub_dict)
        self.nonconservative_matrix = new_B

        # Extend aux_state BEFORE refreshing so source_jacobian_wrt_aux_variables
        # is sized to the full aux vector (the substituted source references the
        # new aux Symbols).
        self.aux_state = list(self.aux_state) + list(name_to_sym.values())
        self.aux_function_map = aux_function_map
        self.refresh_derived_operators(eigenvalues=False)
        if self.eigenvalues is not None:
            self.eigenvalues = self.eigenvalues.xreplace(sub_dict)
        if self.eigenvalues_cfl is not None:
            self.eigenvalues_cfl = self.eigenvalues_cfl.xreplace(sub_dict)
        self.history.append({
            "name": "expose_functions_as_aux",
            "description": (
                f"intercepted {len(atoms)} Function atom(s) → aux "
                f"Symbols {sorted(name_to_sym.keys())}"
            ),
        })


    # ── expose_derivatives_as_aux ─────────────────────────────────────

    def expose_derivatives_as_aux(self):
        """Replace every `Derivative(target, axes...)` atom in the operators by an aux Symbol
        `{target}_{axes}` and record it in `aux_derivative_map` for the solver to fill.
        """
        from itertools import product

        def _matrix_entries(M):
            """Iterate over `(index, entry)` of operator `M`."""
            if isinstance(M, sp.Matrix):
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        yield (i, j), M[i, j]
            else:
                shape = M.shape
                for idx in product(*[range(s) for s in shape]):
                    yield idx, M[idx]

        derivative_subs = {}
        aux_derivative_map = {}

        def _aux_name(d):
            """Aux symbol name `{target}_{axes}` of derivative `d`."""
            target = d.args[0]
            target_name = (target.func.__name__
                           if isinstance(target, sp.Function)
                           else str(target))
            axes = []
            for var, n in d.variable_count:
                axes.extend([str(var)] * int(n))
            return f"{target_name}_" + "_".join(axes)

        for M in (self.flux, self.hydrostatic_pressure,
                  self.nonconservative_matrix, self.source,
                  self.mass_matrix):
            for _, entry in _matrix_entries(M):
                for d in sp.sympify(entry).atoms(sp.Derivative):
                    if d in derivative_subs:
                        continue
                    aux_sym = sp.Symbol(_aux_name(d), real=True)
                    derivative_subs[d] = aux_sym
                    aux_derivative_map[aux_sym] = d

        if not derivative_subs:
            return

        # Apply substitution to every matrix.  sp.Matrix has xreplace;
        # NDimArray (the NCP) does not, so iterate entries.
        self.flux = self.flux.xreplace(derivative_subs)
        self.hydrostatic_pressure = self.hydrostatic_pressure.xreplace(
            derivative_subs)
        self.source = self.source.xreplace(derivative_subs)
        self.mass_matrix = self.mass_matrix.xreplace(derivative_subs)
        B = self.nonconservative_matrix
        shape = B.shape
        new_B = sp.MutableDenseNDimArray.zeros(*shape)
        for idx in product(*[range(s) for s in shape]):
            new_B[idx] = B[idx].xreplace(derivative_subs)
        self.nonconservative_matrix = new_B

        # Extend aux_state with the new derivative Symbols BEFORE refreshing so
        # source_jacobian_wrt_aux_variables is sized to the full aux vector.
        self.aux_state = list(self.aux_state) + list(
            derivative_subs.values())
        self.aux_derivative_map = aux_derivative_map

        self.refresh_derived_operators(eigenvalues=False)
        if self.eigenvalues is not None:
            self.eigenvalues = self.eigenvalues.xreplace(derivative_subs)
        if self.eigenvalues_cfl is not None:
            self.eigenvalues_cfl = self.eigenvalues_cfl.xreplace(derivative_subs)

        self.history.append({
            "name": "expose_derivatives_as_aux",
            "description": (
                f"intercepted {len(derivative_subs)} Derivative atom(s) "
                f"→ aux Symbols "
                f"{[str(s) for s in derivative_subs.values()]}"
            ),
        })


    # ── change_state_variables ────────────────────────────────────────

    def change_state_variables(self, new_state, transform):
        """Change the state variables in place, updating every state-dependent field (operators,
        derived operators, reconstruction pair, BC kernels, aux registry by the chain rule).

        Args:
            new_state: new state vector of the same length.
            transform: maps each replaced OLD state entry to its expression in the new variables.
        """
        n = self.n_state
        if len(new_state) != n:
            raise ValueError(
                f"new_state has length {len(new_state)}, expected {n}"
            )

        old_state = list(self.state)
        full_transform = {}
        for old_s in old_state:
            if old_s in transform:
                full_transform[old_s] = sp.sympify(transform[old_s])
            else:
                if old_s not in new_state:
                    raise ValueError(
                        f"old state {old_s!r} is not in new_state and "
                        f"has no entry in transform — cannot map it"
                    )
                full_transform[old_s] = old_s

        # Jacobian J[i, j] = ∂T_i / ∂new_state[j]
        J = sp.zeros(n, n)
        for i, old_s in enumerate(old_state):
            T_i = full_transform[old_s]
            for j, new_s in enumerate(new_state):
                J[i, j] = sp.diff(T_i, new_s)

        # Substitute T into operators.  ZArray.xreplace returns ZArray;
        # ZArray.expand applies sp.expand element-wise.
        new_flux = self.flux.xreplace(full_transform).expand()
        new_P    = self.hydrostatic_pressure.xreplace(full_transform).expand()
        new_S    = self.source.xreplace(full_transform).expand()

        # NCP: B_new[i, k, d] = Σ_j B_old[i, j, d](T) · J[j, k].  Per-d
        # slab is a Matrix-multiply; convert each slab through sp.Matrix
        # for the multiply and wrap back at the end.
        n_eq = self.n_equations
        n_dim = self.n_dim
        new_B = sp.MutableDenseNDimArray.zeros(n_eq, n, n_dim)
        for d in range(n_dim):
            B_slab = sp.Matrix(
                n_eq, n,
                lambda i, j, _d=d: sp.sympify(
                    self.nonconservative_matrix[i, j, _d]
                ),
            )
            B_slab_sub = B_slab.xreplace(full_transform)
            B_slab_new = sp.expand(B_slab_sub * J)
            for i in range(n_eq):
                for k in range(n):
                    new_B[i, k, d] = B_slab_new[i, k]
        new_B = _to_zarray(new_B)

        # Mass matrix: M_new = M_old(T) · J.  Same Matrix-multiply
        # pattern — ZArray.xreplace → sp.Matrix for multiply → ZArray.
        M_old_M = sp.Matrix(self.mass_matrix.tolist())
        M_old_sub = M_old_M.xreplace(full_transform)
        new_M = _to_zarray(sp.expand(M_old_sub * J))

        self.state = list(new_state)
        self.flux = new_flux
        self.hydrostatic_pressure = new_P
        self.nonconservative_matrix = new_B
        self.source = new_S
        self.mass_matrix = new_M
        # Push the transform through the secondary fields too so they
        # reference the new state, not the old: ``update_variables``
        # (per-cell state remap / Chorin corrector explicit update) and
        # ``eigenvalues`` (preserved under invertible change-of-vars —
        # xreplace, don't re-solve).
        if self.update_variables is not None:
            # ``update_variables`` is a per-cell state-to-state map.
            # Naively xreplacing the OLD state symbols turns
            # ``identity_in_old(Q_old) = Q_old`` into the inverse
            # transform on the NEW state — e.g. ``[h, U_0, U_1, ...]``
            # becomes ``[h, q_U0/h, 3·q_U1/h, ...]`` which actively
            # destroys the conservative-form state at every step.
            # If the old map was the identity (default for Models with
            # no custom clamp), rebuild the identity in the new state.
            old_uv_list = [sp.sympify(self.update_variables[i, 0])
                           for i in range(self.update_variables.shape[0])]
            was_identity = (
                len(old_uv_list) == len(old_state)
                and all(sp.simplify(old_uv_list[i] - old_state[i]) == 0
                        for i in range(len(old_state)))
            )
            if was_identity:
                n_uv = len(new_state)
                self.update_variables = _to_zarray(sp.Matrix(
                    n_uv, 1, lambda r, _c: new_state[r]
                ))
            else:
                self.update_variables = self.update_variables.xreplace(
                    full_transform
                )
        # ``reconstruction_variables`` / ``state_from_reconstruction``:
        # the forward map xreplaces straight through under CoV (it's an
        # expression in state symbols).  The inverse is re-derived from
        # the post-CoV forward + new state — its WB-symbol expressions
        # depend on the *new* state slot names, so a plain xreplace
        # wouldn't suffice.
        if self.reconstruction_variables is not None:
            from zoomy_core.model.reconstruction_inverse import (
                invert_reconstruction,
            )
            self.reconstruction_variables = (
                self.reconstruction_variables.xreplace(full_transform))
            self.state_from_reconstruction = invert_reconstruction(
                self.reconstruction_variables, list(new_state),
            )
        # Explicit-source twin: same shape and same CoV mechanics as
        # ``source`` — just substitute the transform and re-expand.
        if self.source_explicit is not None:
            self.source_explicit = (
                self.source_explicit.xreplace(full_transform).expand()
            )
        # Diffusion-matrix tensors (shape ``(n_eq, n_state, n_dim, n_dim)``).
        # The constitutive contraction
        #   ``F_diff[i, d] = Σ_{j, e} A[i, j, d, e] · ∂_e Q[j]``
        # combined with ``∂_e Q_old[j] = J[j, k] · ∂_e Q_new[k]`` gives
        #   ``A_new[i, k, d, e] = Σ_j A_old[i, j, d, e](T) · J[j, k]``
        # — a per-(i, d, e) slab Matrix-multiply on the state axis,
        # mirroring how the NCP is transformed above.
        self.diffusion_matrix = _cov_diffusion_tensor(
            self.diffusion_matrix, full_transform, J, n_state=n,
        )
        self.diffusion_matrix_explicit = _cov_diffusion_tensor(
            self.diffusion_matrix_explicit, full_transform, J, n_state=n,
        )
        # Symbolic BC kernels — boundary_conditions /
        # aux_boundary_conditions / boundary_gradients.  CoV cannot be
        # done by ``xreplace`` on the existing Piecewise body: an
        # extrapolation slot's body is ``ZArray(Q)`` — the OLD state
        # Symbols as *index placeholders*, not as physical quantities —
        # and a ``{U_0: q_U0/h, …}`` substitution rewrites every such
        # placeholder into the inverse transform of the old symbol.
        # The runtime then divides inflow momentum by the depth at
        # every extrapolation face, NaN-ing ``h`` immediately.
        #
        # Correct treatment: rebuild the BC kernels from their source
        # :class:`zoomy_core.model.boundary_conditions.BoundaryConditions`
        # against the NEW state Zstruct, so extrapolation comes out
        # ``ZArray(new_state)`` (identity on slot symbols) while
        # prescribed-value BCs continue to reference the parameter
        # Symbols they were built with.
        new_variables_zstruct = Zstruct(**{str(s): s for s in new_state})
        new_variables_zstruct._symbolic_name = "Q"
        self._cov_rebuild_bc_kernels(new_variables_zstruct, full_transform,
                                     new_state)
        # Chain-rule propagation for aux derivative entries that target
        # OLD state variables.  Their meaning (∂_x of OLD state) must be
        # rewritten in terms of NEW-state derivatives, otherwise the
        # operator expressions disagree with the runtime aux pipeline
        # (which computes ``∂_x state[i]`` using the NEW state).
        self._propagate_chain_rule_through_aux(
            full_transform, J, old_state, new_state,
        )
        # Re-derive the quasilinear matrix + source jacobian from the new
        # primaries.
        self.refresh_derived_operators(eigenvalues=False)
        if self.eigenvalues is not None:
            self.eigenvalues = self.eigenvalues.xreplace(full_transform)
        if self.eigenvalues_cfl is not None:
            self.eigenvalues_cfl = self.eigenvalues_cfl.xreplace(full_transform)
        self.history.append({
            "name": "change_state_variables",
            "description": (
                "state: ["
                + ", ".join(str(s) for s in old_state)
                + "] → ["
                + ", ".join(str(s) for s in new_state)
                + "]; transform: "
                + ", ".join(f"{k}↦{v}" for k, v in transform.items())
            ),
        })
        return self

    def _cov_rebuild_bc_kernels(self, new_variables_zstruct,
                                full_transform, new_state):
        """Rebuild the boundary, boundary-gradient and aux-boundary kernels against a new state
        Zstruct after a change of variables.
        """
        bc_source = getattr(self, "_bc_source", None)
        aux_bc_source = getattr(self, "_aux_bc_source", None)

        def _reuse_arg(args, name):
            """Read a non-variables arg (time, position, normal, ...) out of an existing kernel's
            args; stable under the change of variables.
            """
            return args[name] if hasattr(args, "__contains__") and name in args.keys() else args[name]

        # boundary_conditions + boundary_gradients share the main BC
        # source (``self._bc_source`` is the model's ``boundary_conditions``).
        if bc_source is not None and self.boundary_conditions is not None:
            old_args = self.boundary_conditions.args
            t   = old_args.time
            X   = old_args.position
            dX  = old_args.distance
            Qx  = old_args.aux_variables
            p_z = old_args.parameters
            n_z = old_args.normal
            self.boundary_conditions = (
                bc_source.get_boundary_condition_function(
                    t, X, dX, new_variables_zstruct, Qx, p_z, n_z,
                    function_name=self.boundary_conditions.name,
                )
            )
            if self.boundary_gradients is not None:
                self.boundary_gradients = (
                    bc_source.get_boundary_gradient_function(
                        t, X, dX, new_variables_zstruct, Qx, p_z, n_z,
                        function_name=self.boundary_gradients.name,
                    )
                )
        elif self.boundary_conditions is not None:
            import warnings
            warnings.warn(
                "SystemModel.change_state_variables: BC source object "
                "not available (this SystemModel was constructed without "
                "from_model); falling back to xreplace-based propagation "
                "of boundary_conditions.  This is correct for BC bodies "
                "that are physical expressions of state but WRONG for "
                "extrapolation slots (placeholder Q[i] symbols).  Rebuild "
                "the BC kernels manually against the new state if your "
                "system uses extrapolation.",
                RuntimeWarning,
                stacklevel=3,
            )
            self.boundary_conditions = _cov_propagate_bc_function(
                self.boundary_conditions, full_transform, new_state,
            )
            self.boundary_gradients = _cov_propagate_bc_function(
                self.boundary_gradients, full_transform, new_state,
            )

        # aux_boundary_conditions: separate source.
        if (aux_bc_source is not None
                and self.aux_boundary_conditions is not None):
            old_args = self.aux_boundary_conditions.args
            t   = old_args.time
            X   = old_args.position
            dX  = old_args.distance
            Qx  = old_args.aux_variables
            p_z = old_args.parameters
            n_z = old_args.normal
            self.aux_boundary_conditions = (
                aux_bc_source.get_boundary_condition_function(
                    t, X, dX, new_variables_zstruct, Qx, p_z, n_z,
                    function_name=self.aux_boundary_conditions.name,
                )
            )
        elif self.aux_boundary_conditions is not None:
            self.aux_boundary_conditions = _cov_propagate_bc_function(
                self.aux_boundary_conditions, full_transform, new_state,
            )

    def _propagate_chain_rule_through_aux(self, full_transform, J,
                                          old_state, new_state):
        """Rewrite the derivative-aux registry entries that target a substituted state slot by the
        chain rule `d_x old[i] = J[i,k] d_x new[k]` (1-D only).
        """
        if not getattr(self, "aux_registry", None):
            return

        from itertools import product

        old_state_names = {str(s): s for s in old_state}
        new_state_names = {str(s): (j, s) for j, s in enumerate(new_state)}
        space = list(self.space)
        if not space:
            return
        axis = space[0]   # only handle 1D for now (matches multi_index = (1,))

        aux_subs = {}
        new_aux_entries = []
        entries_to_remove = []

        for entry in self.aux_registry:
            if entry.get("kind") != "derivative":
                continue
            if entry.get("target_kind") != "state":
                continue
            if entry.get("multi_index") != (1,):
                continue
            target_name = entry["target_name"]
            if target_name not in old_state_names:
                continue
            old_sym = old_state_names[target_name]
            T = full_transform.get(old_sym, old_sym)
            if T == old_sym:
                # Identity transform — state retained; just sync state_index
                # in case the order changed.
                if target_name in new_state_names:
                    entry["state_index"] = new_state_names[target_name][0]
                continue
            # State substituted; build chain-rule expansion.
            old_aux_sym = entry["aux_symbol"]
            chain_expr = sp.S.Zero
            for j, new_sym in enumerate(new_state):
                partial = sp.diff(T, new_sym)
                if partial == 0:
                    continue
                new_aux_name = f"{str(new_sym)}_x"
                new_aux_sym = sp.Symbol(new_aux_name, real=True)
                # Only add if not already in registry or pending.
                already = (any(e.get("name") == new_aux_name
                               for e in self.aux_registry)
                           or any(e["name"] == new_aux_name
                                  for e in new_aux_entries))
                if not already:
                    new_aux_entries.append({
                        "kind": "derivative",
                        "name": new_aux_name,
                        "row": -1,
                        "atom": sp.Derivative(new_sym, axis, evaluate=False),
                        "aux_symbol": new_aux_sym,
                        "target_name": str(new_sym),
                        "multi_index": (1,),
                        "target_kind": "state",
                        "state_index": j,
                    })
                chain_expr += partial * new_aux_sym
            aux_subs[old_aux_sym] = sp.expand(chain_expr)
            entries_to_remove.append(entry["name"])

        if not aux_subs:
            return

        # Apply substitution to all operator matrices.
        self.flux = self.flux.xreplace(aux_subs)
        self.hydrostatic_pressure = self.hydrostatic_pressure.xreplace(aux_subs)
        self.source = self.source.xreplace(aux_subs)
        self.mass_matrix = self.mass_matrix.xreplace(aux_subs)
        B = self.nonconservative_matrix
        shape = B.shape
        new_B = sp.MutableDenseNDimArray.zeros(*shape)
        for idx in product(*[range(s) for s in shape]):
            new_B[idx] = sp.sympify(B[idx]).xreplace(aux_subs)
        self.nonconservative_matrix = _to_zarray(new_B)

        # Update aux_state list + aux_registry: drop the substituted
        # entries, append the new ones, and reindex `row` so they form
        # a contiguous block.
        keep = [e for e in self.aux_registry
                if e["name"] not in entries_to_remove]
        new_aux_state = []
        new_registry = []
        for i, entry in enumerate(keep + new_aux_entries):
            entry = dict(entry)
            entry["row"] = i
            new_registry.append(entry)
            new_aux_state.append(entry["aux_symbol"])
        self.aux_state = new_aux_state
        self.aux_registry = new_registry
        # Stash chain-rule info; ``change_state_variables`` folds it
        # into its single history entry so callers see one event per
        # CoV (matches what tests expect).
        self._last_cov_aux_chain_rule = {
            "removed": list(entries_to_remove),
            "added": [e["name"] for e in new_aux_entries],
        }

    def remove_non_diagonal_h(self, **kwargs):
        """Apply `RemoveNonDiagonalH` to this system."""
        return RemoveNonDiagonalH()(self, **kwargs)

    # ── classification guard ──────────────────────────────────────────

    def assert_all_classified(self):
        """Raise if any equation row carries an operator term the structural extractor could not
        classify into a slot; the strict counterpart of the build-time warning.
        """
        if self.unclassified_terms:
            lines = "\n".join(
                f"  row {i}: {sp.sstr(r)}" for i, r in self.unclassified_terms)
            raise ValueError(
                "SystemModel has unclassified operator term(s) — these matched "
                "no classification branch and are silently dropped from the "
                "balance law:\n" + lines + "\n\nFix the derivation so every "
                "additive term reads as a flux / pressure / NCP / diffusion / "
                "source contribution (or tag it explicitly); do NOT special-"
                "case the classifier.")
        return self

    # ── describe ──────────────────────────────────────────────────────

    def describe_unclosed(self) -> str:
        """List the aux rows nothing computes (the model's open terms), classifying each aux slot
        as computed, external or open.
        """
        unclosed = list(getattr(self, "_unclosed_closures", None) or ())
        external = dict(getattr(self, "external_aux", None) or {})
        computed = [str(a) for a in (self.aux_state or ())
                    if str(a) not in unclosed and str(a) not in external]

        out = [f"UNCLOSED TERMS: {len(unclosed)}"]
        for nm in unclosed:
            out.append(f"  ✗ {nm}   — nothing computes this; it reads ZERO")
        if external:
            out.append(f"external (declared, backend-supplied): {len(external)}")
            for nm, defn in external.items():
                out.append(f"  · {nm}" + (f"   {defn}" if defn is not None else ""))
        if computed:
            out.append(f"computed (aux_registry): {len(computed)}")
            out.append("  " + ", ".join(computed))
        if not unclosed:
            out.append("The system is CLOSED.")
        return "\n".join(out)

    def describe(self, full: bool = False) -> "SystemModelDescription":
        """A `SystemModelDescription` rendering the core operators; `full=True` adds the
        derived slots.
        """
        return SystemModelDescription(self, full=full)


class SystemModelRow:
    """Row view of a `SystemModel`: row `i` of every stored matrix."""

    def __init__(self, parent: SystemModel, i: int):
        """View row `i` of `parent`."""
        self._parent = parent
        self._i = i

    @property
    def flux(self):
        """Row `i` of the flux."""
        return self._parent.flux[self._i, :]

    @property
    def hydrostatic_pressure(self):
        """Row `i` of the hydrostatic pressure."""
        return self._parent.hydrostatic_pressure[self._i, :]

    @property
    def source(self):
        """Entry `i` of the source."""
        return self._parent.source[self._i, 0]

    @property
    def mass_matrix_row(self):
        """Row `i` of the mass matrix."""
        return self._parent.mass_matrix[self._i, :]

    @property
    def nonconservative_matrix_row(self):
        """Row `i` of the nonconservative matrix as an `(n_eq, n_dim)` array."""
        n = self._parent.n_equations
        d = self._parent.n_dim
        out = sp.MutableDenseNDimArray.zeros(n, d)
        for j in range(n):
            for k in range(d):
                out[j, k] = self._parent.nonconservative_matrix[self._i, j, k]
        return out


class SystemModelDescription:
    """Markdown (Jupyter `_repr_markdown_`) and plain-text description of a `SystemModel`."""

    def __init__(self, sm: SystemModel, *, full: bool = False):
        """Describe `sm`; `full` adds the derived slots."""
        self._sm = sm
        self._full = full

    # ── rendering helpers ──────────────────────────────────────────────
    @staticmethod
    def _as_matrix(obj):
        """Coerce a stored operator to a 2-D `sp.Matrix` (a flat vector becomes a column);
        `None` passes through.
        """
        if obj is None:
            return None
        if isinstance(obj, sp.MatrixBase):
            return obj
        lst = obj.tolist() if hasattr(obj, "tolist") else list(obj)
        return sp.Matrix(lst)

    def _render(self, parts, label, mat):
        """Append the labelled operator `mat` (or `= 0`) to `parts`."""
        M = self._as_matrix(mat)
        if M is None or M.is_zero_matrix:
            parts.append(f"**{label} $= 0$**")
        else:
            parts.append(f"**{label}:**")
            parts.append(f"$$\n{sp.latex(M)}\n$$")

    def _render_ndim(self, parts, label, arr, rank):
        """Render a rank-3 NCP or rank-4 diffusion tensor as `n_eq x n_state` slabs per
        direction (pair); all-zero collapses to `= 0`.
        """
        sm = self._sm
        if arr is None:
            parts.append(f"**{label} $= 0$**")
            return
        n, ns, nd = sm.n_equations, sm.n_state, sm.n_dim
        slabs = []
        if rank == 3:
            for d in range(nd):
                slab = sp.Matrix(n, ns, lambda i, j, _d=d: arr[i, j, _d])
                if not slab.is_zero_matrix:
                    tag = f"$_{{{sp.latex(sm.space[d])}}}$" if nd > 1 else ""
                    slabs.append((f"{label}{tag}", slab))
        else:  # rank == 4
            for d in range(nd):
                for e in range(nd):
                    slab = sp.Matrix(n, ns,
                                     lambda i, j, _d=d, _e=e: arr[i, j, _d, _e])
                    if not slab.is_zero_matrix:
                        tag = (f"$_{{{sp.latex(sm.space[d])}{sp.latex(sm.space[e])}}}$"
                               if nd > 1 else "")
                        slabs.append((f"{label}{tag}", slab))
        if not slabs:
            parts.append(f"**{label} $= 0$**")
            return
        for lab, slab in slabs:
            parts.append(f"**{lab}:**")
            parts.append(f"$$\n{sp.latex(slab)}\n$$")

    def _operator_block(self) -> str:
        """The core operators of the balance law (always rendered)."""
        sm = self._sm
        parts = []

        # Canonical equation form (incl. the diffusive flux ∇·(A∇Q)).
        parts.append("**System form:**")
        parts.append(
            r"$$"
            r"M(Q)\,\partial_t Q "
            r"+ \nabla\cdot\!\big(F(Q) + P(Q)\big)"
            r" + \sum_{d} B_{d}(Q)\,\partial_{d} Q "
            r"- \nabla\cdot\!\big(A(Q)\,\nabla Q\big)"
            r" - S(Q) = 0"
            r"$$"
        )

        # State + auxiliary state.
        parts.append("**State $Q$:**")
        parts.append(f"$$\n{sp.latex(sp.Matrix(list(sm.state)))}\n$$")
        if sm.aux_state:
            parts.append("**Auxiliary $Q_{aux}$:**")
            parts.append(
                f"$$\n{sp.latex(sp.Matrix(list(sm.aux_state)))}\n$$")
        else:
            parts.append(r"**Auxiliary $Q_{aux} = \varnothing$**")

        self._render(parts, "Mass matrix $M$", sm.mass_matrix)
        self._render(parts, "Flux $F$", sm.flux)
        self._render(parts, "Hydrostatic pressure $P$",
                     sm.hydrostatic_pressure)
        self._render_ndim(parts, "Diffusive flux $A$",
                          sm.diffusion_matrix, rank=4)
        self._render_ndim(parts, "NCP $B$",
                          sm.nonconservative_matrix, rank=3)
        self._render(parts, "Source $S$ (implicit)", sm.source)
        if sm.source_explicit is not None:
            self._render(parts, "Source $S_\\mathrm{exp}$ (explicit)",
                         sm.source_explicit)
        return "\n\n".join(parts)

    def _extra_block(self) -> str:
        """The rare and derived operators (only with `full=True`)."""
        sm = self._sm
        parts = ["**Extended operators:**"]
        any_shown = False
        for label, val, rank in (
            ("Explicit diffusion $A_\\mathrm{exp}$",
             sm.diffusion_matrix_explicit, 4),
            ("Quasilinear $A_\\mathrm{ql}$", sm.quasilinear_matrix, 3),
            ("Source Jacobian $\\partial S/\\partial Q$",
             sm.source_jacobian_wrt_variables, 2),
            ("Eigenvalues $\\Lambda$", sm.eigenvalues, 1),
            ("Update variables", sm.update_variables, 1),
            ("Reconstruction variables", sm.reconstruction_variables, 1),
            ("State from reconstruction", sm.state_from_reconstruction, 1),
            ("interpolate\\_to\\_3d", sm.interpolate_to_3d, 1),
        ):
            if val is None:
                continue
            any_shown = True
            if rank >= 3:
                self._render_ndim(parts, label, val, rank)
            else:
                self._render(parts, label, val)
        if not any_shown:
            parts.append("_(none set)_")
        return "\n\n".join(parts)

    def _unclassified_block(self) -> str:
        """The classification-completeness report listing operator terms matched to no slot (only
        with `full=True`, empty when clean).
        """
        sm = self._sm
        unc = list(getattr(sm, "unclassified_terms", []) or [])
        parts = ["**Unclassified terms (silent drops):**"]
        if not unc:
            parts.append("_none — every term reached a slot ✓_")
            return "\n\n".join(parts)
        parts.append(
            "⚠ the following terms were classified into NO operator slot and "
            "are dropped from the balance law:")
        for i, r in unc:
            parts.append(f"- row {i} (updates ${sp.latex(sm.state[i])}$): "
                         f"$${sp.latex(r)}$$")
        return "\n\n".join(parts)

    def _repr_markdown_(self) -> str:
        """The Markdown rendering of the system."""
        sm = self._sm
        parts = [
            f"**SystemModel** — {sm.n_equations} equation"
            f"{'s' if sm.n_equations != 1 else ''}, "
            f"{sm.n_dim} spatial dimension{'s' if sm.n_dim != 1 else ''}"
        ]
        if sm.parameters.length():
            parts.append(
                "**Parameters:** "
                + ", ".join(
                    f"${sp.latex(sm.parameters[k])} = "
                    f"{getattr(sm.parameter_values, k, '?')}$"
                    for k in sm.parameters.keys()
                )
            )
        parts.append(self._operator_block())     # CORE operators — always
        if self._full:
            parts.append(self._extra_block())     # rare slots — full=True only
            parts.append(self._unclassified_block())  # completeness report
        if sm.history:
            parts.append(
                "**Operations:** "
                + ", ".join(h["name"] for h in sm.history)
            )
        return "\n\n".join(parts)

    def __str__(self) -> str:
        """Plain-text rendering (Markdown markers stripped)."""
        md = self._repr_markdown_()
        return md.replace("$$", "").replace("$", "").replace("**", "")

    def __repr__(self) -> str:
        """Same as `str`."""
        return self.__str__()


# ── System-level operations ───────────────────────────────────────────

class RemoveNonDiagonalH:
    """System-level op: substitute the mass equation into every row with a nonzero `h` entry in
    the mass matrix, moving the `M[i,h] d_t h` term into the NCP matrix and source.
    The mass row must satisfy `M[m,h] = 1` and zero elsewhere; other mass-matrix entries are
    left alone.
    """

    name = "remove_non_diagonal_h"
    description = (
        "Substitute mass equation into rows with non-zero ∂_t h "
        "coupling; clear M[i, h] for those rows."
    )

    def __call__(self, sm: "SystemModel", *, h_state_index=None,
                 mass_equation_index=None):
        """Clear the `h` column of the mass matrix; `h_state_index` and
        `mass_equation_index` are auto-detected when omitted.
        """
        n_eq = sm.n_equations
        n_st = sm.n_state
        n_dim = sm.n_dim

        if h_state_index is None:
            h_state_index = self._find_h_state_index(sm)
        if mass_equation_index is None:
            mass_equation_index = self._find_mass_row(sm, h_state_index)

        h_col = h_state_index
        m = mass_equation_index

        # Precondition: M[m, :] is e_{h_col}.
        M_mh = sp.simplify(sm.mass_matrix[m, h_col])
        if M_mh != 1:
            raise ValueError(
                f"RemoveNonDiagonalH: mass row {m} has M[{m}, {h_col}] = "
                f"{M_mh}, expected 1.  Supply mass_equation_index "
                f"explicitly or normalise the mass row first."
            )
        for k in range(n_st):
            if k == h_col:
                continue
            entry = sp.simplify(sm.mass_matrix[m, k])
            if entry != 0:
                raise ValueError(
                    f"RemoveNonDiagonalH: mass row {m} is not pure: "
                    f"M[{m}, {k}] = {entry}, expected 0."
                )

        M = sp.Matrix(sm.mass_matrix.tolist())
        S = sp.Matrix(sm.source.tolist())
        B = sp.MutableDenseNDimArray(
            sm.nonconservative_matrix.tolist(),
            shape=tuple(sm.nonconservative_matrix.shape),
        )

        # Pre-compute mass-row spatial pieces.
        mass_F = [sm.flux[m, d] for d in range(n_dim)]
        mass_P = [sm.hydrostatic_pressure[m, d] for d in range(n_dim)]
        # ∂F[m, d]/∂Q_k and ∂P[m, d]/∂Q_k for every (k, d).
        dFmd_dk = [
            [sp.diff(mass_F[d], sm.state[k]) for d in range(n_dim)]
            for k in range(n_st)
        ]
        dPmd_dk = [
            [sp.diff(mass_P[d], sm.state[k]) for d in range(n_dim)]
            for k in range(n_st)
        ]
        mass_B = [
            [sm.nonconservative_matrix[m, k, d] for d in range(n_dim)]
            for k in range(n_st)
        ]
        mass_S = sm.source[m, 0]

        affected = []
        for i in range(n_eq):
            if i == m:
                continue
            coeff = sp.simplify(M[i, h_col])
            if coeff == 0:
                continue
            affected.append(i)
            for k in range(n_st):
                for d in range(n_dim):
                    B[i, k, d] = sp.expand(
                        B[i, k, d]
                        - coeff * (dFmd_dk[k][d]
                                   + dPmd_dk[k][d]
                                   + mass_B[k][d])
                    )
            S[i, 0] = sp.expand(S[i, 0] - coeff * mass_S)
            M[i, h_col] = sp.S.Zero

        sm.mass_matrix = _to_zarray(M)
        sm.nonconservative_matrix = _to_zarray(B)
        sm.source = _to_zarray(S)
        sm.refresh_derived_operators(eigenvalues=False)
        sm.history.append({
            "name": self.name,
            "description": (
                f"substituted mass equation (row {m}) into rows "
                f"{affected}; cleared M[:, {h_col}] on those rows"
            ),
        })
        return sm

    @staticmethod
    def _find_h_state_index(sm: "SystemModel") -> int:
        """Index of the state entry named `h`; raises `ValueError` if absent."""
        for i, s in enumerate(sm.state):
            if getattr(s, "name", None) == "h":
                return i
        raise ValueError(
            "RemoveNonDiagonalH: no state entry named 'h' found; "
            "pass h_state_index explicitly."
        )

    @staticmethod
    def _find_mass_row(sm: "SystemModel", h_col: int) -> int:
        """Row `m` with `M[m, h_col] = 1` and zeros elsewhere."""
        candidates = []
        for i in range(sm.n_equations):
            entry = sp.simplify(sm.mass_matrix[i, h_col])
            if entry != 1:
                continue
            others_zero = all(
                sp.simplify(sm.mass_matrix[i, k]) == 0
                for k in range(sm.n_state) if k != h_col
            )
            if others_zero:
                candidates.append(i)
        if not candidates:
            raise ValueError(
                f"RemoveNonDiagonalH: no row found with M[i, {h_col}] = 1 "
                f"and zero elsewhere; pass mass_equation_index explicitly."
            )
        if len(candidates) > 1:
            raise ValueError(
                f"RemoveNonDiagonalH: multiple candidate mass rows "
                f"{candidates}; pass mass_equation_index explicitly."
            )
        return candidates[0]




class HydrostaticReconstruction:
    """System-level op: repack the gravity term `g h d_x eta` from the chain's `(P, B)` form
    to the standard SWE `P = g h^2/2` form that hydrostatic-reconstruction Riemann solvers
    expect.
    The bathymetry term is then supplied at runtime by the Audusse fluctuation.

    References:
        Audusse et al. 2004, https://doi.org/10.1137/S1064827503431090
    """

    name = "hydrostatic_reconstruction"
    description = (
        "Convert chain's g·h·η in (P, B) to standard SWE P = g·h²/2 "
        "for Audusse HR compatibility (drops -g·h·b_x; HR fluctuation "
        "supplies it at runtime)."
    )

    def __call__(self, sm: "SystemModel", *, h_state_index=None):
        """Repack every row carrying the `g h (b + h)` packaging; `h_state_index` is
        auto-detected when omitted.
        """
        # Locate h state index.
        if h_state_index is None:
            for i, s in enumerate(sm.state):
                if getattr(s, "name", None) == "h":
                    h_state_index = i
                    break
            if h_state_index is None:
                raise ValueError(
                    "PrepareForAudusseHR: no state entry named 'h' found"
                )
        h_sym = sm.state[h_state_index]

        # Locate g (parameter) and b (aux Symbol).
        g_sym = None
        for k in sm.parameters.keys():
            sym = sm.parameters[k]
            if getattr(sym, "name", None) == "g":
                g_sym = sym
                break
        if g_sym is None:
            raise ValueError(
                "PrepareForAudusseHR: no 'g' parameter found"
            )

        # ``b`` may live in either ``aux_state`` (b-as-static-aux) or
        # ``state`` (b-promoted-to-state with ``∂_t b = 0`` row).  HR
        # only needs the Symbol to match against the gravity packaging
        # ``P[i, d]`` and ``B[i, h, d]``; it does not care whether
        # ``b`` evolves.
        b_sym = None
        for s in sm.aux_state:
            if getattr(s, "name", None) == "b":
                b_sym = s
                break
        if b_sym is None:
            for s in sm.state:
                if getattr(s, "name", None) == "b":
                    b_sym = s
                    break
        if b_sym is None:
            raise ValueError(
                "HydrostaticReconstruction: no 'b' symbol found in state "
                "or aux_state"
            )

        n_eq = sm.n_equations
        n_dim = sm.n_dim

        # Pattern we expect on rows with gravity-on-η packaging:
        # P[i, d] contains  +g·h·(b + h) = g·h·b + g·h²
        # B[i, h, d] contains  -g·(b + h) = -g·b - g·h.
        # Repackage = subtract  g·h·b + g·h²/2  from P
        #             add  g·(b + h)  to B[i, h, d].
        # Detection: the *coefficient* of g·h·b in P[i, d] is +1 AND
        # the coefficient of g·b in B[i, h, d] is -1.  Then the
        # row has the gravity-on-η packaging.

        P = sp.Matrix(sm.hydrostatic_pressure.tolist())
        B = sp.MutableDenseNDimArray(
            sm.nonconservative_matrix.tolist(),
            shape=tuple(sm.nonconservative_matrix.shape),
        )

        affected = []
        for i in range(n_eq):
            for d in range(n_dim):
                P_id = sp.expand(P[i, d])
                B_ihd = sp.expand(B[i, h_state_index, d])
                # Coefficient of g·h·b in P[i, d].
                coeff_in_P = P_id.coeff(g_sym * h_sym * b_sym)
                # Coefficient of g·b in B[i, h, d].
                coeff_in_B = B_ihd.coeff(g_sym * b_sym)
                if coeff_in_P == 1 and coeff_in_B == -1:
                    # Repackage: P -= g·h·(b+h) - g·h²/2 = g·h·b + g·h²/2
                    P[i, d] = sp.expand(
                        P_id - g_sym * h_sym * b_sym
                        - g_sym * h_sym**2 / sp.Integer(2)
                    )
                    # B[i, h, d] -= -g·(b+h) = +g·(b+h)
                    B[i, h_state_index, d] = sp.expand(
                        B_ihd + g_sym * (b_sym + h_sym)
                    )
                    affected.append((i, d))

        if not affected:
            sm.history.append({
                "name": self.name,
                "description": "no rows with gravity-on-η packaging found",
            })
            return sm

        sm.hydrostatic_pressure = _to_zarray(P)
        sm.nonconservative_matrix = _to_zarray(B)
        sm.refresh_derived_operators(eigenvalues=False)
        sm.history.append({
            "name": self.name,
            "description": (
                f"repackaged gravity g·h·(b+h) → g·h²/2 on rows/dims "
                f"{affected}; -g·h·b_x source dropped — Audusse HR "
                f"must supply it at runtime"
            ),
        })
        return sm


class InvertMassMatrix:
    """System-level op: divide each evolution row by its diagonal mass entry `M_ii` and set
    `M_ii = 1`.
    Requires a diagonal mass matrix on evolution rows and zero on algebraic rows (check with
    `assert_diagonal_mass_matrix`); state-dependent entries give a symbolic `1/M_ii`.
    """

    name = "invert_mass_matrix"
    description = (
        "Divide each evolution row by its diagonal mass entry to reach "
        "canonical ∂_t Q form."
    )

    def __call__(self, sm: SystemModel):
        """Left-multiply every evolution row of `sm` by `1/M_ii`."""
        # Precondition: mass matrix is diagonal on evolution rows.
        sm.assert_diagonal_mass_matrix()

        n_eq = sm.n_equations
        n_st = sm.n_state
        n_dim = sm.n_dim
        e2s = list(sm.equation_to_state_index)

        # Mutable working copies of the operator tensors.
        M = sp.Matrix(sm.mass_matrix.tolist())
        F = sp.Matrix(sm.flux.tolist())
        P = sp.Matrix(sm.hydrostatic_pressure.tolist())
        S = sp.Matrix(sm.source.tolist())
        B = sp.MutableDenseNDimArray(
            sm.nonconservative_matrix.tolist(),
            shape=tuple(sm.nonconservative_matrix.shape),
        )

        n_flux_cols = F.shape[1]
        n_P_cols = P.shape[1]
        n_S_cols = S.shape[1]

        normalised = []
        for i in range(n_eq):
            diag = sp.simplify(M[i, e2s[i]])
            if diag == 0:
                # Algebraic row — nothing to invert.
                continue
            if diag == 1:
                continue
            inv = 1 / diag
            for k in range(n_flux_cols):
                F[i, k] = sp.simplify(F[i, k] * inv)
            for k in range(n_P_cols):
                P[i, k] = sp.simplify(P[i, k] * inv)
            for k in range(n_S_cols):
                S[i, k] = sp.simplify(S[i, k] * inv)
            for k in range(n_st):
                for d in range(n_dim):
                    B[i, k, d] = sp.simplify(B[i, k, d] * inv)
            M[i, e2s[i]] = sp.S.One
            normalised.append(i)

        sm.mass_matrix = _to_zarray(M)
        sm.flux = _to_zarray(F)
        sm.hydrostatic_pressure = _to_zarray(P)
        sm.source = _to_zarray(S)
        sm.nonconservative_matrix = _to_zarray(B)
        # Derived operators depend on the primaries; refresh.
        sm.refresh_derived_operators(eigenvalues=False)
        sm.history.append({
            "name": self.name,
            "description": (
                f"divided evolution rows {normalised} by their diagonal "
                f"mass entries"
            ),
        })


__all__ = [
    "face_normal_symbols",
    "SystemModel",
    "SystemModelRow",
    "SystemModelDescription",
    "InvertMassMatrix",
    "RemoveNonDiagonalH",
    "HydrostaticReconstruction",
    "UnclosedClosureError",
    "allow_unclosed",
]


#: Set inside the :func:`allow_unclosed` context manager — the ONE way to get a
#: SystemModel back with unclosed terms still in it.
_ALLOW_UNCLOSED = False


@contextmanager
def allow_unclosed():
    """Context manager permitting a build to return a `SystemModel` that still has unclosed
    terms, an explicit and greppable opt-in for inspecting open full-stress models.
    """
    global _ALLOW_UNCLOSED
    prev = _ALLOW_UNCLOSED
    _ALLOW_UNCLOSED = True
    try:
        yield
    finally:
        _ALLOW_UNCLOSED = prev


def _declare_model_hooks(sm, model) -> None:
    """Apply the model's own role and admissibility declarations (`declare_system_roles`,
    `admissible_constraints`), if any.
    """
    if model is None:
        return
    hook = getattr(model, "declare_system_roles", None)
    if callable(hook):
        hook(sm)
    tags = getattr(model, "tags", None)
    if tags:
        from zoomy_core.systemmodel.tags import declare_tags
        declare_tags(sm, tags, f"{type(model).__name__}.tags")
    adm = getattr(model, "admissible_constraints", None)
    if callable(adm):
        constraints = adm()
        if constraints is not None:
            sm.declare_admissible(constraints)


def _declared_aux_definitions(model) -> dict:
    """`{name: definition}` of the aux the model declared itself via ``add_equation(...,
    group="aux")``; the backend fills them (e.g. stay-3D column integrals).
    """
    for owner in (model, getattr(model, "derivation", None)):
        names = getattr(owner, "_aux_names", None)
        if not names:
            continue
        eqs = getattr(owner, "_equations", None) or {}
        return {str(nm): getattr(eqs.get(nm), "expr", None) for nm in names}
    return {}


def _assert_declared_closures_closed(sm, model, canonical_source=None):
    """Raise `UnclosedClosureError` if a closure declared via `Model.declare_closure` was
    never substituted and its aux slot would silently read zero.
    """
    # ``_closure_vars`` lives on the INNER per-layer DModel built inside
    # ``derive_model`` and does NOT survive to the model handed here, so
    # trusting it makes this a silent no-op.  Use the emitted object: an aux
    # slot with NO aux_registry row is one that nothing ever computes.
    computed = set()
    for entry in (getattr(sm, "aux_registry", None) or ()):
        nm = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", None)
        if nm is not None:
            computed.add(str(nm))

    # …and the OTHER way an aux gets a value: a real ``update_aux_variables``
    # row.  A model that declares ``aux_variables=["r_o"]`` and returns the
    # rain rate from ``update_aux_variables()`` has no registry entry — the
    # registry only holds what ``expose_aux_atoms`` minted — so reading the
    # registry alone calls that computed field unclosed.  A row equal to its
    # own aux symbol is the identity passthrough ``register_aux`` writes for
    # rows it does not own, and computes nothing.
    uav = getattr(sm, "update_aux_variables", None)
    if uav is not None:
        for row, sym in enumerate(sm.aux_state or ()):
            try:
                expr = uav[row, 0] if getattr(uav, "shape", None) else uav[row]
            except (IndexError, TypeError):
                continue
            if sp.sympify(expr) != sp.sympify(sym):
                computed.add(str(sym))

    # …UNLESS the model declared it as a genuine auxiliary with a definition.
    # Those have no registry row by construction (a column quadrature is not an
    # algebraic aux update) and are filled by the backend — declared, not
    # forgotten.  Record them on the SystemModel so the distinction is
    # inspectable rather than living only inside this check.
    declared = _declared_aux_definitions(model)
    # Name the model the USER wrote.  On the declarative path ``model`` is the
    # inner derivation object, whose class is the generic ``Model`` — reporting
    # that tells the reader nothing about which of their models is open.
    name = type(canonical_source if canonical_source is not None
                else model).__name__
    aux_names = [str(a) for a in (sm.aux_state or ())]
    sm.external_aux = {nm: defn for nm, defn in declared.items()
                       if nm in aux_names}

    unclosed = sorted(nm for nm in aux_names
                      if nm not in computed and nm not in sm.external_aux)
    sm._unclosed_closures = unclosed
    if not unclosed:
        return
    if _ALLOW_UNCLOSED:
        import warnings
        warnings.warn(
            f"{name}: {len(unclosed)} unclosed term(s) "
            f"({', '.join(unclosed)}) — returned under allow_unclosed(). "
            "The system is OPEN: those rows read ZERO until closed. "
            "See sm.describe_unclosed().", stacklevel=2)
        return
    raise UnclosedClosureError(unclosed, name, system_model=sm)


class UnclosedClosureError(RuntimeError):
    """A declared closure reached the `SystemModel` with nothing closing it; carries the built
    `system_model`.
    """

    def __init__(self, unclosed, model_name="", system_model=None):
        """Store the unclosed terms, the model name and the built system."""
        self.unclosed = list(unclosed)
        self.model_name = model_name
        self.system_model = system_model
        super().__init__(
            f"{model_name}: {len(self.unclosed)} DECLARED closure(s) reached the "
            f"SystemModel unbound: {', '.join(self.unclosed)}.\n"
            "These were swept into aux_state with no update_aux_variables row, "
            "so they would read ZERO — silently absent physics (e.g. an "
            "internal interface traction, leaving the layers frictionless).\n"
            "Fix by EITHER passing a closure that binds them, OR — if the field "
            "really is supplied from outside — declaring it with a definition "
            "via add_equation(name, Eq(sym, expr), group='aux') / "
            "register_aux(name, formula).\n"
            "To inspect the open system instead, rebuild inside "
            "`with allow_unclosed():` and call sm.describe_unclosed().")

    def describe(self) -> str:
        """The open-term listing (see `SystemModel.describe_unclosed`)."""
        if self.system_model is not None:
            return self.system_model.describe_unclosed()
        return "\n".join([f"UNCLOSED TERMS — {self.model_name}"]
                         + [f"  {nm}" for nm in self.unclosed])
