"""SystemModel — operator-form symbolic PDE for analysis + transformation.

`SystemModel` is the runtime/operator-form representation of a PDE
system: a flat container of symbolic ``sp.Matrix`` / sympy-tensor
operators (flux, non-conservative-matrix, source, mass matrix,
hydrostatic pressure) plus state, parameters, coordinates, and
boundary-condition kernels.  It carries no derivation history and no
equation tree — that's the `Model` class's job.

The two are **independent siblings**, not inherited.  `Model` owns
derivation; `SystemModel` owns the operator surface that solvers and
analysis consume.  ``SystemModel.from_model(m)`` extracts the
operators once by calling the model's API methods and freezes the
result; subsequent calls on the SystemModel never re-walk.

Operations on a SystemModel modify the stored matrices in place via
``apply(operation)`` — the most important being ``InvertMassMatrix``
which left-multiplies the system by ``M⁻¹`` to reach canonical
``∂_t Q + ∂_x F + B·∂_x Q − S = 0`` form.

Solvers and analysis routines accept either a ``Model`` or a
``SystemModel`` directly; internally they normalize via
``SystemModel.from_model(m)`` if a Model is passed.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import sympy as sp

from zoomy_core.misc.misc import Zstruct, ZArray


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
    """The canonical outward face-normal symbols ``n0 … n_{n_dim-1}``.

    ONE definition of the name/ordering convention, shared by the
    ``SystemModel.normal`` Zstruct and by every DERIVATION that needs to write
    a geometric (normal-aware) relation — e.g. a model-registered free-slip
    wall, which reflects only the normal component of the momentum vector.
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
}

# The canonical time-step symbol (matches ``to_numpy.DT_SYMBOL`` and the
# ``sp.Symbol("dt", positive=True)`` the C printers bind).
_DT_SYMBOL = sp.Symbol("dt", positive=True)


def _to_zarray(obj):
    """Coerce a sympy Matrix / NDimArray / list into ``ZArray``.

    Accepts ``None`` (returns None) and existing ``ZArray`` (returns
    as-is).  Shape is preserved exactly — column vectors stay
    ``(n, 1)`` Matrix → ``(n, 1)`` ZArray.  Construction sites are
    responsible for choosing the rank they want; this normaliser
    just unifies the storage type.
    """
    if obj is None or isinstance(obj, ZArray):
        return obj
    if isinstance(obj, sp.MatrixBase):
        return ZArray(obj.tolist(), shape=tuple(obj.shape))
    if isinstance(obj, sp.NDimArray):
        return ZArray(obj.tolist(), shape=tuple(obj.shape))
    return ZArray(obj)


def _channel_explicit_diffusion(model, canonical_source):
    """Mirror the basemodel ``from_model`` handling of the EXPLICIT diffusion
    tensor on the declarative (``_from_derivation_model``) path.

    A case-side SME/VAM model may override ``diffusion_matrix_explicit()`` (the
    ``∇·(ν_h h ∇u)`` horizontal-eddy-viscosity slot).  The override lives on the
    user-facing model — the ``canonical_source`` here (the SME/VAM wrapper the
    case constructs), NOT on the derivation ``model`` whose untagged residuals
    the structural extractor reads.  Read the override off either object,
    coerce it to an ``NDimArray`` exactly as the basemodel branch's
    ``_extract_A`` does, and return it.

    Returns ``None`` when no real (non-zero) override is declared — the
    basemodel default ``diffusion_matrix_explicit()`` returns an all-zero
    tensor, so treat that as "no override" and inject NOTHING (a plain SME/VAM
    keeps ``diffusion_matrix_explicit is None``; the implicit, extractor-derived
    ``diffusion_matrix`` slot is never touched here)."""
    def _is_zero(e):
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
    if not shape:
        yield ()
        return
    for i in range(shape[0]):
        for rest in _iter_indices(shape[1:]):
            yield (i,) + rest


def _cov_diffusion_tensor(A_old, sub_map, J, *, n_state):
    """Apply a state-variable change-of-vars to a diffusion tensor.

    ``A`` has shape ``(n_eq, n_state, n_dim, n_dim)`` and enters the
    residual via the contraction
    ``F_diff[i, d] = Σ_{j, e} A[i, j, d, e] · ∂_e Q[j]``.
    Under ``∂_e Q_old[j] = J[j, k] · ∂_e Q_new[k]`` this rewrites to
    ``A_new[i, k, d, e] = Σ_j A_old[i, j, d, e](T) · J[j, k]``.

    Returns ``None`` when ``A_old`` is ``None`` (default-zero slot).
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
    """Apply a state-variable change-of-vars to a symbolic BC kernel.

    The BC kernel is a :class:`zoomy_core.model.basefunction.Function`
    whose ``definition`` is a (typically Piecewise) sympy expression
    referencing state Symbols, and whose ``args`` Zstruct binds named
    positional arguments — including ``variables`` — to those Symbols.

    Under CoV we (i) xreplace the OLD state Symbols in ``definition``
    with their NEW-state expressions, and (ii) rebuild
    ``args.variables`` as a fresh Zstruct over the new state Symbols
    (preserving the ``_symbolic_name`` so the lambdifier sees the same
    positional slot).  Other ``args`` entries (``time``, ``position``,
    ``distance``, ``aux_variables``, ``parameters``, ``normal``,
    ``idx``) stay untouched.

    Returns ``None`` when ``bc_func`` is ``None``.
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
    """Route a custom derivation function-group ``group_name`` to the
    ``SystemModel`` attribute ``sm_attr`` (e.g. a derived wall BC → a
    boundary-condition slot)."""
    _FUNCTION_SLOTS[group_name] = sm_attr


def _grad_aux_name(base, axis="x") -> str:
    """Name a spatial-gradient aux symbol after the field it differentiates and
    the axis/axes it is taken along: ``∂_x h → dhdx``, ``∂_y q_0 → dq0dy``,
    ``∂_x∂_x b → dbdxx`` (``axis`` carries the concatenated axis letters)."""
    if isinstance(base, sp.Function):
        head = base.func.__name__
        idx = "".join(str(int(a)) for a in base.args
                      if getattr(a, "is_Integer", False))
        return f"d{head}{idx}d{axis}"
    return f"d{base}d{axis}"


def _gradients_to_aux(expr, spaces):
    """Replace every PURELY-SPATIAL ``∂(field)`` in ``expr`` (along ANY of the
    ``spaces`` coordinates, to any order) with a fresh gradient-aux Symbol.

    Returns ``(new_expr, grads)`` with ``grads`` an order-stable, de-duplicated
    list of ``(aux_symbol, differentiated_field, multi_index_tuple)`` — the
    runtime aux the solver supplies from its LSQ stencil, and the metadata the
    caller registers in ``aux_registry`` so :meth:`Solver._walk_derivative_aux`
    fills them.  This is what a reconstructed ``w`` (and any derivative-carrying
    reconstruction slot) needs to lower cleanly: it leaves NO raw ``Derivative``
    atoms for the backend printer (which would otherwise drop the slot — e.g.
    the jax ``interpolate_to_3d`` drop) and is dimension-complete (x AND y AND
    z), not the old x-only conversion that left ``∂_y`` raw in the 2-D w slot.
    """
    space_index = {s: d for d, s in enumerate(spaces)}
    n_dim = len(spaces)
    grads: list = []
    seen: set = set()

    def _is_spatial_grad(e):
        return (isinstance(e, sp.Derivative)
                and all(v in space_index for v, _ in e.variable_count))

    def _to_sym(e):
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
    """Collapse applied state/aux fields ``q(i, t, x) → q_i``, ``h(t, x) → h``
    so the stored operator reads in the SystemModel's bare state symbols."""
    from zoomy_core.model.derivation.system_extract import _state_symbol

    def _is_field(e):
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
    """Parse ``model._function_groups`` into the matching ``SystemModel`` slots.

    Each group ``{component_index: rhs}`` becomes a vector operator: the rhs is
    rewritten into bare state symbols, every ``∂_x(state)`` becomes a runtime
    gradient aux (appended to ``aux_state``), and the unit-interval vertical
    coordinate ``ζ`` is exposed via ``sm.position``.

    Groups named ``boundary:<name>`` are model-DERIVED boundary definitions
    (``{state_index: boundary value}``); they land in ``sm.boundary_specs``
    where :class:`~zoomy_core.model.boundary_conditions.FromModel` picks them
    up by ``<name>`` at :meth:`SystemModel.attach_boundary_conditions` time.
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
        """A boundary-group key may be the state FIELD itself — resolve it to
        its slot in ``sm.state`` so registrations never hard-code the layout."""
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
    """``Min`` over ``k ≤ iterations`` of ``(maxᵢ (Mᵏ·1)ᵢ)^{1/k}`` — a strict
    upper bound on ``ρ(M)`` for a NONNEGATIVE symbolic matrix ``M`` (given as
    a list of rows), read off one ``vₖ = M vₖ₋₁`` chain from ``v₀ = 1``.

    See :attr:`SystemModel.spectral_radius_bound` for the statement, the
    safety argument and the measured tightness.  Every candidate is an upper
    bound on its own, so the ``Min`` is one too.

    Deliberately ALL products and NO quotients: the obvious companion bound
    (Collatz–Wielandt, ``ρ(M) ≤ maxᵢ (Mv)ᵢ/vᵢ`` for any ``v > 0``) is tighter
    per iteration but needs the iterate rescaled and floored away from zero —
    and a ``Max`` INSIDE the chain multiplies the symbolic TREE size by
    ``n + 1`` per step on top of the ``n`` the matvec already costs.  Sympy's
    ``Max``/``Min`` constructor sorts its arguments through ``ordered`` →
    ``_node_count``, which walks the TREE and does not see the shared DAG, so
    that turned a 0.8 s foam emission of ``SME(level=1)`` into 70 s.  Products
    only keeps the tree at ``nᵏ`` and the emission at 2 s.

    ``evaluate=False`` on both reductions for the SAME reason
    ``local_max_abs_eigenvalue`` uses it: ``Max``/``Min`` over wave-speed
    scale expressions is a RUNTIME reduction, and letting sympy attempt the
    pairwise ordering expands every branch and explodes setup further.

    No rescaling means ``vₖ`` carries the raw ``ρᵏ``.  In IEEE double that is
    unreachable either way for a wave speed: overflow needs ``‖A‖ > 1e77``
    and underflow ``‖A‖ < 1e-77`` at ``k = 4``.
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


@dataclass
class SystemModel:
    """Symbolic operator-form PDE system.

    Stored matrices follow the contract:

    .. math::

        M(Q) \\; \\partial_t Q
            + \\nabla \\cdot \\big(F(Q) + P(Q)\\big)
            + \\sum_d B(Q)[:,:,d] \\, \\partial_d Q
            - S(Q) = 0

    Operators are indexed ``(equation_row, state_col[, dim])``.  In
    general the system is **rectangular** — there can be fewer (or
    more) equations than state entries; this is the case for the
    splitter sub-systems, where each stage updates only a subset of
    the shared state vector.  ``equation_to_state_index[r]`` records
    which state entry equation ``r`` updates; for square systems the
    default is the identity map ``[0, 1, …, n_state-1]``.

    Every operator the system needs is a **stored field**, frozen at
    construction — there are no derived-on-demand methods.  Alongside
    the five primaries (flux, P, NCP, source, mass matrix) the system
    carries the derived operators ``quasilinear_matrix``, the source
    jacobians ``source_jacobian_wrt_variables`` /
    ``source_jacobian_wrt_aux_variables`` and ``eigenvalues``, plus the
    ``normal`` symbols and the parameter Zstructs.  ``quasilinear_matrix``
    is a cheap derivative computed lazily on first access.  The source
    jacobians are CHANNELED from the source Model in :meth:`from_model`
    (read off the SystemModel, never re-derived by printers); a standalone
    SystemModel falls back to deriving them once from ``source`` in
    ``__post_init__``.  Any in-place operation refreshes them via
    :meth:`refresh_derived_operators`.  ``eigenvalues`` is the one
    operator that may be expensive (a symbolic spectral derivation) or
    deliberately skipped — :meth:`from_model` simply carries over
    whatever ``Model.eigenvalues()`` produced, and ``None`` means
    "skipped" (numerical-eigenvalue mode).

    Shape contract:

    * ``flux``                  — ``(n_eq, n_dim)``
    * ``hydrostatic_pressure``  — ``(n_eq, n_dim)``
    * ``nonconservative_matrix``— ``(n_eq, n_state, n_dim)``
    * ``source``                — ``(n_eq, 1)``
    * ``mass_matrix``           — ``(n_eq, n_state)``
    * ``quasilinear_matrix``    — ``(n_eq, n_state, n_dim)``
    * ``source_jacobian_wrt_variables``     — ``(n_eq, n_state)``
    * ``source_jacobian_wrt_aux_variables`` — ``(n_eq, n_aux)``
    * ``eigenvalues``           — ``(n_eq, 1)`` or ``None`` if skipped

    ``parameters`` is a Zstruct mapping parameter name → symbol (the
    symbolic side the operators reference); ``parameter_values`` maps
    name → numeric default.  ``normal`` is a Zstruct of the face-normal
    component symbols ``n0, n1, …``.
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

    def __post_init__(self):
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
        return self.flux.shape[0]

    @property
    def n_state(self) -> int:
        return len(self.state)

    @property
    def n_dim(self) -> int:
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
        """Q rows carrying exactly one factor of the depth — the rows an
        Audusse-Bristeau-Klein hydrostatic reconstruction MUST rescale by
        ``h*/h``.

        In a depth-averaged system every conserved row except the depth
        itself and the bathymetry is an EXTENSIVE (depth-weighted) quantity:
        ``q_{d,j} = h·alpha_{d,j}`` for a moment model, ``hu`` for SWE,
        ``hK`` / ``hE`` for a k-epsilon closure.  Rescaling all of them is
        what makes the reconstructed pair a STATE — the Audusse positivity
        proof assumes ``q* = h* u``, and a partially rescaled pair
        ``(h* = 0, q != 0)`` is not a state at all: it exports mass out of an
        empty cell.

        ``h`` and ``b`` are excluded when (and only when) they are state
        rows; a model that carries either in ``aux_state`` has nothing to
        exclude here.
        """
        return [i for i, s in enumerate(self.state)
                if str(s) not in ("h", "b")]

    @property
    def discharge_state_indices(self) -> list:
        """Per horizontal direction ``d``, the state row whose value IS the
        depth-integrated mass flux in that direction — ``hu``/``hv`` for SWE,
        ``q_x_0``/``q_y_0`` for a moment model.  ``-1`` where the system has no
        unique such row.

        Read straight off the model's OWN mass balance: the ``h`` equation is
        ``d_t h + d_d (mass flux)_d = 0``, so the direction-``d`` discharge row
        is the single state the mass-row flux depends on.  That is the row
        which

        * exports the mass an Audusse-reconstructed face may not export
          (hence its place in the positivity argument), and
        * carries the bed-slope / hydrostatic force, so it is the row that
          receives the Audusse well-balancing source once the bed NCP has
          been cancelled by ``b_L = b_R = b_face`` at the face.

        ``-1`` is returned per direction when the mass flux is not a single
        state row — a multilayer system, whose mass flux is the SUM of the
        layer discharges, genuinely has no single momentum row and a
        depth-averaged bed source is not defined for it.  Consumers must
        check, not assume.
        """
        names = [str(s) for s in self.state]
        if "h" not in names:
            return [-1] * self.n_dim
        i_h = names.index("h")
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
        return self.n_equations

    @property
    def n_aux_variables(self) -> int:
        return len(self.aux_state)

    @property
    def n_parameters(self) -> int:
        return self.parameters.length()

    @property
    def dimension(self) -> int:
        return self.n_dim

    @property
    def variables(self) -> Zstruct:
        """Zstruct view ``name → state Symbol`` (mirrors ``Model.variables``)."""
        v = Zstruct(**{str(s): s for s in self.state})
        v._symbolic_name = "Q"
        return v

    def update_variables_or_identity(self) -> ZArray:
        """``update_variables`` with the documented ``None ⇒ identity`` fallback
        MADE EXPLICIT — always an ``(n_eq, 1)`` expression, never ``None``.

        Row ``i`` of the identity is ``state[equation_to_state_index[i]]``, i.e.
        exactly "leave Q alone", which is what the numpy runtime already does
        when it sees ``None`` (``solver_numpy._apply_update_variables`` returns
        Q unchanged).  This is the seam the C-family printers emit from, so the
        generated ABI carries the symbol even for a model with no per-cell
        remap.

        Why the ABI must not be conditional: the wet/dry cap is OFF by default,
        so a derived SME(level=0) has ``update_variables = None`` — and foam's
        hand-written ``zoomyFoam.C:352`` calls ``Model::update_variables(...)``
        unconditionally.  Emitting the symbol only when the slot is non-empty
        made every capless model fail to compile with "'update_variables' is
        not a member of 'Model'".  Callers must not have to know whether a
        model happens to clamp.
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
        """Zstruct view ``name → aux Symbol`` (mirrors ``Model.aux_variables``)."""
        v = Zstruct(**{str(s): s for s in self.aux_state})
        v._symbolic_name = "Qaux"
        return v

    @property
    def _parameter_symbols(self) -> Zstruct:
        """Alias of ``parameters`` (name → Symbol) for Model-API parity."""
        return self.parameters

    # ── Operators carried as declared Functions (single signature source) ──

    def position_struct(self) -> Zstruct:
        """The length-3 position group ``(x, y, z)`` — THE accessor for a
        consumer that needs the coordinates (printers, analysis).
        ``self.position`` is set by the derivation / BC path; a bare
        SystemModel resolves the canonical symbols exactly like
        ``attach_boundary_conditions``."""
        if self.position is not None:
            return self.position
        return Zstruct(X0=self.space[0],
                       X1=sp.Symbol("y", real=True),
                       X2=sp.Symbol("z", real=True))

    _position_struct = position_struct

    def operator_signature(self, name: str) -> Zstruct:
        """The declared ``args`` :class:`Zstruct` for operator ``name`` — the
        SINGLE source every backend reads.  Groups are materialised from this
        SystemModel's own symbols in the ``OPERATOR_ARG_SLOTS`` order."""
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
        """The ``WB_<state>`` symbols the reconstruction INVERSE is written in.

        ``state_from_reconstruction`` takes the limiter's reconstructed FACE
        values, not the cell state, so its first argument group is these fresh
        symbols rather than ``variables``."""
        from zoomy_core.model.reconstruction_inverse import reconstruction_symbols
        syms = reconstruction_symbols(list(self.state))
        return Zstruct(**{s.name: s for s in syms})

    def operator(self, name: str) -> "Any":
        """The operator kernel CARRIED as a ``basefunction.Function``: ``args``
        is the declared signature (:meth:`operator_signature`) and ``definition``
        is the live operator array.  Backends read ``operator(name).args`` for
        the signature and only translate syntax — they never route arguments."""
        from zoomy_core.model.basefunction import Function
        defn = getattr(self, name, None)
        return Function(name=name, args=self.operator_signature(name),
                        definition=defn if defn is not None else ZArray([0]))

    @property
    def eigenvalue_mode(self) -> str:
        """``"symbolic"`` when a symbolic spectrum was carried over,
        ``"numerical"`` when it was skipped (``eigenvalues is None``) —
        mirrors ``Model.eigenvalue_mode`` so a solver's wavespeed build
        branches the same way."""
        return "numerical" if self.eigenvalues is None else "symbolic"

    def attach_boundary_conditions(self, bcs, aux_bcs=None):
        """Inject boundary conditions onto a declarative SystemModel using the
        SAME ``BoundaryConditions`` objects as the production models — the hook
        for "use our old BCs as we always did".

        Builds the indexed symbolic BC kernels (``boundary_conditions`` /
        ``boundary_gradients`` / ``aux_boundary_conditions``) the runtime +
        codegen expect, from this SystemModel's own time / position / state /
        aux / parameters / normal.  ``aux_bcs`` defaults to per-tag
        Extrapolation.  Optional — a SystemModel without BCs still builds; the
        solver only needs them at run time.  Returns ``self``.
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
        # Characteristic).  Recurse into any PerFieldBoundary slot BCs.
        def _prepare(bc):
            if hasattr(bc, "resolve"):
                bc.resolve(self)
            bc.bind_on(state_names)
            for inner in {id(v): v for v in
                          getattr(bc, "_slot_bc", {}).values()}.values():
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

    @property
    def is_square(self) -> bool:
        """True when n_equations == n_state and equations map identity-style."""
        return (self.n_equations == self.n_state
                and self.equation_to_state_index == list(range(self.n_state)))

    @property
    def stationary_indices(self) -> frozenset:
        """State indices whose evolution is identically zero by
        construction.

        A field ``Q[i]`` is **stationary** iff its row vanishes in
        every transport / source / diffusion slot:

        - ``flux[i, :] = 0``
        - ``hydrostatic_pressure[i, :] = 0``
        - ``nonconservative_matrix[i, :, :] = 0``
        - ``source[i] = 0``
        - ``source_explicit[i] = 0`` (if present)
        - ``diffusion_matrix[i, :, :, :] = 0`` (if present)
        - ``diffusion_matrix_explicit[i, :, :, :] = 0`` (if present)

        The discrete time-step then writes nothing to ``Q[i]``, so any
        post-step hook (slope limiter, positivity projection, ...)
        that does is introducing **spurious** modification — the
        canonical example is a vertex-based DG slope limiter clipping
        the bathymetry ``b`` row in a shallow-water model, which
        breaks rank-boundary consistency under MPI and silently
        creates or destroys topography.

        Solvers should consult this set to skip such fields when
        applying post-step state transforms.  Users can override (in
        either direction) via per-solver kwargs that resolve handles
        through :meth:`field_index`.
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
        """Return ``True`` if ``tensor[i, ...]`` is identically zero.

        ``None`` (slot absent) counts as zero.  Entries are checked
        with ``== 0`` only (catches ``sp.S.Zero`` / ``sp.Integer(0)``
        / ``0`` — the canonical "stationary row" produced by
        ``Matrix.zeros`` / ``sp.MutableDenseNDimArray.zeros`` plus the
        model author's explicit zero assignment).  We deliberately do
        **NOT** fall back to ``sp.simplify`` here — the source / NCP /
        diffusion rows of non-stationary fields routinely contain
        deeply nested expressions (Manning friction = nested
        ``sqrt`` ∘ ``Max`` ∘ ``Rational``) on which ``sp.simplify``
        runs for many minutes.  Stationary detection runs at solver
        setup, so any pessimisation hits the user-facing startup
        latency.  If a model author writes a non-trivially-simplified
        expression that *happens* to evaluate to zero, they should
        either simplify it in the model or list the field manually
        via ``solver.limiter_exclude_fields``.
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
        """Resolve a state-field handle to its integer index.

        Accepts:

        - ``sp.Symbol`` (e.g. ``self.variables.b``) — matched by name.
        - ``str`` (e.g. ``"b"``) — matched by name.
        - ``int`` — returned as-is (with bounds check).

        Raises ``ValueError`` if the handle does not correspond to a
        registered state variable.  This is the canonical way for
        solvers to translate the *symbolic* field handles that the
        model author writes (``[model.variables.b]``) into the
        *integer* indices the post-processing pipeline uses
        internally — without ever forcing the model author to think
        about state layout.
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
        """Flux Jacobian ``∂F/∂Q + ∂P/∂Q + B`` — LAZY: computed from the
        primaries on first access, then cached.  Assigning re-seeds the cache
        (e.g. ``refresh_derived_operators`` after a state change)."""
        if self._quasilinear_matrix is None:
            self._quasilinear_matrix = _to_zarray(self._compute_quasilinear_matrix())
        return self._quasilinear_matrix

    @quasilinear_matrix.setter
    def quasilinear_matrix(self, value):
        self._quasilinear_matrix = _to_zarray(value)

    def _compute_quasilinear_matrix(self):
        """``∂F/∂Q + ∂P/∂Q + B`` — shape ``(n_eq, n_state, n_dim)``.

        Computed from the CLEAN primaries (with a bare ``1/h``), so a wet
        entry reads e.g. ``∂(q²/h)/∂h = −q²/h²``.  The wet/dry
        desingularization does NOT recompute this from ``q²·hinv`` (which
        would drop the ``−u²`` term, since ``hinv`` is an independent symbol);
        instead :func:`zoomy_core.systemmodel.operations.regularize_pow`
        FREEZES this clean matrix and then substitutes ``1/h**k → hinv**k``
        into it (``−q²/h² → −q²·hinv²``), keeping the wavespeed exact and the
        codegen factored — no ``∂kp_hinv/∂h`` Heaviside/Dirac chain term."""
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
        """``∂S/∂Q`` — shape ``(n_eq, n_state)``.  Fallback derivation used only
        when no Model channeled ``source_jacobian_wrt_variables``."""
        n_eq = self.n_equations
        n_st = self.n_state
        out = sp.zeros(n_eq, n_st)
        for i in range(n_eq):
            for j in range(n_st):
                out[i, j] = sp.diff(self.source[i, 0], self.state[j])
        return out

    def _compute_source_jacobian_wrt_aux_variables(self):
        """``∂S/∂Qaux`` — shape ``(n_eq, n_aux)``.  Fallback derivation used only
        when no Model channeled ``source_jacobian_wrt_aux_variables``."""
        n_eq = self.n_equations
        n_aux = len(self.aux_state)
        out = sp.zeros(n_eq, n_aux)
        for i in range(n_eq):
            for k in range(n_aux):
                out[i, k] = sp.diff(self.source[i, 0], self.aux_state[k])
        return out

    def ensure_jacobians(self):
        """Derive the source/flux jacobians NOW, if they are not already there.

        The fallback derivation is off by default (:attr:`derive_jacobians`)
        because it is ``n_eq x n_state`` sympy.diff calls and is wasted whenever
        nothing downstream asks for them.  A consumer that DOES need them — IMEX
        assembling an implicit source Jacobian, a quasilinear/steady-ODE
        assembly — calls this BEFORE the runtime is lowered, since the printers
        emit whatever is present at that moment.

        Idempotent, and never overwrites a matrix a Model channeled.
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
        """``∂F/∂Qaux`` — shape ``(n_eq, n_aux, n_dim)``.

        The flux twin of :meth:`_compute_source_jacobian_wrt_aux_variables`, and
        needed for the same reason its source counterpart is: a TOTAL derivative
        of the flux along x is

            d/dx F(Q, Qaux) = ∂F/∂Q · dQ/dx + ∂F/∂Qaux · dQaux/dx

        and ``quasilinear_matrix`` carries only the first term (it differentiates
        w.r.t. the state with aux held independent).  Anything that forms a
        steady balance or a quasilinear system from the flux — the
        moving-equilibrium steady-ODE reconstruction is the current consumer —
        needs the second term too, or it silently solves a different equation.

        Note ``P`` (hydrostatic pressure) is deliberately NOT folded in here:
        this is the flux jacobian, and callers that want the full transport
        matrix add ``∂P/∂Qaux`` themselves, exactly as ``quasilinear_matrix``
        assembles ``∂F/∂Q + ∂P/∂Q + B`` from separate pieces.
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
        """Eigenvalues of the normal-projected quasilinear matrix —
        ``(n_eq, 1)``.  This is the (potentially expensive) symbolic
        spectral derivation; callers decide when to pay it."""
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
        """Opaque eigendecomposition of the normal-projected quasilinear
        matrix ``A_n = Σ_d n_d · quasilinear_matrix[:, :, d]``.

        Flattened layout (row-major), length ``n + 2·n²``::

            [ eigenvalues λ (n),
              right eigenvectors R (n·n),
              left  eigenvectors L = R⁻¹ (n·n) ]

        Each component is a call to the opaque
        :class:`zoomy_core.model.kernel_functions.eigensystem` kernel —
        realised numerically per backend (``np.linalg.eig`` / Eigen),
        never a closed-form symbolic spectral derivation.  The Roe
        scheme builds ``|A| = R |Λ| L`` symbolically on top.  This is a
        *derived* operator (cheap symbolic wrapping of
        ``quasilinear_matrix``), so it is exposed as a property — always
        consistent with the current quasilinear matrix — and mirrors
        ``Model.eigensystem`` so a SystemModel-driven solver reaches the
        same operator the root model defines.
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
        """λ-only numerical spectrum of the normal-projected quasilinear matrix
        ``A_n = Σ_d n_d · quasilinear_matrix[:, :, d]`` — ``n`` calls to the
        opaque :class:`zoomy_core.model.kernel_functions.eigenvalues` kernel.

        The light companion of :attr:`eigensystem` (no R / L): the wave-speed /
        CFL bound needs only ``max|λ|``, so this feeds
        ``Numerics.local_max_abs_eigenvalue`` when the model has no closed-form
        ``eigenvalues`` (SME / VAM) — a DIMENSIONALLY CORRECT numerical spectrum
        in place of the old Gershgorin row-sum bound, which was a row-sum of a
        dimensionally-inhomogeneous matrix (mass row ~1, momentum row ~g·h) and
        scaled as ``g·h`` rather than ``sqrt(g·h)`` — ~6-9× too large (REQ-167).
        A *derived* operator (cheap symbolic wrapping of ``quasilinear_matrix``),
        so exposed as a property — always consistent with the current matrix."""
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

    @property
    def spectral_radius_bound(self):
        """CLOSED-FORM upper bound on ``ρ(A_n)`` — the wave speed, without an
        eigen-decomposition.  A scalar expression, not a spectrum.

        The CFL / Rusanov consumer needs ``max|λ|``, and a RADIUS is far
        cheaper than a SPECTRUM: :attr:`numerical_eigenvalues` costs an
        ``Eigen::EigenSolver`` per face per step (an ``O(n³)`` QR iteration
        behind an opaque kernel call), which is what made every emitted C /
        foam SME case run ~40× slow once a model stopped carrying a
        closed-form ``eigenvalues``.

        ONE classical statement, applied at several ``k``::

            ρ(A)ᵏ = ρ(Aᵏ) ≤ ‖Aᵏ‖_∞ ≤ ‖|A|ᵏ‖_∞ = ‖|A|ᵏ·1‖_∞

        so ``ρ(A_n) ≤ (maxᵢ (Mᵏ·1)ᵢ)^{1/k}`` with ``M = |A_n|``, for EVERY
        ``k ≥ 1``.  ``k = 1`` is exactly the Gershgorin row-sum ``‖A_n‖_∞``;
        each further ``k`` is one more matvec on the same chain
        ``vₖ = M vₖ₋₁`` from ``v₀ = 1``, and the bound falls toward
        ``ρ(|A_n|)``.  The emitted kernel is the ``Min`` over ``k ≤
        iterations``.

        The second step is what makes this usable at all, and SWE at rest
        shows why in two lines: ``A = [[0, 1], [c², 0]]`` has ``ρ = c``, the
        row-sum bound reads ``c²`` (a factor ``c`` = 3.13 too large at
        ``g h = 9.81``, and dimensionally wrong — ``g·h``, not ``√(g·h)``,
        exactly the REQ-167 objection to Gershgorin), and ``M²·1 = (c², c²)``
        makes ``k = 2`` return ``c`` EXACTLY.  Squaring is what restores the
        dimensional homogeneity the conservative variables destroyed.

        **Safety is structural, not statistical.**  Every candidate is an
        upper bound on its own — no eigenvalue is ever computed, nothing
        converges, and there is no iterate at which the inequality could fail.
        So ``iterations`` buys TIGHTNESS ONLY, the CFL cannot go unsafe, and
        no safety factor is needed.  Measured on the confluence
        ``SME(level=4, dimension=3)`` (n = 12) against
        ``numpy.linalg.eigvals`` of the SAME matrix over 1688 states — wet,
        near-dry to ``h = 1e-8``, ``|u|`` to 10 m/s, shear to ``3·u``, bed
        steps, axis-aligned and oblique normals — as ``bound / ρ(A_n)``, and
        the codegen cost of the foam ``NumericsKernels.H`` it emits:

        ==========  ==========  ==========  ==========  ==========  =========
        iterations  median wet  p95 wet     median all  min ratio   emit (s)
        ==========  ==========  ==========  ==========  ==========  =========
        1           2.79        6.00        4.86        1.457       ~40
        2           1.40        1.80        2.17        1.224       46
        3           1.38        1.63        1.88        1.126       424
        ==========  ==========  ==========  ==========  ==========  =========

        (5064 bound evaluations, 0 below 1.  For reference the incumbent — the
        opaque eigensolve — is not a bound at all: it returns ``max|Re λ|``,
        which came in BELOW ``ρ(A_n)`` on 2 of those states, both
        non-hyperbolic.)

        Two things set the default at 2.  It is where the runtime win is
        already essentially complete (2 matvecs = ``2n²`` multiply-adds
        against an ``O(n³)`` QR iteration plus an ``Eigen::MatrixXd`` heap
        allocation per call), AND it is the last ``k`` whose CODEGEN is cheap:
        the C printers walk the expression TREE, not the shared DAG, and the
        tree grows by a factor ``n`` per step (``count_ops`` 1.9k / 23k / 239k
        at ``k = 1 / 2 / 3``, ``n = 12``), so ``k = 3`` already costs 424 s to
        emit at ``n = 12`` where ``k = 2`` costs 46 s — and the whole
        ``NumericsKernels.H`` took 69 s on the opaque-eigensolve baseline, so
        at ``k = 2`` this is a codegen SPEED-UP as well.  Raise
        ``spectral_bound_iterations`` on a small system, where ``nᵏ`` is
        affordable and the extra tightness is free.

        The limit of the chain is ``ρ(|A_n|)``, at 1.001 (median, wet) on the
        same sweep — what is left on the table past a few iterations is the
        sign cancellation ``|·|`` throws away, not the iteration.

        A *derived* operator (a cheap symbolic wrapping of
        ``quasilinear_matrix``), so exposed as a property — always consistent
        with the current matrix, exactly like :attr:`numerical_eigenvalues`.
        """
        n = self.n_equations
        qm = self.quasilinear_matrix
        normal = list(self.normal.values())
        M = [[sp.Abs(sum(qm[i, j, d] * normal[d] for d in range(self.n_dim)))
              for j in range(n)] for i in range(n)]
        return _perron_radius_bound(M, self.spectral_bound_iterations)

    def refresh_derived_operators(self, *, eigenvalues: bool = False):
        """Recompute ``quasilinear_matrix`` and the source jacobians from
        the (possibly just-mutated) primary operators, keeping the
        derived operators consistent.  After an in-place CoV / xreplace the
        channeled Model matrices are stale, so the jacobians are re-derived
        from the current ``source`` here.  ``eigenvalues=True`` also
        re-derives the spectrum — only request it when an operation
        genuinely changes the system's characteristic structure and the
        spectrum was not skipped."""
        self.quasilinear_matrix = self._compute_quasilinear_matrix()
        # POINT-IMPLICIT SOURCE: once ``source_treatment='linearized'`` has
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
        if getattr(self, "source_treatment", "explicit") == "linearized":
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
        """Widen ``J`` to ``(n_equations, len(aux_state))`` with zero columns.

        Point-implicit path only, where the aux jacobian is not re-derived (see
        :meth:`refresh_derived_operators`) but must still match the aux width
        the printers lower."""
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
        """Consistency check: verify ``mass_matrix`` is diagonal on
        every evolution row and zero on every algebraic row.

        - Evolution row ``i`` (with ``equation_to_state_index[i] = j``):
          must have ``M[i, k] = 0`` for every ``k != j``, and
          ``M[i, j] != 0``.
        - Algebraic row ``i``: ``M[i, :] = 0``.

        Raises ``ValueError`` with a precise location report on
        violation.  Run this **after** ``change_state_variables`` to
        confirm the variable choice produces a clean diagonal mass
        matrix — otherwise the variable map is wrong for the system
        at hand.

        Once this passes, ``InvertMassMatrix`` is a trivial per-row
        division by the diagonal entry; downstream consumers see
        ``M = I`` on evolution rows.
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
        """Apply a system-level operation in place.  Returns self."""
        op_name = name or getattr(operation, "name", None) or \
                  type(operation).__name__
        op_desc = description or getattr(operation, "description", op_name)
        operation(self)
        self.history.append({"name": op_name, "description": op_desc})
        return self

    def __getitem__(self, i: int) -> "SystemModelRow":
        return SystemModelRow(self, i)

    # ── from_model factory ────────────────────────────────────────────

    @classmethod
    def _from_derivation_model(cls, model, Q=None, Qaux=None,
                               canonical_source=None,
                               dae=False) -> "SystemModel":
        """Build a SystemModel from a declarative
        :class:`zoomy_core.model.derivation.model.Model` by structurally extracting
        the operators from its untagged residuals (see
        :func:`zoomy_core.model.derivation.system_extract.extract_system_operators`).
        ``Q`` (one applied field per evolution row) is required; ``Qaux`` is
        auto-populated when ``None``."""
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
        # Carry the WB reconstruction choice from the declarative Model so the
        # solver can pick the numerics (e.g. SME(equilibrium_reconstruction=...)).
        sm.equilibrium_reconstruction = getattr(
            canonical_source, "equilibrium_reconstruction", "none")
        return sm

    def has_active_source(self) -> bool:
        """``True`` if any source row is non-zero AT THE CONFIGURED PARAMETERS.

        The symbolic source carries free parameters (``e_x``, ``n``, …) that are
        non-zero as symbols but vanish at their configured values (``n=0``
        frictionless, ``e_x=0`` no tilt), so the raw symbolic form is not the
        question — substitute first.  Matching is by symbol NAME because the
        assumptions on a source symbol need not equal a freshly minted
        ``Symbol``, which makes ``subs``-by-object silently miss.
        """
        import sympy as _sp
        pv = dict(self.parameter_values or {})

        def _numify(e):
            e = _sp.sympify(e)
            return e.subs({s: pv[s.name] for s in e.free_symbols
                           if s.name in pv})
        try:
            return any(_sp.simplify(_numify(e)) != 0
                       for e in _sp.flatten(self.source))
        except Exception:                                # noqa: BLE001
            return True          # unknown ⇒ assume active, the safe branch

    def has_higher_moments(self) -> bool:
        """``True`` if the system carries moment rows beyond the depth-averaged
        one — i.e. anything a level-0 shallow-water closed form does not cover."""
        names = [str(s) for s in self.state]
        return len([n for n in names if n not in ("h", "b")]) - 1 >= 1

    def resolve_equilibrium_kernel(self, *, force_steady_ode: bool = False) -> str:
        """Which CONCRETE well-balanced kernel a backend must run.

        This is model knowledge and it is decided HERE, once, so a backend never
        re-derives it.  Backends previously reconstructed the moment level by
        counting non-``h``/``b`` state fields and sniffed the source with
        ``sympy.simplify`` at solver-setup time — the model layer telling the
        model's own story through the numerics layer's mouth.  A backend now
        reads the answer and branches on a plain Python string at trace time, so
        nothing about "level" survives into generated code.

        ``none`` / ``audusse`` pass through unchanged.  The Bernoulli family
        splits: the closed-form specific-energy reconstruction is only valid for
        a level-0, source-free system, so anything with friction or higher
        moments resolves to the fully well-balanced steady-ODE moving-equilibrium
        reconstruction (Pimentel-García).  ``force_steady_ode`` routes a
        frictionless level-0 system there too, which is what makes the WB
        head-to-head comparison possible (cid 157).
        """
        eqr = getattr(self, "equilibrium_reconstruction", "none") or "none"
        if eqr in ("none", "audusse"):
            return eqr
        if force_steady_ode or self.has_higher_moments() or self.has_active_source():
            return "steady_ode"
        return "bernoulli_closed_form"

    def is_vertical_dependent(self, symbol) -> bool:
        """``True`` if the state ``symbol`` is a vertical-dependent (3-D) field
        — e.g. ``ũ(t,x,ζ)`` — and ``False`` for a depth-collapsed field
        (``h(t,x)``, ``b(t,x)``).  Reads :attr:`state_function_map` against
        :attr:`vertical`; ``False`` when either is absent (a model with no
        vertical, e.g. SME/VAM after the moment projection integrates ζ out)."""
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
        """Build a SystemModel from a Model — the ONE entry point (REQ-143).

        A production model that carries a plain-string ``_system_model_kind``
        class attribute (SWE, SME, VAM, MLSWE, MLSME, MLVAM, Sigma3D, KESME,
        QRKESME) is dispatched to its per-model builder in
        :mod:`zoomy_core.systemmodel.model_builders` (lazy import to avoid an
        import cycle) — the builder holds the model-specific Q selection,
        hydrostatic-pressure tag, spectrum/positivity flags and BC resolution.

        Otherwise the LOW-LEVEL path (:meth:`_from_model_impl`) runs, dispatched
        on the *Model class*:

        * the **declarative** clean-redesign
          :class:`zoomy_core.model.derivation.model.Model` (whose equations are plain
          untagged sympy residuals) — the operators are extracted
          STRUCTURALLY from the residuals via
          :func:`zoomy_core.model.derivation.system_extract.extract_system_operators`.
          ``Q`` (and optionally ``Qaux``) are supplied HERE, at the transition,
          one ``Q`` entry per evolution row; the field coverage ``Q ∪ Qaux`` is
          validated (a ``ValueError`` names any uncovered field).
        * the **production** :class:`zoomy_core.model.basemodel.Model` — the
          operator-API methods (``model.flux()`` / ``model.source()`` / …,
          backed by solver-tag extraction) are read once and frozen.  ``Q`` /
          ``Qaux`` are ignored (the state vector is baked into the model).
        """
        if getattr(model, "_system_model_kind", None) is not None:
            from zoomy_core.systemmodel.model_builders import build_system_model
            return build_system_model(model)
        return cls._from_model_impl(
            model, Q=Q, Qaux=Qaux, canonical_source=canonical_source, dae=dae)

    @classmethod
    def _from_model_impl(cls, model, Q=None, Qaux=None,
                         canonical_source=None, dae=False) -> "SystemModel":
        """Low-level Model → SystemModel build (declarative model + Q, or a
        production model whose state vector is baked in).  Unchanged behaviour;
        see :meth:`from_model` for the dispatch."""
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
            """Coerce a model ZArray / sp.Matrix / 1-D ZArray into an
            sp.Matrix with the requested shape."""
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
            """Pull a rank-4 ``(n_eq, n_state, n_dim, n_dim)`` ZArray
            from ``model.<name>()`` if the model defines it; otherwise
            return an explicit zero tensor of that shape."""
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
        return sm

    # ── reconstruct_residuals ─────────────────────────────────────────

    def reconstruct_residuals(self):
        """Re-assemble the row-wise residuals ``M·∂_t Q + ∂_x F + ∂_x P
        + B·∂_x Q − S`` for every equation.  Returns a list of sympy
        expressions, indexed by row.

        Aux Symbols generated by ``expose_aux_atoms`` are back-
        substituted to their original ``Function`` / ``Derivative``
        atoms so the residuals display in their natural form for
        bit-for-bit fixture comparisons.

        Used in tests to compare the operator-form back against the
        original equation residuals (e.g. against Escalante eq (4)).
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
            if isinstance(sym, sp.Function):
                return sym
            args = [t] + list(coords)
            return sp.Function(str(sym), real=True)(*args)

        state_funcs = [_as_function(s) for s in self.state]
        sym_to_func = dict(zip(self.state, state_funcs))
        aux_reverse = self._aux_reverse_map()

        def _restore(expr):
            """First put back the original aux atoms (which still
            reference state as Symbols), then upgrade state Symbols
            to Functions so derivatives display naturally."""
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
        """Auto-scan the operator matrices and route every
        non-state symbolic input into ``aux_state``.

        For each unique:

        * **bare** ``Function`` atom whose name is **not** a state
          variable (e.g. topography ``b(t, x, y)``, externally
          prescribed forcing fields) — substitute with an aux Symbol
          of the same name;
        * ``Derivative`` atom — substitute with an aux Symbol named
          ``{target}_{axes}`` (e.g. ``b_x``, ``h_x``, ``b_x_x``);

        and populate :attr:`aux_registry` with one structured dict per
        new aux entry:

        .. code-block:: python

            {"kind":   "function" | "derivative",
             "name":   <aux symbol name>,
             "row":    <index in self.aux_state>,
             "atom":   <original sympy atom>,
             "aux_symbol": <Symbol>,
             # derivative entries also carry:
             "target_name":  <name of the field being differentiated>,
             "target_kind":  "state" | "function" | "unknown",
             "state_index":  <int>          # if target_kind == "state"
             "function_row": <int>          # if target_kind == "function"
             "multi_index":  <tuple>        # spatial-derivative orders
                                            # in (axis_0, axis_1, …)}

        ``multi_index`` follows the SAME convention as
        :meth:`LSQMesh.compute_derivatives` so a solver can compute
        every derivative-aux entry in one call.  Time derivatives are
        skipped (they live in the mass matrix).

        Idempotent — if ``aux_registry`` is already non-empty (or all
        atoms are state) this is a no-op.
        """
        import sympy as sp
        from itertools import product as _product

        if getattr(self, "aux_registry", None) is not None:
            return

        state_names = {str(s) for s in self.state}
        space_names = {str(s): d for d, s in enumerate(self.space)}
        n_dim = len(self.space)

        def _iter_entries(M):
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
        """Return ``{aux_Symbol: original_atom}`` for every aux entry so
        reconstructed residuals can be displayed in their original
        ``Derivative(…)`` form.

        Covers both ``aux_registry`` (the live, re-derived aux) and
        ``aux_input_registry`` (frozen inputs a split stage reads but does
        not re-derive — e.g. the predictor-produced derivatives a Chorin
        pressure block consumes).  Both appear as aux symbols in the
        operators, so both must reverse-map; otherwise a pruned input
        symbol (``q_0_x``) would survive the reconstruction and fail to
        cancel against its ``Derivative(q_0, x)`` reference."""
        registry = ((getattr(self, "aux_registry", None) or [])
                    + (getattr(self, "aux_input_registry", None) or []))
        return {entry["aux_symbol"]: entry["atom"] for entry in registry}

    # ── (legacy) expose_functions_as_aux / expose_derivatives_as_aux ─

    def expose_functions_as_aux(self, function_names=None):
        """Replace bare :class:`~sympy.Function` atoms with auxiliary
        Symbols of the same name added to :attr:`aux_state`.

        Companion to :meth:`expose_derivatives_as_aux`.  For each
        unique Function atom whose name is in ``function_names``
        (or every free Function if ``function_names is None``), a
        fresh aux Symbol named ``{function_name}`` is substituted in
        every operator matrix.

        Use for free Functions like topography ``b(t, x, y)`` so they
        live as proper aux state (visible in VTK output, fed from a
        per-cell callable at runtime) instead of being symbolically
        deleted via ``_zero_functions_by_name``.

        The mapping ``aux_function_map: Dict[Symbol, Function]`` is
        attached to ``self`` so solvers know which aux entries are
        free-function values (vs state derivatives, vs permanent
        Zstruct aux).
        """
        import sympy as sp
        from itertools import product as _product

        names = (set(function_names)
                 if function_names is not None else None)

        def _matches(a):
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
        """Replace ``Derivative(target, axes...)`` atoms in every
        operator matrix with auxiliary Symbols added to
        ``self.aux_state``.

        Each unique Derivative atom found in ``flux``,
        ``hydrostatic_pressure``, ``nonconservative_matrix``,
        ``source``, or ``mass_matrix`` is replaced with a fresh aux
        Symbol named ``{target}_{axes}``  (e.g. ``h_x``, ``U_0_x_x``,
        ``b_x``).  The Symbols are appended to ``self.aux_state`` and
        recorded in ``self.aux_derivative_map: Dict[Symbol,
        Derivative]`` so a solver can compute their per-cell values
        (state derivatives via mesh + LSQ reconstruction; parameter
        Function derivatives via analytical / tabulated evaluation).

        This makes spatial derivatives — at any order — first-class
        runtime inputs without changing the
        ``(Q, Qaux, p) → ndarray`` operator-API surface.

        Mutates ``self`` in place; no-op if the matrices contain no
        Derivative atoms.
        """
        from itertools import product

        def _matrix_entries(M):
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
        """Apply a state-variable change of variables in place.

        Parameters
        ----------
        new_state : list
            The new state vector (same length as ``self.state``).
            Entries that already appear in the old state stay; entries
            that replace an old state must appear in ``transform``.
        transform : dict
            Maps each OLD state entry that is being replaced to its
            expression in the NEW state variables.  Old states not
            mentioned are assumed to map identically to themselves.

        Updates **every state-dependent field** in place:

        * the state list itself (``state``);
        * primary operators — ``flux``, ``hydrostatic_pressure``,
          ``nonconservative_matrix``, ``source``, ``mass_matrix``;
        * explicit / IMEX twins — ``source_explicit``,
          ``diffusion_matrix``, ``diffusion_matrix_explicit``;
        * derived operators (refreshed) — ``quasilinear_matrix``,
          ``source_jacobian``;
        * eigenvalue spectrum (``eigenvalues`` — invariant under
          invertible CoV, so just xreplace);
        * MUSCL well-balanced pair —
          ``reconstruction_variables`` (xreplaced) and
          ``state_from_reconstruction`` (re-inverted against the new
          state slots);
        * per-cell state remap / explicit Chorin-corrector update
          (``update_variables``);
        * **symbolic BC kernels** — ``boundary_conditions``,
          ``aux_boundary_conditions``, ``boundary_gradients`` —
          definitions are xreplaced through ``transform`` and the
          ``args.variables`` Zstruct is rebuilt to bind the new state
          symbols positionally (otherwise ``state.u`` would still be
          named in the BC args after the CoV substitutes it away);
        * the aux registry — chain-rule propagation through
          ``∂_x state_old[i] = J[i, k] · ∂_x state_new[k]`` for every
          derivative aux entry that targeted a substituted state slot.

        Fields **not** transformed by this method:

        * ``initial_conditions`` and ``aux_initial_conditions`` — these
          are runtime callable objects (``InitialConditions`` subclasses
          like ``Constant``, ``RP``, ``UserFunction``) that produce
          per-cell numeric arrays.  They do not reference state Symbols
          symbolically; the user must re-supply them with values in the
          new state ordering after a CoV (e.g. ``Constant(constants=
          lambda n: [h, q_U0=h·U_0, …])`` instead of the old
          ``[h, U_0, …]``).

        Mechanics.  Let ``T_i(Q_new) = transform[Q_old[i]]`` (identity
        if not in ``transform``) and ``J[i, j] = ∂T_i/∂Q_new[j]``.  In
        operator form the system is invariant under the substitution
        ``Q_old → T(Q_new)`` together with the time-derivative
        identity ``∂_t Q_old = J · ∂_t Q_new`` and the spatial-
        derivative identity ``∂_d Q_old = J · ∂_d Q_new``:

          * ``F_new = F_old(T(Q_new))``,
            ``P_new = P_old(T(Q_new))``,
            ``S_new = S_old(T(Q_new))``.
          * ``B_new[i, k, d] = Σ_j B_old[i, j, d](T) · J[j, k]``.
          * ``M_new = M_old(T(Q_new)) · J``.
          * ``A_new[i, k, d, e] = Σ_j A_old[i, j, d, e](T) · J[j, k]``
            (state-axis multiplication on the diffusion tensor; same
            pattern as the NCP).

        This contract preserves the residual ``M ∂_t Q + ∂_x F + ∂_x P
        + B · ∂_x Q − ∇·(A:∇Q) − S`` under the change of variables.
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
        """Rebuild the symbolic BC kernels against a NEW state Zstruct.

        Called from :meth:`change_state_variables` after the primary
        operator matrices are updated.  Three kernels are rebuilt:
        ``boundary_conditions``, ``boundary_gradients`` (both from
        ``self._bc_source``) and ``aux_boundary_conditions``
        (from ``self._aux_bc_source``).

        The source ``BoundaryConditions`` instances are cached on the
        SystemModel by :meth:`from_model` (``_bc_source`` /
        ``_aux_bc_source``).  When a source is available we call
        ``source.get_boundary_condition_function(...)`` (and the
        gradient variant) with the new state Zstruct — the body
        re-emerges semantically:

          * Extrapolation slots → ``ZArray(new_state)``
            (identity on slot symbols, *not* the inverse transform of
            the old symbols);
          * Prescribed values / pressure rows / wall BCs that compute
            *physical* expressions of state get rebuilt against the
            new symbols by the BC subclasses themselves.

        Fallback: when the source isn't available (e.g. the SystemModel
        was constructed by hand without going through ``from_model``)
        we drop back to the legacy ``xreplace`` propagation with a
        ``stacklevel=3`` warning.  Hand-built BC bodies that were
        *physical* expressions of state survive that fallback
        correctly; extrapolation-shaped bodies do not, which is why
        rebuilding is the default.
        """
        bc_source = getattr(self, "_bc_source", None)
        aux_bc_source = getattr(self, "_aux_bc_source", None)

        def _reuse_arg(args, name):
            """Read a non-variables arg out of an existing BC kernel's
            args Zstruct (time, position, distance, parameters, normal,
            etc.) — these are stable under CoV."""
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
        """After ``change_state_variables`` substitutes state Symbols
        in the operator matrices, the aux-registry entries that
        represent ``∂_x state[i]`` (where ``state[i]`` is an OLD
        variable being substituted) carry stale meaning: the symbol
        in the operator was ``∂_x U_0`` but the runtime computes
        ``∂_x state[i] = ∂_x q_U0`` after the substitution.  Chain-rule
        propagation fixes this:

        ``∂_x T_i(Q_new) = Σ_j (∂T_i/∂Q_new[j]) · ∂_x Q_new[j]``

        For every aux entry whose target was an OLD state variable
        that got substituted, substitute the aux Symbol in the
        operators with the chain-rule expansion, and add new aux
        entries for ``∂_x Q_new[j]`` if not already present.
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
        """Convenience wrapper around :class:`RemoveNonDiagonalH`."""
        return RemoveNonDiagonalH()(self, **kwargs)

    # ── classification guard ──────────────────────────────────────────

    def assert_all_classified(self):
        """Raise if any equation row carries an operator term the structural
        extractor could not classify into a slot (a silent drop).

        The opt-in STRICT counterpart to the default build-time warning: a
        model that quietly loses a term from the balance law — e.g. a
        ``∂_x(A·∂_x b)`` bed-slope routed to an off-diagonal diffusion entry
        the runtime discards — fails here instead of running wrong.  A
        correctly-routed model (``unclassified_terms`` empty) returns ``self``
        so it can be chained."""
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
        """List the aux rows nothing computes — the model's OPEN terms.

        Every aux slot is one of three things, and this says which:

        * **computed** — an ``aux_registry`` row (a derivative from the LSQ
          walk, or a closure evaluated pointwise);
        * **external** — declared by the model with a definition it cannot
          evaluate pointwise (the stay-3D column integrals ``U``, ``ω``);
          a backend fills it;
        * **UNCLOSED** — neither.  Nothing writes the row, so it reads ZERO:
          silently absent physics, which is what
          :class:`UnclosedClosureError` refuses to hand back.

        Written for the reader who just hit that error and wants to know which
        closures the model is still missing.
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
        """Return a Description rendering the operator form.

        By DEFAULT the core balance-law operators are rendered — ``Q``,
        ``Q_aux``, the mass matrix ``M``, flux ``F``, hydrostatic
        pressure ``P``, diffusive flux ``A``, NCP ``B`` and the
        (implicit / explicit) source ``S``.  ``full=True`` additionally
        renders the rare / derived slots (quasilinear matrix, source
        Jacobian, eigenvalues, state-update, reconstruction maps,
        ``interpolate_to_3d``, …)."""
        return SystemModelDescription(self, full=full)


class SystemModelRow:
    """Row view of a :class:`SystemModel` — exposes row ``i`` of
    every stored matrix."""

    def __init__(self, parent: SystemModel, i: int):
        self._parent = parent
        self._i = i

    @property
    def flux(self):
        return self._parent.flux[self._i, :]

    @property
    def hydrostatic_pressure(self):
        return self._parent.hydrostatic_pressure[self._i, :]

    @property
    def source(self):
        return self._parent.source[self._i, 0]

    @property
    def mass_matrix_row(self):
        return self._parent.mass_matrix[self._i, :]

    @property
    def nonconservative_matrix_row(self):
        n = self._parent.n_equations
        d = self._parent.n_dim
        out = sp.MutableDenseNDimArray.zeros(n, d)
        for j in range(n):
            for k in range(d):
                out[j, k] = self._parent.nonconservative_matrix[self._i, j, k]
        return out


class SystemModelDescription:
    """Markdown / plaintext description of a SystemModel.

    Renders via ``_repr_markdown_`` in Jupyter; ``str(...)`` gives a
    plain-text fallback.
    """

    def __init__(self, sm: SystemModel, *, full: bool = False):
        self._sm = sm
        self._full = full

    # ── rendering helpers ──────────────────────────────────────────────
    @staticmethod
    def _as_matrix(obj):
        """Coerce a stored operator (ZArray / list / Matrix) to a 2-D
        ``sp.Matrix`` (a flat vector becomes a column).  ``None`` passes
        through so callers can render it as ``= 0``."""
        if obj is None:
            return None
        if isinstance(obj, sp.MatrixBase):
            return obj
        lst = obj.tolist() if hasattr(obj, "tolist") else list(obj)
        return sp.Matrix(lst)

    def _render(self, parts, label, mat):
        M = self._as_matrix(mat)
        if M is None or M.is_zero_matrix:
            parts.append(f"**{label} $= 0$**")
        else:
            parts.append(f"**{label}:**")
            parts.append(f"$$\n{sp.latex(M)}\n$$")

    def _render_ndim(self, parts, label, arr, rank):
        """Render a rank-3 NCP ``B[i, j, d]`` or rank-4 diffusion
        ``A[i, j, d, e]`` by slicing into ``n_eq × n_state`` slabs per
        direction (pair).  All-zero / ``None`` collapses to ``= 0``."""
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
        """The CORE operator set — always rendered: state, aux state, and
        every operator that is part of the canonical balance law."""
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
        """The rare / derived operators — only with ``full=True``."""
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
        """The classification-completeness report — only with ``full=True``.

        Lists every additive operator term the structural extractor matched no
        slot for (a silent drop from the balance law).  Clearly labelled and
        EMPTY-when-clean so its absence is never mistaken for "not checked"."""
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
        md = self._repr_markdown_()
        return md.replace("$$", "").replace("$", "").replace("**", "")

    def __repr__(self) -> str:
        return self.__str__()


# ── System-level operations ───────────────────────────────────────────

class RemoveNonDiagonalH:
    """System-level op: substitute the mass equation into every row
    whose mass-matrix ``h``-column entry is non-zero, pushing the
    ``M[i, h] · ∂_t h`` term into the NCP matrix ``B`` and the source
    ``S``.  Leaves ``M[i, k]`` untouched for ``k != h``.

    Motivation.  The Galerkin chain for free-surface models produces
    j ≥ 1 momentum rows that carry a state-dependent ``∂_t h``
    cross-term — e.g. ``M[xmom_j1, h] = (q_U1 − q_U0)/h`` in the
    conservative state of VAM(1, 2, 2).  The cross-term reflects the
    test function's time-dependence through ``ζ = (z − b)/h`` and is
    genuine, not a bug.  Substituting the mass equation
    ``∂_t h = −∂_x F[m, :] − ∂_x P[m, :] − Σ_k B[m, k, :] ∂_x Q_k +
    S[m]`` into each affected row pushes the term out of the
    time-derivative slot, where state-dependent ``M`` would otherwise
    silently mislead explicit solvers that assume ``M = I``.

    Scope.  This op clears only the ``h`` column of ``M``.  Other
    diagonal / off-diagonal entries are left intact — normalising
    state-dependent diagonal entries (e.g. ``M[xmom_j1, U_1] = h/3`` in
    primitive form) is the job of an extended
    :class:`InvertMassMatrix`, not this pass.

    Preconditions.  The mass row ``m`` (auto-detected, or supplied via
    ``mass_equation_index``) must satisfy ``M[m, h] = 1`` and
    ``M[m, k] = 0`` for ``k != h``.  Other rows may have arbitrary
    ``M[i, h]`` content.

    Post-conditions.  ``M[i, h_state_index] = 0`` for every previously
    affected row ``i``.  Per affected row, ``B`` and ``S`` are updated
    so the residual is exact modulo the mass equation:

    ``B[i, k, d] ← B[i, k, d] − M_old[i, h] · (∂F[m, d]/∂Q_k
                                                + ∂P[m, d]/∂Q_k
                                                + B[m, k, d])``

    ``S[i] ← S[i] − M_old[i, h] · S[m]``

    The mass row ``m`` itself is never modified.
    """

    name = "remove_non_diagonal_h"
    description = (
        "Substitute mass equation into rows with non-zero ∂_t h "
        "coupling; clear M[i, h] for those rows."
    )

    def __call__(self, sm: "SystemModel", *, h_state_index=None,
                 mass_equation_index=None):
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
        for i, s in enumerate(sm.state):
            if getattr(s, "name", None) == "h":
                return i
        raise ValueError(
            "RemoveNonDiagonalH: no state entry named 'h' found; "
            "pass h_state_index explicitly."
        )

    @staticmethod
    def _find_mass_row(sm: "SystemModel", h_col: int) -> int:
        """Find the row m with M[m, h_col] = 1 and M[m, k] = 0 elsewhere."""
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
    """System-level op: repack `g·h·∂_x η` from the chain's
    ``(P, B)`` form to standard SWE ``P = g·h²/2`` form, for use
    with Audusse hydrostatic-reconstruction Riemann solvers.

    **What this does**

    The chain's ``quadratic_form="escalante"`` output represents
    the depth-integrated gravity term ``g·h·∂_x η`` as

    ``P[i, d] = g·h·(b + h)``  and  ``B[i, h, d] = −g·(b + h)``,

    because ``∂_x(g·h·η) − g·η·h_x ≡ g·h·∂_x η`` (product rule).
    This packaging makes the bathymetry ``b`` appear *inside* the
    hydrostatic-pressure flux, which is correct as a continuous
    PDE — but Audusse's WB cancellation works most cleanly when
    ``P`` has the *standard SWE* form ``g·h²/2`` (no ``b``) and
    the bathymetry-on-momentum contribution ``−g·h·b_x`` is left
    to be supplied at runtime by the Audusse fluctuation
    ``(P_raw − P_star) @ n`` (see
    :meth:`PositiveRusanov.numerical_fluctuations`).

    This op rewrites for every row with that packaging:

    * ``P[i, d]``      →  ``P[i, d] − g·h·(b + h) + g·h²/2``
                       =  remainder + ``g·h²/2``
    * ``B[i, h, d]``   →  ``B[i, h, d] + g·(b + h)``
                       =  ``0`` (chain's escalante form had only
                          the gravity term here on the j=0 row)

    **Trade-off**

    The resulting symbolic SystemModel is *missing* the
    ``−g·h·b_x`` term — its residual no longer represents the
    full PDE on its own.  Audusse HR provides this term via the
    flux fluctuation at runtime; a non-HR Riemann solver (plain
    Rusanov) will be inconsistent on a varying bathymetry.  Only
    apply this op when the runtime is configured to use a
    ``PositiveRusanov``-family flux (which the
    :class:`ChorinSplitVAMSolver` does by default).

    **Detection**

    The op auto-detects every row ``i`` whose
    ``P[i, d]`` contains a ``g·h·(b + h)`` summand AND whose
    ``B[i, h_state_index, d]`` contains a matching ``−g·(b + h)``.
    These together signal the chain's gravity-on-η packaging.
    For VAM(1, 2, 2) this fires on the ``xmom_j0`` row only.
    """

    name = "hydrostatic_reconstruction"
    description = (
        "Convert chain's g·h·η in (P, B) to standard SWE P = g·h²/2 "
        "for Audusse HR compatibility (drops -g·h·b_x; HR fluctuation "
        "supplies it at runtime)."
    )

    def __call__(self, sm: "SystemModel", *, h_state_index=None):
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
    """System-level op: left-multiply each evolution row by ``1/M_ii``.

    Operates on a SystemModel whose mass matrix is **diagonal** on
    every evolution row (with arbitrary state-dependent diagonal
    entries) and zero on every algebraic row.  Use
    :meth:`SystemModel.assert_diagonal_mass_matrix` first to verify
    this precondition.

    Per evolution row ``i`` (with ``equation_to_state_index[i] = j``):
    divide ``flux[i, :]``, ``hydrostatic_pressure[i, :]``,
    ``source[i, 0]``, and ``nonconservative_matrix[i, :, :]`` by the
    diagonal entry ``M[i, j]``.  Set ``M[i, j] = 1``.

    Diagonal entries may be state- or aux-dependent — the division is
    symbolic and produces a ``1/M_ii`` factor in the resulting
    operators (which gets lambdified per-cell at runtime).
    """

    name = "invert_mass_matrix"
    description = (
        "Divide each evolution row by its diagonal mass entry to reach "
        "canonical ∂_t Q form."
    )

    def __call__(self, sm: SystemModel):
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
    """Permit a build to RETURN a SystemModel that still has unclosed terms.

    The open system is a legitimate object to want — it is the honest
    full-stress model, and printing it is how you find out *which* closures a
    model still needs.  What is not legitimate is getting it by accident, so
    the opt-in is explicit and greppable::

        with allow_unclosed():
            sm = SystemModel.from_model(VAM(level=1, dimension=2))
        print(sm.describe_unclosed())

    Inside the block the open terms are reported as a ``UserWarning`` and
    listed on :attr:`SystemModel._unclosed_closures`; outside it they raise
    :class:`UnclosedClosureError`.
    """
    global _ALLOW_UNCLOSED
    prev = _ALLOW_UNCLOSED
    _ALLOW_UNCLOSED = True
    try:
        yield
    finally:
        _ALLOW_UNCLOSED = prev


def _declared_aux_definitions(model) -> dict:
    """``{name: definition}`` for aux the MODEL declared itself.

    ``add_equation(name, Eq(sym, expr), group="aux")`` is how a model says "this
    is a genuine auxiliary field with a definition, filled by the backend" — the
    stay-3D column integrals ``U = ∫₀¹ u dζ`` and ``ω`` are the standing
    example: a running quadrature is not an algebraic ``update_aux_variables``
    row, so it has no registry entry and never will.

    Those names live in ``_aux_names`` on the DERIVATION model, which is reached
    as ``model.derivation`` — the outer user-facing Model does not carry them
    (the same seam that hides ``_closure_vars``), so checking only ``model``
    silently returns nothing and every declared aux looks unclosed.
    """
    for owner in (model, getattr(model, "derivation", None)):
        names = getattr(owner, "_aux_names", None)
        if not names:
            continue
        eqs = getattr(owner, "_equations", None) or {}
        return {str(nm): getattr(eqs.get(nm), "expr", None) for nm in names}
    return {}


def _assert_declared_closures_closed(sm, model, canonical_source=None):
    """Refuse to hand back a SystemModel with a DECLARED closure left unbound.

    ``Model.declare_closure`` marks a field as needing a closure; if no closure
    substitutes it during the derivation it is swept into ``aux_state`` by
    ``expose_aux_atoms`` — with NO ``update_aux_variables`` row.  Nothing then
    ever writes that slot, so the term reads ZERO: not an error, not garbage,
    silently absent physics.  That is how the ML-VAM interface tractions
    ``tau(0)`` / ``tau(1)`` vanish — they ARE derived (the by-parts step emits
    them), but ``apply_layer_stress_closures`` binds them only on the top / bed
    layer, so every internal interface is frictionless and nothing says so.

    An aux that is MEANT to be external is declared explicitly (``register_aux``
    gives it a formula, so it is computed).  This check fires only on the
    difference: declared a closure, landed in aux, and has no row.
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
    """A declared closure reached the SystemModel with nothing closing it.

    Carries the built :attr:`system_model` so the open terms can be inspected
    from the exception itself — ``err.describe()`` is the same listing
    :meth:`SystemModel.describe_unclosed` produces.
    """

    def __init__(self, unclosed, model_name="", system_model=None):
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
        """The open-term listing (see :meth:`SystemModel.describe_unclosed`)."""
        if self.system_model is not None:
            return self.system_model.describe_unclosed()
        return "\n".join([f"UNCLOSED TERMS — {self.model_name}"]
                         + [f"  {nm}" for nm in self.unclosed])
