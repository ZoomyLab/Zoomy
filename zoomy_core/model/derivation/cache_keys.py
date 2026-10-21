"""Content-addressed cache keys (SHA-256 of sympy srepr) for the derivation pipeline. Keys capture
the full symbolic state of an operation or model, so a too-coarse key can only cause a miss, never a
stale hit.
"""
from __future__ import annotations

import hashlib
import os

import sympy as sp

from zoomy_core.model.derivation.basisfunctions import Basisfunction
from zoomy_core.model.derivation.basis_cache import _basis_fingerprint


def derivation_cache_enabled() -> bool:
    """Return whether derivation caching is on (env ZOOMY_DERIVATION_CACHE, default off). This is
    the single switch for every cache tier (derivation_cache, model cache, SystemModel cache).
    """
    return os.environ.get(
        "ZOOMY_DERIVATION_CACHE", "0").strip() not in {"0", "false", "no"}

# Bump when the derivation pipeline changes in a way that invalidates every
# cached entry (new op semantics, changed extraction, etc.).
CACHE_VERSION = "v10"  # v10: VAM's default truncation is the order-consistent one (Fr, De =
#                              O(1): the w rows keep -nu d_x d_z u); a v9 VAM entry holds the
#                              hand rule (now VAMWeakViscosity) under the same spec key.
#                        v9: cid 209 — SystemModel gained the ``eigenvalues_cfl``
#                              field (the dt-only spectrum; gate_eigenvalues_dry
#                              writes it instead of overwriting ``eigenvalues``).
#                              The SM cache PICKLES SystemModel instances and its
#                              key hashes the MODEL's MRO + the builder — NOT
#                              system_model.py — so a v8 blob unpickles into an
#                              object missing the new field and every consumer
#                              that reads it raises AttributeError.  Bump to bust
#                              every stale v8 entry (incl. the shipped _prebuilt).
#                        v8: cid 170/171 — the shared momentum blueprints changed
#                              (equations.py: gravity carried as a VECTOR via
#                              gravity_components; ml_sme/ml_vam interface transfer
#                              now routed through the basis mass matrix).  These
#                              live in equations.py, a derive_model DEPENDENCY that
#                              the per-model spec/MRO hash does NOT cover, so bump
#                              to invalidate every stale v7 entry (incl. _prebuilt).
#                        v7: REQ-188 — the SystemModel cache key now hashes the
#                              FULL source of every class in the model's MRO
#                              (sm_cache.cache_key) instead of only derive_model,
#                              so a case-local operator override (e.g. RainSWE.source)
#                              invalidates the entry; bump busts stale v6 SM entries
#                              (incl. the shipped _prebuilt) keyed the old way.
#                        v6: REQ-176(4) correction — package_viscous now tags its
#                              diffusive flux with the ViscousDiffusion marker so the
#                              in-plane σ-metric h/b cross pieces route to the
#                              diffusion tensor (A[q_k←h] non-zero); changes the
#                              retain-viscous extraction, so bump invalidates v5.
#                        v5: REQ-176-firedrake — SWE base gained a wet/dry
#                              momentum cap (update_variables) not folded into the
#                              spec key (which hashes derive_model, not the model's
#                              update_variables method); bump invalidates stale
#                              identity-update_variables SWE entries on every tier.
#                        v4: REQ-176(4) viscous-retained moment families —
#                              SME/ML-SME/VAM/ML-VAM add_inplane_viscous +
#                              package_viscous + NewtonianInPlane closure (default
#                              path byte-identical, but derive_model changed).
#                        v3: SWE Manning source regularized (vel_eps, REQ-166) —
#                              plain-model operator edits are not captured by the
#                              spec/derive_model hash, so bump to invalidate.
#                        v2: parameter VALUES out of the spec key (REQ-163)

# Fields that never affect the math: display names, verbosity, back-references.
_COSMETIC_FIELDS = frozenset({"name", "description", "log_level", "_model"})

_PROBE = sp.Symbol("__cache_probe__")


def _canonical(value) -> str:
    """Return a deterministic structural string for a value in an operation state or model spec.
    Callables are keyed by their effect on a probe symbol, opaque objects by their __dict__.
    """
    # sympy objects → srepr (structural, deterministic)
    if isinstance(value, sp.Basic):
        return sp.srepr(value)
    # basis instance → reuse the matrix-cache fingerprint (name|level|bounds|…)
    if isinstance(value, Basisfunction):
        return f"Basis({_basis_fingerprint(value)})"
    # a basis CLASS (some ops store the class, not an instance)
    if isinstance(value, type) and issubclass(value, Basisfunction):
        return f"BasisCls({value.__name__})"
    # primitives
    if isinstance(value, (str, int, float, bool, type(None))):
        return repr(value)
    # objects that carry their own cache key (nested op / Closure / StateSpace) —
    # checked BEFORE the callable branch, so a *callable* Operation keys on its
    # config content, not just its __call__ effect/bytecode (which omits config
    # like ElderViscosity(friction=...)).
    if hasattr(value, "_key_content"):
        return f"{type(value).__name__}({value._key_content()})"
    # callables (lambdas / functions) → symbolic effect on a probe symbol
    if callable(value) and not isinstance(value, sp.Basic):
        try:
            return f"fn({sp.srepr(sp.sympify(value(_PROBE)))})"
        except Exception:
            # last resort: the bytecode (stable for an unchanged source lambda)
            code = getattr(value, "__code__", None)
            return f"fn-opaque({getattr(code, 'co_code', repr(value))!r})"
    # ordered containers → element-wise (order matters)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical(v) for v in value) + "]"
    # sets / frozensets → sorted (order does NOT matter)
    if isinstance(value, (set, frozenset)):
        return "{" + ",".join(sorted(_canonical(v) for v in value)) + "}"
    # dicts → sorted by canonical key
    if isinstance(value, dict):
        items = sorted((_canonical(k), _canonical(v)) for k, v in value.items())
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    # a class type that isn't a basis
    if isinstance(value, type):
        return f"type({value.__module__}.{value.__name__})"
    # unknown object → its own __dict__ (recurse), so we never silently drop state
    if hasattr(value, "__dict__"):
        return f"{type(value).__name__}(" + _state_repr(vars(value)) + ")"
    return repr(value)


def _state_repr(state: dict) -> str:
    """Canonical string of an object's __dict__ without cosmetic fields."""
    parts = []
    for key in sorted(state):
        if key in _COSMETIC_FIELDS:
            continue
        parts.append(f"{key}={_canonical(state[key])}")
    return ";".join(parts)


def op_key_content(op) -> str:
    """Return the result-affecting content string of one derivation Operation."""
    return f"{type(op).__name__}({_state_repr(vars(op))})"


def cache_key(obj) -> str:
    """Return the SHA-256 hex key of an Operation, Equation or any value understood by _canonical,
    prefixed with CACHE_VERSION.
    """
    if hasattr(obj, "_key_content"):
        content = obj._key_content()
    elif _looks_like_operation(obj):
        content = op_key_content(obj)
    elif hasattr(obj, "expr"):                       # Equation / Expression
        content = f"Equation({_canonical(obj.expr)})"
    else:
        content = _canonical(obj)
    return hashlib.sha256(f"{CACHE_VERSION}|{content}".encode()).hexdigest()


def _looks_like_operation(obj) -> bool:
    """Detect an Operation structurally, avoiding an import cycle with operations.py."""
    # Operation lives in model.operations; detect structurally to avoid an import
    # cycle (operations.py would otherwise import this module's consumers).
    return hasattr(obj, "_apply_leaf") or hasattr(obj, "apply_to_model") \
        or hasattr(obj, "apply_to_equation")


def argument_key(obj) -> str:
    """Return the content key of a decorated function's argument: spec key for a declarative model,
    cache_key for operations and equations, else canonical srepr.
    """
    if hasattr(obj, "param") and hasattr(getattr(obj, "param"), "values"):
        return model_spec_key(obj)
    if _looks_like_operation(obj) or hasattr(obj, "_key_content") \
            or hasattr(obj, "expr"):
        return cache_key(obj)
    return _canonical(obj)


# ---------------------------------------------------------------------------
# Model-spec + op-sequence keys (the L0 / L-final composite keys)
# ---------------------------------------------------------------------------

# Declarative-model (``param.Parameterized``) params that do NOT change the
# symbolic derivation OR the cached SystemModel's operators — excluded from the
# spec key.  ``name`` is cosmetic; the BC/IC params are runtime-attached and are
# keyed separately by the cache layer when it decides pre/post-attach caching.
_NON_DERIVATION_PARAMS = frozenset({
    "name",
    "boundary_conditions", "aux_boundary_conditions",
    "initial_conditions", "aux_initial_conditions",
    # REQ-163: the `parameters` constructor param records which SYMBOLS the
    # caller passed values for — neither the names nor the values are
    # structural (the class registers its parameter symbols in derive_model
    # regardless; values stay free symbols and go to the solver at runtime).
    "parameters",
})


def model_spec_key(model, *, include_bc_ic=False) -> str:
    """Return the key of a model's derivation identity: class plus constructor params, without
    cosmetic and runtime-attached ones.

    Args:
        include_bc_ic: Also fold the boundary/initial-condition params into the key.
    """
    cls = type(model)
    parts = [f"cls={cls.__module__}.{cls.__name__}"]

    if hasattr(model, "param") and hasattr(model.param, "values"):
        skip = set(_NON_DERIVATION_PARAMS)
        if include_bc_ic:
            skip = {"name"}
        for name in sorted(model.param.values().keys()):
            if name in skip:
                continue
            parts.append(f"{name}=" + _canonical(getattr(model, name)))
        # ``parameter_values`` — numeric VALUES are NOT part of the identity
        # (REQ-163, verified): parameters stay FREE SYMBOLS through the whole
        # derivation and lowering (SWE/SME operators are str-identical across
        # values); the numbers are handed to the solver at runtime.  Only the
        # parameter NAME SET shapes the symbol table, so key on the keys.
        # Exception: a model whose ``derive_model`` consumes a value in a
        # PYTHON branch (structural decision) lists that name in the class
        # allowlist ``_derivation_baked_params`` — those values are folded in.
        pv = getattr(model, "parameter_values", None)
        if pv is not None:
            if hasattr(pv, "items"):
                pv = {str(k): v for k, v in pv.items()}
            elif hasattr(pv, "keys"):
                pv = {str(k): getattr(pv, k) for k in pv.keys()}
            else:
                pv = {}
            baked = tuple(getattr(model, "_derivation_baked_params", ()) or ())
            if baked:
                parts.append("baked_params=" + _canonical(
                    {k: pv[k] for k in sorted(baked) if k in pv}))
        return hashlib.sha256(
            f"{CACHE_VERSION}|spec|{'|'.join(parts)}".encode()).hexdigest()

    # raw derivation Model
    parts.append("coords=" + _canonical(tuple(getattr(model, "_coords", ()) or ())))
    Q = getattr(model, "_Q", None)
    if isinstance(Q, dict):
        parts.append("Q=" + _canonical(dict(Q)))
    params = getattr(model, "parameters", None)
    if params is not None and hasattr(params, "keys"):
        parts.append("params=" + _canonical(sorted(params.keys())))
    for attr in ("coord_relations", "_field_decoration", "_sigma_from", "_vertical"):
        val = getattr(model, attr, None)
        if val:
            parts.append(f"{attr}=" + _canonical(val))
    return hashlib.sha256(
        f"{CACHE_VERSION}|spec|{'|'.join(parts)}".encode()).hexdigest()


def equations_key(model) -> str:
    """Return the key of a model's current equation residuals, order-independent over equation
    names.
    """
    eqs = getattr(model, "_equations", None) or {}
    items = sorted((name, _canonical(eq.expr)) for name, eq in eqs.items())
    body = ";".join(f"{n}={c}" for n, c in items)
    return hashlib.sha256(
        f"{CACHE_VERSION}|eqs|{body}".encode()).hexdigest()


def op_sequence_key(ops) -> str:
    """Return the key of an ordered sequence of operations."""
    body = ">".join(cache_key(op) for op in ops)
    return hashlib.sha256(
        f"{CACHE_VERSION}|ops|{body}".encode()).hexdigest()


def basis_key(basis) -> str:
    """Return the key of a concrete basis from its matrix-cache fingerprint."""
    return hashlib.sha256(
        f"{CACHE_VERSION}|basis|{_basis_fingerprint(basis)}".encode()
    ).hexdigest()


__all__ = [
    "CACHE_VERSION",
    "cache_key",
    "argument_key",
    "op_key_content",
    "model_spec_key",
    "equations_key",
    "op_sequence_key",
    "basis_key",
]
