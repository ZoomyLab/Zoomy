"""Modal separation of variables: turns a field u(t, x, zeta) into an unexpanded sum over a basis,
with the index bookkeeping ResolveBasis needs. Sits at the start of the moment/Galerkin derivation
on a derivation Model.
"""

from __future__ import annotations

import sympy as sp

from zoomy_core.model.operations import Operation


__all__ = [
    "ModalIndexRegistry",
    "reset_modal_indices",
    "separation_of_variables",
    "SeparationOfVariables",
    "TensorSeparationOfVariables",
    "build_modal_sum",
    "modal_bound",
    "modal_index",
    "test_index",
]


# Distinct mode letters, in order.  ``l`` is skipped — it is the Galerkin
# test index.  Indices are memoised by coefficient name, so re-running a
# notebook cell (or expanding the same field twice) reuses the same index
# instead of running off the alphabet.
_INDEX_NAMES = ["i", "j", "k", "m", "n", "o", "p", "q", "r", "s"]


def modal_index(name="i"):
    """Return an integer, non-negative trial summation-index symbol."""
    return sp.Symbol(name, integer=True, nonnegative=True)


def test_index(name="l"):
    """Return an integer, non-negative Galerkin test/moment index symbol; the name l is reserved for
    it.
    """
    return sp.Symbol(name, integer=True, nonnegative=True)


def modal_bound(name="N_u"):
    """Return a fresh abstract integer, non-negative truncation-bound symbol."""
    return sp.Symbol(name, integer=True, nonnegative=True)


def _coeff_key(coeff):
    """Return the coefficient family name of a name, head class or applied coefficient."""
    if isinstance(coeff, str):
        return coeff
    if isinstance(coeff, type):
        return coeff.__name__
    func = getattr(coeff, "func", None)
    if func is not None:
        return func.__name__
    return str(coeff)


# ── ModalIndexRegistry ────────────────────────────────────────────────────


class ModalIndexRegistry:
    """Per-model registry of coeff_name -> (index, basis) plus a field_head -> coeff_name map. Every
    field gets a distinct summation index, otherwise products of two sums would silently lose their
    cross terms.
    """

    def __init__(self):
        """Create an empty registry."""
        self._entries = {}          # coeff_name → {"index": Symbol, "basis": …}
        self._field_to_coeff = {}   # field head class → coeff_name
        self._order = []            # mode letters handed out, in order
        self._registered = {}       # user-registered test/moment index → Symbol

    def _mint_index(self):
        """Return the next unused index symbol (i, j, k, ...; l is skipped)."""
        nm = next((n for n in _INDEX_NAMES if n not in self._order), None)
        if nm is None:
            nm = f"i_{len(self._order)}"
        self._order.append(nm)
        return sp.Symbol(nm, integer=True, nonnegative=True)

    def assign(self, coeff_name, *, basis=None, field_head=None, index=None):
        """Register or update a coefficient entry and return its (memoised) index."""
        if coeff_name in self._entries:
            if basis is not None:
                self._entries[coeff_name]["basis"] = basis
        else:
            self._entries[coeff_name] = {
                "index": index if index is not None else self._mint_index(),
                "basis": basis,
            }
        if field_head is not None:
            self._field_to_coeff[field_head] = coeff_name
        return self._entries[coeff_name]["index"]

    def fresh(self, key=None):
        """Return a new index, or the memoised one for the given coefficient name."""
        if key is None:
            return self._mint_index()
        return self.assign(key)

    def coeff_name_for(self, target):
        """Resolve a name, coefficient or field head (bare or applied) to its coefficient name, or
        None.
        """
        if isinstance(target, str):
            return target if target in self._entries else None
        if isinstance(target, type):                    # head class
            if target in self._field_to_coeff:
                return self._field_to_coeff[target]
            nm = target.__name__
            return nm if nm in self._entries else None
        if getattr(target, "args", None) is not None:   # applied instance
            head = target.func
            if head in self._field_to_coeff:
                return self._field_to_coeff[head]
            nm = head.__name__
            return nm if nm in self._entries else None
        return None

    def index_for(self, target):
        """Return the registered summation index of a field or coefficient, or None."""
        name = self.coeff_name_for(target)
        return self._entries[name]["index"] if name else None

    def basis_for(self, target):
        """Return the registered basis of a field or coefficient, or None."""
        name = self.coeff_name_for(target)
        return self._entries[name]["basis"] if name else None

    def register(self, name):
        """Mint a user test/moment index, raising if the name collides with an existing index."""
        if name in self._registered:
            return self._registered[name]
        if name in self._order:
            raise ValueError(
                f"index {name!r} is already an auto-assigned modal (trial) "
                f"index on this model; pick another test-index name.")
        sym = sp.Symbol(name, integer=True, nonnegative=True)
        self._registered[name] = sym
        return sym

    def reset(self):
        """Clear all entries and indices."""
        self._entries.clear()
        self._field_to_coeff.clear()
        self._order.clear()
        self._registered.clear()


def reset_modal_indices(model):
    """Clear the model's modal-index registry so index assignment is deterministic."""
    model._modal_registry.reset()


# ── build the unexpanded modal sum ─────────────────────────────────────────


def build_modal_sum(field, coeff, basis, order, index):
    """Build the unexpanded modal sum `Sum_i coeff(i, *coords) * phi(i, zeta)` for a field.

    Args:
        field: Applied field; its last argument is the separation coordinate zeta.
        coeff: Coefficient head, or applied coefficient carrying its own coords.
        basis: Basis supplying the opaque phi head, or a bare phi head.
        order: Truncation bound N.
        index: Summation index.
    """
    zeta = field.args[-1]
    if isinstance(coeff, type):
        coeff_head, coords = coeff, tuple(field.args[:-1])
    elif getattr(coeff, "args", None):
        coeff_head, coords = coeff.func, tuple(coeff.args)
    else:
        coeff_head, coords = getattr(coeff, "func", coeff), tuple(field.args[:-1])
    phi = getattr(basis, "phi", None) or getattr(basis, "phi_fn", basis)
    return sp.Sum(coeff_head(index, *coords) * phi(index, zeta),
                  (index, 0, order))


# ── separation_of_variables (the Model op) ─────────────────────────────────


class SeparationOfVariables(Operation):
    """Model operation that replaces a field's decorated head by its unexpanded modal sum and swaps
    the unknown family in Q. Built by separation_of_variables; it rewrites every equation and
    registers the index and basis in model._modal_registry.

    References:
        Kowalski, Torrilhon 2019, https://doi.org/10.4208/cicp.OA-2017-0263
    """

    whole_model_op = True

    def __init__(self, field, coeff, basis, order, *, index=None,
                 name="separation_of_variables"):
        """Store the field, coefficient, basis, truncation order and optional index."""
        self._field = field
        self._coeff = coeff
        self._basis = basis
        self._order = order
        self._index = index
        super().__init__(
            name=name,
            description=(f"separation of variables {field} → "
                         f"Σ {_coeff_key(coeff)}·φ"),
        )

    def apply_to_model(self, model):
        """Expand every application of the field head into a modal sum, each with a distinct dummy
        index, and redeclare the unknown as the coefficient family. Boundary evaluations u(t, x,
        0/1) expand too, so the ansatz may follow a field-level boundary condition.
        """
        # The field's DECORATED head (post-PDE).  Read it off the model's
        # ``_field_decoration`` map; if the model never ran a PDE transform
        # the original head is used.
        deco = model._field_decoration or {}
        head = deco.get(self._field.func, self._field.func)

        coeff_name = _coeff_key(self._coeff)
        idx = model._modal_registry.assign(
            coeff_name, basis=self._basis, field_head=head, index=self._index)
        # Register the ORIGINAL (pre-PDE) field head too, so a later
        # ``model.modal_index(u)`` keyed on the physical-z field resolves to
        # the same index as the decorated head.
        if self._field.func is not head:
            model._modal_registry.assign(
                coeff_name, field_head=self._field.func)

        # The coefficient family head + its (horizontal) coords.
        coeff_head = (self._coeff if isinstance(self._coeff, type)
                      else getattr(self._coeff, "func", self._coeff))
        coords = (tuple(self._coeff.args) if getattr(self._coeff, "args", None)
                  else tuple(self._field.args[:-1]))
        phi = (getattr(self._basis, "phi", None)
               or getattr(self._basis, "phi_fn", self._basis))
        order = self._order

        # Replace EVERY application of the (decorated) field head — at ANY
        # argument list — with the modal sum ``Σ_i a_i(coords)·φ(i, <basis arg>)``.
        # Using a HEAD-level ``replace`` (not ``xreplace`` on a single ``ũ(t,x,ζ)``
        # target) is what lets the ansatz be inserted AFTER a field-level
        # ``KinematicBC``: the BC substitutes boundary evaluations ``ũ(t,x,0)`` /
        # ``ũ(t,x,1)`` whose last arg is the concrete boundary value, and those
        # must expand to ``Σ_i a_i·φ(i, 0/1)`` too.  ``call_args[-1]`` is the
        # separation (basis) coordinate of each application — ζ in the bulk, 0/1
        # at a boundary.
        #
        # FRESH DUMMY PER FIELD OCCURRENCE.  ``Basic.replace`` fires the callback
        # once per DISTINCT subexpression, so two structurally different
        # applications of the same head — e.g. ``u(t,x,ζ)`` in the bulk and
        # ``u(t,x,0)`` at a bed BC, or a nested ``u·u`` whose factors differ by
        # argument — used to collapse onto ONE shared summation dummy ``idx``.
        # ``sympy`` then keeps the two ``Sum``s sharing that index and SILENTLY
        # DROPS the cross terms (``i ≠ j``) when they multiply.  We hand the FIRST
        # occurrence the canonical registry index (so single-field expansions and
        # the ``Q`` redeclaration are byte-identical to before) and mint a fresh
        # DISTINCT index for every subsequent occurrence, drawn from the same
        # ``i, j, k, …`` pool.  A ``u·u`` that ``sympy`` has already folded to a
        # ``Pow`` (one base node) still relies on :class:`ExpandSums` to relabel —
        # this fix removes the collision for the DISTINCT-node cross terms that
        # ``ExpandSums`` never sees (they are handed to it pre-multiplied).
        def _make_expander():
            """Return a per-equation expander; the first occurrence keeps the registry index and
            later ones get fresh distinct indices.
            """
            used = {idx}
            first = {"done": False}

            def _next_index():
                """Return the next unused summation index."""
                if not first["done"]:
                    first["done"] = True
                    return idx
                for nm in _INDEX_NAMES:
                    cand = sp.Symbol(nm, integer=True, nonnegative=True)
                    if cand not in used:
                        used.add(cand)
                        return cand
                n = 1
                while True:
                    cand = sp.Symbol(f"{idx.name}_{n}", integer=True,
                                     nonnegative=True)
                    if cand not in used:
                        used.add(cand)
                        return cand
                    n += 1

            def _expand(*call_args):
                """Return the modal sum for one application of the field head."""
                j = _next_index()
                return sp.Sum(coeff_head(j, *coords) * phi(j, call_args[-1]),
                              (j, 0, order))
            return _expand

        for eq in model._equations.values():
            _expand = _make_expander()
            eq.expr = eq.expr.replace(head, _expand)
            # An ORIENTED relation (e.g. ``ω = …`` after ``SolveFor``) keeps a
            # separate ``_as_relation`` dict; rewrite its lhs/rhs too so the
            # stored relation stays consistent with ``expr`` and a later
            # ``apply(eq)`` substitutes the ansatz-expanded form, not the stale
            # field head.
            rel = getattr(eq, "_as_relation", None)
            if rel:
                eq._as_relation = {k.replace(head, _expand): v.replace(head, _expand)
                                   for k, v in rel.items()}

        # Swap the unknown family: the field's head leaves Q, the coeff
        # family (``a(i, *coords)``) enters.
        coeff_applied = coeff_head(idx, *coords)
        # A genuine family rename (u → a): the Q key follows the new coeff family.
        model.redeclare_unknown(head, coeff_applied, rename_key=True)
        model._refresh_unknowns()
        return model


class TensorSeparationOfVariables(Operation):
    """Model operation for a concretely expanded tensor-product Galerkin ansatz over a vertical and
    a lateral separation coordinate. Heads are matched by name (the field and its sigma-decorated
    sibling) and the sum is expanded with integer orders, so there are no shared dummies.

    Args:
        field: Original field application, e.g. u(t, x, y, z).
        coeff_head: Tensor coefficient family, applied as `q(j, i, *kept_args)` with lateral j
            outer, vertical i inner.
        pos_z, basis_z, order_z: Position, basis and truncation order of the vertical coordinate.
        pos_y, basis_y, order_y: Same for the lateral coordinate.
        scale_y: Lateral slot carries y = scale_y * eta, so the coefficient's eta is arg / scale_y.
    """

    whole_model_op = True

    def __init__(self, field, coeff_head, *, pos_z, basis_z, order_z,
                 pos_y, basis_y, order_y, scale_y=1, name="tensor_sov"):
        """Store the field, coefficient family and per-direction positions, bases and orders."""
        self._field, self._coeff = field, coeff_head
        self._pz, self._bz, self._oz = pos_z, basis_z, order_z
        self._py, self._by, self._oy = pos_y, basis_y, order_y
        self._sy = scale_y
        super().__init__(
            name=name,
            description=(f"tensor separation of variables {field} → "
                        f"Σ_ji {_coeff_key(coeff_head)}·φ_i·ψ_j"))

    def apply_to_model(self, model):
        """Replace the field head and its decorated sibling by the tensor-product sum in every
        equation.
        """
        # Match heads BY NAME — field + its σ-decorated ``\tilde{field}`` sibling
        # (decorated heads are custom Function subclasses; identity is unreliable).
        names = {self._field.func.__name__,
                 "\\tilde{%s}" % self._field.func.__name__}
        heads = {f.func for eq in model._equations.values()
                 for f in eq.expr.atoms(sp.Function) if f.func.__name__ in names}
        pz, py = self._pz, self._py
        phi = getattr(self._bz, "phi", None) or getattr(self._bz, "phi_fn",
                                                         self._bz)
        psi = getattr(self._by, "phi", None) or getattr(self._by, "phi_fn",
                                                         self._by)

        def _expand(*call_args):
            """Return the double sum for one application of the field head."""
            kept = tuple(a for p, a in enumerate(call_args) if p not in (pz, py))
            eta_val = sp.cancel(call_args[py] / self._sy)   # W·η→η, 0→0, W→1
            return sp.Add(*[
                self._coeff(j, i, *kept)
                * phi(i, call_args[pz]) * psi(j, eta_val)
                for i in range(self._oz + 1) for j in range(self._oy + 1)])

        for eq in model._equations.values():
            for head in heads:
                eq.expr = eq.expr.replace(head, _expand)
                rel = getattr(eq, "_as_relation", None)
                if rel:
                    eq._as_relation = {k.replace(head, _expand):
                                       v.replace(head, _expand)
                                       for k, v in rel.items()}
        return model


def separation_of_variables(field, coeff, basis, order, *, index=None):
    """Return a SeparationOfVariables operation approximating `field` by
    `Sum_i coeff(i, *coords) * phi(i, zeta)` with a distinct registry index.

    Args:
        field: Original field application, e.g. u(t, x, z).
        coeff: Coefficient head or applied coefficient, e.g. a(t, x).
        basis: Basis supplying the opaque phi.
        order: Truncation bound N.
        index: Override of the summation index.

    Returns:
        Operation for model.apply(...).

    References:
        Kowalski, Torrilhon 2019, https://doi.org/10.4208/cicp.OA-2017-0263
    """
    return SeparationOfVariables(field, coeff, basis, order, index=index)
