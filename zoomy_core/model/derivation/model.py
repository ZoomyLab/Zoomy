"""The derivation ``Model``: ordered coordinates, parameters, scalar/vector equations, the unknown
registry Q/Qaux and the operation history. It is the entry point of the derivation chain Model ->
SystemModel (``SystemModel.from_model``).
"""

from __future__ import annotations

import sympy as sp

from zoomy_core import coords as _coords
from zoomy_core.misc.misc import Zstruct
from zoomy_core.misc.description import Description
from zoomy_core.model.equation import Equation
from zoomy_core.model.operations import Operation


__all__ = ["Model", "VectorEquation", "MomentFamily", "resolve_modes",
           "ResolveModes"]


# Component index → attribute name, by vector length.
#   length 2 → (x, z)   [vertical-plane models: horizontal + vertical]
#   length 3 → (x, y, z)
_COMPONENT_NAMES = {
    1: ("x",),
    2: ("x", "z"),
    3: ("x", "y", "z"),
}


class VectorEquation:
    """Vector equation: ordered component Equations with named-axis (``.x/.y/.z``) and index access.
    Components are also flattened into the model's ``_equations`` as ``<name>_<axis>``.
    """

    def __init__(self, name, shape, components):
        """Wrap the component equations of a vector row of length 1, 2 or 3."""
        self.name = name
        self.shape = tuple(shape)
        n = len(components)
        names = _COMPONENT_NAMES.get(n)
        if names is None:
            raise ValueError(
                f"VectorEquation {name!r}: no axis names for length {n} "
                f"(supported: {sorted(_COMPONENT_NAMES)}).")
        self._axis_names = names
        self._components = list(components)  # list[Equation]

    # ── named-axis access ────────────────────────────────────────────
    def __getattr__(self, item):
        """Named-axis access ``.x`` / ``.y`` / ``.z``."""
        # Only consulted when normal attribute lookup fails.
        names = self.__dict__.get("_axis_names", ())
        if item in names:
            return self._components[names.index(item)]
        raise AttributeError(item)

    # ── integer / term indexing ──────────────────────────────────────
    def __getitem__(self, i):
        """Component by index or axis name, or ``(component, term)``."""
        if isinstance(i, tuple) and len(i) == 2:
            comp, term = i
            if isinstance(comp, str):
                comp = self._axis_names.index(comp)
            return self._components[comp][term]
        return self._components[i]

    def __iter__(self):
        """Iterate over the component equations."""
        return iter(self._components)

    def __len__(self):
        """Number of components."""
        return len(self._components)

    @property
    def components(self):
        """List of component equations."""
        return list(self._components)

    def replace_axis(self, axis, replacement):
        """Swap the component at ``axis`` (name or index), e.g. for a MomentFamily after a
        moment-axis bump.
        """
        i = (self._axis_names.index(axis) if isinstance(axis, str) else axis)
        self._components[i] = replacement
        return replacement

    def apply(self, *args, **kwargs):
        """Broadcast an operation across every component."""
        for comp in self._components:
            comp.apply(*args, **kwargs)
        return self

    def describe(self, strip_args=False):
        """Markdown description of the components."""
        parts = [f"**{self.name}** (vector, {len(self)} components)"]
        for comp in self._components:
            parts.append(str(comp.describe(strip_args=strip_args))
                         if isinstance(comp, MomentFamily)
                         else comp.describe_line(strip_args=strip_args,
                                                 bullet=True))
        return Description("\n".join(parts))

    def __repr__(self):
        """Short repr with name, shape and component count."""
        return (f"VectorEquation({self.name!r}, shape={self.shape}, "
                f"{len(self)} components)")


class MomentFamily:
    """Indexed family of moment rows produced by ``resolve_modes`` (``model.mass[l]``,
    ``model.momentum.x[l]``). Rows are also flattened into the model's ``_equations`` as
    ``<base>_<mode>``.
    """

    def __init__(self, name, modes, components, index=None):
        """Store the family name, mode list, per-mode rows and the symbolic index."""
        self.name = name
        self.modes = list(modes)
        self.index = index
        self._components = list(components)   # list[Equation], in mode order

    # ── integer-mode indexing ────────────────────────────────────────
    def __getitem__(self, i):
        """Moment row by position."""
        return self._components[i]

    def __iter__(self):
        """Iterate over the moment rows."""
        return iter(self._components)

    def __len__(self):
        """Number of moments."""
        return len(self._components)

    @property
    def components(self):
        """List of moment rows."""
        return list(self._components)

    # ── ``.eq0 / .eq1 / …`` and ``.eq_l`` aliases ────────────────────
    def __getattr__(self, item):
        """Aliases ``.eq0``, ``.eq1``, ... for the mode rows and ``.eq_l`` for the whole family."""
        # Consulted only on normal-lookup failure.  ``eqK`` → mode-K row.
        comps = self.__dict__.get("_components", [])
        if item.startswith("eq"):
            tail = item[2:]
            if tail.isdigit():
                return comps[int(tail)]
            # ``eq_l`` — the symbolic-index alias for the whole family.
            if tail in ("_l", "_" + str(self.__dict__.get("index", ""))):
                return self
        raise AttributeError(item)

    def apply(self, *args, **kwargs):
        """Broadcast an operation across every moment row."""
        for comp in self._components:
            comp.apply(*args, **kwargs)
        return self

    def describe(self, strip_args=False):
        """Markdown description of the moment rows."""
        parts = [f"**{self.name}** (moment family, {len(self)} modes)"]
        for comp in self._components:
            parts.append(comp.describe_line(strip_args=strip_args, bullet=True))
        return Description("\n".join(parts))

    def __repr__(self):
        """Short repr with name, mode count and index."""
        return (f"MomentFamily({self.name!r}, {len(self)} modes, "
                f"index={self.index})")


class FieldHandle:
    """Handle on a model field (typically the sigma-mapped applied form) exposing ``.head``,
    ``.expr``, ``.at(value)`` and call syntax. Sympifiable; handles live on ``Model.functions``
    only.
    """

    def __init__(self, applied):
        """Wrap the applied field."""
        self._applied = applied

    @property
    def head(self):
        """Function class of the field."""
        return self._applied.func

    @property
    def expr(self):
        """Applied field expression."""
        return self._applied

    def at(self, value):
        """Field evaluated at ``value`` in its last (vertical) argument."""
        return self._applied.subs(self._applied.args[-1], value)

    def __call__(self, *args):
        """Re-apply the head to new arguments."""
        return self._applied.func(*args)

    def _sympy_(self):
        """Sympy conversion hook."""
        return self._applied

    def __repr__(self):
        """Short repr."""
        return f"FieldHandle({self._applied})"


class Model:
    """Computational-graph spine of a symbolic derivation: coordinates, parameters, equations,
    unknowns (Q declared, Qaux derived) and history. Operations are applied with
    ``model.apply(op)``; the result is lifted to a SystemModel.
    """

    def __init__(self, coords=(_coords.t, _coords.x, _coords.z),
                 parameters=None, name="model"):
        """Create an empty model over ``coords`` (default ``t, x, z``) with positive-Symbol
        ``parameters`` given as ``{name: default}``.
        """
        self.name = name
        self._coords = tuple(coords)

        # Parameters → Zstruct of positive Symbols; remember the numeric
        # defaults the caller passed for description / later substitution.
        parameters = parameters or {}
        params = Zstruct()
        self.parameter_values = Zstruct()
        for key in parameters:
            setattr(params, key, sp.Symbol(key, positive=True))
            setattr(self.parameter_values, key, parameters[key])
        self.parameters = params

        # Ordered equation registry.  Vector rows store BOTH the proxy (under
        # the vector name) and the flattened component rows (under
        # ``<name>_<axis>``); ``_equations`` only holds scalar Equation rows.
        self._equations: dict[str, Equation] = {}
        self._vector_equations: dict[str, VectorEquation] = {}
        self._equation_shapes: dict[str, tuple] = {}
        # Moment families produced by ``resolve_modes`` (a Galerkin moment-axis
        # SHAPE BUMP).  Keyed by the proxy name a user reaches for: a scalar
        # bump registers ``{"mass": MomentFamily}``; a vector-component bump
        # registers ``{"momentum_x": MomentFamily}`` and the component proxy on
        # the VectorEquation is replaced in place so ``model.momentum.x[l]``
        # resolves to the family.
        self._moment_families: dict[str, MomentFamily] = {}

        # Stored ASSUMPTIONS / Relations (kinematic BCs, hydrostatic, …).  An
        # Assumption holds a ``subs_map`` (not a single ``.expr``), so it lives
        # in its own registry rather than ``_equations``.  ``PDETransformation``
        # rewrites their subs_maps alongside the equations, and ``model.<name>``
        # / ``model.apply(model.<name>)`` reach + apply them.
        self._assumptions: dict = {}
        # Names of equation rows flagged ``group="aux"`` — auxiliary relations
        # (w-reconstruction, ω-definition, …) that are derived/used but are NOT
        # part of the final PDE system handed to a SystemModel.  Oriented
        # relations in ``_assumptions`` (KinematicBCs) are auxiliary too.
        self._aux_names: set = set()
        # Field HEADS explicitly demoted to auxiliary via :meth:`move_to_aux`
        # (e.g. the bed ``b`` — given data that appears in the primary momentum
        # slope ``g h ∂_x b`` yet has no evolution equation).  A head listed here
        # is kept OUT of ``Q`` even when it is present in a primary equation, and
        # surfaces in ``Qaux`` instead.
        self._aux_fields: set = set()

        # Function groups — named slots that hold explicit DEFINITIONS of
        # derived model operators (the vertical reconstruction → ``interpolate``,
        # its Galerkin inverse → ``project``, derived BCs, …) rather than PDE
        # residuals or auxiliary fields.  ``{slot_name: {component_index: rhs}}``;
        # populated by :meth:`register_group` and parsed straight into the
        # matching ``SystemModel`` slot by ``from_model``.
        self._function_groups: dict = {}

        # Derivation history + per-model modal-index registry.  The registry
        # hands out a distinct summation index per coefficient family and is
        # read back by ``separation_of_variables`` / ``ResolveBasis``.
        self.history: list[dict] = []
        from zoomy_core.model.derivation.modal import ModalIndexRegistry
        self._modal_registry = ModalIndexRegistry()

        # Coordinate-transformation state.  ``PDETransformation`` records the
        # σ-map here so later steps (``kinematic_bc``, depth projection) can
        # read the decorated-head map, the source vertical coord, the mapped
        # reference coord, and the explicit ``z = b + h·ζ`` relation.
        #   _field_decoration: {original Function head → decorated Function head}
        #   _sigma_from:       the physical vertical coord that was mapped (z)
        #   _vertical:         the mapped reference coord (ζ) — None pre-map
        #   coord_relations:   {z: b + h·ζ}
        self._field_decoration: dict = {}
        self._sigma_from = None
        self._vertical = None
        self.coord_relations: dict = {}

        # Declared unknowns (Q).  Maps field-head-name → applied field Function.
        self._Q: dict[str, sp.Expr] = {}
        # Heads that have appeared in some equation at least once.  A declared
        # unknown is auto-dropped from Q only once it has *appeared* and then
        # *vanished* — declaring an unknown before its equation exists (the
        # SME cell sets ``Q = [h, u, w, p]`` before the balances are added) is
        # legitimate and must not be eaten by the refresh.
        self._ever_seen_heads: set = set()

    # ── parameters helpers ───────────────────────────────────────────
    def parameter(self, name, default=0.0):
        """Return the positive Symbol for parameter ``name``, creating it with ``default`` if it is
        not yet declared.
        """
        if name in self.parameters.keys():
            return getattr(self.parameters, name)
        sym = sp.Symbol(name, positive=True)
        setattr(self.parameters, name, sym)
        setattr(self.parameter_values, name, default)
        return sym

    @property
    def coords(self):
        """Ordered tuple of coordinates."""
        return self._coords

    @property
    def vertical(self):
        """Vertical coordinate: the mapped ``zeta`` after a PDETransformation, else the last
        coordinate.
        """
        if self._vertical is not None:
            return self._vertical
        return self._coords[-1] if self._coords else None

    @property
    def horizontal(self):
        """Horizontal coordinates: all but time and vertical."""
        vert = self.vertical
        return tuple(c for c in self._coords[1:] if c != vert)

    # ── history ──────────────────────────────────────────────────────
    def _history(self, op_name, target, level="major", description=None,
                 log_level=1):
        """Append an operation record to the history."""
        self.history.append({
            "op": op_name,
            "target": target,
            "level": level,
            "log_level": log_level,
            "description": description,
        })

    # ── declarative variable registration (equation-blueprint API) ──────
    def declare_state(self, *fields):
        """Add fields to ``Q`` as primary (state) unknowns; used by equation blueprints."""
        for f in fields:
            self._Q[self._field_name(f)] = f
        self._refresh_unknowns()
        return fields[0] if len(fields) == 1 else fields

    def declare_closure(self, *fields):
        """Record fields as closure variables (present in the equations, routed to ``Qaux`` unless
        substituted).
        """
        if not hasattr(self, "_closure_vars"):
            self._closure_vars = set()
        for f in fields:
            self._closure_vars.add(self._field_name(f))
        return fields[0] if len(fields) == 1 else fields

    # ── equation registration ────────────────────────────────────────
    def add_equation(self, name, shape_or_expr=None, expr=None, *, group="model",
                 _refresh=True):
        """Register a scalar equation, a vector equation (``name, shape, [exprs]``), a blueprint, or
        an oriented relation such as a KinematicBC (stored as an assumption).

        Args:
            group: ``"model"`` for the final PDE system or ``"aux"`` for auxiliary relations that
                never reach the SystemModel.

        Returns:
            The Equation, VectorEquation or stored relation.
        """
        if getattr(name, "_is_blueprint", False):
            return name.add_to(self)
        if group not in ("model", "aux"):
            raise ValueError(
                f"add_equation(group=...): 'model' | 'aux', got {group!r}.")
        if group == "aux":
            self._aux_names.add(name)
        # Store an ORIENTED relation (KinematicBC / a solved Equation — anything
        # carrying ``_as_relation``) AS the object: it holds ``{lhs: rhs}``, not
        # a bare residual.  ``PDETransformation`` σ-maps its rule alongside the
        # equations, and ``model.<name>`` / ``model.apply(model.<name>)`` reach
        # and apply it as a substitution.
        if expr is None and getattr(shape_or_expr, "_as_relation", None):
            self._assumptions[name] = shape_or_expr
            return shape_or_expr

        # Disambiguate the two-arg scalar call from the three-arg vector call.
        if expr is None and shape_or_expr is not None and not (
                isinstance(shape_or_expr, tuple)
                and all(isinstance(k, int) for k in shape_or_expr)):
            # scalar: add_equation(name, expr).  Accept an existing Equation /
            # Expression and COPY its residual — e.g. duplicate continuity for
            # the w-reconstruction: ``m.add_equation("w", m.mass)``.  The
            # ``.expr`` unwrap must NOT fire on sympy objects: ``Derivative``
            # also carries ``.expr`` (its differentiated argument), and
            # unwrapping it silently turns ``∂_t b`` into ``b``.
            scalar_expr = (shape_or_expr.expr
                           if hasattr(shape_or_expr, "expr")
                           and not isinstance(shape_or_expr, sp.Basic)
                           else shape_or_expr)
            eq = Equation(scalar_expr, name=name, model=self)
            self._equations[name] = eq
            self._equation_shapes[name] = (1,)
            if _refresh:
                self._refresh_unknowns()
            return eq

        if expr is None:
            # scalar with explicit shape omitted / zero row.
            eq = Equation(sp.S.Zero if shape_or_expr is None else shape_or_expr,
                          name=name, model=self)
            self._equations[name] = eq
            self._equation_shapes[name] = (1,)
            if _refresh:
                self._refresh_unknowns()
            return eq

        # vector: add_equation(name, shape, [exprs])
        shape = shape_or_expr
        components = []
        n = len(expr)
        names = _COMPONENT_NAMES.get(n)
        if names is None:
            raise ValueError(
                f"add_equation({name!r}): no axis names for {n} components.")
        for axis, comp_expr in zip(names, expr):
            comp_name = f"{name}_{axis}"
            comp_eq = Equation(comp_expr, name=comp_name, model=self)
            self._equations[comp_name] = comp_eq
            self._equation_shapes[comp_name] = (1,)
            components.append(comp_eq)
        vec = VectorEquation(name, shape, components)
        self._vector_equations[name] = vec
        self._equation_shapes[name] = tuple(shape)
        self._refresh_unknowns()
        return vec

    def remove(self, name):
        """Remove a named equation, vector row or stored assumption; returns ``self``."""
        if name in self._assumptions:
            del self._assumptions[name]
            self._refresh_unknowns()
        else:
            self._remove_equation(name)
        return self

    def _remove_equation(self, name):
        """Drop a scalar row, or a vector proxy with its components; removing one component also
        prunes it from the parent VectorEquation.
        """
        if name in self._vector_equations:
            vec = self._vector_equations.pop(name)
            for comp in vec.components:
                comp_name = getattr(comp, "name", None)
                self._equations.pop(comp_name, None)
                self._equation_shapes.pop(comp_name, None)
            self._equation_shapes.pop(name, None)
            self._refresh_unknowns()
            return
        # A standalone scalar OR a vector component — drop the scalar row and,
        # if it was a component, prune it from the parent proxy.
        self._equations.pop(name, None)
        self._equation_shapes.pop(name, None)
        for vname, vec in list(self._vector_equations.items()):
            for i, comp in enumerate(vec._components):
                if getattr(comp, "name", None) == name:
                    del vec._components[i]
                    vec._axis_names = tuple(
                        ax for j, ax in enumerate(vec._axis_names) if j != i)
                    if vec._components:
                        self._equation_shapes[vname] = (len(vec._components),)
                        vec.shape = (len(vec._components),)
                    else:
                        self._vector_equations.pop(vname, None)
                        self._equation_shapes.pop(vname, None)
                    break
        self._refresh_unknowns()

    def _collapse_moment_family(self, base, keep):
        """Shrink moment family ``base`` to the modes in ``keep``; a single survivor is re-keyed
        from ``<base>_<mode>`` to ``<base>`` but stays a family.
        """
        fam = self._moment_families[base]
        keep = list(keep)
        survivors = [fam.components[fam.modes.index(k)] for k in keep]
        fam._components = survivors
        fam.modes = keep
        if len(keep) == 1:
            (k,) = keep
            old_name = f"{base}_{k}"
            row = self._equations.pop(old_name)
            self._equation_shapes.pop(old_name, None)
            row.name = base
            self._equations[base] = row
            self._equation_shapes[base] = (len(keep),)
        else:
            self._equation_shapes[base] = (len(keep),)
        self._refresh_unknowns()
        return fam

    # ── attribute access to equations ────────────────────────────────
    def __getattr__(self, item):
        """Resolve ``model.<name>`` to a vector proxy, moment family, scalar equation or stored
        assumption.
        """
        # Consulted only on normal-lookup failure.  Resolve vector proxies
        # first, then moment families, then scalar rows.
        vec = self.__dict__.get("_vector_equations", {})
        if item in vec:
            return vec[item]
        fam = self.__dict__.get("_moment_families", {})
        if item in fam:
            return fam[item]
        eqs = self.__dict__.get("_equations", {})
        if item in eqs:
            return eqs[item]
        asmpt = self.__dict__.get("_assumptions", {})
        if item in asmpt:
            return asmpt[item]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {item!r}")

    # ── apply (broadcast / route model-level ops) ────────────────────
    def apply(self, op, **kwargs):
        """Apply an operation: ``system_level`` ops once to the model, ``whole_model_op`` ops via
        ``apply_to_model``, all others broadcast over every scalar equation.
        """
        lvl = getattr(op, "log_level", 1)
        if getattr(op, "system_level", False):
            result = op(self)
            self._history(getattr(op, "name", None) or type(op).__name__,
                          "*system*", log_level=lvl)
            return result if result is not None else self
        if getattr(op, "whole_model_op", False):
            op.apply_to_model(self)
            self._history(getattr(op, "name", None) or type(op).__name__,
                          "*model*", log_level=lvl)
            return self
        for eq in list(self._equations.values()):
            eq.apply(op, _no_history=True, **kwargs)
        self._history(getattr(op, "name", None) or type(op).__name__, "*all*",
                      log_level=lvl)
        self._refresh_unknowns()
        return self

    # ── unknowns: Q (declared) / Qaux (derived) ──────────────────────
    @property
    def Q(self):
        """Declared unknowns as a Zstruct."""
        out = Zstruct()
        for nm, field in self._Q.items():
            setattr(out, nm, field)
        out._symbolic_name = "Q"
        return out

    @Q.setter
    def Q(self, fields):
        """Declare the unknowns from a list (or Zstruct) of applied fields, auto-named by head."""
        if isinstance(fields, Zstruct):
            fields = fields.values()
        new_q = {}
        for f in fields:
            new_q[self._field_name(f)] = f
        self._Q = new_q
        self._refresh_unknowns()

    @staticmethod
    def _as_field_list(value):
        """Normalise a Q entry (single field or list of mode fields) to a list."""
        return list(value) if isinstance(value, (list, tuple)) else [value]

    @property
    def Qaux(self):
        """Derived auxiliaries: every field in the equations that is not in ``Q``, not a coordinate
        and not a parameter.
        """
        q_heads = {self._head(f)
                   for v in self._Q.values() for f in self._as_field_list(v)}
        out = Zstruct()
        seen = set()
        for f in sorted(self._collect_fields(), key=str):
            head = self._head(f)
            if head in q_heads:
                continue
            nm = self._field_name(f)
            if nm in seen:
                continue
            seen.add(nm)
            setattr(out, nm, f)
        out._symbolic_name = "Qaux"
        return out

    def explicit_state(self):
        """One applied field per time-derivative row of the primary equations, with modal families
        flattened; the state vector for ``SystemModel.from_model``.
        """
        t = self.coords[0]
        out, seen = [], set()
        for name, eq in self._equations.items():
            if name in self._aux_names:
                continue
            cands = []
            for der in sp.expand(eq.expr).atoms(sp.Derivative):
                if der.variables == (t,):
                    for f in der.args[0].atoms(sp.Function):
                        if self._is_field_application(f) and f not in seen:
                            cands.append(f)
            if cands:
                f = sorted(cands, key=sp.srepr)[0]
                out.append(f)
                seen.add(f)
        return out

    @property
    def functions(self):
        """Registry of every field by its original name as a FieldHandle (bulk form, vertical
        argument normalised to ``zeta`` after the sigma map).
        """
        zeta = sp.Symbol("zeta", real=True)
        deco_to_orig = {deco: orig.__name__
                        for orig, deco in self._field_decoration.items()}

        def _rank(f):
            """Rank an applied field: canonical ``zeta`` form over other Symbol arguments over
            boundary/numeric forms.
            """
            # Prefer the canonical ζ form, then any Symbol/Dummy vertical arg,
            # then boundary / numeric forms — so the handle wraps the bulk field.
            last = f.args[-1] if f.args else None
            if last == zeta:
                return 2
            return 1 if isinstance(last, sp.Symbol) else 0

        reps: dict = {}
        for f in self._collect_fields():
            head = self._head(f)
            name = deco_to_orig.get(head, getattr(head, "__name__", str(head)))
            if name not in reps or _rank(f) > _rank(reps[name]):
                reps[name] = f

        out = Zstruct()
        for name in sorted(reps):
            f = reps[name]
            head = self._head(f)
            # σ-mapped fields: normalise the vertical arg to the canonical ζ so
            # ``.expr`` / ``.at`` read off ``\tilde f(t, x, ζ)`` regardless of
            # which form (``ζ`` / ``\hat ζ`` / ``0`` / ``1``) was present.
            if (head in deco_to_orig and f.args
                    and isinstance(f.args[-1], sp.Symbol)):
                f = head(*f.args[:-1], zeta)
            setattr(out, name, FieldHandle(f))
        out._symbolic_name = "functions"
        return out

    def _refresh_unknowns(self):
        """Drop from ``Q`` any unknown that has appeared and then left the primary equations or was
        demoted via ``move_to_aux``.
        """
        primary_heads = {self._head(f)
                         for f in self._collect_fields(primary_only=True)}
        self._ever_seen_heads |= {self._head(f) for f in self._collect_fields()}

        def _kept(v):
            """True while the entry is in a primary equation or has never been seen."""
            heads = [self._head(f) for f in self._as_field_list(v)]
            # Explicitly demoted (move_to_aux) → never a state, even if it still
            # appears in a primary equation (the bed-slope ``b``).
            if any(hd in self._aux_fields for hd in heads):
                return False
            # Keep while present in the PRIMARY system, or while never seen yet
            # (declared-early case, e.g. ``h`` before its balance is added).
            return (any(hd in primary_heads for hd in heads)
                    or all(hd not in self._ever_seen_heads for hd in heads))

        self._Q = {nm: v for nm, v in self._Q.items() if _kept(v)}

    def move_to_aux(self, *fields):
        """Demote fields (given data such as the bed ``b``) from ``Q`` to ``Qaux`` even if they stay
        in a primary equation; returns ``self``.
        """
        for field in fields:
            self._aux_fields.add(self._resolve_head(field))
        self._refresh_unknowns()
        return self

    def register_group(self, slot, index, relation):
        """Record ``relation`` as component ``index`` of function group ``slot`` (e.g.
        ``'interpolate'``, ``'project'``), parsed into the matching SystemModel slot by
        ``from_model``.

        Args:
            relation: oriented relation, Equation, FieldHandle or bare sympy expression; only its
                right-hand side is stored.

        Returns:
            ``self``
        """
        key = (int(index) if isinstance(index, int)
               or getattr(index, "is_Integer", False) else index)
        self._function_groups.setdefault(slot, {})[key] = \
            self._relation_rhs(relation)
        return self

    @staticmethod
    def _relation_rhs(relation):
        """Defining right-hand side of a relation, Equation, FieldHandle or expression."""
        rel = getattr(relation, "_as_relation", None)
        if isinstance(rel, dict) and rel:
            return sp.sympify(next(iter(rel.values())))
        if hasattr(relation, "rhs"):
            return sp.sympify(relation.rhs)
        return sp.sympify(getattr(relation, "expr", relation))

    def _resolve_head(self, field):
        """Function head of a field given as head, applied Function or name."""
        if isinstance(field, str):
            for f in self._collect_fields():
                if self._field_name(f) == field:
                    return self._head(f)
            for orig, deco in self._field_decoration.items():
                if getattr(orig, "__name__", None) == field:
                    return deco
            raise KeyError(f"move_to_aux: no field named {field!r}")
        return self._head(field)

    def redeclare_unknown(self, old_field, new_fields, rename_key=False):
        """Transfer unknown status from ``old_field`` to ``new_fields``, keeping the logical key
        (decoration) or renaming it to the new family (``rename_key=True``).
        """
        old_head = self._head(old_field)
        if not isinstance(new_fields, (list, tuple)):
            new_fields = [new_fields]
        value = new_fields[0] if len(new_fields) == 1 else list(new_fields)

        def _is_old(v):
            """True if a Q entry contains the old head."""
            return any(self._head(f) == old_head
                       for f in self._as_field_list(v))

        # The LOGICAL key(s) the old family currently occupies.
        old_keys = [nm for nm, v in self._Q.items() if _is_old(v)]
        if len(old_keys) == 1:
            # In-place swap (preserves dict ORDER): keep the key for a
            # decoration, or rename it to the new family for a rename.
            new_key = (self._field_name(new_fields[0]) if rename_key
                       else old_keys[0])
            self._Q = {(new_key if nm == old_keys[0] else nm):
                       (value if nm == old_keys[0] else v)
                       for nm, v in self._Q.items()}
        else:
            # Spread across several keys (or none yet): drop the old entries and
            # key each new field by its own family name.
            self._Q = {nm: v for nm, v in self._Q.items() if not _is_old(v)}
            for nf in new_fields:
                self._Q[self._field_name(nf)] = nf

    # ── field introspection ──────────────────────────────────────────
    @staticmethod
    def _is_field_application(atom):
        """True for applications of undefined-like Function heads (user fields and minted decorated
        heads); False for sympy builtins and basis/bracket machinery.
        """
        from sympy.core.function import AppliedUndef

        if isinstance(atom, AppliedUndef):
            return True
        func = getattr(atom, "func", None)
        if func is None or not isinstance(func, sp.FunctionClass):
            return False
        # Basis machinery (opaque ``φ(k, ζ)`` / weight ``c(ζ)``) and named
        # Galerkin brackets (``Gram`` / ``Weight``) are derivation machinery,
        # never unknowns — they carry ``_is_basis_head`` / ``_is_bracket``.
        if getattr(func, "_is_basis_head", False):
            return False
        if getattr(func, "_is_bracket", False):
            return False
        # Undefined-like: a direct subclass of bare ``Function`` with no
        # evaluation rule (the minted decorated heads) — distinguished from
        # sympy builtins which carry their own ``eval``.
        return (sp.Function in getattr(func, "__bases__", ())
                and func.__dict__.get("eval") is None)

    def _collect_fields(self, primary_only=False):
        """Every field application across the equations whose head is not a parameter;
        ``primary_only`` skips auxiliary rows.
        """
        param_names = set(self.parameters.keys())
        fields = set()
        for name, eq in self._equations.items():
            if primary_only and name in self._aux_names:
                continue
            for atom in eq.expr.atoms(sp.Function):
                if not self._is_field_application(atom):
                    continue
                if atom.func.__name__ in param_names:
                    continue
                fields.add(atom)
        return fields

    # ── index registration (Galerkin test / moment index) ────────────
    def register_index(self, name):
        """Mint and register a Galerkin test/moment index Symbol; the same name returns the same
        symbol and clashes with modal indices raise.
        """
        return self._modal_registry.register(name)

    # ── modal-index / basis accessors (per-field Resolve) ─────────────
    def modal_index(self, target):
        """Modal-mode index assigned to a field or coefficient family by
        ``separation_of_variables``.
        """
        idx = self._modal_registry.index_for(target)
        if idx is None:
            from zoomy_core.model.derivation.modal import _coeff_key
            idx = self._modal_registry.fresh(_coeff_key(target))
        return idx

    def modal_basis(self, target):
        """Basis registered with the field's separation-of-variables expansion, or None."""
        return self._modal_registry.basis_for(target)

    def _coord_symbols(self):
        """Set of coordinate symbols."""
        return set(self._coords)

    @staticmethod
    def _head(field):
        """Function head of an applied field or head."""
        if isinstance(field, sp.FunctionClass):
            return field
        func = getattr(field, "func", None)
        return func if func is not None else field

    @classmethod
    def _field_name(cls, field):
        """Field name taken from its head."""
        head = cls._head(field)
        return getattr(head, "__name__", None) or str(head)

    # ── describe ─────────────────────────────────────────────────────
    def describe(self, strip_args=False, show="model", show_tags=False,
                 show_history=False, max_log_level=5):
        """Markdown description of the model: parameters, Q and equations; ``show`` selects
        ``"model"``, ``"aux"`` or ``"all"`` rows.
        """
        if show not in ("model", "aux", "all"):
            raise ValueError(
                f"describe(show=...): expected 'model' | 'aux' | "
                f"'all', got {show!r}.")
        n_eq = len(self._equations)
        n_vec = len(self._vector_equations)
        n_ops = len(self.history)
        parts = [
            f"**{self.name}** — {n_eq} equation{'s' if n_eq != 1 else ''} "
            f"({n_vec} vector), {n_ops} op{'s' if n_ops != 1 else ''} "
            f"— coords {tuple(str(c) for c in self._coords)}"
        ]
        if self.parameters.length():
            kv = ", ".join(
                f"${sp.latex(self.parameters[k])} = "
                f"{getattr(self.parameter_values, k, '?')}$"
                for k in self.parameters.keys())
            parts.append(f"**Parameters:** {kv}")
        if self._Q:
            parts.append("**Q (declared):** "
                         + ", ".join(f"`{k}`" for k in self._Q))
        # Equations: render vector proxies as a group, then any remaining
        # standalone scalar rows.
        if show in ("model", "all"):
            rendered = set()
            parts.append("\n**Equations:**")
            for vname, vec in self._vector_equations.items():
                parts.append(f"- **{vname}** (vector):")
                for comp in vec.components:
                    if isinstance(comp, MomentFamily):
                        for row in comp.components:
                            parts.append("  " + row.describe_line(
                                strip_args=strip_args, bullet=True))
                            rendered.add(row.name)
                    else:
                        parts.append("  " + comp.describe_line(
                            strip_args=strip_args, bullet=True, show_tags=show_tags))
                        rendered.add(comp.name)
            for base, fam in self._moment_families.items():
                if base in self._vector_equations or any(
                        fam is c for v in self._vector_equations.values()
                        for c in v.components):
                    continue
                parts.append(f"- **{base}** (moment family):")
                for row in fam.components:
                    parts.append("  " + row.describe_line(
                        strip_args=strip_args, bullet=True, show_tags=show_tags))
                    rendered.add(row.name)
            for name, eq in self._equations.items():
                if name in rendered or name in self._aux_names:
                    continue          # aux rows render under the Auxiliary heading
                parts.append(eq.describe_line(strip_args=strip_args, bullet=True, show_tags=show_tags))
        # Auxiliary: ``group="aux"`` equation rows (w, ω, …) PLUS the oriented
        # relations / BCs in ``_assumptions`` (KinematicBC …, ``lhs = rhs``).
        if show in ("aux", "all"):
            aux_rows = [(nm, self._equations[nm]) for nm in self._equations
                        if nm in self._aux_names]
            if aux_rows or self._assumptions:
                from zoomy_core.model.operations import _StripArgsLatexPrinter
                pr = _StripArgsLatexPrinter() if strip_args else None
                tex = (lambda e: pr.doprint(e)) if pr else sp.latex
                parts.append("\n**Auxiliary:**")
                for nm, eq in aux_rows:
                    parts.append(eq.describe_line(strip_args=strip_args, bullet=True, show_tags=show_tags))
                for name, a in self._assumptions.items():
                    rules = getattr(a, "subs_map", {}) or {}
                    body = " \\\\ ".join(f"{tex(l)} = {tex(r)}"
                                         for l, r in rules.items())
                    parts.append(f"- **{name}:** ${body}$")
        if show_history and self.history:
            shown = [h for h in self.history
                     if h.get("log_level", 1) <= max_log_level]
            parts.append(f"**History** (≤ level {max_log_level}, "
                         f"{len(shown)}/{len(self.history)} ops):")
            for h in shown:
                parts.append(f"- `{h['op']}` (L{h.get('log_level', 1)}) "
                             f"→ {h['target']}")
        return Description("\n".join(parts))

    def __repr__(self):
        """Short repr with name and counts."""
        return (f"Model({self.name!r}, {len(self._equations)} equations, "
                f"{len(self.history)} ops)")


# ── resolve_modes: the Galerkin moment-axis SHAPE BUMP ─────────────────────


def resolve_modes(equation, *, index, modes, test_weight=None,
                  basis_cls=None, level=None, var=None, resolver=None):
    """Moment-axis shape bump: specialise the abstract Galerkin test index of a row to concrete
    modes and group the rows as a MomentFamily (scalar ``(1,) -> (N+1,)``, vector component ``(n,)
    -> (n, N+1)``).

    Args:
        equation: row carrying the abstract ``index`` (``model.mass`` or ``model.momentum.x``).
        modes: concrete moment indices.
        test_weight, basis_cls, level, var: if given, each row is also closed by
            ``closure.Resolve``.
        resolver: custom per-mode closer ``(row, test_weight_k, k)`` used instead of ``Resolve``.

    Returns:
        The MomentFamily.
    """
    model = equation._model
    base = equation.name                      # "mass" or "momentum_x"
    modes = list(modes)

    # Locate a parent VectorEquation if this row is a component.
    parent_vec = None
    axis = None
    for vname, vec in model._vector_equations.items():
        for ax, comp in zip(vec._axis_names, vec._components):
            if comp is equation or getattr(comp, "name", None) == base:
                parent_vec, axis = vec, ax
                break
        if parent_vec is not None:
            break

    src_expr = equation.expr

    # Build one moment row per mode via the outer-product specialisation, then
    # (optionally) close each with the concrete Resolve.
    do_close = (test_weight is not None and basis_cls is not None
                and level is not None)
    if do_close:
        from zoomy_core.model.derivation.closure import Resolve

    comps = []
    for k in modes:
        row_name = f"{base}_{k}"
        # Specialise the abstract test index ``l`` → concrete mode ``k``
        # BEFORE creating the row: one exact ``xreplace`` + one
        # canonicalization per mode.  Routing the substitution through
        # ``row.apply({index: k})`` instead costs a per-TERM pass plus a
        # full-expression simplify over the giant unspecialised sum for
        # EVERY mode — the dominant cost of high-level builds (SME(4)
        # spent ~70% of its 26 s build here; the integrals are cheap).
        # ``_refresh=False``: the unknowns registry is refreshed ONCE after
        # the loop — the per-add refresh re-walks every equation in the
        # model (the other half of the build cost).
        model.add_equation(row_name, src_expr.xreplace({index: k}),
                           _refresh=False)
        row = model._equations[row_name]
        if do_close:
            tw = test_weight.xreplace({index: k})
            if resolver is not None:
                resolver(row, tw, k)
            else:
                row.apply(Resolve(tw, basis_cls, level, var=var),
                          _no_history=True)
        comps.append(row)

    family = MomentFamily(base, modes, comps, index=index)
    model._moment_families[base] = family

    # Shape bump: scalar (1,) → (N+1,); vector component (n,) → (n, N+1).
    if parent_vec is not None:
        # The component is being REPLACED by the family, not removed — drop
        # only its flattened scalar row and install the family in the proxy
        # slot (do NOT route through ``_remove_equation``, which would prune
        # the axis we are about to fill).
        model._equations.pop(base, None)
        model._equation_shapes.pop(base, None)
        parent_vec.replace_axis(axis, family)
        old_shape = model._equation_shapes.get(parent_vec.name,
                                               (len(parent_vec),))
        model._equation_shapes[parent_vec.name] = (*old_shape, len(modes))
        parent_vec.shape = (*parent_vec.shape, len(modes))
    else:
        # Drop the abstract-index template scalar row.
        model._remove_equation(base)
        model._equation_shapes[base] = (len(modes),)

    model._history("resolve_modes", base)
    model._refresh_unknowns()
    return family


class ResolveModes(Operation):
    """Tracked-operation form of ``resolve_modes``; it does not close the basis brackets."""

    def __init__(self, index, modes, name="resolve_modes"):
        """Store the abstract ``index`` and the concrete ``modes``."""
        self._index = index
        self._modes = list(modes)
        super().__init__(
            name=name,
            description=f"moment shape-bump {index} → {self._modes}")

    def apply_to_equation(self, equation):
        """Bump the equation via ``resolve_modes``."""
        return resolve_modes(equation, index=self._index, modes=self._modes)
