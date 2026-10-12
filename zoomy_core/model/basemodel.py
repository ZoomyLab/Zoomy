"""Base class ``Model`` of the Model layer, where the user writes the physics as symbolic
equations or operator methods. A Model is lifted to a SystemModel by ``SystemModel.from_model``.
"""

import sympy as sp
import numpy as np
import param

from zoomy_core.model.boundary_conditions import BoundaryConditions

# Import the new Aux specific classes
import zoomy_core.model.aux_boundary_conditions as AuxBC
from zoomy_core.model.initial_conditions import Constant, InitialConditions
from zoomy_core.misc.misc import Zstruct, ZArray
from zoomy_core.model.basefunction import Function, SymbolicRegistrar

sp.init_printing()

# Spec-keyed cache of the heavy symbolic derivation (task 0018), shared across
# model instances.  Populated by ``Model._run_derive_model`` for models that opt
# in via ``_cacheable_derivation``.  Keyed by
# ``zoomy_core.model.derivation.cache_keys.model_spec_key``.
_DERIVATION_MODEL_CACHE: dict = {}


def clear_derivation_model_cache():
    """Drop all cached symbolic derivations."""
    _DERIVATION_MODEL_CACHE.clear()


def _derivation_byproduct(model, on_derivation, legacy_attr):
    """Fetch a derivation byproduct (interpolate/project/reconstruction rows) from the
    derivation object, else from the legacy ``self.<legacy_attr>`` stash, else ``None``.
    """
    deriv = getattr(model, "derivation", None)
    val = getattr(deriv, on_derivation, None) if deriv is not None else None
    if val is None:
        val = getattr(model, legacy_attr, None)
    return val


def register_sympy_attribute(definition, prefix="q"):
    """Turn an int or list field spec into a ``Zstruct`` of sympy Symbols (legacy; prefer
    :func:`parse_definition_to_zstruct`).
    """
    if isinstance(definition, Zstruct):
        return definition
    if isinstance(definition, int):
        entries = [(f"{prefix}_{i}", "real") for i in range(definition)]
    elif isinstance(definition, (list, tuple)):
        entries = [_split_declaration(e) for e in definition]
    else:
        raise TypeError(f"Unsupported definition type: {type(definition)}")
    z = Zstruct(**{str(n): _constrained_symbol(n, c) for n, c in entries})
    z._symbolic_name = prefix
    return z


def eigenvalue_dict_to_matrix(eigenvals_dict):
    """Flatten a sympy ``eigenvals()`` dict ``{eigenvalue: multiplicity}`` into a ``ZArray``."""
    result = []
    for ev, mult in eigenvals_dict.items():
        result.extend([ev] * int(mult))
    return ZArray(result)


def default_simplify(expr):
    """Default simplifier, ``powsimp`` with ``combine='all'``."""
    return sp.powsimp(expr, combine="all", force=False, deep=True)


#: Declared constraint name → extra sympy assumptions on the Symbol.
#: A declaration carries the constraint as the SECOND element of a pair —
#: ``("h", "nonnegative")`` in a variable list, ``"g": (9.81, "positive")``
#: in a parameter dict.  The assumption is what lets the CAS pull depth out
#: of a radical (``sqrt(h**2) → h``, ``sqrt(g*h**5) → sqrt(g)*h**(5/2)``), so
#: a dropped constraint is a silent loss of algebraic simplification, not
#: cosmetic.
SYMBOL_CONSTRAINTS = {
    "real": {},
    "nonnegative": {"nonnegative": True},   # x >= 0 — wet/dry depth admits h = 0
    "positive": {"positive": True},         # x >  0
}


def _constrained_symbol(name, constraint):
    """Real ``sp.Symbol`` carrying the sign assumption of the declared constraint; an unknown
    constraint raises.
    """
    if constraint not in SYMBOL_CONSTRAINTS:
        raise ValueError(
            f"unknown constraint {constraint!r} on symbol {str(name)!r}; "
            f"expected one of {sorted(SYMBOL_CONSTRAINTS)}")
    return sp.Symbol(str(name), real=True, **SYMBOL_CONSTRAINTS[constraint])


def _split_declaration(entry):
    """Split a declaration entry into ``(payload, constraint)``; a bare entry gets the
    constraint ``"real"``.
    """
    if isinstance(entry, (list, tuple)) and len(entry) > 1:
        return entry[0], entry[1]
    return entry, "real"


def parse_definition_to_zstruct(definition, prefix="q_"):
    """Turn an int, list, dict or ``Zstruct`` spec into a ``Zstruct`` of constrained sympy
    Symbols. List entries are ``name`` or ``(name, constraint)``, dict entries ``name: default``
    or ``name: (default, constraint)``; see :data:`SYMBOL_CONSTRAINTS`.
    """
    attributes = {}
    if isinstance(definition, int):
        for i in range(definition):
            name = f"{prefix}{i}"
            attributes[name] = sp.Symbol(name, real=True)
    elif isinstance(definition, (list, tuple)):
        for entry in definition:
            name, constraint = _split_declaration(entry)
            attributes[str(name)] = _constrained_symbol(name, constraint)
    elif isinstance(definition, dict):
        for name, data in definition.items():
            _, constraint = _split_declaration(data)
            attributes[str(name)] = _constrained_symbol(name, constraint)
    elif isinstance(definition, Zstruct):
        return parse_definition_to_zstruct(definition.as_dict(), prefix=prefix)

    return Zstruct(**attributes)


def merge_parameter_overrides(declared, overrides):
    """Merge user ``parameters=`` over the declared parameter dict, keeping the declared
    constraint unless the override carries its own ``(value, constraint)`` pair.
    """
    merged = dict(declared)
    for name, value in (overrides or {}).items():
        if isinstance(value, (list, tuple)) and len(value) > 1:
            merged[name] = value          # override states its own constraint
            continue
        _, constraint = _split_declaration(merged.get(name))
        merged[name] = (value, constraint)
    return merged


def extract_parameter_defaults(definition):
    """Numeric defaults of a parameter declaration (feeds ``model.parameter_values``)."""
    defaults = {}
    if isinstance(definition, dict):
        for name, val in definition.items():
            if isinstance(val, (int, float)):
                defaults[name] = val
            elif isinstance(val, (list, tuple)) and len(val) == 2:
                if isinstance(val[0], (int, float)):
                    defaults[name] = val[0]
            else:
                defaults[name] = 0.0
    elif isinstance(definition, Zstruct):
        return extract_parameter_defaults(definition.as_dict())
    else:
        assert False
    return defaults


class Model(param.Parameterized, SymbolicRegistrar):
    """Base class of the Model layer: named equations plus variables, parameters, boundary and
    initial conditions, lifted to a SystemModel by ``SystemModel.from_model``. Subclasses
    populate equations in ``derive_model`` or override the operator methods (``flux``,
    ``source``, ...).
    """
    name = param.String(default="Model")
    dimension = param.Integer(default=1)
    disable_differentiation = param.Boolean(default=False)

    # ── General Gauss-Legendre quadrature order (REQ-160) ──────────────
    # BASE-LEVEL capability inherited by EVERY moment model: any Galerkin
    # integral with no closed form (non-polynomial material closures —
    # Bingham/HB, k-ε, rational eddy-viscosity) is resolved numerically by a
    # ``GaussQuadrature(var=ζ, order=quadrature_order)`` inserted after the
    # analytic bracket machinery.  0 (default) = off: an unresolvable integral
    # raises at extraction.  Each moment model (SME, MLSME, MLVAM, VAM, …)
    # routes this SAME order into EVERY per-layer / per-column reduction, so a
    # non-polynomial closure is never special-cased per class.
    quadrature_order = param.Integer(default=0, bounds=(0, None), doc=(
        "Gauss-Legendre order for numerical integration of Galerkin integrals "
        "the analytic machinery cannot resolve (non-polynomial closures); 0 = off."))

    eigenvalue_mode = param.Selector(
        default="symbolic",
        objects=["symbolic", "numerical"],
        doc="'symbolic': solve characteristic polynomial in SymPy (exact, can be slow). "
            "'numerical': defer to np.linalg.eigvals at runtime (fast, required for large systems)."
    )

    # Well-balanced interface reconstruction onto a steady manifold.  Built into
    # every model (default off); meaningful for free-surface models that define
    # the equilibrium.  'none' = standard scheme; 'audusse' = lake-at-rest
    # hydrostatic reconstruction (the existing PositiveNonconservativeRusanov,
    # exact for still water); 'bernoulli' = moving-equilibrium WB (single-signed
    # profiles, discharge-fraction streamline label); 'projected_bernoulli' =
    # layered moving WB (σ-parameterized |u| thickness weighting + signed
    # velocity, handles sign reversal over depth).  The solver reads this off the
    # SystemModel to pick the numerics.
    equilibrium_reconstruction = param.Selector(
        default="none",
        objects=["none", "audusse", "bernoulli", "projected_bernoulli"],
        doc="Well-balanced reconstruction strategy: "
            "none | audusse | bernoulli | projected_bernoulli."
    )

    variables = param.Parameter(default=1)
    aux_variables = param.Parameter(default=0)
    parameters = param.Parameter(default={})

    # Accept EITHER the legacy BoundaryConditions container OR the new flat
    # per-field list (boundary_conditions=[Wall("left", on="momentum"), …]),
    # resolved against the model's state in `system_model`.
    boundary_conditions = param.ClassSelector(
        class_=(BoundaryConditions, list), default=None)
    aux_boundary_conditions = param.ClassSelector(
        class_=BoundaryConditions, default=None
    )

    initial_conditions = param.ClassSelector(class_=InitialConditions, default=None)
    aux_initial_conditions = param.ClassSelector(class_=InitialConditions, default=None)

    # ── Lazy-finalization flag ─────────────────────────────────────
    # Subclasses whose ``derive_model`` leaves equations in Function
    # form (e.g. SME, VAM) set ``_finalize_lazy = True``.  For those:
    # ``__init__`` stops after the pipeline; the Function → Symbol
    # substitution + auto-tagging + ``_initialize_functions`` happen
    # later — either via an explicit ``model._finalize_for_systemmodel()``
    # call or automatically when ``SystemModel.from_model(model)``
    # runs.  This lets the user apply optional closures
    # (e.g. ``apply_slip_newton_friction``) on Function-form equations
    # without breaking ``Derivative(g·h², x)`` (which would evaluate
    # to 0 on bare Symbols).  Default ``False`` preserves the
    # existing flow for hand-written subclasses (SWE etc.).
    _finalize_lazy = False

    # ── Derivation cache (task 0018) ───────────────────────────────
    # Models whose ``derive_model`` is a PURE function of their spec —
    # it RETURNS the derivation object and stashes nothing else on
    # ``self`` (no ``self._bed`` / ``self._rows`` byproducts) — set
    # ``_cacheable_derivation = True``.  The heavy symbolic build is then
    # memoised across instances keyed on the model spec (level, dimension,
    # closures, parameters, …); the cheap ``system_model`` tail still
    # rebuilds a FRESH SystemModel per access, so callers may safely mutate
    # the result's ICs/BCs.  Default ``False`` leaves a model uncached.
    _cacheable_derivation = False

    def __init__(self, init_functions=True, **params):
        """Declare variables and parameters, run the derivation, and (unless ``_finalize_lazy``)
        finalize for the SystemModel.

        Args:
            init_functions: register the operator methods as symbolic functions during
                finalization.
        """
        super().__init__(**params)
        self.functions, self.call = Zstruct(), Zstruct()
        self._equations: dict = {}
        self.history: list = []
        # ``_variable_map`` maps equation names to row indices in the
        # operator-API matrices (e.g. {"continuity_0": [0],
        # "momentum_x_0": [1], ...}).  Subclasses set this in
        # ``_finalize_for_systemmodel`` *after* the pipeline produces
        # the final equation set.  Default: empty (no extraction).
        self._variable_map: dict = {}
        # Set by ``_finalize_for_systemmodel`` so subsequent calls
        # short-circuit; reset by mutating methods like
        # ``apply_slip_newton_friction``.
        self._finalized = False
        self._initialize_derived_properties()
        # Subclass derivation hook — populates ``self._equations``
        # via ``self.add_equation`` + ``self.apply(Op(...))`` and may
        # extend ``self.variables`` / ``self.parameters``.
        # REQ-163: for cacheable (pure, RETURN-style) derivations the heavy
        # symbolic build is DEFERRED to first ``.derivation`` access — a
        # SystemModel cache hit (sm_cache) then never derives at all.  Impure
        # models (derive_model mutates self) keep the eager call.
        if not self._cacheable_derivation:
            self._run_derive_model()
        # After the derivation, every Symbol used in any equation must
        # have a numeric value declared in ``self.parameter_values``.
        self._assert_parameter_values_supplied()
        if self._finalize_lazy:
            # Stop here — equations stay in Function form; user can
            # apply closures, then ``SystemModel.from_model`` will
            # auto-call ``_finalize_for_systemmodel``.
            return
        # Non-lazy (default) path: tag + register functions now.
        self._finalize_for_systemmodel(init_functions=init_functions)

    def _finalize_for_systemmodel(self, *, init_functions=True):
        """Idempotent hand-off preparation: subclass substitution, auto-tagging and optional
        function registration.
        """
        if self._finalized:
            return self
        self._prepare_for_systemmodel()
        self._auto_tag_equations()
        if init_functions:
            self._initialize_functions()
        self._finalized = True
        return self

    def _prepare_for_systemmodel(self):
        """Subclass hook run before auto-tagging; no-op on the base class."""
        return None

    @property
    def derivation(self):
        """The symbolic derivation object, built or restored lazily on first access for
        cacheable models.
        """
        d = self.__dict__.get("_derivation_obj")
        if d is None:
            self._run_derive_model()
            d = self.__dict__.get("_derivation_obj")
        return d

    @derivation.setter
    def derivation(self, value):
        """Set the derivation object."""
        self.__dict__["_derivation_obj"] = value

    def _run_derive_model(self):
        """Run the derivation, or restore it from the spec-keyed cache for
        ``_cacheable_derivation`` models.
        """
        if not self._cacheable_derivation:
            self.derive_model()
            return
        import os
        from zoomy_core.model.derivation.cache_keys import (
            model_spec_key, derivation_cache_enabled)
        cache_on = derivation_cache_enabled()
        rebuild = os.environ.get(
            "ZOOMY_DERIVATION_REBUILD", "").strip() in {"1", "true", "yes"}
        key = model_spec_key(self)
        cached = _DERIVATION_MODEL_CACHE.get(key) if (cache_on and not rebuild) else None
        if cached is not None:
            self.derivation = cached
            return
        result = self.derive_model()
        if result is not None:
            self.derivation = result
        _DERIVATION_MODEL_CACHE[key] = self.derivation

    # ── derivation hook + minimal Model surface ───────────────────
    #
    # The Model class exposes ONLY four public methods:
    # `add_equation / remove_equation / apply / describe`.  All
    # transformations are `Operation` subclasses (see
    # `zoomy_core.model.operations`) and are applied via
    # `model.apply(SomeOp(args))`.  In particular: there is NO
    # `multiply` or `resolve_dummy` convenience method — those become
    # `Multiply(...)` and `ResolveDummy(...)` Operations.
    #
    # Equations live in the private `self._equations` dict and are
    # accessed externally via `__getattr__` (named-attribute style):
    # `self.momentum_x.apply(...)`, `self.continuity.apply(...)`.
    # Iteration is via `for eq in self: ...` (`__iter__`).  There is
    # NO public `.equations` collection on Model.

    def derive_model(self):
        """Subclass hook that populates equations via ``add_equation`` and ``apply``; no-op on
        the base class.
        """
        return None

    def add_equation(self, name, expression=None, shape=None):
        """Insert an equation under ``name``, accessible as ``self.<name>``.

        Args:
            shape: ``None`` for a scalar equation, else a shape or list of component labels
                giving one sub-equation ``name_i_j`` per index.
        """
        from zoomy_core.model.equation import Equation
        if shape is None:
            expr = expression if expression is not None else sp.S.Zero
            self._equations[name] = Equation(expr, name=name, model=self)
            self._history("add_equation", name)
            return self
        # Branched: vector / tensor add_equation.
        import itertools as _it
        if all(isinstance(c, str) for c in shape):
            combos = [(c,) for c in shape]
        else:
            combos = list(_it.product(*shape))
        for combo in combos:
            sub = "_".join([name] + list(combo))
            self._equations[sub] = Equation(sp.S.Zero, name=sub, model=self)
            self._history("add_equation", sub)
        return self

    def remove_equation(self, name):
        """Remove the equation ``name``."""
        del self._equations[name]
        self._history("remove_equation", name)
        return self

    def apply(self, op, *, level="major", description=None):
        """Broadcast an Operation, substitution dict, Relation or callable over every equation
        (whole-model operations are applied once).

        Args:
            level: history level, ``"major"`` or ``"minor"``.
            description: free text recorded in the history.
        """
        # Whole-model dispatch: e.g. ResolveDummy([v1, v2]) needs
        # access to the whole equation dict to branch.
        whole_model = getattr(op, "whole_model_op", False)
        if whole_model and hasattr(op, "apply_to_model"):
            op.apply_to_model(self)
        else:
            for eq in self._equations.values():
                eq.apply(op, _no_history=True)
        self._history(
            getattr(op, "name", None) or type(op).__name__,
            "*", level=level, description=description,
        )
        return self

    def describe(self, *, show_history=False, include_minor=False):
        """Return a :class:`ModelDescription` that renders the equation set (LaTeX in Jupyter,
        plain text otherwise).
        """
        return ModelDescription(
            self,
            show_history=show_history,
            include_minor=include_minor,
        )

    def _history(self, op_label, target, *, level="major", description=None,
                 log_level=1):
        """Append one entry to the operation history."""
        self.history.append({
            "op": op_label, "target": target,
            "level": level, "log_level": log_level,
            "description": description or "",
        })

    # ── Dunders ──────────────────────────────────────────────────
    def __getattr__(self, name):
        """Named-attribute access to equations; also raises the migration message for the
        removed ``Model.system_model``.
        """
        # Avoid infinite recursion during `__init__` when
        # `self._equations` itself doesn't yet exist.
        eqs = self.__dict__.get("_equations", None)
        if eqs is not None and name in eqs:
            return eqs[name]
        if name == "system_model":
            # REQ-143/REQ-164 — REMOVED, not shimmed: a Model does not know
            # SystemModel (hierarchy is strictly Model → SystemModel →
            # NumericalSystemModel).  ``hasattr`` stays False; this only
            # turns the broken call sites' bare AttributeError into the
            # migration instruction.  (Must live HERE, not in a raising
            # property: a property's AttributeError falls back into this
            # ``__getattr__`` and its message would be swallowed.)
            raise AttributeError(
                "Model.system_model was REMOVED (REQ-143/REQ-164). Build it "
                "explicitly:\n"
                "    from zoomy_core.systemmodel.system_model import "
                "SystemModel\n"
                "    sm = SystemModel.from_model(model)\n"
                "(verified field-identical for all models INCLUDING "
                "overrides; use NumericalSystemModel.from_model(model) for "
                "the numerics-ready variant)")
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute "
            f"or equation {name!r}"
        )

    def __iter__(self):
        """Iterate over equations in insertion order."""
        return iter(self._equations.values())

    def __len__(self):
        """Number of equations."""
        return len(self._equations)

    def _assert_parameter_values_supplied(self):
        """Raise if an equation contains a free Symbol that is neither a coordinate or state
        symbol nor has a numeric default in ``parameter_values``.
        """
        if not self._equations:
            return
        # Symbols that legitimately appear in equations but are not
        # parameters (coordinates, distance, normal axes, state).
        skip = set()
        for attr in ("time", "distance"):
            v = getattr(self, attr, None)
            if v is not None:
                skip.add(v)
        for zstruct_attr in ("position", "normal", "variables", "aux_variables"):
            z = getattr(self, zstruct_attr, None)
            if z is None:
                continue
            for k in z.keys():
                v = z[k]
                if isinstance(v, sp.Symbol):
                    skip.add(v)
        # StateSpace exposes reference coordinates that propagate
        # through σ-mapped equations (zeta, zeta_ref, optionally y, z).
        # Parameters (state.g, state.rho) are *not* skipped — they're
        # exactly what we want to flag when the user forgot to supply
        # a numeric default.  (In the new design, state.g / state.rho
        # don't exist on StateSpace anyway — parameters live on
        # `self.parameters`.)
        state = getattr(self, "state", None)
        if state is not None:
            for nm in ("t", "x", "y", "z", "zeta", "zeta_ref"):
                v = getattr(state, nm, None)
                if isinstance(v, sp.Symbol):
                    skip.add(v)
        param_value_names = set(self.parameter_values.keys()) if hasattr(self, "parameter_values") else set()
        param_symbol_names = set(self.parameters.keys()) if hasattr(self, "parameters") else set()
        missing: dict = {}
        for name, eq in self._equations.items():
            try:
                fsyms = eq.expr.free_symbols
            except AttributeError:
                continue
            for sym in fsyms:
                if sym in skip:
                    continue
                sym_name = str(sym)
                if sym_name in param_value_names:
                    continue
                if sym_name in param_symbol_names:
                    # Declared as a Symbol on the Model but no value —
                    # still record as missing (user must declare value).
                    if sym_name not in param_value_names:
                        missing.setdefault(sym_name, sym)
                    continue
                # Unknown free Symbol (neither declared nor in skip).
                missing.setdefault(sym_name, sym)
        if missing:
            names = ", ".join(sorted(missing.keys()))
            raise ValueError(
                f"symbol {sorted(missing.keys())[0]!r} found in {type(self).__name__} "
                f"equations but no numeric default supplied in "
                f"parameters={{...}}; missing: [{names}]. Supply via "
                f"`{type(self).__name__}(..., parameters={{...}})`."
            )

    def _auto_tag_equations(self):
        """Tag every untagged equation with solver tags via ``auto_solver_tag`` so the default
        operator extractors can pull tagged terms.
        """
        if not self._equations:
            return
        from zoomy_core.model.derivation.tag_extraction import auto_solver_tag
        # State atoms = self.variables (Symbols after derive_model
        # finishes its substitution pass).
        state_atoms = [v for v in self.variables.values()
                       if isinstance(v, sp.Symbol)]
        if not state_atoms:
            return
        t_sym = self.time
        coords = [self.position[d] for d in range(self.dimension)]
        x_sym = coords[0] if coords else None
        if x_sym is None:
            return
        # Gravity parameter Symbol for the hydrostatic_pressure
        # classifier.  In the new design, equation atoms reference
        # ``self.parameters.g`` directly (parameters live on the
        # Model and are propagated into MassMomentum at construction
        # time — no Symbol-identity mismatch).
        gravity_param = None
        if hasattr(self, "parameters") and "g" in self.parameters.keys():
            gravity_param = self.parameters.g
        for name, eq in self._equations.items():
            sg = getattr(eq, "_solver_groups", None)
            if sg:  # already tagged
                continue
            try:
                # Pass the primary horizontal coord only: ``coords`` here
                # would also contain the vertical ``z`` (``position`` spans
                # all dims), and ``∂_z`` is not a flux in the averaged
                # system.  Declarative models (VAM/SME/MLVAM) carry no
                # ``_equations`` so this fallback is inert for them; the
                # splitter's N-D ``auto_solver_tag`` call uses the
                # SystemModel's horizontal ``space`` directly.
                tagged = auto_solver_tag(
                    eq.expr, state_funcs=state_atoms,
                    t=t_sym, coords=[x_sym], gravity_param=gravity_param,
                )
            except Exception:
                # Heuristic can fail on degenerate shapes; leave
                # untagged so the extractor sees zero.
                continue
            # Stash solver groups on the equation.  The Expression
            # returned by auto_solver_tag carries ``_solver_groups``;
            # the Equation interface expects ``get_solver_tag`` /
            # ``solver_tags`` / ``untagged_remainder``.
            eq._solver_groups = getattr(tagged, "_solver_groups", {})

    def _resolve_input(self, val):
        """Resolve a declaration given as a ``param.Parameter``, a callable (of the model) or a
        plain value.
        """
        if isinstance(val, param.Parameter):
            val = val.default
        if callable(val):
            try:
                return val(self)
            except TypeError:
                return val()
        return val

    def _initialize_derived_properties(self):
        """Parse parameters, variables and aux variables into Zstructs and create time,
        position, normal and default BC/IC objects.
        """
        p_def = self._resolve_input(self.parameters)
        var_def = self._resolve_input(self.variables)
        aux_def = self._resolve_input(self.aux_variables)

        # 1. Parameters — symbolic / numeric split (canonical naming).
        # ``self.parameters`` — Zstruct of sympy Symbols used in
        #   equations and in the operator API (``flux``, ``source``,
        #   …).  This is the *symbolic identity* of each parameter.
        # ``self.parameter_values`` — Zstruct of numeric floats,
        #   user-mutable (e.g. ``model.parameter_values.g = 12.0``).
        #   Carried into ``SystemModel.parameter_values`` and lifted
        #   into the runtime callable argument named ``parameter`` at
        #   the printer / ``lambdify`` boundary.
        self.parameters = parse_definition_to_zstruct(p_def, "p")
        self.parameters._symbolic_name = "p"
        defaults = extract_parameter_defaults(p_def)
        self.parameter_values = Zstruct(
            **{k: float(defaults.get(k, 0.0)) for k in self.parameters.keys()}
        )
        self.parameter_values._symbolic_name = "p"

        # 2. Parse Variables
        self.variables = parse_definition_to_zstruct(var_def, "q")
        self.variables._symbolic_name = "Q"

        # 3. Parse Aux
        self.aux_variables = parse_definition_to_zstruct(aux_def, "qaux")
        self.aux_variables._symbolic_name = "Qaux"

        self.n_variables = self.variables.length()
        self.n_aux_variables = self.aux_variables.length()
        # Equal cardinality on the symbol Zstruct and the values Zstruct
        # — same keys, just different value types.
        self.n_parameters = self.parameters.length()

        self.time, self.distance = sp.symbols("t dX", real=True)
        # Position carries x / y / z (matches StateSpace + SystemModel
        # + sympy convention).  The `_symbolic_name = "X"` is preserved
        # for the codegen path which prefixes positional args as `X[d]`.
        self.position = parse_definition_to_zstruct(
            ["x", "y", "z"], "X")
        self.position._symbolic_name = "X"
        self.normal = parse_definition_to_zstruct(
            ["n" + str(i) for i in range(self.dimension)]
        )
        self.normal._symbolic_name = "n"

        self.boundary_conditions = self.boundary_conditions or BoundaryConditions([])

        self.initial_conditions = self.initial_conditions or Constant()
        self.aux_initial_conditions = self.aux_initial_conditions or Constant()

        self._simplify = default_simplify

    def _initialize_functions(self):
        """Register the operator methods and the boundary-condition kernels as symbolic
        functions with their call signatures.
        """
        std_sig = Zstruct(
            variables=self.variables,
            aux_variables=self.aux_variables,
            p=self.parameters,
        )
        eig_sig = Zstruct(**std_sig.as_dict(), normal=self.normal)
        # The per-cell update kernels carry an explicit trailing scalar ``dt``
        # (uniform across backends; full models ignore it, a Chorin corrector's
        # projection update is load-bearing in it).
        upd_sig = Zstruct(**std_sig.as_dict(), dt=sp.Symbol("dt", positive=True))
        res_sig = Zstruct(
            time=self.time,
            position=self.position,
            distance=self.distance,
            **std_sig.as_dict(),
        )
        base_proj = (
            Zstruct(Z=self.position[2]) if self.position.length() > 2 else Zstruct()
        )
        proj_sig = Zstruct(position=self.position, **std_sig.as_dict())

        ic_sig = Zstruct(position=self.position, p=self.parameters)

        # Gradient symbols — still produced for code-generation backends
        # (generic_c / amrex) that carry ``∇Q`` as an LDG-style explicit
        # auxiliary field.  The standard operator API
        # (``diffusion_matrix``, ``flux``, ``source``, …) is purely a
        # function of ``(Q, Qaux, p)``; derivatives that diffusion
        # depends on enter via ``Qaux`` (auto-exposed by
        # :meth:`SystemModel.expose_aux_atoms`).
        grad_syms = []
        for v in self.variables.keys():
            for d in range(self.dimension):
                grad_syms.append(sp.Symbol(f"dQ_{v}_d{d}", real=True))
        self.gradient_variables = parse_definition_to_zstruct(
            [str(s) for s in grad_syms], "gradQ"
        )
        self.gradient_variables._symbolic_name = "gradQ"

        regs = [
            ("flux", self.flux, std_sig),
            ("diffusion_matrix", self.diffusion_matrix, std_sig),
            ("diffusion_matrix_explicit",
             self.diffusion_matrix_explicit, std_sig),
            ("source_explicit", self.source_explicit, std_sig),
            ("dflux", self.dflux, std_sig),
            ("hydrostatic_pressure", self.hydrostatic_pressure, std_sig),
            ("nonconservative_matrix", self.nonconservative_matrix, std_sig),
            ("quasilinear_matrix", self.quasilinear_matrix, std_sig),
            ("eigenvalues", self.eigenvalues, eig_sig),
            ("left_eigenvectors", self.left_eigenvectors, eig_sig),
            ("right_eigenvectors", self.right_eigenvectors, eig_sig),
            ("eigensystem", self.eigensystem, eig_sig),
            ("source", self.source, std_sig),
            (
                "source_jacobian_wrt_variables",
                self.source_jacobian_wrt_variables,
                std_sig,
            ),
            (
                "source_jacobian_wrt_aux_variables",
                self.source_jacobian_wrt_aux_variables,
                std_sig,
            ),
            ("interpolate_to_3d", self.interpolate_to_3d, proj_sig),
            ("project_from_3d", self.project_from_3d, std_sig),
            ("residual", self.residual, res_sig),
            ("interpolate", self.interpolate, std_sig),
            ("initial_condition", self.initial_condition, ic_sig),
            ("initial_aux_condition", self.initial_aux_condition, ic_sig),
            ("update_variables", self.update_variables, upd_sig),
            ("update_aux_variables", self.update_aux_variables, upd_sig),
            ("reconstruction_variables",
             self.reconstruction_variables, std_sig),
            ("state_from_reconstruction",
             self.state_from_reconstruction, std_sig),
            (
                "update_variables_jacobian_wrt_variables",
                self.update_variables_jacobian_wrt_variables,
                upd_sig,
            ),
            (
                "update_aux_variables_jacobian_wrt_variables",
                self.update_aux_variables_jacobian_wrt_variables,
                upd_sig,
            ),
        ]
        for name, method, sig in regs:
            self.register_symbolic_function(name, method, sig)

        # --- Boundary Conditions Setup ---

        # 1. Main Boundary Conditions — value + face-normal gradient.
        # Both are built as indexed ``Piecewise`` ``Function`` kernels
        # so the runtime can call them as
        # ``rt.boundary_conditions(bc_idx, …)`` /
        # ``rt.boundary_gradients(bc_idx, …)`` per boundary face.
        self._boundary_conditions = (
            self.boundary_conditions.get_boundary_condition_function(
                self.time,
                self.position,
                self.distance,
                self.variables,
                self.aux_variables,
                self.parameters,
                self.normal,
                function_name="boundary_conditions",
            )
        )
        self._boundary_gradients = (
            self.boundary_conditions.get_boundary_gradient_function(
                self.time,
                self.position,
                self.distance,
                self.variables,
                self.aux_variables,
                self.parameters,
                self.normal,
                function_name="boundary_gradients",
            )
        )

        # 2. Aux Boundary Conditions Synchronization
        #    Collect all tags used in the main BCs
        main_tags = set(
            bc.tag for bc in self.boundary_conditions.boundary_conditions_list
        )

        # Initialize aux_boundary_conditions if None
        if self.aux_boundary_conditions is None:
            # Default: Create AuxBC.Extrapolation for every tag found in main BCs
            default_aux = [
                AuxBC.Extrapolation(tag=bc.tag)
                for bc in self.boundary_conditions.boundary_conditions_list
            ]
            self.aux_boundary_conditions = BoundaryConditions(default_aux)
        else:
            # If user provided some, ensure we cover any missing tags with AuxBC.Extrapolation
            aux_tags = set(
                bc.tag for bc in self.aux_boundary_conditions.boundary_conditions_list
            )
            missing = main_tags - aux_tags
            if missing:
                current_list = list(
                    self.aux_boundary_conditions.boundary_conditions_list
                )
                for t in missing:
                    current_list.append(AuxBC.Extrapolation(tag=t))
                # Re-create to ensure sorting by tag happens in __init__
                self.aux_boundary_conditions = BoundaryConditions(current_list)

        # 3. Aux Boundary Conditions Generation
        #    We use standard argument order (variables, aux_variables).
        #    The objects in self.aux_boundary_conditions are instances of AuxBC.Extrapolation/Lambda,
        #    so they will correctly return Qaux.
        self._aux_boundary_conditions = (
            self.aux_boundary_conditions.get_boundary_condition_function(
                self.time,
                self.position,
                self.distance,
                self.variables,
                self.aux_variables,
                self.parameters,
                self.normal,
                function_name="aux_boundary_conditions",  # [FIX] Pass the name here!
            )
        )
        self._aux_boundary_conditions.name = "aux_boundary_conditions"

    def print_boundary_conditions(self):
        """Definition of the boundary-condition function."""
        return self._boundary_conditions.definition

    # --- Physics Methods ---
    # Default implementations route through ``tag_extraction``: every
    # equation in ``self.equations`` is walked, terms are pulled by
    # canonical solver tag, and the resulting matrix is translated
    # from derivation Symbols / Function calls (e.g. ``h(t, x)``,
    # ``q(k, t, x)``, ``Symbol("g", positive=True)``) to Model Symbols
    # (``self.variables.h``, ``self.variables.q_k``,
    # ``self.parameters.g``) via ``self._symbol_map()``.
    #
    # Subclasses that prefer hand-written operators (legacy SWE-style)
    # may still override these methods directly — the override wins
    # and ``self.equations`` need not be populated.

    def _extract_via_tag(self, canonical_tag):
        """Collect all terms carrying the canonical solver tag into an operator matrix.

        Returns:
            The matrix, or ``None`` if the model has no equations or variable map.
        """
        if not self._equations or not self._variable_map:
            return None
        from zoomy_core.model.derivation.tag_extraction import (
            collect_solver_tag,
            canonical_solver_tag,
        )
        canonical = canonical_solver_tag(canonical_tag)
        state_atoms = [v for v in self.variables.values()
                       if isinstance(v, sp.Symbol)]
        coords = [self.position[d] for d in range(self.dimension)]
        n_dir = self.dimension if canonical in (
            "flux", "hydrostatic_pressure", "nonconservative_flux") else 1
        # `collect_solver_tag` walks `system._equations` natively (we
        # wrap self in an adapter only because the legacy signature
        # expected `.equations` dict-attr — Model exposes `_equations`
        # now).
        class _EqAdapter:
            """Exposes the equation dict under the ``equations`` attribute expected by
            ``collect_solver_tag``.
            """
            equations = self._equations
        return collect_solver_tag(
            _EqAdapter(), canonical,
            variable_map=self._variable_map,
            n_variables=self.n_variables,
            n_directions=n_dir,
            state_variables=state_atoms,
            coords=coords,
            policy="warn",
        )

    def flux(self):
        """Flux ``F(Q, Qaux, p)``, rank-2 ``ZArray`` ``(n_eq, n_dim)``; by default tag-extracted
        from the equations, else zeros.
        """
        raw = self._extract_via_tag("flux")
        if raw is None:
            return ZArray.zeros(self.n_variables, self.dimension)
        return ZArray(raw)

    def diffusion_matrix(self):
        """Implicitly treated diffusion tensor ``A(Q, Qaux, p)`` of shape
        ``(n_var, n_var, dim, dim)`` with ``F_diff[i, d] = sum_{j,e} A[i,j,d,e] d_e Q[j]``.
        Default is zero; see :meth:`diffusion_matrix_explicit` for the explicit slot.
        """
        return ZArray.zeros(
            self.n_variables, self.n_variables,
            self.dimension, self.dimension,
        )

    def diffusion_matrix_explicit(self):
        """Explicitly treated diffusion tensor with the same shape and contract as
        :meth:`diffusion_matrix` (subject to the parabolic CFL); default zero.
        """
        return ZArray.zeros(
            self.n_variables, self.n_variables,
            self.dimension, self.dimension,
        )

    def dflux(self):
        """Legacy flux derivative slot; zeros."""
        return ZArray.zeros(self.n_variables, self.dimension)
    
    def hydrostatic_pressure(self):
        """Hydrostatic pressure flux ``P(Q, Qaux, p)``, rank-2 ``(n_eq, n_dim)``; tag-extracted,
        else zeros.
        """
        raw = self._extract_via_tag("hydrostatic_pressure")
        if raw is None:
            return ZArray.zeros(self.n_variables, self.dimension)
        return ZArray(raw)

    def nonconservative_matrix(self):
        """Nonconservative matrix ``B(Q, Qaux, p)``, rank-3 ``(n_eq, n_state, n_dim)``;
        tag-extracted, else zeros.
        """
        raw = self._extract_via_tag("nonconservative_flux")
        if raw is None:
            return ZArray.zeros(self.n_variables, self.n_variables, self.dimension)
        return ZArray(raw)

    def source(self):
        """Implicitly treated source (friction, stiff terms), length ``n_variables``;
        tag-extracted, else zeros.
        """
        raw = self._extract_via_tag("implicit_source")
        if raw is None:
            return ZArray.zeros(self.n_variables)
        return ZArray(raw)

    def source_explicit(self):
        """Explicitly treated source (non-stiff body forces), length ``n_variables``;
        tag-extracted, else zeros.
        """
        raw = self._extract_via_tag("explicit_source")
        if raw is None:
            return ZArray.zeros(self.n_variables)
        return ZArray(raw)

    def mass_matrix(self):
        """Mass matrix ``M[i, j]`` = coefficient of ``d_t Q[j]`` in equation ``i``, extracted
        from ``time_derivative``-tagged terms. Rows without a time derivative (elliptic
        constraints) are zero, so ``M`` is singular for such models.
        """
        raw = self._extract_via_tag("time_derivative")
        if raw is None:
            return ZArray.zeros(self.n_variables, self.n_variables)
        state_atoms = [v for v in self.variables.values()
                       if isinstance(v, sp.Symbol)]
        n_eq = self.n_variables
        n_state = len(state_atoms)
        M = sp.zeros(n_eq, n_state)
        t = self.time
        for i, row_expr in enumerate(raw):
            if row_expr == 0:
                continue
            expanded = sp.expand(row_expr)
            for j, q_j in enumerate(state_atoms):
                coef = expanded.coeff(sp.Derivative(q_j, t))
                if coef != 0:
                    M[i, j] = coef
        return ZArray(M)

    def residual(self):
        """Residual slot; zeros."""
        return ZArray.zeros(self.n_variables)

    def interpolate(self):
        """Interpolation slot; the state itself."""
        return ZArray(self.variables)

    def interpolate_to_3d(self):
        """Vertical reconstruction ``{profile_slot: expr}``, taken from the derivation
        byproduct, else zeros.
        """
        rows = _derivation_byproduct(self, "interpolate_rows", "_interpolate_rows")
        if rows is not None:
            return rows
        return ZArray.zeros(6)

    def project_from_3d(self):
        """Profile-to-state projection ``{state_field_or_slot: expr}``; see
        :meth:`interpolate_to_3d`.
        """
        rows = _derivation_byproduct(self, "project_rows", "_project_rows")
        if rows is not None:
            return rows
        return ZArray.zeros(self.n_variables)

    def initial_condition(self):
        """Initial state expression; zeros."""
        return ZArray.zeros(self.n_variables)

    def initial_aux_condition(self):
        """Initial aux-state expression; zeros."""
        return ZArray.zeros(self.n_aux_variables)

    def update_variables(self):
        """Per-cell state update applied after each step; identity."""
        return ZArray(self.variables)

    def update_aux_variables(self):
        """Per-cell aux update applied after each step; identity."""
        return ZArray(self.aux_variables)

    def reconstruction_variables(self):
        """Symbolic map from the state to the (primitive, well-balanced) variables the MUSCL
        reconstruction limits; identity by default. Override to limit primitives such as
        ``eta = h + b`` and ``u = q/h`` instead of ``(h, q)``.

        Returns:
            ``ZArray`` of length ``n_variables``.
        References:
            Audusse et al. 2004, https://doi.org/10.1137/S1064827503431090
        """
        rows = _derivation_byproduct(self, "reconstruction_rows",
                                     "_reconstruction_rows")
        if rows is not None:
            return rows
        return ZArray(self.variables)

    def state_from_reconstruction(self):
        """Symbolic inverse of :meth:`reconstruction_variables`, auto-derived by
        ``invert_reconstruction`` in terms of ``WB_<state>`` symbols; override only if sympy
        cannot close the inverse.
        """
        from zoomy_core.model.reconstruction_inverse import invert_reconstruction
        return invert_reconstruction(
            self.reconstruction_variables(),
            list(self.variables.get_list()),
        )

    def update_variables_jacobian_wrt_variables(self):
        """Jacobian of ``update_variables`` with respect to the state."""
        if self.disable_differentiation:
            return ZArray.zeros(self.n_variables, self.n_variables)
        return self._simplify(
            sp.derive_by_array(self.update_variables(), self.variables.get_list())
        )

    def update_aux_variables_jacobian_wrt_variables(self):
        """Jacobian of ``update_aux_variables`` with respect to the state."""
        if self.disable_differentiation:
            return ZArray.zeros(self.n_aux_variables, self.n_variables)
        return self._simplify(
            sp.derive_by_array(self.update_aux_variables(), self.variables.get_list())
        )

    def quasilinear_matrix(self):
        """Quasilinear matrix ``dF/dQ + dP/dQ + B`` of shape ``(n_var, n_var, dim)``."""
        if self.disable_differentiation:
            return ZArray.zeros(self.n_variables, self.n_variables, self.dimension)
        JacF = ZArray(sp.derive_by_array(self.flux(), self.variables.get_list()))
        for d in range(self.dimension):
            JacF[:, :, d] = ZArray(JacF[:, :, d].tomatrix().T)
        JacP = ZArray(sp.derive_by_array(self.hydrostatic_pressure(), self.variables.get_list()))
        for d in range(self.dimension):
            JacP[:, :, d] = ZArray(JacP[:, :, d].tomatrix().T)
        return self._simplify(JacF + JacP + self.nonconservative_matrix())

    def source_jacobian_wrt_variables(self):
        """Jacobian of the implicit source with respect to the state."""
        if self.disable_differentiation:
            return ZArray.zeros(self.n_variables, self.n_variables)
        return self._simplify(
            sp.derive_by_array(self.source(), self.variables.get_list())
        )

    def source_jacobian_wrt_aux_variables(self):
        """Jacobian of the implicit source with respect to the aux variables."""
        if self.disable_differentiation:
            return ZArray.zeros(self.n_variables, self.n_aux_variables)
        return self._simplify(
            sp.derive_by_array(self.source(), self.aux_variables.get_list())
        )

    def eigenvalues(self):
        """Eigenvalues of the normal-projected quasilinear matrix, from the characteristic
        polynomial; a placeholder of zeros in ``'numerical'`` mode.
        """
        if self.eigenvalue_mode == "numerical":
            return ZArray([sp.Integer(0)] * self.n_variables)
        q_mat = self.quasilinear_matrix()
        A = self.normal[0] * q_mat[:, :, 0]
        for i in range(1, self.dimension):
            A += self.normal[i] * q_mat[:, :, i]
        A = A.tomatrix()
        lam = sp.symbols("lam")
        char_poly = A.charpoly(lam)
        evs = sp.solve(char_poly, lam)
        return ZArray([self._simplify(ev) for ev in evs])

    def left_eigenvectors(self):
        """Left eigenvector slot; zeros."""
        return ZArray.zeros(self.n_variables, self.n_variables)

    def right_eigenvectors(self):
        """Right eigenvector slot; zeros."""
        return ZArray.zeros(self.n_variables, self.n_variables)

    def eigensystem(self):
        """Packed eigendecomposition ``[Lambda(n), R(n*n), L(n*n)]`` of the normal-projected
        quasilinear matrix, as opaque ``eigensystem`` kernel calls evaluated numerically in the
        backends.
        """
        from zoomy_core.model.kernel_functions import (
            eigensystem as _es, pick as _pick)
        n = self.n_variables
        qm = self.quasilinear_matrix()
        A_flat = [
            sum(qm[i, j, d] * self.normal[d] for d in range(self.dimension))
            for i in range(n) for j in range(n)
        ]
        stack = _es(*A_flat)
        return ZArray([_pick(stack, sp.Integer(idx))
                       for idx in range(n + 2 * n * n)])

    def print_model_functions(self, function_names=None):
        """Print the definitions of the named model functions (default: flux,
        nonconservative_matrix, source, residual).
        """
        if function_names is None:
            function_names = ["flux", "nonconservative_matrix", "source", "residual"]

        print("=" * 80, flush=True)
        print(f"Model: {self.__class__.__name__}", flush=True)
        print("variables:", self.variables.keys(), flush=True)
        print("aux_variables:", self.aux_variables.keys(), flush=True)
        if hasattr(self, "derivative_specs"):
            specs = [f"{s.field}:{''.join(s.axes)}" for s in self.derivative_specs]
            print("derivative_specs:", specs, flush=True)
        print("-" * 80, flush=True)

        available = set(self.functions.keys()) if hasattr(self, "functions") else set()
        for name in function_names:
            if name not in available:
                continue
            print(f"{name}:", flush=True)
            print(self.functions[name].definition, flush=True)
            print("-" * 80, flush=True)

    def summarize_model(self, tex=False):
        """Model summary with the PDE form, ``Q``, ``Qaux``, ``F``, ``P``, ``B``, ``S``,
        eigenvalues, parameters and configuration.

        Args:
            tex: return sympy objects as LaTeX strings.
        Returns:
            dict keyed by those names.
        """
        fmt = (lambda e: sp.latex(e)) if tex else (lambda e: e)

        Q_vec = sp.Matrix(list(self.variables.values()))

        F_def = self.flux() if hasattr(self, "functions") and "flux" in self.functions.keys() else self.flux()
        P_def = self.hydrostatic_pressure()
        S_def = self.source()

        B_defs = {}
        nc = self.nonconservative_matrix()
        dim_labels = ["x", "y", "z"][:self.dimension]
        for d in range(self.dimension):
            B_d = sp.Matrix([[nc[r, c, d] for c in range(self.n_variables)]
                             for r in range(self.n_variables)])
            B_defs[dim_labels[d]] = B_d

        has_nc = any(B_defs[k] != sp.zeros(self.n_variables) for k in B_defs)
        has_source = not all(S_def[i] == 0 for i in range(len(S_def)))
        has_pressure = not all(P_def[i] == 0 for i in range(len(P_def._array)))

        pde_parts = [r"\partial_t \mathbf{Q}"]
        for d, label in enumerate(dim_labels):
            pde_parts.append(rf"+ \partial_{label} \mathbf{{F}}_{label}(\mathbf{{Q}})")
        if has_pressure:
            for d, label in enumerate(dim_labels):
                pde_parts.append(rf"+ \partial_{label} \mathbf{{P}}_{label}(\mathbf{{Q}})")
        if has_nc:
            for d, label in enumerate(dim_labels):
                pde_parts.append(rf"+ \mathbf{{B}}_{label}(\mathbf{{Q}}) \partial_{label} \mathbf{{Q}}")
        pde_parts.append(r"= \mathbf{S}(\mathbf{Q})" if has_source else r"= \mathbf{0}")
        pde_form = " ".join(pde_parts)

        eig_mode = getattr(self, "eigenvalue_mode", "symbolic")
        if eig_mode == "numerical":
            eig_summary = "numerical (np.linalg.eigvals at runtime)"
            eig_list = None
        else:
            try:
                evs = self.functions.eigenvalues.definition if hasattr(self, "functions") and "eigenvalues" in self.functions.keys() else self.eigenvalues()
                eig_list = list(evs)
                eig_summary = "symbolic"
            except Exception:
                eig_list = None
                eig_summary = "symbolic (failed to compute)"

        params = dict(self.parameter_values.as_dict(recursive=False))

        config = {
            "class": self.__class__.__name__,
            "dimension": self.dimension,
            "n_variables": self.n_variables,
            "n_aux_variables": self.n_aux_variables,
            "n_parameters": self.n_parameters,
            "eigenvalue_mode": eig_mode,
        }
        for attr in ["n_layers", "level", "basis_type"]:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if hasattr(val, "name"):
                    config[attr] = val.name
                elif hasattr(val, "__name__"):
                    config[attr] = val.__name__
                else:
                    config[attr] = val
        if hasattr(self, "basisfunctions"):
            config["basis_name"] = self.basisfunctions.name

        F_mat = sp.Matrix([[F_def[r, d] for d in range(self.dimension)]
                           for r in range(self.n_variables)]) if F_def.rank() == 2 else sp.Matrix(list(F_def))
        P_mat = sp.Matrix([[P_def[r, d] for d in range(self.dimension)]
                           for r in range(self.n_variables)]) if P_def.rank() == 2 else sp.Matrix(list(P_def))

        result = {
            "pde_form": pde_form,
            "Q": fmt(Q_vec),
            "Qaux": fmt(sp.Matrix(list(self.aux_variables.values()))) if self.n_aux_variables > 0 else None,
            "F": fmt(F_mat),
            "P": fmt(P_mat) if has_pressure else None,
            "B": {k: fmt(v) for k, v in B_defs.items()} if has_nc else None,
            "S": fmt(sp.Matrix(list(S_def))) if has_source else None,
            "eigenvalues": [fmt(e) for e in eig_list] if eig_list else eig_summary,
            "parameters": params,
            "config": config,
        }
        return result


class ModelDescription:
    """Markdown / plain-text rendering of a :class:`Model` returned by :meth:`Model.describe`."""

    def __init__(self, model, *, show_history=False, include_minor=False):
        """Store the model and the history options."""
        self._model = model
        self._show_history = show_history
        self._include_minor = include_minor

    # ── Jupyter LaTeX path ──────────────────────────────────────────
    def _repr_markdown_(self):
        """Markdown with LaTeX equations for Jupyter."""
        m = self._model
        n_eq = len(m._equations)
        n_ops = len(m.history)
        parts = [
            f"**{type(m).__name__}** `{m.name}` — "
            f"{n_eq} equation{'s' if n_eq != 1 else ''}, "
            f"{n_ops} op{'s' if n_ops != 1 else ''}"
        ]
        # State + parameters one-line summaries (only if non-empty).
        if hasattr(m, "variables") and m.variables.length():
            state_syms = ", ".join(
                f"${sp.latex(v)}$" for v in m.variables.values()
            )
            parts.append(f"**State $Q$:** {state_syms}")
        if hasattr(m, "parameters") and m.parameters.length():
            param_kv = ", ".join(
                f"${sp.latex(m.parameters[k])} = "
                f"{getattr(m.parameter_values, k, '?')}$"
                for k in m.parameters.keys()
            )
            parts.append(f"**Parameters:** {param_kv}")
        # Equations as LaTeX, one per line.
        if n_eq:
            parts.append("**Equations:**")
            for name, eq in m._equations.items():
                latex = sp.latex(eq.expr)
                parts.append(
                    f"- `{name}`: $\\displaystyle {latex} \\;=\\; 0$"
                )
        # Optional history.
        if self._show_history and m.history:
            parts.append("**History:**")
            for h in m.history:
                if not self._include_minor and h.get("level") == "minor":
                    continue
                desc = (f" — {h['description']}"
                        if h.get("description") else "")
                parts.append(
                    f"- `[{h['op']}]` target=`{h['target']}`{desc}"
                )
        return "\n\n".join(parts)

    # ── Plain-text fallback ─────────────────────────────────────────
    def __str__(self):
        """Plain-text fallback."""
        m = self._model
        lines = [f"{type(m).__name__}({m.name!r}) — "
                 f"{len(m._equations)} equations, "
                 f"{len(m.history)} ops"]
        for name, eq in m._equations.items():
            lines.append(f"  {name}  :  {eq.expr}  =  0")
        if self._show_history and m.history:
            lines.append("")
            lines.append("history:")
            for h in m.history:
                if not self._include_minor and h.get("level") == "minor":
                    continue
                desc = (f" — {h['description']}"
                        if h.get("description") else "")
                lines.append(f"  [{h['op']}] target={h['target']}{desc}")
        return "\n".join(lines)

    def __repr__(self):
        """Same as ``str``."""
        return self.__str__()

