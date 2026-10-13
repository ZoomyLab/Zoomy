"""Module `zoomy_core.transformation.to_numpy`."""

from typing import Callable, Dict, Optional
import sympy as sp
import numpy as np

from zoomy_core.misc.custom_types import FArray
from zoomy_core.misc.misc import Zstruct
from zoomy_core.model.basefunction import Function
from zoomy_core.model.basemodel import Model
from zoomy_core.fvm import userfunctions as _uf
from zoomy_core.transformation.vectorize import uniform_rank


# Canonical time-step symbol — the trailing argument of the per-cell update
# kernels (``update_variables`` / ``update_aux_variables``).  Same name +
# assumptions as the splitter / VAM ``chorin_split(dt)`` symbol, so it compares
# equal under sympy and resolves the corrector's baked ``dt``.
DT_SYMBOL = sp.Symbol("dt", positive=True)

# The numpy UserFunctions kernels live in ``zoomy_core.fvm.userfunctions`` (the
# python mirror of the C++ ``UserFunctions.H``, REQ-168).  Re-exported here for
# back-compat with callers that imported them from this module.
_eigensystem_numpy = _uf.eigensystem
_eigenvalues_numpy = _uf.eigenvalues
_solve_numpy = _uf.solve


def _broadcast_array(x, *args, **kwargs):
    """`numpy.array` that BROADCASTS its entries to a common shape: the
    vectorised boundary kernels stack per-face rows next to scalar constants
    (a Dirichlet value, a time-only expression), which plain `np.array`
    rejects as an inhomogeneous shape."""
    if isinstance(x, (list, tuple)):
        rows = [_broadcast_array(e) for e in x]
        if not rows:
            return np.array(rows, *args, **kwargs)
        return np.array(np.broadcast_arrays(*rows), *args, **kwargs)
    return np.asarray(x, *args, **kwargs)


def _branch_groups(bc_idx):
    """Static `[(branch, face positions), …]` of an integer branch vector,
    plus the inverse permutation that restores face order after the groups
    are concatenated."""
    bc_idx = np.asarray(bc_idx, dtype=int).ravel()
    groups = [(int(k), np.flatnonzero(bc_idx == k))
              for k in np.unique(bc_idx)]
    order = (np.concatenate([g for _k, g in groups]) if groups
             else np.zeros(0, dtype=int))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return groups, inverse


class _TemplateSlotsMixin:
    """New-template slots of the numpy runtime (lazy, built on first use so
    the old solvers never pay for them): `admissible` and the D5/D6
    boundary GHOST kernels."""

    def _template_model(self):
        sm = getattr(self, "model", None)
        if sm is None or not hasattr(sm, "boundary_ghost_kernel"):
            raise TypeError("the template slots need a runtime built from a "
                            "SystemModel/NSM (NumpyRuntimeModel.from_system_model)")
        return sm

    def _template_modules(self, broadcast=False):
        module = dict(self.module)
        if broadcast:
            module["array"] = _broadcast_array
        mods = [module]
        if self.printer:
            mods.append(self.printer)
        return mods

    # ── admissibility ────────────────────────────────────────────────
    def admissible(self, Q, Qaux, parameters):
        """The admissibility list evaluated per cell: ``(n_constraints,
        n_cells)` — every row must stay `>= 0`` (decision D1).  ``(0,
        n_cells)`` for a system that declares none."""
        fn = getattr(self, "_admissible_fn", None)
        if fn is None:
            sm = self._template_model()
            exprs = list(sm.admissible_constraints())
            if not exprs:
                def fn(Q, Qaux, p):
                    Q = np.asarray(Q)
                    return np.zeros((0,) + Q.shape[1:])
            else:
                kernel = Function(name="admissible",
                                  args=sm.operator_signature("admissible"),
                                  definition=sp.Array(exprs))
                lowered = self._lambdify_function(
                    kernel, self._template_modules(broadcast=True))

                def fn(Q, Qaux, p, _f=lowered, _n=len(exprs)):
                    Q = np.asarray(Q, dtype=float)
                    out = np.asarray(_f(Q, Qaux, p), dtype=float)
                    return out.reshape((_n,) + Q.shape[1:])
            self._admissible_fn = fn
        return fn(Q, Qaux, parameters)

    # ── boundary ghost state (D5) + ghost aux (D6) ───────────────────
    def _ghost_lowered(self):
        cached = getattr(self, "_ghost_cache", None)
        if cached is not None:
            return cached
        sm = self._template_model()
        fn, branches = sm.boundary_ghost_kernel()
        mods = self._template_modules(broadcast=True)
        whole = self._lambdify_function(fn, mods)
        a = fn.args
        branch_args = Zstruct(**{k: a[k] for k in a.keys() if k != "idx"})
        lowered = [self._lambdify_function(
            Function(name=f"boundary_ghost_state__branch{i}",
                     args=branch_args, definition=br), mods)
            for i, br in enumerate(branches)]
        self._ghost_cache = (whole, lowered, len(sm.state))
        return self._ghost_cache

    def boundary_ghost_state(self, bc_idx, time, position, distance, Q, Qaux,
                             parameters, normal, bdata=None):
        """The indexed D5 ghost kernel, evaluated as ONE vectorised Piecewise:
        works per face (scalar `bc_idx`, `(n,)` state) and batched over
        faces (`bc_idx` `(n_f,)`, state `(n, n_f)`).  `bdata=None`
        means the BC input state (extrapolation fallback of coupled faces)."""
        whole, _br, n = self._ghost_lowered()
        Q = np.asarray(Q, dtype=float)
        bdata = Q if bdata is None else np.asarray(bdata, dtype=float)
        out = np.asarray(whole(bc_idx, time, position, distance, Q, Qaux,
                               parameters, normal, bdata), dtype=float)
        return out.reshape(Q.shape)

    def boundary_ghost(self, bc_idx):
        """BATCHED ghost kernel for a STATIC branch vector `bc_idx` (one
        entry per boundary face, routed by tag name — see
        `zoomy_core.model.boundary_conditions.boundary_face_branches`).

        Returns ``ghost(time, position, distance, Q_in, Qaux_in, parameters,
        normal, bdata=None) -> (Q_ghost, Qaux_ghost)`` with every face array
        carrying a trailing face axis (`position` `(3, n_f)`, `distance`
        `(n_f,)`, `Q_in` `(n, n_f)`, `normal` `(≥dim, n_f)`).  Each
        BC branch is evaluated ONCE on its own faces (no Python loop over
        faces, no evaluation of a branch on foreign faces).  The ghost aux is
        the model's own aux update rule applied to the ghost state (D6); aux
        rows that rule does not define keep the input value."""
        _whole, lowered, n = self._ghost_lowered()
        groups, inverse = _branch_groups(bc_idx)
        uav = getattr(self, "update_aux_variables", None)

        def ghost(time, position, distance, Q, Qaux, parameters, normal,
                  bdata=None):
            Q = np.asarray(Q, dtype=float)
            Qaux = np.asarray(Qaux, dtype=float)
            position = np.asarray(position, dtype=float)
            normal = np.asarray(normal, dtype=float)
            distance = np.asarray(distance, dtype=float)
            bdata = Q if bdata is None else np.asarray(bdata, dtype=float)
            parts = []
            for k, sel in groups:
                out = lowered[k](time, position[:, sel], distance[sel],
                                 Q[:, sel], Qaux[:, sel], parameters,
                                 normal[:, sel], bdata[:, sel])
                parts.append(np.asarray(_broadcast_array(list(out)),
                                        dtype=float).reshape(n, sel.size))
            Qg = (np.concatenate(parts, axis=1)[:, inverse] if parts
                  else np.zeros_like(Q))
            if uav is None or Qaux.shape[0] == 0:
                return Qg, Qaux
            Ag = np.asarray(uav(Qg, Qaux, parameters, time, position),
                            dtype=float).reshape(Qaux.shape)
            return Qg, Ag

        return ghost


class NumpyRuntimeModel(_TemplateSlotsMixin):
    """Runtime model generated from a symbolic Model.

    Instead of assuming hardcoded attributes (e.g. `_flux`), this class
    compiles all functions registered in `model.functions` and exposes them
    as callable runtime attributes.
    """

    # --- Constants ---
    # Built from the numpy UserFunctions table (REQ-168): arithmetic helpers +
    # the backend-supplied kernel contract (compute_derivative / eigensystem /
    # eigenvalues / solve).  compute_derivative is None here — the solver injects
    # the mesh-bound impl (mesh.compute_derivatives) before update_aux_variables
    # is compiled.
    module = _uf.numpy_module()
    printer = "numpy"
    
    def _flatten_signature_args(self, arg_struct):
        """Internal helper `_flatten_signature_args`."""
        flat_args = []

        def _flatten(value):
            """Internal helper `_flatten`."""
            if hasattr(value, "values") and callable(value.values):
                for item in value.values():
                    _flatten(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    _flatten(item)
            else:
                flat_args.append(value)

        _flatten(arg_struct)
        return flat_args

    def _lower_opaque_kernels(self, expr):
        """Backend seam for the opaque-kernel calling convention (no-op for
        numpy / jax).

        The producers emit every component read of the opaque kernels in the
        CANONICAL `pick(K(*a), idx)` form (K ∈ {eigensystem, eigenvalues,
        solve}).  numpy and jax register `K` (returns the whole packed stack)
        and `pick` (trailing-axis index read) in their module dict, so the
        form lowers directly and lambdify's cse hoists the shared `K(*a)` node
        to ONE call per face — no rewrite needed here.  The per-component
        backends override this seam: :meth:`to_ufl.UFLRuntimeModel.
        _lower_opaque_kernels` restores the `K(idx, *a)` form their UFL
        bindings expect."""
        return expr

    def _lambdify_function(self, function_obj, modules):
        """Internal helper `_lambdify_function`."""
        args = self._flatten_signature_args(function_obj.args)
        expr = self._vectorize_expression(function_obj.definition, function_obj.args)
        expr = self._lower_opaque_kernels(expr)

        use_cse = getattr(self, 'use_cse', True)
        try:
            compiled = sp.lambdify(args, expr, modules=modules, cse=use_cse)
        except (TypeError, Exception):
            compiled = sp.lambdify(args, expr, modules=modules)

        fast_flatten = self._compile_flattener(function_obj.args)
        # REQ-185: ``source`` / ``update_aux_variables`` carry trailing
        # coordinate/time groups (time, [dt], position).  A caller with no
        # coordinate context (autonomous evaluation — e.g. the DAE/IMEX solvers
        # or a Newton residual) may omit them; the missing trailing groups
        # default to the coordinate origin (t=0, x=y=z=0), which is EXACT for
        # operators that do not reference them.  ``0.0`` flattens onto every
        # leaf of a missing group (the scalar time, or the three position
        # components) via ``fast_flatten``.
        sig_vals = (function_obj.args.values()
                    if hasattr(function_obj.args, "values")
                    else function_obj.args)
        n_groups = len(list(sig_vals))

        def runtime_callable(*runtime_args):
            """Runtime callable."""
            if len(runtime_args) < n_groups:
                runtime_args = runtime_args + (0.0,) * (n_groups - len(runtime_args))
            return compiled(*fast_flatten(runtime_args))

        return runtime_callable

    @staticmethod
    def _compile_flattener(signature):
        """
        Pre-compute the arg extraction plan at compile time.

        Returns a fast closure that converts runtime args to a flat list
        using only integer indexing — no Zstruct iteration at runtime.
        """
        plan = []

        def _build_plan(expected, arg_idx, path):
            if hasattr(expected, "values") and callable(expected.values):
                keys = list(expected.keys()) if hasattr(expected, "keys") else None
                for i, child in enumerate(expected.values()):
                    child_path = path + (("key", keys[i], i),)
                    _build_plan(child, arg_idx, child_path)
            elif isinstance(expected, (list, tuple)):
                for i, child in enumerate(expected):
                    _build_plan(child, arg_idx, path + (("idx", None, i),))
            else:
                plan.append((arg_idx, path))

        sig_values = list(signature.values()) if hasattr(signature, "values") else list(signature)
        for arg_idx, expected in enumerate(sig_values):
            _build_plan(expected, arg_idx, ())

        def _safe_hasattr(obj, name):
            """`hasattr` shadowed by `getattr` with broad exception
            handling — UFL exposes `.T` as a property that raises
            `ValueError` on rank-1 tensors, and Python's `hasattr`
            returns `False` only on AttributeError, propagating
            everything else.  Treat any exception as 'attribute not
            usable here'."""
            try:
                getattr(obj, name)
                return True
            except Exception:
                return False

        def fast_flatten(runtime_args):
            result = []
            for arg_idx, path in plan:
                val = runtime_args[arg_idx]
                for step_type, step_key, step_idx in path:
                    if isinstance(val, np.ndarray):
                        val = val[step_idx]
                    # ARRAYS INDEX POSITIONALLY, never by attribute: a state
                    # field named like an array attribute (``T``, ``real``,
                    # ``at``, ``size`` …) would otherwise pick ``Q.T`` off a
                    # jax array instead of ``Q[i]`` (the jax BC-kernel crash
                    # for a state named ``T``).  Keyed access is for
                    # struct-like containers (Zstruct) only.
                    elif (step_type == "key" and not hasattr(val, "shape")
                          and _safe_hasattr(val, step_key)):
                        val = getattr(val, step_key)
                    else:
                        try:
                            val = val[step_idx]
                        except (TypeError, IndexError, KeyError):
                            pass
                result.append(val)
            return result

        return fast_flatten

    def _flatten_runtime_args(self, signature, runtime_args):
        """Internal helper `_flatten_runtime_args`."""
        expected_args = signature.values() if hasattr(signature, "values") else signature
        expected_args = list(expected_args)
        if len(runtime_args) != len(expected_args):
            raise TypeError(
                f"Expected {len(expected_args)} runtime args, got {len(runtime_args)}"
            )

        flat = []
        for expected, value in zip(expected_args, runtime_args):
            flat.extend(self._flatten_runtime_value(expected, value))
        return flat

    def _flatten_runtime_value(self, expected, value):
        """Internal helper `_flatten_runtime_value`."""
        if hasattr(expected, "values") and callable(expected.values):
            out = []
            keys = list(expected.keys()) if hasattr(expected, "keys") else None
            for i, child in enumerate(expected.values()):
                child_value = self._extract_component(value, i, keys[i] if keys else None)
                out.extend(self._flatten_runtime_value(child, child_value))
            return out

        if isinstance(expected, (list, tuple)):
            out = []
            for i, child in enumerate(expected):
                child_value = self._extract_component(value, i, None)
                out.extend(self._flatten_runtime_value(child, child_value))
            return out

        return [value]

    def _collect_vector_symbols(self, signature):
        """Internal helper `_collect_vector_symbols`."""
        vector_symbols = []
        # Prefix match: the Riemann kernels carry two-sided state groups
        # (``variables_minus`` / ``variables_plus`` / ``aux_variables_*`` /
        # ``normal``); they broadcast over faces exactly like the std groups
        # broadcast over cells, so their constants need the same
        # ones_like/zeros_like wrapping.  ``parameters`` stays scalar.
        vector_bases = ("variables", "aux_variables", "normal", "position",
                        "q", "aux")    # q_minus/q_plus etc. (Riemann kernels)

        def _is_vector_key(key):
            return key is not None and any(
                key == b or key.startswith(b + "_") for b in vector_bases)

        def _collect(node, key=None, in_vector_context=False):
            """Internal helper `_collect`."""
            context = in_vector_context or _is_vector_key(key)
            if hasattr(node, "values") and callable(node.values):
                keys = list(node.keys()) if hasattr(node, "keys") else [None] * len(node.values())
                for child_key, child_value in zip(keys, node.values()):
                    _collect(child_value, key=child_key, in_vector_context=context)
            elif isinstance(node, (list, tuple)) or (
                    not isinstance(node, sp.Basic)
                    and hasattr(node, "__iter__")):
                # ZArray groups (the Riemann kernels' q_minus/q_plus/...)
                # iterate their member symbols without exposing .values()
                for child_value in node:
                    _collect(child_value, key=None, in_vector_context=context)
            else:
                if context and isinstance(node, sp.Basic):
                    vector_symbols.append(node)

        _collect(signature)
        return tuple(vector_symbols)

    def _get_anchor_symbol(self, signature):
        """Internal helper `_get_anchor_symbol`."""
        if hasattr(signature, "contains") and signature.contains("variables"):
            variables = signature["variables"]
            if hasattr(variables, "values") and len(variables.values()) > 0:
                return list(variables.values())[0]

        vector_symbols = self._collect_vector_symbols(signature)
        return vector_symbols[0] if vector_symbols else None

    def _vectorize_expression(self, expr, signature):
        """Internal helper `_vectorize_expression`."""
        if not (hasattr(expr, "tolist") and callable(expr.tolist)):
            return expr
        shape = getattr(expr, "shape", ())
        if shape and any(int(s) == 0 for s in shape):
            # SymPy currently fails tolist() for empty dimensions.
            # Return an explicit SymPy empty array so lambdify can handle it.
            return sp.Array([], tuple(int(s) for s in shape))

        arr = sp.Array(expr.tolist())
        vector_symbols = self._collect_vector_symbols(signature)
        anchor = self._get_anchor_symbol(signature)
        # REQ-84: shared constant-entry rank-normalization seam so every
        # vector backend (numpy here, jax via jax_runtime) gets uniform-rank
        # rows by construction.  ``uniform_rank`` is a byte-for-byte lift of
        # the old inline loop.
        return uniform_rank(arr, vector_symbols, anchor)

    def _lower_ndarray_operator(self, name, arr, n_cols, n_eq, n_dim,
                                op_sig, modules):
        """Lower a rank-3 operator `arr(Q, Qaux, p)` of shape
        `(n_eq, n_cols, n_dim)` (NCP / quasilinear) to a runtime callable.

        Backend hook consumed by `from_system_model`.  The numpy
        lowering emits one `(n_eq, n_cols)` matrix per spatial axis
        (each routed through `_lambdify_function` so each slab is
        vectorised), then `np.stack`\\ s them along the last axis to
        rebuild the grid-broadcast `(n_eq, n_cols, n_dim)` array.  The
        UFL backend overrides this to emit a single `ufl.as_tensor`."""
        slab_fns = []
        for d in range(n_dim):
            slab = sp.Matrix(n_eq, n_cols, lambda i, j, _d=d: arr[i, j, _d])
            fn = Function(name=f"{name}__d{d}", args=op_sig, definition=slab)
            slab_fns.append(self._lambdify_function(fn, modules))

        # No hand-built arg list: the runtime forwards ``*op_args`` and the
        # slab callables bind them per the operator's DECLARED ``Function.args``
        # (``sm.operator_signature(name)``).
        def _runtime(*op_args, _slab_fns=slab_fns):
            slabs = [np.asarray(f(*op_args), dtype=float)
                     for f in _slab_fns]
            return np.stack(slabs, axis=-1)
        return _runtime

    def _lower_rank4_operator(self, name, A_arr, n_eq, n_st, n_dim,
                              op_sig, modules):
        """Lower a rank-4 constitutive tensor `A(Q, Qaux, p)` of shape
        `(n_eq, n_state, n_dim, n_dim)` (the `div(A : grad Q)`
        diffusion tensor) to a runtime callable.

        Backend hook consumed by `from_system_model`.  The numpy
        lowering emits one `(n_eq, n_state)` matrix per ``(d_flux,
        d_grad)`` pair, stacks along the trailing axis, and reshapes back
        to rank-4.  The UFL backend overrides this to emit a single
        `ufl.as_tensor`."""
        slab_fns_4d = []
        for d in range(n_dim):
            for e in range(n_dim):
                slab = sp.Matrix(
                    n_eq, n_st,
                    lambda i, j, _d=d, _e=e: A_arr[i, j, _d, _e],
                )
                fn = Function(name=f"{name}__d{d}_e{e}", args=op_sig,
                              definition=slab)
                slab_fns_4d.append(self._lambdify_function(fn, modules))

        # Same *op_args forwarding as ``_lower_ndarray_operator`` — the arg
        # list is the slab Functions' declared args, never re-listed here.
        def _runtime_A(*op_args, _fns=slab_fns_4d,
                       _n_dim=n_dim, _n_eq=n_eq, _n_st=n_st):
            slabs = [np.asarray(f(*op_args), dtype=float)
                     for f in _fns]
            stacked = np.stack(slabs, axis=-1)
            new_shape = stacked.shape[:-1] + (_n_dim, _n_dim)
            return stacked.reshape(new_shape)
        return _runtime_A

    @staticmethod
    def _extract_component(value, index, key):
        """Internal helper `_extract_component`."""
        if key is not None and hasattr(value, key):
            return getattr(value, key)
        try:
            return value[index]
        except Exception:
            return value

    def __init__(
        self,
        model: Model,
        module: Optional[Dict[str, Callable]] = None,
        printer: Optional[str] = None,
        kernel=None,
    ):
        """Initialize the instance."""
        # Dispatch on model style.  A legacy ``Model`` carries the
        # ``.functions`` registry that the body below lambdifies.  A new-style
        # BaseModel (e.g. the thesis ``MalpassetSME``) builds via
        # ``SystemModel.from_model(model)`` but has NO
        # ``.functions``/``.name``/``.variables`` — and a bare
        # ``SystemModel`` / ``NumericalSystemModel`` likewise.  For any of those,
        # build through the SystemModel/NSM path so ``RuntimeModel(model)`` (the
        # entry point every backend/solver already calls — e.g. Firedrake's
        # ``UFLRuntimeModel(model)``) works for every model style without each
        # caller choosing a factory.  ``type(self).from_system_model`` keeps the
        # backend-specific operator lowering (UFL emission for UFLRuntimeModel).
        # (REQ-90)
        if not hasattr(model, "functions"):
            built = type(self).from_system_model(
                model, module=module, printer=printer)
            self.__dict__.update(built.__dict__)
            return
        self.model = model
        self.name = model.name
        self.dimension = model.dimension
        self.n_variables = model.n_variables
        self.n_aux_variables = model.n_aux_variables
        self.n_parameters = model.n_parameters
        # Extract numeric parameter values from the symbolic model's Zstruct
        # at compile time.  Per the canonical naming convention,
        # ``model.parameter_values`` is the Zstruct of numeric floats
        # (``model.parameters`` is the Zstruct of sympy Symbols).
        self.parameters: FArray = np.array(list(model.parameter_values.values()), dtype=float)

        # Copy the class-level mapping to avoid shared mutable state.
        self.module = dict(type(self).module) if module is None else dict(module)
        self.printer = type(self).printer if printer is None else printer

        # If a Kernel is provided, compile its functions and merge into module
        if kernel is not None:
            kernel_rt = NumpyRuntimeSymbolic(kernel, module=self.module, printer=self.printer)
            for name, fn in kernel_rt.runtime_functions.items():
                self.module[name] = fn

        modules = [self.module]
        if self.printer:
            modules.append(self.printer)

        self.runtime_functions: Dict[str, Callable] = {}
        for name, function_obj in model.functions.items():
            self.runtime_functions[name] = self._lambdify_function(function_obj, modules)

        # Boundary condition wrappers are kept as separate members on the
        # symbolic model and are not part of model.functions.  They are
        # *required* — if the model has no ``_boundary_conditions`` /
        # ``_aux_boundary_conditions`` / ``_boundary_gradients``, the
        # access fails loudly (per the zoomy "prefer breaking over silent
        # skip" rule).
        self.runtime_functions["boundary_conditions"] = self._lambdify_function(
            model._boundary_conditions, modules
        )
        self.runtime_functions["aux_boundary_conditions"] = self._lambdify_function(
            model._aux_boundary_conditions, modules
        )
        self.runtime_functions["boundary_gradients"] = self._lambdify_function(
            model._boundary_gradients, modules
        )

        # Keep attribute-style access for existing solver code paths.
        for name, function in self.runtime_functions.items():
            setattr(self, name, function)

    @classmethod
    def from_system_model(cls, sm, *, module=None, printer=None):
        """Build a runtime by lambdifying a `SystemModel`'s
        stored matrices.

        Lightweight adapter that mirrors the operator-API surface of
        the runtime built from a `Model`: the resulting object has
        `flux`, `nonconservative_matrix`, `source`,
        `hydrostatic_pressure`, `mass_matrix` callable attributes,
        each accepting its DECLARED operator args (read off
        `sm.operator_signature(name)` — e.g. `(Q, Qaux, p)` for flux,
        `(Q, Qaux, p, t, [dt], x)` for source) and returning the
        corresponding numpy array.

        For most solver code paths the existing `NumpyRuntimeModel(model)`
        flow is still preferred because it carries the model's full
        function registry (`boundary_conditions` kernel etc.) — this
        factory is the right entry point when the analysis or
        transformation pipeline starts from a SystemModel that may not
        have a backing Model (e.g. SystemModel constructed from
        scratch).
        """
        # Normalise the entry: accept a Model, a SystemModel, or an NSM and
        # always lambdify from an NSM (the canonical front door).
        from zoomy_core.numerics.numerical_system_model import (
            to_numerical_system_model)
        sm = to_numerical_system_model(sm)
        rt = cls.__new__(cls)
        # The SystemModel *is* the symbolic model for this runtime —
        # solvers unwrap it via ``_get_symbolic_model`` to build their
        # operators.
        rt.model = sm
        rt.name = "SystemModelRuntime"
        rt.dimension = sm.n_dim
        rt.n_variables = sm.n_equations
        rt.n_aux_variables = len(sm.aux_state)
        rt.n_parameters = sm.parameters.length()
        rt.parameters = np.array(list(sm.parameter_values.values()),
                                 dtype=float)
        # REQ-190: standard numerical timestep cap, carried on the runtime so
        # the numpy/UFL "emit" exposes the same ``dt_max`` every other backend
        # reads (``sm`` is an NSM after the front-door coercion above).
        rt.dt_max = sm.dt_max
        rt.module = dict(cls.module) if module is None else dict(module)
        rt.printer = cls.printer if printer is None else printer

        modules = [rt.module]
        if rt.printer:
            modules.append(rt.printer)

        # Every operator is lambdified through ``_lambdify_function`` so
        # ``_vectorize_expression`` is applied first — that wraps every
        # constant entry in ``zeros_like(anchor)`` / ``c·ones_like(anchor)``
        # so the runtime broadcasts cleanly when called with full-grid
        # arrays (Q shape ``(n_vars, n_cells)`` → matrix output
        # ``(n_eq, n_state, n_cells)``).  Without this, IMEX
        # implicit-source / source-jacobian calls (and any vectorised
        # full-grid pattern) collapse to 2-D and break.
        #
        # Operator signatures (kwarg keys feed
        # ``_collect_vector_symbols``: ``variables`` / ``aux_variables``
        # / ``normal`` / ``position`` are the *vector* groups whose
        # symbols anchor the broadcast).
        p_struct = sm.parameters         # Zstruct of parameter Symbols
        # Operator signatures are DECLARED once on the SystemModel
        # (``OPERATOR_ARG_SLOTS``) and carried as ``Function.args``; this runtime
        # reads ``sm.operator_signature(name)`` and only handles the one numpy
        # lowering-SYNTAX constraint: ``dt`` cannot appear twice in a lambdify
        # arg list, so drop the declared ``dt`` group when a split sub-system
        # baked ``dt`` in as a model PARAMETER (Chorin/VAM).  ``time`` /
        # ``position`` (source, update_aux) and the trailing ``dt`` (source,
        # update_variables) all flow from the declaration — no local re-listing.
        dt_in_params = bool(getattr(p_struct, "contains", lambda *_: False)("dt"))

        def _op_sig(name):
            """Declared `args` for `name`, minus the `dt` group when
            `dt` is already a parameter (duplicate lambdify argument)."""
            args = sm.operator_signature(name)
            if dt_in_params and args.contains("dt"):
                args = Zstruct(**{k: args[k] for k in args.keys()
                                  if k != "dt"})
            return args

        rt.runtime_functions = {}

        def _register(name, definition, sig_name=None):
            """Route the operator through `_lambdify_function` with its
            DECLARED signature (read off the SystemModel Function)."""
            if definition is None:
                return
            fn = Function(name=name, args=_op_sig(sig_name or name),
                          definition=definition)
            rt.runtime_functions[name] = rt._lambdify_function(fn, modules)
            setattr(rt, name, rt.runtime_functions[name])

        def _column_to_rank1(mat):
            """`(n, 1)` Matrix → rank-1 `sp.Array(n)`.  `_to_matrix`
            in `from_model` coerces column-like ZArrays to `(n, 1)`;
            consumers (IMEX `S[:, c]`, the source operator broadcast
            pattern) expect rank-1 → `(n_eq, n_cells)` after
            vectorisation, not 3-D `(n_eq, 1, n_cells)`."""
            if mat is None:
                return None
            return sp.Array([mat[i, 0] for i in range(mat.shape[0])])

        _register("flux", sm.flux)
        _register("hydrostatic_pressure", sm.hydrostatic_pressure)
        # ``source(Q, Qaux, p, t, [dt], x)`` — the declaration carries time +
        # position always, and ``dt`` unless it collides with a model parameter
        # (handled by ``_op_sig``).  ``source_needs_dt`` tells the solver to
        # pass ``dt`` explicitly at call time (True iff ``dt`` survives as a
        # first-class source arg, i.e. it is NOT already a parameter).
        _register("source", _column_to_rank1(sm.source))
        rt.source_needs_dt = not dt_in_params
        _register("source_explicit",
                  _column_to_rank1(sm.source_explicit))
        _register("mass_matrix", sm.mass_matrix)
        _register("source_jacobian_wrt_variables",
                  sm.source_jacobian_wrt_variables)
        _register("source_jacobian_wrt_aux_variables",
                  sm.source_jacobian_wrt_aux_variables if sm.aux_state else None)
        # Flux twin of the above.  Emitted on the SAME terms so a consumer that
        # needs a total d/dx of the flux gets both halves -- without it,
        # dF/dQaux * dQaux/dx is silently dropped from any quasilinear or steady
        # balance assembled from the flux.
        _register("flux_jacobian_wrt_aux_variables",
                  sm.flux_jacobian_wrt_aux_variables if sm.aux_state else None)
        # ``update_variables(Q, Qaux, p, dt)`` — per-cell state remap.  For a
        # full model it returns the whole state; for a Chorin corrector
        # sub-system it returns one value per ``equation_to_state_index`` row
        # (the closed-form projection, where ``dt`` is load-bearing).  Same
        # broadcast pattern as ``source``.
        # When a split sub-system threads ``dt`` in as a MODEL PARAMETER
        # (``chorin_split(dt)``), the declared ``dt`` group is the SAME symbol as
        # the parameter, so ``_op_sig`` drops it (else lambdify raises
        # ``SyntaxError: duplicate argument 'dt'`` — REQ-151 defect C); the
        # parameter already supplies ``dt`` and the flattener ignores any
        # trailing runtime ``dt`` the solver still passes.
        _register("update_variables",
                  _column_to_rank1(sm.update_variables))
        _register("update_aux_variables",
                  _column_to_rank1(getattr(sm, "update_aux_variables", None)))
        _register("eigenvalues", _column_to_rank1(sm.eigenvalues))
        # The WB reconstruction pair — NSM operators, so the runtime lowers them
        # like any other (cid 214).  numpy previously lowered NEITHER, which is
        # why its solver hand-rolled FreeSurface*/Surface*/Primitive
        # reconstructions instead of using the map the model emitted.
        # Already rank-1 ``(n_state,)`` — NOT the ``(n, 1)`` column shape
        # ``eigenvalues`` carries — so they register as-is.
        _register("reconstruction_variables",
                  getattr(sm, "reconstruction_variables", None))
        _register("state_from_reconstruction",
                  getattr(sm, "state_from_reconstruction", None))
        # ``eigenvalues_cfl`` — the dt-only spectrum published by
        # ``gate_eigenvalues_dry`` (cid=209).  ``None`` when no gate was opted
        # into; the CFL consumer then falls back to ``eigenvalues``.  The
        # Rusanov dissipation always reads ``eigenvalues``.
        _register("eigenvalues_cfl",
                  _column_to_rank1(getattr(sm, "eigenvalues_cfl", None)))

        # NDimArray operators (NCP, quasilinear) — per-axis slab as a
        # Matrix, each routed through ``_lambdify_function`` (so each
        # slab is vectorised), then ``np.stack`` along the last axis.
        n_dim = sm.n_dim
        n_eq = sm.n_equations
        n_st = sm.n_state

        def _register_ndarray(name, arr, n_cols):
            fn = rt._lower_ndarray_operator(
                name, arr, n_cols, n_eq, n_dim, _op_sig(name), modules)
            rt.runtime_functions[name] = fn
            setattr(rt, name, fn)

        _register_ndarray("nonconservative_matrix",
                          sm.nonconservative_matrix, n_eq)
        _register_ndarray("quasilinear_matrix",
                          sm.quasilinear_matrix, n_st)

        # ``diffusion_matrix``: rank-4 ``A(Q, Qaux, p)`` of shape
        # ``(n_eq, n_state, n_dim, n_dim)`` — the constitutive tensor in
        # ``div(A : grad Q)``.  Lambdify per ``(d_flux, d_grad)`` slab as
        # an ``(n_eq, n_state)`` matrix and stack along the trailing two
        # axes.  Skipped when the SystemModel does not carry diffusion.
        def _register_rank4(name, A_arr):
            if A_arr is None:
                return
            fn = rt._lower_rank4_operator(
                name, A_arr, n_eq, n_st, n_dim, _op_sig(name), modules)
            rt.runtime_functions[name] = fn
            setattr(rt, name, fn)

        _register_rank4("diffusion_matrix", sm.diffusion_matrix)
        _register_rank4("diffusion_matrix_explicit", sm.diffusion_matrix_explicit)

        # Indexed boundary-condition kernels — lambdified via
        # ``_lambdify_function`` so the per-face call is
        # ``rt.boundary_conditions(bc_idx, time, position, distance,
        # Q_cell, Qaux_cell, parameters, normal) → q_face``.  The
        # SystemModel must carry both (per the zoomy rule: prefer
        # breaking over silent skip).
        rt.runtime_functions["boundary_conditions"] = rt._lambdify_function(
            sm.boundary_conditions, modules)
        rt.boundary_conditions = rt.runtime_functions["boundary_conditions"]
        rt.runtime_functions["aux_boundary_conditions"] = rt._lambdify_function(
            sm.aux_boundary_conditions, modules)
        rt.aux_boundary_conditions = rt.runtime_functions[
            "aux_boundary_conditions"]
        rt.runtime_functions["boundary_gradients"] = rt._lambdify_function(
            sm.boundary_gradients, modules)
        rt.boundary_gradients = rt.runtime_functions["boundary_gradients"]

        return rt


class NumpyRuntimeSymbolic(NumpyRuntimeModel):
    """
    Runtime wrapper for generic symbolic registrars (e.g. Numerics).

    Compiles all entries from `symbolic_obj.functions` using the same
    lambdify/argument-flattening machinery as `NumpyRuntimeModel`.
    """

    def __init__(
        self,
        symbolic_obj,
        module: Optional[Dict[str, Callable]] = None,
        printer: Optional[str] = None,
    ):
        """Initialize the instance."""
        self.symbolic_obj = symbolic_obj
        self.module = dict(type(self).module) if module is None else dict(module)
        self.printer = type(self).printer if printer is None else printer

        modules = [self.module]
        if self.printer:
            modules.append(self.printer)

        self.runtime_functions: Dict[str, Callable] = {}
        for name, function_obj in symbolic_obj.functions.items():
            self.runtime_functions[name] = self._lambdify_function(function_obj, modules)

        for name, function in self.runtime_functions.items():
            setattr(self, name, function)
