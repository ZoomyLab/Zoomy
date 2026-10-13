"""Module `zoomy_core.transformation.to_ufl`."""

import sympy as sp
from zoomy_core.misc.misc import Zstruct
from zoomy_core.systemmodel.system_model import OPERATOR_ARG_SLOTS
from zoomy_core.transformation.to_numpy import (
    DT_SYMBOL,
    NumpyRuntimeModel,
    NumpyRuntimeSymbolic,
)


# Spatial-coordinate-name → UFL grad axis index used by the
# Derivative-resolver path.  Matches Firedrake's default mesh
# coordinate ordering ``(x, y, z)``.
_UFL_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


class _ZdGrad(sp.Function):
    """Symbolic placeholder for a sympy ``Derivative`` atom.

    Pre-substituted by :meth:`UFLRuntimeModel._lambdify_function` so
    that lambdify can serialise the expression through the default
    printer — sympy's ``PythonCodePrinter`` has no ``_print_Derivative``
    method and raises on inline derivatives.  At runtime the UFL
    module dict maps ``_ZdGrad(field, axis_idx0, axis_idx1, ...)`` to
    ``ufl.grad(field)[axis_idx0][axis_idx1]...`` (chained for higher
    orders).
    """

    @classmethod
    def from_derivative(cls, deriv):
        """Build ``_ZdGrad(field, idx0, idx1, ...)`` from a
        ``sp.Derivative(field, var0, var1, ...)`` atom.  Each
        coordinate Symbol's name resolves to a fixed-index integer
        via :data:`_UFL_AXIS_INDEX`."""
        target = deriv.args[0]
        idxs = []
        for var, n in deriv.variable_count:
            try:
                idx = _UFL_AXIS_INDEX[str(var)]
            except KeyError as e:
                raise KeyError(
                    f"_ZdGrad: spatial coordinate {var!r} not in "
                    f"{sorted(_UFL_AXIS_INDEX)}."
                ) from e
            for _ in range(int(n)):
                idxs.append(sp.Integer(idx))
        return cls(target, *idxs)


def _zd_grad_resolver(field, *axis_indices):
    """Runtime resolver: ``ufl.grad(field)[axis_indices...]`` with one
    ``ufl.grad`` per axis index.  Single-index = first-order
    derivative; repeated = higher orders / mixed partials."""
    import ufl

    val = field
    for idx in axis_indices:
        val = ufl.grad(val)[int(idx)]
    return val


def _ufl_conditional(condition, true_val, false_val):
    """Internal helper `_ufl_conditional`."""
    import ufl

    if condition is True:
        return true_val
    elif condition is False:
        return false_val
    elif isinstance(true_val, tuple):
        return ufl.conditional(condition, ufl.as_vector(true_val), ufl.as_vector(false_val))
    else:
        return ufl.conditional(condition, true_val, false_val)


class _ZoomyVector(sp.Function):
    """Symbolic wrapper so lambdify emits ``_ZoomyVector(e0, e1, ...)``
    which the UFL module maps to ``ufl.as_vector([e0, e1, ...])``.

    This distinguishes 1-D arrays (vectors) from 2-D arrays (tensors) at
    lambdify time, because both would otherwise be serialised as
    ``ImmutableDenseMatrix(...)`` and be indistinguishable.
    """
    pass


def _ufl_module():
    """The lambdify module dict mapping SymPy names to UFL; imports ufl."""
    import ufl

    return {
        'ones_like': lambda x: 0*x + 1,
        'zeros_like': lambda x:  0*x,
        'array': ufl.as_vector,
        'squeeze': lambda x: x,
        'conditional': _ufl_conditional,

        # --- Elementary arithmetic ---
        "Abs": abs,
        "sign": ufl.sign,
        # SymPy's ``Min`` / ``Max`` accept N arguments; ``ufl.min_value`` /
        # ``ufl.max_value`` are strictly binary.  Reduce N-ary calls to a
        # pairwise chain so any arity lambdifies cleanly.
        "Min": lambda *a: (a[0] if len(a) == 1
                           else __import__("functools").reduce(ufl.min_value, a)),
        "Max": lambda *a: (a[0] if len(a) == 1
                           else __import__("functools").reduce(ufl.max_value, a)),
        # --- Powers and roots ---
        "sqrt": ufl.sqrt,
        "exp": ufl.exp,
        "ln": ufl.ln,
        "pow": lambda x, y: x**y,
        # --- Trigonometric functions ---
        "sin": ufl.sin,
        "cos": ufl.cos,
        "tan": ufl.tan,
        "asin": ufl.asin,
        "acos": ufl.acos,
        "atan": ufl.atan,
        "atan2": ufl.atan2,
        # --- Hyperbolic functions ---
        "sinh": ufl.sinh,
        "cosh": ufl.cosh,
        "tanh": ufl.tanh,
        # --- Piecewise / conditional logic ---
        # SymPy emits ``Heaviside(x, h0)`` (2-arg) when it arises from a
        # subgradient, e.g. d/dx Max(x, c) = Heaviside(x - c, 1/2): the 2nd
        # arg is the value AT x == 0.  That set is measure-zero under FEM
        # quadrature, but the lambda must still accept it (else lowering the
        # symbolic source Jacobian of a floored Manning friction fails).
        "Heaviside": lambda x, *h0: ufl.conditional(
            x > 0, 1.0, ufl.conditional(x < 0, 0.0, h0[0] if h0 else 0.5)),
        "Piecewise": ufl.conditional,
        "signum": ufl.sign,
        # --- Vector & tensor ops ---
        "dot": ufl.dot,
        "inner": ufl.inner,
        "outer": ufl.outer,
        "cross": ufl.cross,
        # --- Differential operators (used in forms) ---
        "grad": ufl.grad,
        "div": ufl.div,
        "curl": ufl.curl,
        # --- Common constants ---
        "pi": ufl.pi,
        "E": ufl.e,
        # --- Matrix and linear algebra ---
        "transpose": ufl.transpose,
        "det": ufl.det,
        "inv": ufl.inv,
        "tr": ufl.tr,
        # --- Python builtins (SymPy may emit these) ---
        "abs": abs,
        "min": lambda *a: (a[0] if len(a) == 1
                           else __import__("functools").reduce(ufl.min_value, a)),
        "max": lambda *a: (a[0] if len(a) == 1
                           else __import__("functools").reduce(ufl.max_value, a)),
        "sqrt": ufl.sqrt,
        "sum": lambda x: ufl.Constant(sum(x)) if isinstance(x, (list, tuple)) else x,
        # --- Array / matrix serialisation ---
        "_ZoomyVector": lambda *args: ufl.as_vector(list(args)),
        "ImmutableDenseMatrix": lambda data: ufl.as_tensor(data),
        # --- Derivative atom resolver
        # Replaces ``sp.Derivative(field, x, ...)`` substituted in by
        # ``_substitute_derivative_atoms`` with ``ufl.grad`` chains.
        "_ZdGrad": _zd_grad_resolver,
    }


class _UFLModule:
    """`UFLRuntimeModel.module`: built on first access, so importing this module needs no ufl."""

    _dict = None

    def __get__(self, obj, owner):
        if _UFLModule._dict is None:
            _UFLModule._dict = _ufl_module()
        return _UFLModule._dict


class UFLRuntimeModel(NumpyRuntimeModel):
    """UFLRuntimeModel. (class)."""
    printer = None
    use_cse = True

    # -----------------------------------------------------------------
    # Array -> Matrix conversion so that lambdify can serialise the expr
    # -----------------------------------------------------------------

    @staticmethod
    def _array_to_matrix(expr):
        """Convert ``ImmutableDenseNDimArray`` to a lambdify-friendly form.

        SymPy's ``lambdify`` (without a code printer) cannot serialise
        ``ImmutableDenseNDimArray`` but *can* handle ``ImmutableDenseMatrix``
        and custom ``Function`` subclasses.

        * 0-D  -> bare scalar symbol
        * 1-D  -> ``_ZoomyVector(e0, e1, ...)``  -> ``ufl.as_vector``
        * 2-D  -> ``ImmutableDenseMatrix``        -> ``ufl.as_tensor``
        * 3-D+ -> reshaped to 2-D Matrix (product-of-leading, last-dim)
        """
        from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray

        if not isinstance(expr, (ImmutableDenseNDimArray, sp.Array)):
            return expr

        shape = tuple(int(s) for s in expr.shape)
        flat = list(expr._array)  # always a flat list of SymPy exprs

        if len(shape) == 0:
            return flat[0] if flat else expr
        elif len(shape) == 1:
            return _ZoomyVector(*flat)
        elif len(shape) == 2:
            return sp.ImmutableDenseMatrix(shape[0], shape[1], flat)
        else:
            import functools, operator
            nrows = functools.reduce(operator.mul, shape[:-1], 1)
            ncols = shape[-1]
            return sp.ImmutableDenseMatrix(nrows, ncols, flat)

    @staticmethod
    def _distribute_piecewise_over_array(expr):
        """Pull ``Piecewise(NDimArray, …)`` out into an NDimArray of
        scalar ``Piecewise`` expressions.

        SymPy's ``PythonCodePrinter`` (which lambdify falls back to when
        no explicit printer is passed) does not know how to serialise an
        ``ImmutableDenseNDimArray`` *inside* a ``Piecewise`` branch.
        The fix is to swap the nesting: if every branch is an
        ``NDimArray`` of the same shape, return an ``Array`` of length
        ``n`` where each entry is the element-wise ``Piecewise``.

        Then :meth:`_array_to_matrix` (called from
        :meth:`_vectorize_expression`) turns the resulting flat array
        into the appropriate ``_ZoomyVector`` / ``ImmutableDenseMatrix``
        wrapper for the UFL module dict to lower.
        """
        from sympy.tensor.array.dense_ndim_array import (
            ImmutableDenseNDimArray,
        )

        if not isinstance(expr, sp.Piecewise):
            return expr

        branches = []
        for branch in expr.args:
            e, c = branch.expr, branch.cond
            if not isinstance(e, (ImmutableDenseNDimArray, sp.Array)):
                # mixed shape — bail and let the default printer try.
                return expr
            branches.append((list(sp.Array(e)._array), c))

        if not branches:
            return expr

        # All branches must agree on length.
        n = len(branches[0][0])
        if not all(len(b[0]) == n for b in branches):
            return expr

        shape = tuple(int(s) for s in sp.Array(expr.args[0].expr).shape)
        flat = []
        for i in range(n):
            per_branch = [(b[0][i], b[1]) for b in branches]
            flat.append(sp.Piecewise(*per_branch))
        return sp.Array(flat, shape)

    def _vectorize_expression(self, expr, signature):
        """Override: distribute any top-level ``Piecewise`` over its
        ``NDimArray`` branches, vectorise, then convert Array → Matrix
        for lambdify."""
        expr = self._distribute_piecewise_over_array(expr)
        result = super()._vectorize_expression(expr, signature)

        from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
        if isinstance(result, (ImmutableDenseNDimArray, sp.Array)):
            return self._array_to_matrix(result)
        return result

    def _lambdify_function(self, function_obj, modules):
        """Skip functions with zero-size output (e.g. Jacobian when n_aux=0).

        Otherwise re-derive slot-operator ``args`` from the declaration
        registry (legacy raw-Model path only — see
        :meth:`_with_declared_operator_args`), then pre-substitute inline
        ``sp.Derivative`` atoms with the :class:`_ZdGrad` placeholder
        Function so lambdify's default printer can serialise the
        expression.  The UFL module dict binds ``_ZdGrad`` to a resolver
        that chains ``ufl.grad`` to produce the requested partial.
        """
        expr = function_obj.definition
        if hasattr(expr, 'shape') and any(int(s) == 0 for s in expr.shape):
            import numpy as np
            shape = tuple(int(s) for s in expr.shape)
            return lambda *a, _s=shape: np.empty(_s)
        function_obj = self._with_declared_operator_args(function_obj)
        # Replace every Derivative atom with its _ZdGrad image.  This
        # is structurally equivalent to ``SystemModel.expose_aux_atoms``
        # but stays at the Model-level lambdify path — needed because
        # the Firedrake backend builds the runtime from Model directly
        # (via UFLRuntimeModel(model)) before SystemModel substitution
        # runs.
        function_obj = self._substitute_derivative_atoms(function_obj)
        return super()._lambdify_function(function_obj, modules)

    def _with_declared_operator_args(self, function_obj):
        """Re-derive a slot operator's ``args`` from ``OPERATOR_ARG_SLOTS``
        (the declaration table behind ``sm.operator(name).args`` /
        ``sm.operator_signature(name)``) on the LEGACY raw-Model path.

        ``Model._initialize_functions`` still registers the pre-REQ-185
        signatures (``source(Q, Qaux, p)``, ``update_aux_variables(Q, Qaux,
        p, dt)``), so a Firedrake runtime built via ``UFLRuntimeModel(model)``
        would emit them stale.  Materialising the declared groups from the
        model's own symbols — exactly like ``SystemModel.operator_signature``
        — makes the UFL emit gain the REQ-185 trailing groups:

        * ``source(Q, Qaux, p, time, [dt], X)``
        * ``update_aux_variables(Q, Qaux, p, time, X)`` (and its jacobian)

        with everything else unchanged (the other declared slots carry the
        same groups the legacy registry already binds).  Mirrors
        ``NumpyRuntimeModel.from_system_model._op_sig``'s one
        lowering-SYNTAX constraint: the ``dt`` group is dropped when ``dt``
        is already a model parameter (duplicate lambdify argument —
        Chorin/VAM split sub-systems).

        The SystemModel/NSM path already receives the declared args from
        ``from_system_model`` (which reads ``sm.operator_signature``), so it
        passes through untouched — only a runtime whose ``model`` carries a
        ``.functions`` registry (a legacy ``Model``) is rewritten.  Non-slot
        registry functions (``residual``, ``eigensystem``, the BC kernels,
        ``interpolate_to_3d``, …) keep their registered signatures.
        """
        model = getattr(self, "model", None)
        if model is None or not hasattr(model, "functions"):
            return function_obj
        slots = OPERATOR_ARG_SLOTS.get(function_obj.name)
        if slots is None:
            return function_obj
        if bool(getattr(model.parameters, "contains",
                        lambda *_: False)("dt")):
            slots = tuple(s for s in slots if s != "dt")
        group = {
            "variables": model.variables,
            "aux_variables": model.aux_variables,
            "parameters": model.parameters,
            "normal": model.normal,
            "time": model.time,
            "dt": DT_SYMBOL,
            "position": model.position,
        }
        from zoomy_core.model.basefunction import Function
        return Function(
            name=function_obj.name,
            args=Zstruct(**{s: group[s] for s in slots}),
            definition=function_obj.definition,
        )

    def _lower_opaque_kernels(self, expr):
        """UFL keeps the opaque kernels in the per-component
        ``eigensystem(idx, …)`` convention: the Firedrake module hooks
        (``eigensystem_hook(idx, *a)`` / ``eigenvalues_hook(idx, *a)``) are
        per-component, with no binding for the packed-stack ``K(*a)`` /
        ``pick``.  The producers emit the canonical ``pick(K(*a), idx)``, so
        restore the ``K(idx, *a)`` form here before lambdify."""
        from zoomy_core.model.kernel_functions import lower_pick_to_component
        return lower_pick_to_component(expr)

    @staticmethod
    def _substitute_derivative_atoms(function_obj):
        """Return a function object with every ``sp.Derivative`` atom in
        its definition replaced by the corresponding ``_ZdGrad`` placeholder.

        Inline derivatives of state w.r.t. spatial coordinates (``∂_z T``,
        ``∂_z U``, ...) are common in eddy-viscosity closures — e.g. the
        MY-2.5 Galperin G_H argument depends on ``∂_z T``.  Without this
        substitution the default lambdify printer raises on the
        Derivative atom.
        """
        from zoomy_core.model.basefunction import Function
        expr = function_obj.definition
        if hasattr(expr, "tolist"):
            entries = sp.flatten(expr.tolist())
        else:
            entries = [expr]
        deriv_atoms = set()
        for e in entries:
            deriv_atoms.update(sp.sympify(e).atoms(sp.Derivative))
        if not deriv_atoms:
            return function_obj
        subs = {d: _ZdGrad.from_derivative(d) for d in deriv_atoms}
        new_def = (expr.xreplace(subs) if hasattr(expr, "xreplace")
                   else sp.sympify(expr).xreplace(subs))
        return Function(
            name=function_obj.name,
            args=function_obj.args,
            definition=new_def,
        )

    # -----------------------------------------------------------------
    # SystemModel/NSM-driven operator lowering (Firedrake runs new-style
    # BaseModel models / NumericalSystemModels)
    # -----------------------------------------------------------------
    #
    # The scalar/vector operators (flux / hydrostatic_pressure / source /
    # eigenvalues / mass_matrix / update_variables / update_aux_variables /
    # the BC kernels) are lowered by the inherited
    # ``NumpyRuntimeModel.from_system_model`` verbatim — it routes each
    # through ``rt._lambdify_function`` (this class's UFL override), so they
    # already emit ``ufl`` forms.  Only the rank-3 (NCP / quasilinear) and
    # rank-4 (diffusion) operators need a UFL-specific lowering: the numpy
    # hook stacks per-axis slabs with ``np.stack`` (invalid on symbolic
    # UFL), whereas UFL emits the WHOLE array as one lambdified function.
    # ``_array_to_matrix`` (called inside ``_vectorize_expression``)
    # reshapes the ``(n_eq, n_cols, n_dim)`` / ``(n_eq, n_st, n_dim, n_dim)``
    # array to a 2-D ``ufl.as_tensor`` — the EXACT shape convention the
    # legacy raw-Model UFL path (``UFLRuntimeModel(model)``) already emits
    # (NCP → ``(n_eq·n_cols, n_dim)``, diffusion → ``(n_eq·n_st·n_dim,
    # n_dim)``), so the Firedrake form assembly consumes NSM-driven
    # operators identically to the legacy Model-driven ones.

    def _lower_ndarray_operator(self, name, arr, n_cols, n_eq, n_dim,
                                std_sig, modules):
        """UFL override: emit the full rank-3 operator as ONE lambdified
        function → ``ufl.as_tensor``.  See the class comment above."""
        from zoomy_core.model.basefunction import Function
        fn = Function(name=name, args=std_sig, definition=arr)
        return self._lambdify_function(fn, modules)

    def _lower_rank4_operator(self, name, A_arr, n_eq, n_st, n_dim,
                              std_sig, modules):
        """UFL override: emit the full rank-4 diffusion tensor as ONE
        lambdified function → ``ufl.as_tensor``.  See the class comment
        above."""
        from zoomy_core.model.basefunction import Function
        fn = Function(name=name, args=std_sig, definition=A_arr)
        return self._lambdify_function(fn, modules)

    @classmethod
    def from_nsm(cls, nsm, *, module=None, printer=None):
        """Build a UFL runtime from a :class:`NumericalSystemModel`.

        Thin front door mirroring ``JaxRuntime.from_nsm``: the operator set
        lives on ``nsm.sm`` (a :class:`SystemModel`), so this just delegates
        to :meth:`from_system_model`, which normalises any Model /
        SystemModel / NSM through ``to_numerical_system_model``."""
        return cls.from_system_model(nsm, module=module, printer=printer)

    module = _UFLModule()


class UFLRuntimeSymbolic(UFLRuntimeModel, NumpyRuntimeSymbolic):
    """UFL runtime for symbolic registrars (e.g. :class:`Numerics`).

    The Firedrake counterpart to :class:`NumpyRuntimeSymbolic`.  It
    walks ``symbolic_obj.functions`` and lambdifies each registered
    symbolic function (``numerical_flux``, ``numerical_fluctuations``,
    ``local_max_abs_eigenvalue``, …) through the UFL module dict, so
    calling, e.g., ``runtime.numerical_flux(qL, qR, auxL, auxR, p, n)``
    returns a UFL expression suitable for direct assembly in a
    Firedrake form.

    The MRO is ``UFLRuntimeSymbolic → UFLRuntimeModel →
    NumpyRuntimeSymbolic → NumpyRuntimeModel``; ``__init__`` is
    inherited from ``NumpyRuntimeSymbolic`` (no other class defines
    one), and ``module`` / ``printer`` /
    ``_vectorize_expression`` / ``_lambdify_function`` come from
    ``UFLRuntimeModel``.
    """
    pass
