"""Module `zoomy_core.transformation.to_glsl`.

SymPy → GLSL ES 3.00 (WebGL2) code printer for generating kernels from
any Zoomy symbolic model.

Subclasses `OutParamCodePrinter` — generated kernels return
`void` and write their result through a trailing `out float res[N]`
parameter, so the body generation, Piecewise handling and kernel/BC
wrapping are shared with the JS printer.  Only the GLSL-specific output
formatting lives here.

Two GLSL ES 3.00 facts shape the output:

* **No array return types.**  Kernels write through `out float res[N]`.
  The result is laid out row-major in the natural shape of the symbolic
  definition (e.g. `flux` is `(n_variables, dimension)` →
  `res[var * dimension + dim]`).
* **No implicit int→float conversion.**  Every numeric literal is
  emitted in float form (`0` → `0.0`).  The one exception is
  comparisons between integer quantities (e.g. a boundary-condition
  index), which are detected and printed with plain int literals.

Usage:

```
    from zoomy_core.transformation.to_glsl import GlslModel
    print(GlslModel(model).generate())
```
"""

import sympy as sp
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanFunction

from zoomy_core.transformation.generic_c import OutParamCodePrinter


# =========================================================================
#  1. GLSL BASE PRINTER
# =========================================================================


class GenericGlslBase(OutParamCodePrinter):
    """GLSL ES 3.00 code printer.

    Inherits the out-parameter body generation, Piecewise handling and
    kernel / boundary-condition wrapping from
    `OutParamCodePrinter`; overrides the GLSL-specific output:
    static types, a bare math namespace, explicit float literals and
    `out float res[N]` result parameters.
    """

    math_namespace = ""
    real_type = "float"
    gpu_enabled = False

    # Symbols compared inside relational expressions print as plain ints
    # while this flag is set (see ``_print_Relational``).
    _int_literal_ctx = False

    c_functions = {
        "conditional": lambda p, c, t, f: (
            f"(({p.doprint(c)}) ? ({p.doprint(t)}) : ({p.doprint(f)}))"
        ),
        "Min": lambda p, *args: p._print_min_max("min", args),
        "Max": lambda p, *args: p._print_min_max("max", args),
        "Abs": lambda p, a: f"abs({p.doprint(a)})",
    }

    # ── CSE temp typing (GLSL is strictly typed) ─────────────────────

    def _temp_decl(self, lhs, rhs):
        """A CSE temp may capture a boolean condition (the predicate of a
        `conditional`); GLSL needs it declared `bool`, not `float`."""
        kw = (
            "bool"
            if isinstance(rhs, (Relational, BooleanFunction))
            else "float"
        )
        return f"{kw} {self.doprint(lhs)} = {self.doprint(rhs)};"

    # ── Numeric literals (GLSL has no implicit int→float) ────────────

    def _print_Integer(self, expr):
        """Internal helper `_print_Integer`."""
        if self._int_literal_ctx:
            return str(int(expr))
        return f"{int(expr)}.0"

    def _print_Rational(self, expr):
        """Internal helper `_print_Rational`."""
        return f"({expr.p}.0 / {expr.q}.0)"

    def _print_Float(self, expr):
        """Internal helper `_print_Float`."""
        s = super()._print_Float(expr)
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s

    def _print_Relational(self, expr):
        """Print a comparison, choosing int vs float literal context."""
        int_ctx = bool(getattr(expr.lhs, "is_integer", False)) and bool(
            getattr(expr.rhs, "is_integer", False)
        )
        old = self._int_literal_ctx
        self._int_literal_ctx = int_ctx
        try:
            lhs = self._print(expr.lhs)
            rhs = self._print(expr.rhs)
        finally:
            self._int_literal_ctx = old
        return f"({lhs} {expr.rel_op} {rhs})"

    # ── Pow (GLSL ``pow`` is undefined for negative base) ────────────

    def _print_Pow(self, expr):
        """Print powers — inline multiplication for integer exponents."""
        base, exp = expr.as_base_exp()
        b = self._print(base)
        if exp == sp.Rational(1, 2) or exp == 0.5:
            return f"sqrt({b})"
        if exp == sp.Rational(-1, 2):
            return f"inversesqrt({b})"
        if exp.is_Integer:
            n = int(exp)
            if n == 0:
                return "1.0"
            if n == 1:
                return f"({b})"
            if n > 0:
                return "(" + " * ".join([f"({b})"] * n) + ")"
            denom = " * ".join([f"({b})"] * (-n))
            return f"(1.0 / ({denom}))"
        return f"pow({b}, {self._print(exp)})"

    # ── Argument formatting ──────────────────────────────────────────

    @staticmethod
    def _is_array_arg(obj):
        """True if a Function argument is a vector (vs a scalar symbol)."""
        if hasattr(obj, "get_list") or hasattr(obj, "_symbolic_name"):
            return True
        if hasattr(obj, "values") and callable(obj.values):
            return True
        return isinstance(obj, (list, tuple, sp.NDimArray, sp.MatrixBase))

    @staticmethod
    def _arg_length(obj):
        """Number of components of a vector Function argument."""
        if hasattr(obj, "length"):
            return obj.length()
        if hasattr(obj, "get_list"):
            return len(obj.get_list())
        try:
            return len(obj)
        except TypeError:
            return 1

    def _generate_args(self, func_obj, shape):
        """GLSL parameter list for a Function object, with the trailing
        `out float res[N]` result parameter appended.

        Zero-length array arguments (e.g. `Qaux` for a model with no
        auxiliary variables) are omitted — GLSL ES 3.00 forbids
        zero-sized arrays, and such a model has no symbols referencing
        the argument anyway.
        """
        parts = []
        for key, obj in func_obj.args.items():
            name = self.ARG_MAPPING.get(key, key)
            if key == "idx":
                # Boundary-condition dispatch index — integer scalar.
                parts.append("int bc_idx")
            elif self._is_array_arg(obj):
                length = self._arg_length(obj)
                if length == 0:
                    continue
                parts.append(f"float {name}[{length}]")
            else:
                parts.append(f"float {name}")
        parts.append(f"out float res[{self._shape_size(shape)}]")
        return ", ".join(parts)

    def wrap_function_signature(self, name, args_str, body_str, shape):
        """Wrap a GLSL body in a `void` function declaration."""
        return f"void {name}({args_str}) {{\n{body_str}\n}}"

    # ── No includes / struct wrappers for GLSL ───────────────────────

    def get_includes(self):
        """No includes needed for GLSL."""
        return ""

    def get_simple_array_def(self):
        """No SimpleArray struct needed for GLSL."""
        return ""


# =========================================================================
#  2. GLSL MODEL GENERATOR
# =========================================================================


class GlslModel(GenericGlslBase):
    """Generate GLSL ES 3.00 kernels from a Zoomy symbolic model.

    Produces standalone `void` functions for `flux`, `source`,
    `eigenvalues`, etc.  Each takes the model's flat `float[]`
    argument arrays plus an `out float res[N]` result parameter.
    Mirrors `zoomy_core.transformation.to_js.JsModel`.
    """

    # Functions to generate (in order).  Skipped if the definition is
    # missing or trivially zero.
    KERNEL_NAMES = [
        "flux",
        "source",
        "eigenvalues",
        "source_jacobian_wrt_variables",
        "quasilinear_matrix",
    ]

    def __init__(self, model):
        """Initialize with a Zoomy Model instance."""
        super().__init__()
        self.model = model
        self.n_dof_q = model.n_variables
        self.n_dof_qaux = model.n_aux_variables
        self.n_parameters = model.n_parameters

        self.register_map("Q", model.variables.values())
        self.register_map("Qaux", model.aux_variables.values())
        self.register_map("n", model.normal.values())
        if hasattr(model, "position"):
            self.register_map("X", model.position.values())
        # Map the *symbolic* parameters; per the canonical naming
        # convention ``model.parameters`` is the Zstruct of sympy
        # Symbols (``model.parameter_values`` is the Zstruct of
        # numeric floats).
        self.register_map("p", model.parameters.values())

    def generate(self):
        """Generate all GLSL kernels as a single code string."""
        blocks = [self._generate_metadata()]

        func_dict = (
            self.model.functions.as_dict()
            if hasattr(self.model.functions, "as_dict")
            else dict(self.model.functions.items())
        )
        for name in self.KERNEL_NAMES:
            if name not in func_dict:
                continue
            func_obj = func_dict[name]
            defn = func_obj.definition
            if defn is None:
                continue
            if hasattr(defn, "is_zero_matrix") and defn.is_zero_matrix:
                continue
            kernel = self._generate_kernel(name, func_obj)
            if kernel is not None:
                blocks.append(kernel)

        return "\n\n".join(blocks)

    # ── Internal ─────────────────────────────────────────────────────

    def _generate_metadata(self):
        """Generate a GLSL comment block with model metadata."""
        param_names = list(self.model.parameters.keys())
        param_vals = [float(v) for v in self.model.parameters.values()]
        lines = [
            f"// Generated GLSL ES 3.00 model: {self.model.__class__.__name__}",
            f"// Dimension: {self.model.dimension}",
            f"// Variables ({self.n_dof_q}): {list(self.model.variables.keys())}",
            f"// Aux variables ({self.n_dof_qaux}): "
            f"{list(self.model.aux_variables.keys())}",
            f"// Parameters ({self.n_parameters}): "
            f"{dict(zip(param_names, param_vals))}",
        ]
        return "\n".join(lines)


# =========================================================================
#  3. GLSL NUMERICS GENERATOR
# =========================================================================


class GlslNumerics(GenericGlslBase):
    """Generate GLSL ES 3.00 kernels from a Zoomy symbolic `Numerics`.

    Produces `numerical_flux` (and `numerical_fluctuations` /
    `local_max_abs_eigenvalue`) as `void` functions taking the
    per-face `Q_minus` / `Q_plus` states plus an
    `out float res[N]` result.  Mirrors
    `zoomy_core.transformation.generic_c.GenericCppNumerics`.
    """

    def __init__(self, numerics):
        """Initialize with a Zoomy Numerics instance."""
        super().__init__()
        self.numerics = numerics
        self.model = numerics.model               # a SystemModel
        self.n_dof_q = self.model.n_equations
        self.n_dof_qaux = len(self.model.aux_state)

        self.register_map("Q", self.model.state)
        self.register_map("Qaux", self.model.aux_state)
        self.register_map("n", self.model.normal.values())
        self.register_map("Q_minus", numerics.variables_minus)
        self.register_map("Q_plus", numerics.variables_plus)
        self.register_map("Qaux_minus", numerics.aux_variables_minus)
        self.register_map("Qaux_plus", numerics.aux_variables_plus)
        self.register_map("flux_minus", numerics.flux_minus)
        self.register_map("flux_plus", numerics.flux_plus)
        self.register_map("source_term", numerics.source_term)
        self.register_map("p", self.model.parameters.values())

    def generate(self):
        """Generate all numerics kernels as a single GLSL code string."""
        blocks = [
            f"// Generated GLSL ES 3.00 numerics: "
            f"{self.numerics.__class__.__name__}",
            f"// n_dof_q = {self.n_dof_q}",
        ]
        for name, func_obj in self.numerics.functions.items():
            if func_obj.definition is None:
                continue
            kernel = self._generate_kernel(name, func_obj)
            if kernel is not None:
                blocks.append(kernel)
        return "\n\n".join(blocks)
