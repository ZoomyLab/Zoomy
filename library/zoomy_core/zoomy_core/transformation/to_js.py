"""Module `zoomy_core.transformation.to_js`.

SymPy → JavaScript code printer for generating Model2D-compatible
kernels from any Zoomy symbolic model.

Subclasses `OutParamCodePrinter` — generated kernels write their
result through a trailing `res` array parameter the caller owns,
instead of returning a freshly-allocated `Float64Array`.  That keeps
the browser solver's hot loop allocation-free; the body generation,
Piecewise handling and kernel/BC wrapping are shared with the GLSL
printer.  Only the JS-specific output formatting lives here.

Usage:

```
    from zoomy_core.transformation.to_js import JsModel
    print(JsModel(model).generate())
```
"""

import sympy as sp

from zoomy_core.transformation.generic_c import OutParamCodePrinter


# =========================================================================
#  1. JS BASE PRINTER
# =========================================================================


class GenericJsBase(OutParamCodePrinter):
    """JavaScript code printer.

    Inherits the out-parameter body generation, Piecewise handling and
    kernel / boundary-condition wrapping from
    `OutParamCodePrinter`; overrides only the JS-specific output
    formatting.
    """

    math_namespace = "Math."
    # JS is untyped — ``Math.max(0, x)`` is fine; never cast literals.
    _typed_numeric_literals = False

    c_functions = {
        "conditional": lambda p, c, t, f: (
            f"(({p.doprint(c)}) ? ({p.doprint(t)}) : ({p.doprint(f)}))"
        ),
        "Min": lambda p, *args: p._print_min_max("min", args),
        "Max": lambda p, *args: p._print_min_max("max", args),
        "Abs": lambda p, a: f"Math.abs({p.doprint(a)})",
    }

    # ── Function arguments / wrapping ────────────────────────────────

    def _generate_args(self, func_obj, shape):
        """JS parameter list for a Function object, with the trailing
        `res` out-parameter appended.  JS imposes no type or
        zero-length-array restrictions, so every argument is kept."""
        parts = []
        for key, val in func_obj.args.items():
            if key == "idx":
                # Boundary-condition dispatch index.
                parts.append("bc_idx")
            else:
                parts.append(self.ARG_MAPPING.get(key, key))
        parts.append("res")
        return ", ".join(parts)

    def wrap_function_signature(self, name, args_str, body_str, shape):
        """Wrap a JS body in a function declaration."""
        return f"function {name}({args_str}) {{\n{body_str}\n}}"

    # ── Pow printing (JS uses ** or Math.pow) ────────────────────────

    def _print_Pow(self, expr):
        """Print power expressions using Math.pow / Math.sqrt / inline multiply."""
        base, exp = expr.as_base_exp()
        b = self._print(base)
        if exp == sp.Rational(1, 2) or exp == 0.5:
            return f"Math.sqrt({b})"
        if exp == sp.Rational(-1, 2):
            return f"(1.0 / Math.sqrt({b}))"
        if exp.is_Integer:
            n = int(exp)
            if n == 0:
                return "1.0"
            if n == 1:
                return b
            if n == 2:
                return f"(({b}) * ({b}))"
            if n == -1:
                return f"(1.0 / ({b}))"
            if n < 0:
                return f"(1.0 / Math.pow({b}, {abs(n)}))"
            return f"Math.pow({b}, {n})"
        return f"Math.pow({b}, {self._print(exp)})"

    # ── No includes / struct wrappers for JS ─────────────────────────

    def get_includes(self):
        """No includes needed for JS."""
        return ""

    def get_simple_array_def(self):
        """No SimpleArray struct needed for JS."""
        return ""


# =========================================================================
#  2. JS MODEL GENERATOR
# =========================================================================


class JsModel(GenericJsBase):
    """Generate JavaScript Model2D kernels from a Zoomy symbolic model.

    Produces standalone JS functions for `flux`, `source`,
    `eigenvalues`, etc.  Each takes the model's flat argument arrays
    plus a trailing `res` out-parameter the caller owns.
    """

    # Functions to generate (in order). Skipped if definition is None/zero.
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
        # Map the *symbolic* parameters; ``model.parameters`` now
        # holds Symbols (post-rename), ``model.parameter_values``
        # holds floats.
        self.register_map("p", model.parameters.values())

    def generate(self):
        """Generate all JS kernels as a single code string."""
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
        """Generate a JS comment block with model metadata."""
        param_names = list(self.model.parameters.keys())
        param_vals = [float(v) for v in self.model.parameter_values.values()]
        lines = [
            f"// Generated JS model: {self.model.__class__.__name__}",
            f"// Dimension: {self.model.dimension}",
            f"// Variables ({self.n_dof_q}): {list(self.model.variables.keys())}",
            f"// Aux variables ({self.n_dof_qaux}): "
            f"{list(self.model.aux_variables.keys())}",
            f"// Parameters ({self.n_parameters}): "
            f"{dict(zip(param_names, param_vals))}",
        ]
        return "\n".join(lines)


# =========================================================================
#  3. JS NUMERICS GENERATOR
# =========================================================================


class JsNumerics(GenericJsBase):
    """Generate JavaScript kernels from a Zoomy symbolic `Numerics`.

    Produces `numerical_flux` (and `numerical_fluctuations` /
    `local_max_abs_eigenvalue`) as standalone JS functions taking the
    per-face `Q_minus` / `Q_plus` states plus a trailing `res`
    out-parameter.  Mirrors
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
        """Generate all numerics kernels as a single JS code string."""
        blocks = [
            f"// Generated JS numerics: {self.numerics.__class__.__name__}",
            f"// nVars = {self.n_dof_q}",
        ]
        for name, func_obj in self.numerics.functions.items():
            if func_obj.definition is None:
                continue
            kernel = self._generate_kernel(name, func_obj)
            if kernel is not None:
                blocks.append(kernel)
        return "\n\n".join(blocks)
