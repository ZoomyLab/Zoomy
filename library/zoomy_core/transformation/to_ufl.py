from attrs import define
import ufl
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel



@define(kw_only=False, slots=True, frozen=True)
class UFLRuntimeModel(NumpyRuntimeModel):
    printer=None
    
    module = {
        'ones_like': lambda x: 0*x + 1,
        'zeros_like': lambda x: 0*x,
        'array': ufl.as_vector,
        'squeeze': lambda x: x,
        
        # --- Elementary arithmetic ---
        "Abs": abs,
        "sign": ufl.sign,
        "Min": ufl.min_value,
        "Max": ufl.max_value,
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
        "Heaviside": lambda x: ufl.conditional(x >= 0, 1.0, 0.0),
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
        "min": ufl.min_value,
        "max": ufl.max_value,
        "sqrt": ufl.sqrt,
        "sum": lambda x: ufl.Constant(sum(x)) if isinstance(x, (list, tuple)) else x,
        "ImmutableDenseMatrix": lambda x: x,
    }
