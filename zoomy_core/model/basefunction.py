"""Module `zoomy_core.model.basefunction`."""

import sympy as sp
import param
from sympy import Tuple
from zoomy_core.misc.misc import Zstruct, ZArray


def listify(expr):
    """Listify."""
    if isinstance(expr, (ZArray, sp.NDimArray)):
        return Tuple(*expr.tolist())
    elif hasattr(expr, "args") and expr.args:
        return expr.func(*[listify(a) for a in expr.args])
    else:
        return expr


class Function(param.Parameterized):
    """Function. (class)."""
    name = param.String(default="Function")
    args = param.ClassSelector(class_=Zstruct, default=None)
    definition = param.Parameter(default=None)

    def __init__(self, **params):
        """Initialize the instance."""
        super().__init__(**params)
        self._no_definition = self.definition is None
        if self.args is None:
            self.args = Zstruct()
        if self.definition is None:
            self.definition = ZArray([0])

        # Safety Check: Allow ZArray, NDimArray, MatrixBase, AND generic Expressions/Lists
        # This covers Piecewise definitions and simple lists of symbols.
        allowed_types = (ZArray, sp.NDimArray, sp.MatrixBase, sp.Expr, list, tuple)

        if not isinstance(self.definition, allowed_types):
            raise TypeError(
                f"Function '{self.name}' definition has invalid type '{type(self.definition).__name__}'.\n"
                "Allowed types: ZArray, sp.NDimArray, sp.MatrixBase, sp.Expr (e.g. Piecewise), list, tuple."
            )

    @property
    def is_empty(self):
        """True when there is no definition, an empty array, or every entry
        is identically zero (a zero kernel contributes nothing)."""
        shape = tuple(getattr(self.definition, "shape", ()) or ())
        if self._no_definition or 0 in shape:
            return True
        d = self.definition
        entries = sp.flatten(d.tolist() if hasattr(d, "tolist") else [d])
        return all(sp.sympify(e) == 0 for e in entries)


class KernelRef(sp.Symbol):
    """A registered function used as a VALUE (e.g. handed to a solver block
    that maps it over cells or faces): printed as ``<owner>.<name>``; the
    backend resolves the name to its compiled kernel."""

    def __new__(cls, owner, name):
        return super().__new__(cls, f"{owner}.{name}")

    @property
    def owner(self):
        return self.name.split(".", 1)[0]

    @property
    def slot(self):
        return self.name.split(".", 1)[1]


def placeholder_call(full_name, definition, *input_args):
    """The opaque call ``full_name(*args)`` standing for a registered
    function: one element read per entry when the definition is an array.
    The printer emits it as a call, or splices the definition back in
    (``resolve_opaque``).  No arguments -> the definition itself."""
    if not input_args:
        return definition
    final_args = []
    for arg in input_args:
        if hasattr(arg, "_symbolic_name") and arg._symbolic_name:
            final_args.append(sp.Symbol(arg._symbolic_name))
        elif hasattr(arg, "get_list"):
            final_args.extend(arg.get_list())
        elif isinstance(arg, (list, tuple)):
            final_args.extend(arg)
        else:
            final_args.append(arg)
    shape = getattr(definition, "shape", ())
    main_call = sp.Function(full_name)(*final_args)
    if not shape or shape == (1,):
        return main_call
    size = 1
    for s in shape:
        size *= s
    base = sp.IndexedBase(main_call, shape=(size,))
    return ZArray([base[i] for i in range(size)]).reshape(*shape)


class KernelCall:
    """An entry of a registrar's ``call`` namespace; wraps one registered
    ``Function`` (looked up by name, so a refreshed registry is seen).

    * called with symbolic arguments -> the placeholder call;
    * used as a value in a sympy expression -> the printable reference
      ``KernelRef('<owner>.<name>')`` (sympy's ``_sympy_`` hook).  This is
      how a solver template hands a kernel to a block."""

    def __init__(self, owner, functions, name, call_name=None):
        self.owner, self.functions, self.name = owner, functions, name
        self.call_name = call_name or name

    def __call__(self, *args):
        return placeholder_call(self.call_name,
                                self.functions[self.name].definition, *args)

    def _sympy_(self):
        return KernelRef(self.owner, self.name)

    def __repr__(self):
        return f"KernelCall({self.owner}.{self.name})"


class SymbolicRegistrar:
    """SymbolicRegistrar. (class)."""
    def register_symbolic_function(self, name, method_ref, sig_struct):
        """``functions[name]`` holds the definition, ``call[name]`` the
        placeholder caller / kernel reference."""
        self.functions[name] = Function(
            name=name, definition=method_ref(), args=sig_struct
        )
        prefix = "Model<T>::" if self.name == "Model" else ""
        self.call[name] = KernelCall(self.name, self.functions, name,
                                     f"{prefix}{name}")
