"""Symbols as the user-facing names of state fields and parameters.

`FieldSymbols` holds the state symbols of a model (`Model.state(...)`, `sm.fields`) with named
groups (`.momentum`) and optional user aliases. `Parameters` is the parameter vector of a solver,
read and set by symbol, and a pytree for JAX. A symbol is identified by its name; the model's own
symbols carry its assumptions, symbols made elsewhere with the same name refer to the same field.
"""
from collections import namedtuple

import numpy as np
import sympy as sp

__all__ = ["FieldGroup", "FieldSymbols", "Parameters", "RunResult", "symbol_name"]

#: Result of `solver.run`: unpack it (`time, Q, Qaux = ...`) or read `.Q`.
RunResult = namedtuple("RunResult", ["time", "Q", "Qaux"])


def symbol_name(s, what="field"):
    """The name of a symbol; anything else raises TypeError."""
    if isinstance(s, sp.Symbol):
        return s.name
    raise TypeError(f"{what} must be a sympy Symbol (e.g. sm.fields.h, sm.parameters.n), "
                    f"got {type(s).__name__} {s!r}")


class FieldGroup(tuple):
    """A named group of state symbols, e.g. `fields.momentum` (all momentum components)."""

    def __new__(cls, name, symbols):
        """Build the group `name` from `symbols`."""
        obj = super().__new__(cls, tuple(symbols))
        obj.name = name
        return obj

    def __repr__(self):
        return f"{self.name}({', '.join(str(s) for s in self)})"


def momentum_symbols(symbols):
    """The momentum components among `symbols`: the family `q` (`q_0`, `q_x_1`, ...)."""
    return [s for s in symbols if s.name.startswith("q_")]


class FieldSymbols:
    """The state symbols of a model by name, its groups and optional aliases.

    `fields.h` is the depth symbol, `fields.momentum` the group of momentum components, and with
    `names=("b", "h", "hu", "hv")` also `fields.hu` (the same symbol as `fields.q_x_0`).
    """

    def __init__(self, symbols, groups=None, names=None):
        """Hold `symbols` (the state in model order), `groups` {name: symbols} and the user
        `names` (one per state field, in model order) as aliases."""
        self._symbols = list(symbols)
        self._by_name = {s.name: s for s in self._symbols}
        mom = momentum_symbols(self._symbols)
        self._groups = {"momentum": FieldGroup("momentum", mom)} if mom else {}
        for k, v in (groups or {}).items():
            self._groups[k] = FieldGroup(k, v)
        self._aliases = {}
        if names is not None:
            names = list(names)
            if len(names) != len(self._symbols):
                raise ValueError(
                    f"names: {len(names)} given, the state has {len(self._symbols)} fields "
                    f"({', '.join(s.name for s in self._symbols)})")
            for alias, s in zip(names, self._symbols):
                if alias in self._groups or (alias in self._by_name and self._by_name[alias] is not s):
                    raise ValueError(f"names: {alias!r} is already a field or group name")
                if alias != s.name:
                    self._aliases[alias] = s

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        for table in (self._by_name, self._aliases, self._groups):
            if name in table:
                return table[name]
        raise AttributeError(
            f"no state field {name!r}; fields: {', '.join(self._by_name)}"
            + (f"; aliases: {', '.join(self._aliases)}" if self._aliases else "")
            + (f"; groups: {', '.join(self._groups)}" if self._groups else ""))

    def __iter__(self):
        return iter(self._symbols)

    def __len__(self):
        return len(self._symbols)

    def __contains__(self, s):
        return isinstance(s, sp.Symbol) and s.name in self._by_name

    def names(self):
        """The field names in model order."""
        return list(self._by_name)

    def slot(self, s):
        """The row of field `s` (a symbol of this state) in `Q`: `Q[fields.slot(fields.h)]`."""
        name = symbol_name(s)
        if name not in self._by_name:
            raise KeyError(f"{name!r} is not a state field; fields: {', '.join(self._by_name)}")
        return self.names().index(name)

    def __repr__(self):
        rev = {s.name: a for a, s in self._aliases.items()}
        fields = ", ".join(f"{n} ({rev[n]})" if n in rev else n for n in self._by_name)
        groups = "; ".join(repr(g) for g in self._groups.values())
        return f"FieldSymbols({fields}" + (f" | {groups}" if groups else "") + ")"


class Parameters:
    """The parameter vector of a solver, read and set by symbol: `p[n]`, `p.set({n: 0.04})`.

    A pytree for JAX (registered by zoomy_jax), so `jax.grad` with respect to all parameters
    returns a `Parameters` of gradients: `dJ[n]`. `np.asarray(p)` and `jnp.asarray(p)` give the
    vector.
    """

    def __init__(self, names, values):
        """Hold the parameter `names` (vector order) and the `values` vector."""
        self._names = tuple(names)
        self._index = {n: i for i, n in enumerate(self._names)}
        self.values = values

    def _slot(self, key):
        name = symbol_name(key, "parameter")
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(f"no parameter {name!r}; parameters: {', '.join(self._names)}") from None

    def __getitem__(self, key):
        if isinstance(key, sp.Symbol):
            return self.values[self._slot(key)]
        return self.values[key]                       # index, slice: the vector as before

    def __getattr__(self, name):                      # .at, .shape, .dtype, ...: the vector's
        if name.startswith("_") or name == "values":
            raise AttributeError(name)
        return getattr(self.values, name)

    def __iter__(self):
        return iter(self.values)

    def set(self, mapping):
        """A copy with the parameters in `mapping` {symbol: value} replaced."""
        return Parameters(self._names, self.vector(mapping))

    def vector(self, mapping=None):
        """The vector, with the parameters in `mapping` {symbol: value} replaced."""
        v = self.values
        for key, val in (mapping or {}).items():
            i = self._slot(key)
            if hasattr(v, "at"):                      # JAX array (traced values pass)
                v = v.at[i].set(val)
            else:
                v = np.array(v, dtype=float, copy=True)
                v[i] = val
        return v

    def names(self):
        """The parameter names in vector order."""
        return list(self._names)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)

    def __jax_array__(self):
        return self.values

    def __len__(self):
        return len(self._names)

    def __repr__(self):
        vals = np.asarray(self.values) if not hasattr(self.values, "aval") else ["traced"] * len(self)
        return "Parameters(" + ", ".join(f"{n}={v}" for n, v in zip(self._names, vals)) + ")"


for _op in ("add", "sub", "mul", "truediv", "pow", "neg"):
    def _make(op):
        def f(self, *other):
            return getattr(self.values, f"__{op}__")(*[np.asarray(o) if isinstance(o, Parameters)
                                                       else o for o in other])
        f.__name__ = f"__{op}__"
        return f
    setattr(Parameters, f"__{_op}__", _make(_op))
    if _op != "neg":
        def _rmake(op):
            def f(self, other):
                return getattr(self.values, f"__r{op}__")(other)
            return f
        setattr(Parameters, f"__r{_op}__", _rmake(_op))


def parameter_vector(default, parameters):
    """The vector for `run(..., parameters=...)`: None gives `default` (a `Parameters`), a
    `Parameters` its vector, a mapping {symbol: value} `default` with those replaced, a vector
    passes through."""
    if parameters is None:
        return default.values
    if isinstance(parameters, Parameters):
        return parameters.values
    if isinstance(parameters, dict):
        return default.vector(parameters)
    return parameters
