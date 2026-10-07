"""Emitted named constants.

No numerical constant is hard-coded in a backend: CFL factors, tableau
coefficients, tolerances (D9) and the write-gate epsilon are named values that
every walker EMITS at the head of the generated code (Python module constants,
C++ ``constexpr``).  Template expressions reference them by name.  A value of
``None`` means "not known yet" (template shown before setup); compiling
requires every referenced constant to be valued.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import sympy as sp

from zoomy_core.solver.template.kinds import SCALAR_KINDS


class ConstantError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Const:
    name: str
    value: Any
    kind: str = "real"
    provenance: str = ""

    def __post_init__(self):
        if not self.name.isidentifier():
            raise ConstantError(f"constant name must be an identifier: {self.name!r}")
        if self.kind not in SCALAR_KINDS:
            raise ConstantError(f"constant {self.name}: kind must be scalar")
        if self.value is not None:
            v = self.value
            if self.kind == "int":
                if isinstance(v, bool) or int(v) != v:
                    raise ConstantError(f"constant {self.name} must be an int")
                object.__setattr__(self, "value", int(v))
            elif self.kind == "bool":
                object.__setattr__(self, "value", bool(v))
            else:
                object.__setattr__(self, "value", float(v))


class ConstTable:
    """Ordered ``name -> Const``."""

    def __init__(self, consts=()):
        self._c: dict = {}
        for c in consts:
            self.add(c)

    def add(self, c: Const, *, replace=False) -> Const:
        old = self._c.get(c.name)
        if old is not None and not replace and old.value is not None \
                and c.value is not None and old.value != c.value:
            raise ConstantError(f"constant {c.name} redefined: {old.value!r} "
                                f"-> {c.value!r}")
        if old is not None and c.value is None and not replace:
            return old
        self._c[c.name] = c
        return c

    def define(self, name, value, kind="real", provenance="", *, replace=False):
        self.add(Const(name, value, kind, provenance), replace=replace)
        return sp.Symbol(name)

    def set(self, name, value):
        c = self._c[name]
        self._c[name] = dataclasses.replace(c, value=value)

    def symbol(self, name) -> sp.Symbol:
        if name not in self._c:
            raise ConstantError(f"{name!r} is not a declared constant")
        return sp.Symbol(name)

    def __contains__(self, name):
        return name in self._c

    def __getitem__(self, name) -> Const:
        return self._c[name]

    def get(self, name, default=None):
        return self._c.get(name, default)

    def __iter__(self):
        return iter(self._c.values())

    def names(self):
        return tuple(self._c)

    def merged(self, other: "ConstTable") -> "ConstTable":
        t = ConstTable(self)
        for c in other:
            t.add(c)
        return t

    def copy(self) -> "ConstTable":
        return ConstTable(self)

    def missing(self, names) -> tuple:
        return tuple(n for n in names if n in self._c and self._c[n].value is None)

    def report(self) -> str:
        rows = []
        for c in self._c.values():
            rows.append(f"{c.name:24s} = {format_value(c.value, c.kind):<22} "
                        f"[{c.kind}] {c.provenance}")
        return "\n".join(rows)


def format_value(v, kind) -> str:
    """Python spelling of a constant value (exact for floats)."""
    if v is None:
        return "None"
    if kind == "bool":
        return "True" if v else "False"
    if kind == "int":
        return str(int(v))
    v = float(v)
    if math.isinf(v):
        return 'float("inf")' if v > 0 else 'float("-inf")'
    if math.isnan(v):
        return 'float("nan")'
    return repr(v)


__all__ = ["Const", "ConstTable", "ConstantError", "format_value"]
