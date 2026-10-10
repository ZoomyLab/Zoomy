"""Feasibility prototype: a string-free authoring layer over the existing IR.

Names are class attributes (``__set_name__``), so they autocomplete, typos
raise ``AttributeError ... Did you mean``, and the generated identifiers come
from the attribute names.  The IR, checker and walkers stay as they are.
"""
from __future__ import annotations

import contextvars
import difflib
import enum
import inspect

import sympy as sp

from zoomy_core.solver.template.ir import Assign, Call, Hook, IfStatic
from zoomy_core.solver.template.kinds import Port
from zoomy_core.solver.template.registry import BlockRegistry, BlockSpec


class Kind(str, enum.Enum):
    real = "real"
    int = "int"
    params = "params"
    geom = "geom"
    cell_state = "cell_state"
    cell_aux = "cell_aux"
    cell_mask = "cell_mask"
    cell_grad = "cell_grad"
    face_state = "face_state"
    face_aux = "face_aux"
    bc_state = "bc_state"
    bc_aux = "bc_aux"


class Comm(str, enum.Enum):
    none = "none"
    neighbor = "neighbor"
    global_ = "global"


class Var(sp.Symbol):
    """A typed template variable: a sympy Symbol carrying its kind."""

    vkind: Kind


class _VarDecl:
    def __init__(self, kind: Kind):
        self.kind = kind
        self.var: Var

    def __set_name__(self, owner, name):
        self.var = Var(name)
        self.var.vkind = self.kind

    def __get__(self, obj, owner=None) -> Var:
        return self.var


def var(kind: Kind) -> Var:
    """Declare a variable inside a namespace class (typed as Var for IDEs)."""
    return _VarDecl(kind)  # type: ignore[return-value]


class V:
    """Every template variable name, declared once with its kind."""

    Q = var(Kind.cell_state)
    Qaux = var(Kind.cell_aux)
    p = var(Kind.params)
    geom = var(Kind.geom)
    time = var(Kind.real)
    dt = var(Kind.real)
    o1 = var(Kind.cell_mask)
    bc_Q = var(Kind.bc_state)
    bc_aux = var(Kind.bc_aux)
    Q_L = var(Kind.face_state)
    Q_R = var(Kind.face_state)
    Qaux_L = var(Kind.face_aux)
    Qaux_R = var(Kind.face_aux)
    grad = var(Kind.cell_grad)
    F = var(Kind.face_state)
    Dp = var(Kind.face_state)
    Dm = var(Kind.face_state)
    cell_term = var(Kind.cell_state)
    Q_cand = var(Kind.cell_state)
    troubled = var(Kind.cell_mask)


# ── recording builder ─────────────────────────────────────────────────────────

_BODY: contextvars.ContextVar = contextvars.ContextVar("template_body")


class body:
    """``with body() as stmts:`` collects the statements recorded inside."""

    def __enter__(self):
        self.stmts = []
        self._tok = _BODY.set(self.stmts)
        return self.stmts

    def __exit__(self, *exc):
        _BODY.reset(self._tok)


def _emit(stmt):
    _BODY.get().append(stmt)


def _names(vs):
    return tuple(v.name for v in vs)


def _ret(outs):
    return outs[0] if len(outs) == 1 else tuple(outs)


def assign(target: Var, expr, note=""):
    _emit(Assign(target.name, expr, note=note))
    return target


class Block:
    """A block signature; calling it inside ``body()`` records a ``Call``."""

    def __init__(self, ins, outs, comm=Comm.none, doc=""):
        self.ins, self.outs, self.comm, self.doc = tuple(ins), tuple(outs), comm, doc

    def __set_name__(self, owner, name):
        self.name = name

    @property
    def spec(self) -> BlockSpec:
        return BlockSpec(self.name,
                         tuple(Port(v.name, v.vkind.value) for v in self.ins),
                         tuple(Port(v.name, v.vkind.value) for v in self.outs),
                         comm=self.comm.value, doc=self.doc)

    def __call__(self, *args: Var, note=""):
        if len(args) != len(self.ins):
            raise TypeError(f"{self.name} takes {len(self.ins)} inputs "
                            f"{_names(self.ins)}, got {len(args)}")
        for a, p in zip(args, self.ins):
            if a.vkind != p.vkind:
                raise TypeError(f"{self.name}: input {p.name} wants {p.vkind.value}, "
                                f"got {a.name}: {a.vkind.value}")
        _emit(Call(self.name, _names(args), _names(self.outs), note=note))
        return _ret(self.outs)


class HookPoint:
    """A named extension point; calling it records a ``Hook``."""

    def __init__(self, ins, outs, doc=""):
        self.ins, self.outs, self.doc = tuple(ins), tuple(outs), doc

    def __set_name__(self, owner, name):
        self.name = name

    def __call__(self, *args: Var, default=None, note=""):
        dflt = ()
        if default is not None:
            with body() as dflt:
                default()
        _emit(Hook(self.name, _names(args), _names(self.outs), default=dflt,
                   note=note or self.doc))
        return _ret(self.outs)


class Flag:
    def __init__(self, doc=""):
        self.doc = doc

    def __set_name__(self, owner, name):
        self.name = name


def if_static(flag: Flag, then, otherwise=None):
    with body() as t:
        then()
    o = ()
    if otherwise is not None:
        with body() as o:
            otherwise()
    _emit(IfStatic(flag.name, t, o))


class BlockSet:
    """Base of a block namespace.  Subclass to add blocks."""

    @classmethod
    def blocks(cls) -> dict:
        return {k: v for k, v in vars_mro(cls).items() if isinstance(v, Block)}

    @classmethod
    def registry(cls) -> BlockRegistry:
        return BlockRegistry([b.spec for b in cls.blocks().values()])

    @classmethod
    def bodies(cls, impl) -> dict:
        """Bind an implementation object (methods named after the blocks):
        every block needs a method of the right arity; a public method that is
        not a block is a typo -> closest-match error."""
        blocks = cls.blocks()
        table, errors = {}, []
        for name in dir(impl):
            if name.startswith("_") or not callable(getattr(impl, name)):
                continue
            if name not in blocks:
                near = difflib.get_close_matches(name, blocks, n=1)
                errors.append(f"{type(impl).__name__}.{name} is not a block of "
                              f"{cls.__name__}" + (f"; did you mean {near[0]!r}?"
                                                    if near else ""))
        for name, b in blocks.items():
            fn = getattr(impl, name, None)
            if fn is None:
                continue
            n = len(inspect.signature(fn).parameters)
            if n != len(b.ins):
                errors.append(f"{type(impl).__name__}.{name} takes {n} args, "
                              f"block wants {len(b.ins)} {_names(b.ins)}")
            table[name] = fn
        if errors:
            raise TypeError("\n".join(errors))
        return table


def vars_mro(cls):
    out = {}
    for c in reversed(cls.__mro__):
        out.update(vars(c))
    return out
