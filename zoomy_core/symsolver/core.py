"""Symbolic solvers: the machinery under the solver templates.

A solver template is a plain Python function.  Run once with sympy symbols as
inputs, it returns ONE sympy expression graph; a printer turns that graph into
source code for a language, and a backend runtime supplies what every called
name means.

Three kinds of names appear in the graph:

* a kernel -- ``nsm.call.<slot>`` (flux, fluctuations, source, BC, ...), the
  registrar ``call`` entry every Model / Numerics / NSM carries.  Handed to a
  block it becomes ``KernelRef`` and prints as ``nsm.<slot>``; the existing
  kernel printers generate the kernel itself.
* ``Block`` -- an operation on the whole mesh or a solve (gather/scatter,
  gradient, reduction, ``while_loop``, ``cond``, ``root``, ``linsolve``).  One
  declared class in core, one implementation per backend.
* ``Effect`` -- a block that talks to the outside (output, preCICE).  The
  builder threads an order token through effects so they run in program order.

Python control flow in a template (``for`` over an RK tableau, ``if order ==
2``) runs at BUILD time and is unrolled.  Run-time control flow is spelled
``While`` / ``Cond``, which trace their Python callables into ``sp.Lambda``.
"""
from __future__ import annotations

import inspect
import os
import sys
import threading

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.printing.cxx import CXX17CodePrinter
from sympy.printing.pycode import PythonCodePrinter

from zoomy_core.model.basefunction import KernelRef  # noqa: F401  (printed below)


# ── names in the graph ──────────────────────────────────────────────────────
class Block(sp.Function):
    """Opaque building block; every backend supplies a callable of this name."""


class Get(sp.Function):
    """Component ``i`` of a block that returns several values."""
    nargs = 2


class while_loop(Block):
    """``(cond, body, init) -> final carry``; cond/body take the carry."""


class cond(Block):
    """``(pred, then, other) -> value``; the branches take no arguments."""


class root(Block):
    """``(residual, x0) -> x`` with ``residual(x) = 0``."""


class linsolve(Block):
    """``(matvec, b) -> x`` with ``matvec(x) = b``."""


class EffectNode(Block):
    pass


class Effect:
    """Declared effect: ``name = Effect(n_out)`` inside a namespace class.
    The backend name is ``<namespace>_<attr>``; it receives the order token
    first and returns ``(token, *outputs)``."""

    def __init__(self, n_out=0):
        self.n_out = n_out

    def __set_name__(self, owner, name):
        self.node = type(f"{owner.__name__}_{name}", (EffectNode,), {})

    def __call__(self, *args):
        node = self.node(_CTX.tok, *args)
        res = unpack(node, self.n_out + 1)
        _CTX.tok, outs = res[0], res[1:]
        return outs[0] if self.n_out == 1 else (outs or None)


# ── building the graph ──────────────────────────────────────────────────────
class _Ctx(threading.local):
    """The current effect token of the build running in this thread."""
    tok = None


_CTX = _Ctx()


_ARITY = {}


def unpack(call, n):
    _ARITY[call] = n
    return tuple(Get(call, i) for i in range(n))


def _tuple(out):
    return tuple(out) if isinstance(out, (tuple, list)) else (out,)


def _lam(f, names=None):
    names = names or list(inspect.signature(f).parameters)
    args = tuple(sp.Dummy(n) for n in names)
    out = f(*args)
    return sp.Lambda(args, sp.Tuple(*out) if isinstance(out, (tuple, list)) else out)


def _pure(f, names=None):
    before = _CTX.tok
    L = _lam(f, names)
    assert _CTX.tok is before, "effects are not allowed inside this callable"
    return L


def While(c, body, init):
    """Run-time loop with an explicit carry: ``body(*carry) -> carry``."""
    init, outer = tuple(init), _CTX.tok
    names = list(inspect.signature(body).parameters)
    n = len(init)

    def b(*a):
        _CTX.tok = a[n]
        return (*_tuple(body(*a[:n])), _CTX.tok)
    Lc = _pure(lambda *a: c(*a[:n]), names + ["tok"])
    Lb = _lam(b, names + ["tok"])
    res = unpack(while_loop(Lc, Lb, sp.Tuple(*init, outer)), n + 1)
    _CTX.tok = res[n]
    return res[:n] if n > 1 else res[0]


def Cond(pred, then, other):
    """Run-time branch; only the taken branch executes."""
    outer, toks = _CTX.tok, []

    def wrap(f):
        def g():
            _CTX.tok = outer
            out = _tuple(f())
            toks.append(_CTX.tok)
            return (*out, _CTX.tok)
        return g
    La, Lb = _lam(wrap(then)), _lam(wrap(other))
    n = len(La.expr) - 1
    res = unpack(cond(pred, La, Lb), n + 1)
    _CTX.tok = res[n] if any(t is not outer for t in toks) else outer
    return res[:n] if n > 1 else res[0]


def Root(residual, x0):
    return root(_pure(residual), x0)


def LinSolve(matvec, b):
    return linsolve(_pure(matvec), b)


# ── scoped common-subexpression elimination ─────────────────────────────────
def _children(e):
    if isinstance(e, sp.Lambda):
        return [e.expr]
    return [] if e.is_Atom else list(e.args)


class _Plan:
    """Which nodes become statements, and in which scope (closure) each one is
    placed: the innermost scope that contains every use, so nothing is hoisted
    out of a ``Cond`` branch or a loop body it does not belong to."""

    def __init__(self, outputs):
        self.post, seen, users = [], set(), {}

        def dfs(e):
            if e in seen:
                return
            seen.add(e)
            for c in _children(e):
                users.setdefault(c, []).append(e)
                dfs(c)
            self.post.append(e)
        for o in outputs:
            dfs(o)
        cnt = {e: len(u) for e, u in users.items()}
        self.gets = {}
        for e in self.post:
            if isinstance(e, Get):
                self.gets.setdefault(e.args[0], {})[int(e.args[1])] = e
        self.only_gets = {e: all(isinstance(u, Get) for u in users.get(e, ()))
                          for e in self.gets}
        self.named = [e for e in self.post if not e.is_Atom and (
            isinstance(e, (sp.Lambda, AppliedUndef, Block))
            or (cnt.get(e, 0) >= 2 and not isinstance(e, (sp.Tuple, Get))))]
        named = set(self.named)
        self.parent, self.place, ctx = {None: None}, {}, {}
        for o in outputs:
            ctx.setdefault(o, set()).add(None)
        for e in reversed(self.post):
            sc = ctx.get(e, set())
            if e in named:
                p = self.place[e] = self._lca(sc)
                if isinstance(e, sp.Lambda):
                    self.parent[e] = p
                inner = {e} if isinstance(e, sp.Lambda) else {p}
            else:
                inner = {e} if isinstance(e, sp.Lambda) else sc
            for c in _children(e):
                ctx.setdefault(c, set()).update(inner)

    def _depth(self, s):
        d = 0
        while s is not None:
            s, d = self.parent[s], d + 1
        return d

    def _lca(self, scopes):
        scopes = list(scopes)
        a = scopes[0]
        for b in scopes[1:]:
            da, db = self._depth(a), self._depth(b)
            while da > db:
                a, da = self.parent[a], da - 1
            while db > da:
                b, db = self.parent[b], db - 1
            while a is not b:
                a, b = self.parent[a], self.parent[b]
        return a

    def in_scope(self, s):
        return [e for e in self.named if self.place[e] is s]


_HERE = os.path.dirname(os.path.abspath(__file__))


def _build_with_names(build, args, labels, roots):
    """Run the templates and remember the local variable name each value had,
    so the printed code reuses the author's names (``dt``, ``Qb``, ``bad``)."""
    def harvest(frame):
        for k, v in frame.f_locals.items():
            if isinstance(v, sp.Basic) and not v.is_Atom and v not in labels:
                labels[v] = k

    def local(frame, event, arg):
        if event in ("line", "return"):
            harvest(frame)
        return local

    def glob(frame, event, arg):
        f = frame.f_code.co_filename
        if f == __file__ or not any(f.startswith(r) for r in roots):
            return None
        return local
    old = sys.gettrace()
    sys.settrace(glob)
    try:
        return build(*args)
    finally:
        sys.settrace(old)


# ── printers: syntax only ───────────────────────────────────────────────────
class _Common:
    kernel_prefix = {}

    def _print(self, e, **kw):
        if e in self.names and e is not self.defining:
            return self.names[e]
        return super()._print(e, **kw)

    def _fresh(self, base):
        nm, k = base, 1
        while nm in self.used:
            k += 1
            nm = f"{base}_{k}"
        self.used.add(nm)
        return nm

    def _print_Dummy(self, e):
        self.names[e] = self._fresh(e.name)
        return self.names[e]

    def _print_KernelRef(self, e):
        return self.kernel_prefix.get(e.owner, e.owner + ".") + e.slot

    def _print_Function(self, e):
        return f"{e.func.__name__}({', '.join(self._print(a) for a in e.args)})"

    def _print_Abs(self, e):
        return f"abs({self._print(e.args[0])})"

    def _print_Min(self, e):
        return self._fold("minimum", e.args)

    def _print_Max(self, e):
        return self._fold("maximum", e.args)

    def _fold(self, f, a):
        s = self._print(a[0])
        for x in a[1:]:
            s = f"{f}({s}, {self._print(x)})"
        return s

    def emit(self, plan, scope, ind):
        lines = []
        for e in plan.in_scope(scope):
            if e in self.labels:
                nm = self._fresh(self.labels[e])
            else:
                self.k += 1
                nm = self._fresh(("f" if isinstance(e, sp.Lambda) else "x")
                                 + str(self.k))
            if isinstance(e, sp.Lambda):
                args = [self._print(v) for v in e.variables]
                body = self.emit(plan, e, ind + "    ")
                lines += self.lambda_def(nm, args, body, self._print(e.expr), ind)
            else:
                self.defining = e
                rhs = self._print(e)
                self.defining = None
                gets = plan.gets.get(e, {})
                n = _ARITY.get(e)
                if gets and n and plan.only_gets[e] and self.destructure:
                    outs = []
                    for i in range(n):
                        g = gets.get(i)
                        if g is None:
                            outs.append("_")
                            continue
                        self.k += 1
                        gn = self._fresh(self.labels.get(g, f"x{self.k}"))
                        self.names[g] = gn
                        outs.append(gn)
                    lines.append(self.assign_many(outs, rhs, ind))
                    continue
                lines.append(self.assign(nm, rhs, ind))
                self.names[e] = nm
                for i, g in sorted(gets.items()):
                    if g in self.labels:
                        gn = self._fresh(self.labels[g])
                        lines.append(self.assign(gn, self._print(g), ind))
                        self.names[g] = gn
                continue
            self.names[e] = nm
        return lines

    def procedure(self, name, args, build, roots=(_HERE,)):
        """Print ``build(*args)`` as one function.  The effect token is the
        last argument and the last output.  Values keep the variable names
        they had in template code under ``roots``."""
        self.names, self.defining, self.k, self.labels = {}, None, 0, {}
        self.used = {str(a) for a in args} | {"tok"}
        tok = sp.Symbol("tok")
        _CTX.tok = tok
        outs = _build_with_names(build, args, self.labels, tuple(roots))
        outputs = [sp.sympify(o) for o in _tuple(outs)] + [_CTX.tok]
        for a in (*args, tok):
            self.names[a] = str(a)
        body = self.emit(_Plan(outputs), None, "    ")
        return self.proc_def(name, [str(a) for a in (*args, tok)], body,
                             self._print(sp.Tuple(*outputs)))


class PyPrinter(_Common, PythonCodePrinter):
    """Python family: numpy and JAX print the same text."""
    destructure = True

    def assign_many(self, names, rhs, ind):
        lhs = ", ".join(names) + ("," if len(names) == 1 else "")
        return f"{ind}{lhs} = {rhs}"

    def __init__(self):
        super().__init__({"allow_unknown_functions": True,
                          "fully_qualified_modules": False})

    def _print_Get(self, e):
        return f"{self._print(e.args[0])}[{e.args[1]}]"

    def _print_Tuple(self, e):
        return "(" + "".join(self._print(a) + ", " for a in e.args) + ")"

    def assign(self, nm, rhs, ind):
        return f"{ind}{nm} = {rhs}"

    def lambda_def(self, nm, args, body, ret, ind):
        return [f"{ind}def {nm}({', '.join(args)}):", *body, f"{ind}    return {ret}"]

    def proc_def(self, name, args, body, ret):
        return "\n".join([f"def {name}({', '.join(args)}):", *body,
                          f"    return {ret}"])


class CxxPrinter(_Common, CXX17CodePrinter):
    """C++ family: foam and amrex print the same text.  (No structured
    bindings: C++17 lambdas cannot capture them.)  ``nsm.source`` is a member
    of the backend's kernel struct; a kernel map with several outputs carries
    the count, ``faces<2>(...)``, since C++ fixes the result type at compile
    time."""
    destructure = False

    def __init__(self):
        super().__init__({"allow_unknown_functions": True})

    def _print_Function(self, e):
        n = _ARITY.get(e, 1)
        tmpl = f"<{n}>" if n > 1 and e.args and isinstance(e.args[0], KernelRef) else ""
        return f"{e.func.__name__}{tmpl}({', '.join(self._print(a) for a in e.args)})"

    def _print_Get(self, e):
        return f"std::get<{e.args[1]}>({self._print(e.args[0])})"

    def _print_Tuple(self, e):
        return f"std::make_tuple({', '.join(self._print(a) for a in e.args)})"

    def _print_Pow(self, e):
        b, x = e.args
        if x == -1:
            return f"(1.0/{self.parenthesize(b, 1000)})"
        return f"pow({self._print(b)}, {self._print(x)})"

    def assign(self, nm, rhs, ind):
        return f"{ind}auto {nm} = {rhs};"

    def lambda_def(self, nm, args, body, ret, ind):
        a = ", ".join(f"auto {x}" for x in args)
        return [f"{ind}auto {nm} = [&]({a}) {{", *body, f"{ind}    return {ret};",
                f"{ind}}};"]

    def proc_def(self, name, args, body, ret):
        a = ", ".join(f"auto {x}" for x in args)
        return "\n".join([f"inline auto {name} = []({a}) {{", *body,
                          f"    return {ret};", "};"])


def contract():
    """Every name a backend runtime must define (blocks + effects)."""
    out, todo = set(), [Block]
    while todo:
        for c in todo.pop().__subclasses__():
            todo.append(c)
            if c is not EffectNode:
                out.add(c.__name__)
    return sorted(out)
