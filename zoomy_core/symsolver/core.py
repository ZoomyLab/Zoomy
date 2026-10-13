"""The machinery under the solver templates: `Block` and `Effect` names in a sympy
graph, scoped common-subexpression planning, and the Python and C++ printers. A
template runs once on sympy symbols and a printer turns the resulting graph into
source code.
"""
from __future__ import annotations

import inspect
import os
import sys
import threading

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.printing.pycode import PythonCodePrinter

from zoomy_core.model.basefunction import KernelRef  # noqa: F401  (printed below)
from zoomy_core.transformation.cxx_base import CxxBase


# ── names in the graph ──────────────────────────────────────────────────────
class Block(sp.Function):
    """Opaque building block; every backend supplies a callable of this name."""


class Get(sp.Function):
    """Component `i` of a block that returns several values."""
    nargs = 2


class while_loop(Block):
    """`(cond, body, init) -> final carry`; cond and body take the carry."""


class cond(Block):
    """`(pred, then, other) -> value`; the branches take no arguments."""


class root(Block):
    """`(residual, x0) -> x` with `residual(x) = 0`."""


class linsolve(Block):
    """`(matvec, b) -> x` with `matvec(x) = b`, solved matrix-free (restarted GMRES
    by default).

    References:
        Saad, Schultz 1986, https://doi.org/10.1137/0907058
    """


class EffectNode(Block):
    """Base of the graph nodes created by `Effect`."""
    pass


class Effect:
    """Declared effect, `name = Effect(n_out)` inside a namespace class; the backend
    name is `<namespace>_<attr>`. It receives the order token first and returns
    `(token, *outputs)`.
    """

    def __init__(self, n_out=0):
        """`n_out`: number of values the effect returns."""
        self.n_out = n_out

    def __set_name__(self, owner, name):
        """Create the node class `<owner>_<attr>`."""
        self.node = type(f"{owner.__name__}_{name}", (EffectNode,), {})

    def __call__(self, *args):
        """Record the effect in program order and return its outputs."""
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
    """Return `n` `Get` nodes for the components of a multi-output `call`.
    """
    _ARITY[call] = n
    return tuple(Get(call, i) for i in range(n))


def _tuple(out):
    """Wrap a non-sequence in a 1-tuple."""
    return tuple(out) if isinstance(out, (tuple, list)) else (out,)


def _lam(f, names=None):
    """Trace `f` into an `sp.Lambda` over fresh dummies."""
    names = names or list(inspect.signature(f).parameters)
    args = tuple(sp.Dummy(n) for n in names)
    out = f(*args)
    return sp.Lambda(args, sp.Tuple(*out) if isinstance(out, (tuple, list)) else out)


def _pure(f, names=None):
    """As `_lam`, asserting that `f` runs no effect."""
    before = _CTX.tok
    L = _lam(f, names)
    assert _CTX.tok is before, "effects are not allowed inside this callable"
    return L


def While(c, body, init):
    """Run-time loop with an explicit carry: `body(*carry) -> carry`; returns the
    final carry.
    """
    init, outer = tuple(init), _CTX.tok
    names = list(inspect.signature(body).parameters)
    n = len(init)

    def b(*a):
        """Loop body with the effect token threaded as an extra carry."""
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
        """Thread the effect token through a branch."""
        def g():
            """The branch as a no-argument callable returning `(*outputs, token)`."""
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
    """Solve `residual(x) = 0` from `x0` at run time."""
    return root(_pure(residual), x0)


def LinSolve(matvec, b):
    """Solve `matvec(x) = b` at run time (see `linsolve`)."""
    return linsolve(_pure(matvec), b)


# ── scoped common-subexpression elimination ─────────────────────────────────
def _children(e):
    """The child nodes of a graph node (a `Lambda`'s body)."""
    if isinstance(e, sp.Lambda):
        return [e.expr]
    return [] if e.is_Atom else list(e.args)


class _Plan:
    """Decide which graph nodes become statements and in which scope each is placed. A
    node goes to the innermost scope containing all its uses, then out of every
    function whose parameters it does not depend on, so it is computed once.
    """

    def __init__(self, outputs):
        """Plan the statements for the graph reaching `outputs`."""
        self.post, seen, users = [], set(), {}

        def dfs(e):
            """Depth-first walk recording post-order and the users of each node."""
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
        # the lambda parameters each node depends on (bottom-up)
        free = {}
        for e in self.post:
            if isinstance(e, sp.Dummy):
                free[e] = frozenset((e,))
            elif isinstance(e, sp.Lambda):
                free[e] = free[e.expr] - set(e.variables)
            else:
                free[e] = frozenset().union(*(free[c] for c in _children(e)))
        self.parent, self.place, ctx = {None: None}, {}, {}
        for o in outputs:
            ctx.setdefault(o, set()).add(None)
        for e in reversed(self.post):
            sc = ctx.get(e, set())
            if e in named:
                p = self._lca(sc)
                while (isinstance(p, sp.Lambda) and p.variables
                       and not free[e] & set(p.variables)):
                    p = self.parent[p]
                self.place[e] = p
                if isinstance(e, sp.Lambda):
                    self.parent[e] = p
                inner = {e} if isinstance(e, sp.Lambda) else {p}
            else:
                inner = {e} if isinstance(e, sp.Lambda) else sc
            for c in _children(e):
                ctx.setdefault(c, set()).update(inner)

    def _depth(self, s):
        """Nesting depth of scope `s`."""
        d = 0
        while s is not None:
            s, d = self.parent[s], d + 1
        return d

    def _lca(self, scopes):
        """Innermost scope containing all `scopes`."""
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
        """The named nodes placed in scope `s`."""
        return [e for e in self.named if self.place[e] is s]


_HERE = os.path.dirname(os.path.abspath(__file__))


def _build_with_names(build, args, labels, roots):
    """Run the templates and remember each value's local variable name, so printed code
    reuses the author's names.
    """
    def harvest(frame):
        """Record the names of the sympy values in `frame`."""
        for k, v in frame.f_locals.items():
            if isinstance(v, sp.Basic) and not v.is_Atom and v not in labels:
                labels[v] = k

    def local(frame, event, arg):
        """Trace function recording local names on every line."""
        if event in ("line", "return"):
            harvest(frame)
        return local

    def glob(frame, event, arg):
        """Trace function that follows only frames under `roots`."""
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
    """Printer behaviour shared by the Python and C++ printers; subclasses supply the
    assignment and function forms.
    """
    kernel_prefix = {}

    def _print(self, e, **kw):
        """Print a named node by its variable name."""
        if e in self.names and e is not self.defining:
            return self.names[e]
        return super()._print(e, **kw)

    def _fresh(self, base):
        """A variable name based on `base` not yet used."""
        nm, k = base, 1
        while nm in self.used:
            k += 1
            nm = f"{base}_{k}"
        self.used.add(nm)
        return nm

    def _print_Dummy(self, e):
        """Print a lambda parameter under a fresh name."""
        self.names[e] = self._fresh(e.name)
        return self.names[e]

    def _print_KernelRef(self, e):
        """Print a kernel as `<owner>.<slot>`."""
        return self.kernel_prefix.get(e.owner, e.owner + ".") + e.slot

    def _print_Float(self, e):
        """Print the exact double (`repr` round-trips; sympy prints 15 digits)."""
        # the double itself (repr round-trips; sympy prints 15 digits): a
        # constant dt printed short ended its run 4e-20 early -> one more
        # step of 4e-20 and a singular pressure solve
        return repr(float(e))

    def _print_Abs(self, e):
        """Print `abs(x)`."""
        return f"abs({self._print(e.args[0])})"

    def emit(self, plan, scope, ind):
        """Emit the statements of `scope` as source lines, naming each node after the
        author's variable where known.
        """
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
        """Print `build(*args)` as one function whose last argument and last output is
        the effect token.

        Args:
            name: function name.
            args: input symbols.
            build: the template callable.
            roots: source directories whose local variable names the printed code
                reuses.
        """
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
        """Tuple-unpacking assignment."""
        lhs = ", ".join(names) + ("," if len(names) == 1 else "")
        return f"{ind}{lhs} = {rhs}"

    def __init__(self):
        """Allow unknown (block) functions and unqualified module names."""
        super().__init__({"allow_unknown_functions": True,
                          "fully_qualified_modules": False})

    def _print_Get(self, e):
        """Print component `i` as `x[i]`."""
        return f"{self._print(e.args[0])}[{e.args[1]}]"

    def _print_Function(self, e):
        """Print an unknown function as a plain call."""
        return f"{e.func.__name__}({', '.join(self._print(a) for a in e.args)})"

    def _fold(self, f, a):
        """Left-nested binary call `f(f(a, b), c)`."""
        s = self._print(a[0])
        for x in a[1:]:
            s = f"{f}({s}, {self._print(x)})"
        return s

    def _print_Min(self, e):
        """Print `Min` as nested `minimum`."""
        return self._fold("minimum", e.args)

    def _print_Max(self, e):
        """Print `Max` as nested `maximum`."""
        return self._fold("maximum", e.args)

    def _print_And(self, e):          # traced booleans: no Python ``and``
        """Print `And` as `logical_and` (traced booleans admit no Python `and`).
        """
        return self._fold("logical_and", e.args)

    def _print_Tuple(self, e):
        """Print a tuple with trailing commas."""
        return "(" + "".join(self._print(a) + ", " for a in e.args) + ")"

    def assign(self, nm, rhs, ind):
        """Assignment statement."""
        return f"{ind}{nm} = {rhs}"

    def lambda_def(self, nm, args, body, ret, ind):
        """Nested `def` returning `ret`."""
        return [f"{ind}def {nm}({', '.join(args)}):", *body, f"{ind}    return {ret}"]

    def proc_def(self, name, args, body, ret):
        """Top-level `def` returning `ret`."""
        return "\n".join([f"def {name}({', '.join(args)}):", *body,
                          f"    return {ret}"])


class CxxPrinter(_Common, CxxBase):
    """C++ family: foam and AMReX print the same text, with the runtime's free functions
    `pow` / `minimum` / `maximum` and no structured bindings (C++17 lambdas
    cannot capture them).
    """
    destructure = False
    math_namespace = ""
    _min_name, _max_name, _nest_left = "minimum", "maximum", True

    def __init__(self):
        """Allow unknown (block) functions."""
        super().__init__({"allow_unknown_functions": True})

    def _callee(self, e):
        """Callee name; a kernel map with several outputs carries the count,
        `faces<2>(...)`, since C++ fixes the result type at compile time.
        """
        n = _ARITY.get(e, 1)
        tmpl = f"<{n}>" if n > 1 and e.args and isinstance(e.args[0], KernelRef) else ""
        return f"{e.func.__name__}{tmpl}"

    def _print_Get(self, e):
        """Print component `i` as `std::get<i>(x)`."""
        return f"std::get<{e.args[1]}>({self._print(e.args[0])})"

    def _print_Tuple(self, e):
        """Print a tuple as `std::make_tuple`."""
        return f"std::make_tuple({', '.join(self._print(a) for a in e.args)})"

    def _print_Pow(self, e):
        """Print `pow(b, x)`, a reciprocal as `1.0/b`."""
        b, x = e.args
        if x == -1:
            return f"(1.0/{self.parenthesize(b, 1000)})"
        return f"pow({self._print(b)}, {self._print(x)})"

    def assign(self, nm, rhs, ind):
        """`auto` declaration."""
        return f"{ind}auto {nm} = {rhs};"

    def lambda_def(self, nm, args, body, ret, ind):
        """Nested `[&]` lambda returning `ret`."""
        a = ", ".join(f"auto {x}" for x in args)
        return [f"{ind}auto {nm} = [&]({a}) {{", *body, f"{ind}    return {ret};",
                f"{ind}}};"]

    def proc_def(self, name, args, body, ret):
        """Top-level `inline auto` lambda returning `ret`."""
        a = ", ".join(f"auto {x}" for x in args)
        return "\n".join([f"inline auto {name} = []({a}) {{", *body,
                          f"    return {ret};", "};"])


def contract():
    """Return the sorted names every backend runtime must define (blocks and effects).
    """
    out, todo = set(), [Block]
    while todo:
        for c in todo.pop().__subclasses__():
            todo.append(c)
            if c is not EffectNode:
                out.add(c.__name__)
    return sorted(out)
