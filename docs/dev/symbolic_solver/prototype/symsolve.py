"""Symbolic solver core: declared building blocks + two printers.

A solver is ONE sympy expression DAG built by plain Python functions.
- Block: a sympy Function the backend implements under the same name.
- Effect: a Block with side effects (I/O, preCICE); calls are ordered by an
  implicit token the builder threads through While / Cond for you.
- While / Cond / Root / LinSolve take Python callables (-> sp.Lambda).
The printer knows only: Lambda -> closure, Tuple, Get, scoped CSE.
"""
import inspect
import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.printing.pycode import PythonCodePrinter
from sympy.printing.cxx import CXX17CodePrinter


class Block(sp.Function):
    """Opaque building block; every backend supplies a callable of this name."""


class Get(sp.Function):
    nargs = 2


# ── control blocks ──────────────────────────────────────────────────────────
class while_loop(Block): pass
class cond(Block): pass
class root(Block): pass
class linsolve(Block): pass


class _Ctx:
    tok = None


_CTX = _Ctx()


class EffectNode(Block):
    pass


class Effect:
    """Declared effect: `name = Effect(n_out)` inside a namespace class.
    Backend name = <namespace>_<attr>; it receives the token first and
    returns (token, *outputs)."""

    def __init__(self, n_out=0):
        self.n_out = n_out

    def __set_name__(self, owner, name):
        self.node = type(f"{owner.__name__}_{name}", (EffectNode,), {})

    def __call__(self, *args):
        node = self.node(_CTX.tok, *args)
        _CTX.tok = Get(node, 0)
        outs = tuple(Get(node, i + 1) for i in range(self.n_out))
        return outs[0] if self.n_out == 1 else (outs or None)


class Kernel:
    """Declared pointwise kernel handle (printed by name, resolved by backend)."""

    def __set_name__(self, owner, name):
        self.sym = sp.Symbol(name)

    def __get__(self, obj, owner):
        return self.sym


def unpack(call, n):
    return tuple(Get(call, i) for i in range(n))


def _tuple(out):
    return tuple(out) if isinstance(out, (tuple, list)) else (out,)


def lam(f, names=None):
    names = names or list(inspect.signature(f).parameters)
    args = tuple(sp.Dummy(n) for n in names)
    out = f(*args)
    return sp.Lambda(args, sp.Tuple(*out) if isinstance(out, (tuple, list)) else out)


def _pure(f, names=None):
    before = _CTX.tok
    L = lam(f, names)
    assert _CTX.tok is before, "effects are not allowed inside this callable"
    return L


def While(c, body, init):
    init, outer = tuple(init), _CTX.tok
    names = list(inspect.signature(body).parameters)
    n = len(init)

    def b(*a):
        _CTX.tok = a[n]
        return (*_tuple(body(*a[:n])), _CTX.tok)
    Lc = _pure(lambda *a: c(*a[:n]), names + ["tok"])
    Lb = lam(b, names + ["tok"])
    res = unpack(while_loop(Lc, Lb, sp.Tuple(*init, outer)), n + 1)
    _CTX.tok = res[n]
    return res[:n] if n > 1 else res[0]


def Cond(pred, then, other):
    outer = _CTX.tok
    toks = []

    def wrap(f):
        def g():
            _CTX.tok = outer
            out = _tuple(f())
            toks.append(_CTX.tok)
            return (*out, _CTX.tok)
        return g
    La, Lb = lam(wrap(then)), lam(wrap(other))
    n = len(La.expr) - 1
    res = unpack(cond(pred, La, Lb), n + 1)
    _CTX.tok = res[n] if any(t is not outer for t in toks) else outer
    return res[:n] if n > 1 else res[0]


def Root(residual, x0):
    return root(_pure(residual), x0)


def LinSolve(matvec, b):
    return linsolve(_pure(matvec), b)


# ── scoped CSE / let-floating ───────────────────────────────────────────────
def _children(e):
    if isinstance(e, sp.Lambda):
        return [e.expr]
    return [] if e.is_Atom else list(e.args)


_CONTROL = (while_loop, cond, root, linsolve)


class _Plan:
    def __init__(self, outputs, mode="cse", labels=()):
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
        if mode == "nocse":      # only what the language needs as statements
            keep = lambda e: isinstance(e, (sp.Lambda, EffectNode) + _CONTROL)
        else:                    # cse / debug: every block call + shared values
            keep = lambda e: (isinstance(e, (sp.Lambda, AppliedUndef, Block))
                              or e in labels
                              or (cnt.get(e, 0) >= 2
                                  and not isinstance(e, (sp.Tuple, Get))))
        self.named = [e for e in self.post if not e.is_Atom and keep(e)]
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


# ── printers ────────────────────────────────────────────────────────────────
class _Common:
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
                nm = self._fresh(("f" if isinstance(e, sp.Lambda) else "x") + str(self.k))
            if isinstance(e, sp.Lambda):
                args = [self._print(v) for v in e.variables]
                body = self.emit(plan, e, ind + "    ")
                lines += self.lambda_def(nm, args, body, self._print(e.expr), ind)
            else:
                self.defining = e
                lines.append(self.assign(nm, self._print(e), ind))
                self.defining = None
            self.names[e] = nm
        return lines

    def procedure(self, name, args, build, mode="cse", sources=()):
        """build(*args) -> outputs; the final effect token is returned last.
        mode: 'cse' (default) | 'nocse' (nested expressions) | 'debug'
        (every value keeps the variable name it had in `sources` files)."""
        self.names, self.defining, self.k, self.labels = {}, None, 0, {}
        self.used = {str(a) for a in args} | {"tok"}
        tok = sp.Symbol("tok")
        _CTX.tok = tok
        if mode == "debug":
            outs = _with_labels(build, args, sources, self.labels)
        else:
            outs = build(*args)
        outputs = [sp.sympify(o) for o in _tuple(outs)] + [_CTX.tok]
        for a in (*args, tok):
            self.names[a] = str(a)
        plan = _Plan(outputs, mode, self.labels)
        body = self.emit(plan, None, "    ")
        return self.proc_def(name, [str(a) for a in (*args, tok)], body,
                             self._print(sp.Tuple(*outputs)))


def _with_labels(build, args, sources, labels):
    """Run the solver functions and remember each value's local variable name."""
    import sys
    files = set(sources)

    def harvest(frame):
        for k, v in frame.f_locals.items():
            if isinstance(v, sp.Basic) and not v.is_Atom and v not in labels:
                labels[v] = k

    def local(frame, event, arg):
        if event in ("line", "return"):
            harvest(frame)
        return local

    def glob(frame, event, arg):
        return local if frame.f_code.co_filename in files else None
    sys.settrace(glob)
    try:
        return build(*args)
    finally:
        sys.settrace(None)


class PyPrinter(_Common, PythonCodePrinter):
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
        return "\n".join([f"def {name}({', '.join(args)}):", *body, f"    return {ret}"])


class CxxPrinter(_Common, CXX17CodePrinter):
    def __init__(self):
        super().__init__({"allow_unknown_functions": True})

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
        return [f"{ind}auto {nm} = [&]({a}) {{", *body, f"{ind}    return {ret};", f"{ind}}};"]

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
