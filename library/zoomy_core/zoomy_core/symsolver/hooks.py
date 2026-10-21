"""Function hooks: a model's operators call an opaque function `hook("name")` (a SymPy function
the derivation keeps opaque, never an aux field); the solver setting `functions={"name": callable}`
implements it on the Python backends (NumPy, JAX).  The C++ backends (foam, AMReX) refuse a hook
at setup.

A derivative the pipeline takes of a hook (source Jacobians, the flux's quasilinear matrix)
appears as `Derivative(f(...), arg)` or `Subs(Derivative(...))`; `HookFunction.derivative`
says how it is printed: `HookDerivative.zero`, the user's partial derivatives (printed like the
function), or `HookDerivative.auto` (a SymPy `Lambda`: SymPy differentiates it on every Python
backend; a callable: JAX AD, so JAX only).
"""
from __future__ import annotations

import re

import param
import sympy as sp
from sympy.core.function import AppliedUndef

from zoomy_core.misc.options import Option, did_you_mean, typed

#: the module-dict key under which a backend hands its `Hooks` to the lambdify sites
HOOKS_KEY = "__zoomy_hooks__"
_SUPPORT = "numpy, jax"


def hook(name):
    """The opaque function `name` for a model's operators (real valued); the solver's
    `functions[name]` implements it."""
    return sp.Function(name, real=True, _is_hook=True)


class HookDerivative(Option):
    """How the derivative of a hook is printed."""
    #: d f / d arg = 0
    zero = "zero"
    #: a SymPy `Lambda`: SymPy differentiates it; a callable: JAX AD (JAX only)
    auto = "auto"


class HookFunction:
    """A hook's implementation `function` (a callable of the backend's arrays, or a SymPy
    `Lambda`) and its `derivative`: a `HookDerivative`, a sequence of callables (one partial per
    argument), or one callable returning all partials.
    """

    def __init__(self, function, derivative=HookDerivative.auto):
        """Validate `function` and `derivative`."""
        if not callable(function):
            raise TypeError(f"HookFunction: {function!r} is not callable")
        derivative = typed(HookDerivative, derivative, "derivative")
        ok = (isinstance(derivative, HookDerivative) or callable(derivative)
              or (isinstance(derivative, (list, tuple)) and all(map(callable, derivative))))
        if not ok:
            raise TypeError(f"derivative={derivative!r}: a HookDerivative, a callable or a "
                            "sequence of callables (one per argument)")
        self.function, self.derivative = function, derivative

    def __call__(self, *args):
        """Evaluate the function."""
        return self.function(*args)

    @property
    def symbolic(self):
        """Whether the function is a SymPy `Lambda` (inlined, differentiated by SymPy)."""
        return isinstance(self.function, sp.Lambda)


def hook_function(derivative=HookDerivative.auto):
    """Decorator: `@hook_function(derivative=...)` makes a `HookFunction`."""
    return lambda function: HookFunction(function, derivative)


class FunctionsParam(param.Dict):
    """`functions`: hook name (an identifier) -> callable, `sp.Lambda` or `HookFunction`."""

    def _validate(self, val):
        super()._validate(val)
        for k, v in (val or {}).items():
            if not (isinstance(k, str) and k.isidentifier()):
                raise ValueError(f"functions: the name {k!r} is not an identifier")
            if not callable(v):
                raise TypeError(f"functions[{k!r}]: {v!r} is not callable")


class HookError(NotImplementedError):
    """A hook (or its derivative) a backend cannot print."""


def hook_names(expr):
    """The names of the hooks `expr` calls."""
    return {a.func.__name__ for a in sp.sympify(expr).atoms(AppliedUndef)
            if getattr(a.func, "_is_hook", False)}


class Hooks:
    """A backend's hooks: `lower(expr)` replaces hook derivatives (and inlines SymPy hooks);
    `module` holds the callables the lowered expressions call.  `grad(f, i)` differentiates a
    callable (JAX: `jax.grad`); `None`: `HookDerivative.auto` of a callable is refused.
    """

    def __init__(self, functions, grad=None):
        """Wrap every plain callable as a `HookFunction` (`HookDerivative.auto`)."""
        self.hooks = {k: v if isinstance(v, HookFunction) else HookFunction(v)
                      for k, v in functions.items()}
        self.grad = grad
        self.module = {k: h.function for k, h in self.hooks.items() if not h.symbolic}

    def _partial(self, f, counts):
        """`d^|counts| f / d args^counts` at `f.args` (`counts` per argument)."""
        name, h = f.func.__name__, self.hooks[f.func.__name__]
        order = [i for i, c in enumerate(counts) for _ in range(c)]
        if h.derivative is HookDerivative.zero:
            return sp.S.Zero
        if h.symbolic:
            return sp.diff(h.function(*f.args), *[f.args[i] for i in order])
        if h.derivative is HookDerivative.auto:
            if self.grad is None:
                raise HookError(f"hook {name!r} needs its derivative: HookDerivative.auto "
                                "differentiates a callable on jax only (a SymPy Lambda on "
                                f"{_SUPPORT}); declare HookDerivative.zero or the partials")
            g = h.function
            for i in order:
                g = self.grad(g, i)
        elif len(order) > 1:
            raise HookError(f"hook {name!r}: a derivative of order {len(order)}; user "
                            "partials give the first derivatives only")
        elif callable(h.derivative):
            g = lambda *a, d=h.derivative, i=order[0]: d(*a)[i]     # noqa: E731
        else:
            g = h.derivative[order[0]]
        dname = f"{name}__d{'_'.join(map(str, order))}"
        self.module[dname] = g
        return sp.Function(dname)(*f.args)

    def _derivative(self, d):
        """A `Derivative` of a hook, lowered."""
        f, counts = d.expr, [0] * len(d.expr.args)
        for v, k in d.variable_count:
            at = [i for i, a in enumerate(f.args) if a == v]
            if len(at) != 1:
                raise HookError(f"hook {f.func.__name__!r}: d/d{v} where {v} is not exactly "
                                "one argument")
            counts[at[0]] += int(k)
        return self._partial(f, counts)

    def _is_hook(self, e):
        return isinstance(e, AppliedUndef) and e.func.__name__ in self.hooks

    def lower(self, expr):
        """`expr` with the hook derivatives printable and SymPy hooks inlined (unchanged when it
        calls no hook)."""
        expr = sp.sympify(expr)
        if not any(self._is_hook(a) for a in expr.atoms(AppliedUndef)):
            return expr
        rep = {}
        for s in expr.atoms(sp.Subs):
            if isinstance(s.expr, sp.Derivative) and self._is_hook(s.expr.expr):
                rep[s] = self._derivative(s.expr).xreplace(dict(zip(s.variables, s.point)))
        for d in expr.atoms(sp.Derivative):
            if self._is_hook(d.expr):
                rep[d] = self._derivative(d)
        expr = expr.xreplace(rep)
        for name, h in self.hooks.items():
            if h.symbolic:
                expr = expr.replace(lambda e, n=name: isinstance(e, AppliedUndef)
                                    and e.func.__name__ == n,
                                    lambda e, h=h: h.function(*e.args))
        return expr


def _called(systems):
    """`{hook name: {operator}}` over the kernels of `systems` (NSMs)."""
    called = {}
    for nsm in systems:
        for slot, kc in nsm.call.items():
            if kc is not None and hasattr(kc, "functions"):
                d = kc.functions[kc.name].definition
                entries = (d.tolist() if hasattr(d, "tolist")
                           else list(d) if isinstance(d, (list, tuple)) else [d])
                for e in sp.flatten(entries):
                    for name in hook_names(e):
                        called.setdefault(name, set()).add(slot)
    return called


def check_functions(functions, systems):
    """Every hook the operators of `systems` (NSMs) call has a function and every function is
    called; raises `ValueError` (with "did you mean")."""
    if not functions:
        return
    called = _called(systems)
    for name in sorted(set(called) - set(functions)):
        raise ValueError(f"the operators call {name!r} but functions has no {name!r}"
                         f"{did_you_mean(name, functions)}")
    for name in sorted(set(functions) - set(called)):
        raise ValueError(f"functions[{name!r}]: no operator calls {name!r}"
                         f"{did_you_mean(name, called)}")


def refuse_hooks(systems, backend, functions=None):
    """The C++ backends: raise `NotImplementedError` for a hook the kernels of `systems` (NSMs)
    call, or any `functions` set."""
    if functions:
        raise NotImplementedError(f"functions ({', '.join(functions)}): {backend} cannot call "
                                  f"Python functions; supported on {_SUPPORT}")
    for name, slots in sorted(_called(systems).items()):
        raise NotImplementedError(
            f"the function {name!r} (called by {', '.join(sorted(slots))}): {backend} "
            f"implements no user functions; supported on {_SUPPORT} (functions={{{name!r}: ...}})")


def refuse_used(refused, owner, source):
    """Raise the `HookError` of a refused kernel (`{slot: error}`) the printed `source` calls
    as `owner.slot`."""
    for slot, err in refused.items():
        if re.search(rf"\b{re.escape(owner)}\.{slot}\b", source):
            raise HookError(f"operator {slot!r}: {err}")
