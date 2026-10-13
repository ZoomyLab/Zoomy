"""One display call that works in every environment a case is run in.

A case is executed in at least three places and they disagree about output:

* a **Jupyter / GUI kernel** — rich output, `display` is an IPython builtin;
* **plain** `python case.py` — there is no IPython at all, so `display` is
  simply undefined and a cell using it dies with `NameError`;
* a **headless worker** — importable IPython may exist, but no kernel.

On top of that, sympy renders as LaTeX only through its `_repr_latex_` hook,
which is not reliably active: equations came out as plain text inside a
notebook.  Routing through `IPython.display.Math(sympy.latex(...))` renders
via MathJax unconditionally.

So cases call `show` and stop carrying their own shim:

```
    from zoomy_core.misc.show import show

    show(model.describe())                      # any object
    show(eq=(sympy.Symbol("h^*"), h_expr))      # a rendered equation
```

Keeping this here rather than pasted into each case is the same rule the rest
of the framework follows: a case selects behaviour, it does not reimplement it.
"""
from __future__ import annotations

__all__ = ["show", "in_notebook"]


class _Latex:
    """A LaTeX payload that renders without IPython being installed.

    `IPython.display.Math` is only a thin wrapper around `_repr_latex_`, and
    depending on it made rendering fail in exactly the environment that matters:
    a lean kernel (Pyodide, a bare venv) has a working `display` but no
    IPython, so importing `Math` failed and equations fell back to ASCII
    pretty-print -- the very bug this module exists to fix.  Rich-repr hooks are
    a protocol, not a library, so implement them directly.
    """

    __slots__ = ("tex",)

    def __init__(self, tex: str):
        self.tex = tex

    def _repr_latex_(self) -> str:
        return f"${self.tex}$"

    # Hosts that prefer markdown/html still get MathJax-able output.
    def _repr_markdown_(self) -> str:
        return f"${self.tex}$"

    def __repr__(self) -> str:                           # plain-text fallback
        return self.tex


def _host_display():
    """A `display` provided by the HOST, if there is one.

    Checked BEFORE IPython because the Theia/Pyodide GUI runs its own kernel:
    it injects `display` into the cell namespace but provides no
    `get_ipython()`, so an IPython-only probe reported "not a notebook" and
    equations came out as ASCII pretty-print in the GUI.

    Looked for in the caller's frames (where the GUI injects it) and in
    builtins.  `ZOOMY_DISPLAY=rich|plain` overrides everything, for a host
    that wants to state the answer rather than be sniffed."""
    import builtins
    import os
    import sys

    forced = os.environ.get("ZOOMY_DISPLAY", "").strip().lower()
    if forced == "plain":
        return None

    d = getattr(builtins, "display", None)
    if callable(d):
        return d
    # walk out of this module into the caller's globals
    try:
        f = sys._getframe(1)
        while f is not None:
            if f.f_globals.get("__name__") != __name__:
                d = f.f_globals.get("display")
                if callable(d) and getattr(d, "__module__", "") != __name__:
                    return d
            f = f.f_back
    except Exception:                                    # noqa: BLE001
        pass
    if forced == "rich":
        try:
            from IPython.display import display as _d
            return _d
        except Exception:                                # noqa: BLE001
            return None
    return None


def in_notebook() -> bool:
    """True when SOMETHING can render rich output.

    A host-injected `display` counts (the GUI kernel), as does a live
    IPython instance.  Importable IPython alone does not — a headless worker
    can import it without a kernel."""
    if _host_display() is not None:
        return True
    try:
        from IPython import get_ipython
    except Exception:                                    # noqa: BLE001
        return False
    try:
        return get_ipython() is not None
    except Exception:                                    # noqa: BLE001
        return False


def show(obj=None, *, eq=None, precision: int = 6):
    """Render `obj`, or the equation `eq=(lhs, rhs)`, wherever we are.

    `eq` is rendered as real LaTeX in a kernel and pretty-printed otherwise.
    `precision` rounds sympy floats for display only — an un-rounded
    manufactured source prints 15-digit rationals and is unreadable.
    """
    if eq is not None:
        import sympy as sp
        lhs, rhs = eq
        try:
            rhs = sp.N(rhs, precision)
        except Exception:                                # noqa: BLE001
            pass
        tex = f"{sp.latex(lhs)} = {sp.latex(rhs)}"
        _d = _host_display()
        if _d is None and in_notebook():
            from IPython.display import display as _d    # noqa: F811
        if _d is not None:
            _d(_Latex(tex))
        else:
            print(sp.pretty(sp.Eq(lhs, rhs, evaluate=False)))
        return
    _d = _host_display()
    if _d is None and in_notebook():
        from IPython.display import display as _d        # noqa: F811
    if _d is not None:
        _d(obj)
    else:
        print(obj)
