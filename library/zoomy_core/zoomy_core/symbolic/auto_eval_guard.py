"""Context manager that bans sympy auto-evaluation inside its block.

Used in tests + the slim walkthrough to verify the primitive layer
never silently fires a math rule.  When the guard is active, calling
`.doit()` on any sympy `Basic` object raises
`AutoEvalForbidden` with a stack trace pointing at the offender.

Usage:

```
    from zoomy_core.symbolic.auto_eval_guard import AutoEvalGuard

    with AutoEvalGuard():
        # Any .doit() call here will raise.
        ...
```

The guard also bans `sp.simplify`, `sp.cancel`, `sp.together`,
`sp.factor`, `sp.collect`, `sp.diff`, `sp.integrate` — the
non-structural calls the redesign forbids inside the symbolic
package.  `sp.expand` is permitted (Canonicalise uses it) — restrict
its kwargs at call sites instead.
"""

from __future__ import annotations

import sympy as sp


class AutoEvalForbidden(RuntimeError):
    """Raised when a forbidden sympy auto-evaluation call fires.

    The redesign's invariant is: every math rule is invoked by an
    explicit primitive.  Sympy's auto-evaluation calls
    (`.doit()`, `sp.simplify`, etc.) bypass that surface and
    cause subtle order-of-operations bugs.  The guard makes any such
    call a hard failure.
    """


def _make_banned(name: str):
    def _banned(*args, **kwargs):
        raise AutoEvalForbidden(
            f"{name} called inside AutoEvalGuard — this should be an "
            f"explicit primitive call instead"
        )
    return _banned


class AutoEvalGuard:
    """Monkeypatch `sympy.Basic.doit` and the eval-style sympy
    helpers to raise.  Restored on exit.
    """

    _patched_attrs = (
        ("Basic", "doit"),
    )
    _patched_module_funcs = (
        "simplify",
        "cancel",
        "together",
        "factor",
        "collect",
        "diff",
        "integrate",
        "powsimp",
        "combsimp",
        "trigsimp",
    )

    def __init__(self):
        self._saved = {}

    def __enter__(self):
        for cls_name, attr in self._patched_attrs:
            cls = getattr(sp, cls_name)
            self._saved[("attr", cls_name, attr)] = getattr(cls, attr)
            setattr(cls, attr, _make_banned(f"{cls_name}.{attr}"))
        for fn_name in self._patched_module_funcs:
            self._saved[("module", fn_name)] = getattr(sp, fn_name)
            setattr(sp, fn_name, _make_banned(f"sp.{fn_name}"))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, original in self._saved.items():
            kind = key[0]
            if kind == "attr":
                _, cls_name, attr = key
                setattr(getattr(sp, cls_name), attr, original)
            else:
                _, fn_name = key
                setattr(sp, fn_name, original)
        self._saved.clear()
        return False
