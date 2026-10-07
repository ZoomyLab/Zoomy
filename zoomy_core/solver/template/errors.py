"""Errors shared by the template backends."""
from __future__ import annotations


class GuardError(RuntimeError):
    """A template ``Guard`` failed (D9: fatal, never a silent retry)."""


def format_guard(msg, names, values) -> str:
    parts = []
    for n, v in zip(names, values):
        try:
            import numpy as np
            a = np.asarray(v)
            parts.append(f"{n}={a.item() if a.ndim == 0 else a!r}")
        except Exception:  # noqa: BLE001 - diagnostics only
            parts.append(f"{n}={v!r}")
    return f"{msg}" + (f" [{', '.join(parts)}]" if parts else "")


__all__ = ["GuardError", "format_guard"]
