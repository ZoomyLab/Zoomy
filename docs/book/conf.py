"""Sphinx config merged by Jupyter Book: editable installs use site-packages; this aids local src checkouts."""

from __future__ import annotations

import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent
_repo = _here.parent.parent
for _name in ("zoomy_core", "zoomy_jax"):
    _lib = _repo / "library" / _name
    if _lib.is_dir():
        sys.path.insert(0, str(_lib))
