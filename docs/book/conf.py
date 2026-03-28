"""Sphinx config merged by Jupyter Book: make ``zoomy_core`` importable for autodoc."""

from __future__ import annotations

import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent
_repo = _here.parent.parent
_zoomy_core = _repo / "library" / "zoomy_core"
if _zoomy_core.is_dir():
    sys.path.insert(0, str(_zoomy_core))
