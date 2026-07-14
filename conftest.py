"""Superproject-root conftest — AUTOMATIC test collection & classification.

Nothing here is a hand-maintained list.  Three automatic rules:

1. **Collection is capability-based** (``pytest_ignore_collect``): a library
   suite ``library/zoomy_<stack>/tests`` is collected iff the package
   ``zoomy_<stack>`` is importable in the current environment/container.
   Every stack container therefore collects exactly the suites it can run —
   adding a new ``library/zoomy_new/tests`` requires NO config change
   (``pytest.ini`` testpaths glob ``library/*/tests`` picks it up).

2. **Stack marker from the path** (``pytest_collection_modifyitems``): a test
   under ``library/zoomy_<stack>/tests`` gets the ``<stack>`` marker when that
   marker is registered in ``pytest.ini`` (alias: ``foam`` → ``openfoam``),
   else it falls back to ``core`` (pure-python helper libs run in the core
   container).  Mirrors the dmplex/fenicsx path rule in ``tests/conftest.py``.

3. **Size default**: a test with NO size marker (``small``/``medium``/
   ``large``/``benchmark``) is ``small`` by definition.  Size rule (per user,
   2026-07-14): **any test taking more than 5 minutes individually is
   large** — mark violators ``@pytest.mark.large``; the report runner prints
   ``SIZE-RULE VIOLATION`` for small-suite tests exceeding 300 s.
"""
import importlib.util
import re

import pytest

_LIB_TESTS_RE = re.compile(r"/library/(zoomy_[A-Za-z0-9_]+)/tests(/|$)")
_MARKER_ALIASES = {"foam": "openfoam"}
_SIZE_MARKERS = {"small", "medium", "large", "benchmark"}


def _stack_of(norm_path):
    m = _LIB_TESTS_RE.search(norm_path)
    return m.group(1) if m else None          # e.g. "zoomy_jax"


def pytest_ignore_collect(collection_path, config):
    """Skip a library suite whose backend package is not importable here."""
    stack = _stack_of(str(collection_path).replace("\\", "/"))
    if stack is None:
        return None
    try:
        found = importlib.util.find_spec(stack) is not None
    except (ImportError, ValueError):
        found = False
    return None if found else True


def _registered_markers(config):
    names = set()
    for line in config.getini("markers"):
        names.add(line.split(":", 1)[0].strip())
    return names


def pytest_collection_modifyitems(config, items):
    registered = _registered_markers(config)
    for item in items:
        try:
            path = str(item.path)
        except AttributeError:
            path = str(item.fspath)
        norm = path.replace("\\", "/")
        stack = _stack_of(norm)
        if stack is not None:
            short = stack.removeprefix("zoomy_")
            marker = _MARKER_ALIASES.get(short, short)
            if marker not in registered:
                marker = "core"
            item.add_marker(getattr(pytest.mark, marker))
        if not _SIZE_MARKERS.intersection(item.keywords):
            item.add_marker(pytest.mark.small)
