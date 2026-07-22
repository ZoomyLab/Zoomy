"""Superproject-root conftest — AUTOMATIC test collection & classification.

Nothing here is a hand-maintained list.  Three automatic rules:

1. **Collection is capability-based** (``pytest_ignore_collect``): a suite that
   belongs to ``zoomy_<stack>`` is collected iff the package ``zoomy_<stack>``
   is importable in the current environment/container.  Every stack container
   therefore collects exactly the suites it can run — adding a new
   ``library/zoomy_new/tests`` requires NO config change (``pytest.ini``
   testpaths globs ``library/*/tests``).

   A suite "belongs to" a stack by PATH, in either of the two homes:
     * ``library/zoomy_<stack>/tests/...``   (the library's own suite)
     * ``tests/**/zoomy_<stack>/...``        (the superproject's suite)
   Both must be gated.  Covering only the first one was a real defect: the
   superproject ships ``tests/unit/zoomy_{jax,mesh,firedrake}``, which every
   container collected unconditionally and then died importing at COLLECTION
   time — before any ``-m`` marker expression could deselect them.  That is
   what made the core job report "0 tests, 4 errors: No module named 'jax'".

2. **Stack marker from the path** (``pytest_collection_modifyitems``): a test
   in either home gets the ``<stack>`` marker when that marker is registered in
   ``pytest.ini`` (alias: ``foam`` → ``openfoam``), else it falls back to
   ``core`` (pure-python helper libs run in the core container).  Mirrors the
   dmplex/fenicsx path rule in ``tests/conftest.py``.  This is the same
   ``_stack_of`` used by rule 1 — so ``tests/unit/zoomy_jax`` is now correctly
   marked ``jax`` instead of falling back to ``core`` and being pulled into the
   core job.

3. **Size default**: a test with NO size marker (``small``/``medium``/
   ``large``/``benchmark``) is ``small`` by definition.  Size rule (per user,
   2026-07-14): **any test taking more than 5 minutes individually is
   large** — mark violators ``@pytest.mark.large``; the report runner prints
   ``SIZE-RULE VIOLATION`` for small-suite tests exceeding 300 s.
"""
import importlib.util
import os
import re

import pytest

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _importable(pkg):
    try:
        return importlib.util.find_spec(pkg) is not None
    except (ImportError, ValueError):
        return False


def _unavailable_suites():
    """Suite dirs to drop BEFORE pytest descends into them.

    ``pytest_ignore_collect`` is too late for this: pytest loads every
    ``conftest.py`` under the testpaths during startup, before that hook is
    consulted.  So an absent stack still got its conftest imported — and
    ``library/zoomy_amrex/tests/`` is itself a package named ``tests``, which
    collides with the superproject's ``tests`` package and aborted the whole
    session with "Plugin already registered under a different name".
    ``collect_ignore`` is evaluated early enough to prevent that.
    """
    ignore = []
    lib = os.path.join(_ROOT, "library")
    if os.path.isdir(lib):
        for entry in sorted(os.listdir(lib)):
            if not entry.startswith("zoomy_"):
                continue
            if os.path.isdir(os.path.join(lib, entry, "tests")) and not _importable(entry):
                ignore.append(os.path.join("library", entry, "tests"))
    return ignore


# Consumed by pytest at conftest-load time.
collect_ignore = _unavailable_suites()

# A library's own suite:            library/zoomy_jax/tests/...
_LIB_TESTS_RE = re.compile(r"/library/(zoomy_[A-Za-z0-9_]+)/tests(/|$)")
# The superproject's per-stack suite: tests/unit/zoomy_jax/..., tests/regression/zoomy_jax/...
# Anchored on a "/tests/" segment so it can never match a source tree
# (library/zoomy_core/zoomy_core/... has no /tests/ component).
_ROOT_TESTS_RE = re.compile(r"/tests/(?:[^/]+/)*?(zoomy_[A-Za-z0-9_]+)(/|$)")
_MARKER_ALIASES = {"foam": "openfoam"}
_SIZE_MARKERS = {"small", "medium", "large", "benchmark"}


def _stack_of(norm_path):
    """Which zoomy_<stack> a test path belongs to, in either test home."""
    m = _LIB_TESTS_RE.search(norm_path)
    if m is None:
        m = _ROOT_TESTS_RE.search(norm_path)
    return m.group(1) if m else None          # e.g. "zoomy_jax"


def pytest_ignore_collect(collection_path, config):
    """Skip a suite whose stack package is not importable here.

    Applies to BOTH test homes (see module docstring rule 1). Without this,
    a container that lacks a stack fails at collection with ImportError, which
    no marker expression can prevent.
    """
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
