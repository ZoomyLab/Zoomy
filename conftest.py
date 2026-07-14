"""Superproject-root conftest — CI report classification (REQ: webpage test reports).

Applies to EVERYTHING collected from the repo root, including the library
suites (``library/zoomy_core/tests`` …) that live OUTSIDE ``tests/`` and
therefore never see ``tests/conftest.py``.

Two auto-classifications so the Smart Tests marker expressions
(``-m "small and (core or ...)"``) pick up the full suite without hand-marking
hundreds of tests:

1. **Stack marker by path** — mirrors the dmplex/fenicsx path rule in
   ``tests/conftest.py``: a test under ``library/zoomy_<stack>/tests`` carries
   that stack's marker.
2. **Size default** — a test with NO size marker (``small``/``medium``/
   ``large``/``benchmark``) is ``small`` by definition.  The size rule (per
   user, 2026-07-14): **any test taking more than 5 minutes individually is
   large** — mark it ``@pytest.mark.large`` explicitly; the report runner
   flags violators (see ``tests/reporting/generate_test_report.py``).
"""
import pytest

_PATH_MARKERS = (
    ("/library/zoomy_core/tests", "core"),
    ("/library/zoomy_jax/tests", "jax"),
    ("/library/zoomy_firedrake/tests", "firedrake"),
    ("/library/zoomy_foam/tests", "openfoam"),
    ("/library/zoomy_prepost/tests", "core"),
    ("/library/zoomy_server/tests", "core"),
    ("/library/zoomy_gui/tests", "core"),
)

_SIZE_MARKERS = {"small", "medium", "large", "benchmark"}


def pytest_collection_modifyitems(config, items):
    for item in items:
        try:
            path = str(item.path)
        except AttributeError:
            path = str(item.fspath)
        norm = path.replace("\\", "/")
        for fragment, marker in _PATH_MARKERS:
            if fragment in norm:
                item.add_marker(getattr(pytest.mark, marker))
        if not _SIZE_MARKERS.intersection(item.keywords):
            item.add_marker(pytest.mark.small)
