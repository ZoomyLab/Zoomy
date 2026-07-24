"""Execute each published tutorial notebook as a test — the Notebooks section
of the CI report is built from these.

Every notebook that ships on the docs site (``docs/scripts/
generate_ci_test_report.py::PUBLISHED_TUTORIALS``) runs here end to end, so a
broken tutorial fails a real test instead of publishing a frozen traceback to
the live site. Each case carries the ``notebook`` marker plus the backend
marker it needs, so the Smart Tests matrix runs it in the right container
(``-m "notebook and core"`` etc.).

Keep ``PUBLISHED`` below in sync with ``PUBLISHED_TUTORIALS`` in the docs
generator — the same set is executed here and mirrored onto the site.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

# (path under tutorials/, backend marker the notebook needs)
PUBLISHED: tuple[tuple[str, str], ...] = (
    ("swe/simple_numpy.ipynb", "core"),
    ("sme/moments_2d.ipynb", "core"),
    ("amrex/minimal.ipynb", "amrex"),
)


def _cases():
    for rel, backend in PUBLISHED:
        yield pytest.param(rel, marks=getattr(pytest.mark, backend), id=rel)


@pytest.mark.notebook
@pytest.mark.parametrize("rel", list(_cases()))
def test_tutorial_notebook_executes(rel):
    jupytext = pytest.importorskip("jupytext")
    pytest.importorskip("nbclient")
    from nbclient import NotebookClient

    nb_path = ROOT / "tutorials" / rel
    assert nb_path.is_file(), f"published tutorial missing: {nb_path}"

    nb = jupytext.read(nb_path)
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    client.execute()          # raises CellExecutionError on any cell failure
