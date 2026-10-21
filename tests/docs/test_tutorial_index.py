"""The vocabulary and the page of docs/scripts/generate_tutorial_index.py (offline, no notebooks run)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "scripts"))
import generate_tutorial_index as index  # noqa: E402

pytestmark = [pytest.mark.small, pytest.mark.unittest]


def notebook(src, rel, meta):
    path = src / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cells": [], "metadata": {"zoomy": meta} if meta is not None else {},
                                "nbformat": 4, "nbformat_minor": 5}))


def meta(*tags, **extra):
    return {"title": "T", "summary": "one line", "tags": list(tags), **extra}


def test_filter_categories_and_harness():
    assert list(index.FILTERS) == ["type", "dimension", "physics", "component", "method",
                                   "feature", "backend"]
    assert index.FILTERS["physics"] == ("viscous", "inviscid", "hydrostatic", "non-hydrostatic",
                                        "compressible", "incompressible", "free-surface")
    assert index.HARNESS == {"requires": ("cpu", "gpu", "mpi", "container")}


def test_requires_defaults_to_cpu_and_gates_the_runner():
    cpu = index.parse("a.ipynb", meta("type:tutorial"))
    assert not cpu.errors and cpu.requires == ("cpu",) and cpu.runs_on_cpu
    gpu = index.parse("b.ipynb", meta("type:use-case", "requires:gpu", "requires:mpi"))
    assert not gpu.errors and gpu.requires == ("gpu", "mpi") and not gpu.runs_on_cpu


@pytest.mark.parametrize("tags, fragment", [
    ([], "exactly one type"),
    (["type:tutorial", "type:use-case"], "exactly one type"),
    (["dimension:1d"], "exactly one type"),
    (["type:tutorial", "dimension:4d"], "unknown tag 'dimension:4d'"),
    (["type:tutorial", "physics:viscus"], "did you mean 'viscous'"),
    (["type:tutorial", "viscosity:viscous"], "unknown tag 'viscosity:viscous'"),
    (["type:tutorial", "task:simulation"], "unknown tag 'task:simulation'"),
])
def test_bad_tags_are_errors(tags, fragment):
    nb = index.parse("x.ipynb", meta(*tags))
    assert any(fragment in e for e in nb.errors), nb.errors


def test_bad_metadata_is_an_error():
    assert index.parse("x.ipynb", None).errors
    assert any("status" in e for e in index.parse("x.ipynb", meta("type:tutorial", status="legacy")).errors)
    assert any("summary" in e for e in index.parse("x.ipynb", {"title": "T", "tags": ["type:tutorial"]}).errors)


def test_load_stops_on_errors(tmp_path):
    src, book = tmp_path / "tutorials", tmp_path / "book"
    notebook(src, "good.ipynb", meta("type:tutorial"))
    notebook(src, "bad.ipynb", meta("dimension:1d"))
    book.mkdir()
    with pytest.raises(SystemExit) as err:
        index.load(src, book)
    assert "bad.ipynb" in str(err.value) and "good.ipynb" not in str(err.value)
    notebook(book, "stray.ipynb", meta("type:tutorial"))
    (src / "bad.ipynb").unlink()
    with pytest.raises(SystemExit, match="notebooks live under tutorials/"):
        index.load(src, book)


def test_page_is_built_from_the_metadata(tmp_path):
    src, book = tmp_path / "tutorials", tmp_path / "book"
    book.mkdir()
    src.mkdir()
    assert "No notebooks are published yet" in index.render(index.load(src, book))
    notebook(src, "a/one.ipynb", meta("type:use-case", "dimension:2d", "physics:viscous"))
    notebook(src, "b/two.ipynb", meta("type:tutorial", "requires:container"))
    notebooks = index.load(src, book)
    assert [nb.rel for nb in notebooks] == ["b/two.ipynb", "a/one.ipynb"]       # tutorials first
    index.mirror(notebooks, src, book)
    copy = json.loads((book / "ipynb" / "a" / "one.ipynb").read_text())
    assert copy["metadata"]["orphan"] is True and copy["metadata"]["zoomy"]["title"] == "T"
    assert "orphan" not in json.loads((src / "a" / "one.ipynb").read_text())["metadata"]
    page = index.render(notebooks)
    assert page.count('class="zt-select"') == len(index.FILTERS)
    assert 'data-cat="requires"' not in page
    assert '<option value="dimension:2d">2d (1)</option>' in page
    assert '<option value="dimension:3d" disabled>3d (0)</option>' in page
    assert ":link: ipynb/a/one" in page and ":link-type: doc" in page
    assert 'data-tags="type:use-case dimension:2d physics:viscous requires:cpu"' in page
