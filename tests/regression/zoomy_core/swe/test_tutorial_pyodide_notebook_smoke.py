from pathlib import Path

import pytest


@pytest.mark.small
@pytest.mark.regression
@pytest.mark.tutorial
@pytest.mark.pyodide
@pytest.mark.core
def test_pyodide_tutorial_notebook_has_browser_install_cell():
    notebook_path = Path("tutorials/swe/pyodide.ipynb")
    content = notebook_path.read_text(encoding="utf-8")
    assert "micropip.install(\"zoomy-core\")" in content
