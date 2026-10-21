"""zoomy_core runs in the browser (Pyodide: GUI, JupyterLite, QR notebooks). Every runtime
dependency must therefore exist in Pyodide. A new dependency is added to PYODIDE_OK only after
checking it there (built-in Pyodide package or a pure-Python wheel)."""
import re
import tomllib
from pathlib import Path

PYODIDE_OK = {"sympy", "numpy", "scipy", "param", "loguru"}


def test_runtime_dependencies_exist_in_pyodide():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    names = {re.split(r"[<>=!~\[; ]", d, maxsplit=1)[0].lower() for d in deps}
    assert names <= PYODIDE_OK, f"not checked in Pyodide: {sorted(names - PYODIDE_OK)}"
