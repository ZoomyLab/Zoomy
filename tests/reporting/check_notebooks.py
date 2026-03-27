#!/usr/bin/env python3
"""
Notebook CI checks:
  1) structural validation for .ipynb files (fast, PR-friendly)
  2) optional execution for a curated smoke list (slow, scheduled/manual)
"""

from __future__ import annotations

import argparse
import py_compile
import tempfile
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", nargs="*", default=[], help="Explicit notebook paths.")
    p.add_argument(
        "--execute-smoke",
        action="store_true",
        help="Execute notebooks listed in smoke list after validation.",
    )
    p.add_argument(
        "--smoke-list",
        default="tests/notebooks/smoke_notebooks.txt",
        help="Text file with one notebook path per line.",
    )
    p.add_argument(
        "--jupytext-check",
        action="store_true",
        help=(
            "Convert selected notebooks to temporary Python files with jupytext "
            "and run syntax compile checks. No files are written to the repo."
        ),
    )
    return p.parse_args()


def _discover_notebooks() -> list[Path]:
    roots = [
        Path("tutorials"),
        Path("notebooks"),
        Path("web/jupyter-lite/notebooks"),
    ]
    out: list[Path] = []
    for root in roots:
        if root.exists():
            out.extend(root.rglob("*.ipynb"))
    return sorted(set(out))


def _load_smoke_list(path: Path) -> list[Path]:
    if not path.exists():
        return []
    notebooks: list[Path] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        notebooks.append(Path(line))
    return notebooks


def _validate_notebook(path: Path) -> None:
    with path.open("r", encoding="utf-8") as fh:
        nb = nbformat.read(fh, as_version=4)
    nbformat.validate(nb)


def _execute_notebook(path: Path) -> None:
    with path.open("r", encoding="utf-8") as fh:
        nb = nbformat.read(fh, as_version=4)
    client = NotebookClient(nb, timeout=900, kernel_name="python3")
    client.execute()


def _jupytext_compile_check(path: Path) -> None:
    nb = jupytext.read(path)
    py_text = jupytext.writes(nb, fmt="py:percent")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="nbcheck_", delete=True, encoding="utf-8"
    ) as fh:
        fh.write(py_text)
        fh.flush()
        py_compile.compile(fh.name, doraise=True)


def main() -> int:
    args = _parse_args()
    if args.paths:
        notebooks = [Path(p) for p in args.paths if p.endswith(".ipynb")]
    else:
        notebooks = _discover_notebooks()

    if not notebooks:
        print("No notebooks selected.")
        return 0

    print(f"Validating {len(notebooks)} notebook(s)...")
    for nb in notebooks:
        _validate_notebook(nb)
    print("Notebook structure validation passed.")

    if args.jupytext_check:
        print(f"Running jupytext compile check for {len(notebooks)} notebook(s)...")
        for nb in notebooks:
            _jupytext_compile_check(nb)
        print("Jupytext compile check passed.")

    if args.execute_smoke:
        smoke = _load_smoke_list(Path(args.smoke_list))
        smoke = [p for p in smoke if p.exists()]
        print(f"Executing {len(smoke)} smoke notebook(s)...")
        for nb in smoke:
            _execute_notebook(nb)
        print("Notebook smoke execution passed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

