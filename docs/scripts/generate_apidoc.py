#!/usr/bin/env python3
"""Regenerate Sphinx apidoc stubs for Zoomy Python packages (Bryne-style layout)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_apidoc(output: Path, module_root: Path, excludes: list[Path]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    cmd = [
        "sphinx-apidoc",
        "-f",
        "-e",
        "-M",
        "-d",
        "4",
        "-o",
        str(output),
        str(module_root),
        *[str(p) for p in excludes],
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output directories before generating.",
    )
    args = parser.parse_args()

    root = _repo_root()
    book_api = root / "docs" / "book" / "api"
    core_pkg = root / "library" / "zoomy_core" / "zoomy_core"
    jax_pkg = root / "library" / "zoomy_jax" / "zoomy_jax"

    out_core = book_api / "_apidoc_zoomy_core"
    out_jax = book_api / "_apidoc_zoomy_jax"

    if args.clean:
        shutil.rmtree(out_core, ignore_errors=True)
        shutil.rmtree(out_jax, ignore_errors=True)

    if not core_pkg.is_dir():
        raise SystemExit(f"missing package path: {core_pkg}")
    if not jax_pkg.is_dir():
        raise SystemExit(f"missing package path: {jax_pkg}")

    _run_apidoc(out_core, core_pkg, [])
    _run_apidoc(
        out_jax,
        jax_pkg,
        [jax_pkg / "gnn_blueprint"],
    )

    # Root package pages are zoomy_core.rst / zoomy_jax.rst; modules.rst is redundant for JB TOC.
    for extra in (out_core / "modules.rst", out_jax / "modules.rst"):
        if extra.is_file():
            extra.unlink()


if __name__ == "__main__":
    main()
