#!/usr/bin/env python3
"""Regenerate the Sphinx apidoc stubs of the API reference.

    python docs/scripts/generate_apidoc.py            # always from scratch (clean)

Covered (``PACKAGES``): zoomy_core's model, systemmodel, numerics, mesh and
symsolver; zoomy_jax.symsolver (and ``gpurank``); the Python packages of
zoomy_foam and zoomy_amrex.  Everything else of a package is excluded, and the
output directories are removed first, so a removed module never leaves a stale
stub behind.  Package pages link their children with ``:doc:`` (the book's TOC
is ``_toc.yml``; toctree directives would be ignored).  Heavy or optional imports are mocked in
``docs/book/_config.yml`` (``autodoc_mock_imports``).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "library"
API = ROOT / "docs" / "book" / "api"

# package -> (source dir, kept top-level entries; None = everything)
PACKAGES = {
    "zoomy_core": (LIB / "zoomy_core" / "zoomy_core",
                   {"model", "systemmodel", "numerics", "mesh", "symsolver"}),
    "zoomy_jax": (LIB / "zoomy_jax" / "zoomy_jax", {"symsolver", "gpurank.py"}),
    "zoomy_foam": (LIB / "zoomy_foam" / "zoomy_foam", None),
    "zoomy_amrex": (LIB / "zoomy_amrex" / "zoomy_amrex", None),
}


def _excluded(src: Path, keep) -> list[str]:
    if keep is None:
        return []
    return sorted(str(p) for p in src.iterdir()
                  if p.name not in keep and p.name != "__init__.py"
                  and (p.is_dir() or p.suffix == ".py"))


_TOCTREE = re.compile(r"\.\. toctree::\n(?:   :\S+:.*\n)*\n((?:   \S+\n)+)")


def _links(rst: Path) -> None:
    """The book's TOC is ``_toc.yml`` (external-toc ignores toctree
    directives), so a package page lists its children as ``:doc:`` links."""
    text = rst.read_text()
    new = _TOCTREE.sub(lambda m: "".join(f"* :doc:`{e.strip()}`\n" for e in m.group(1).splitlines()),
                       text)
    if new != text:
        rst.write_text(new)


def main() -> None:
    for name, (src, keep) in PACKAGES.items():
        if not src.is_dir():
            raise SystemExit(f"missing package path: {src}")
        out = API / f"_apidoc_{name}"
        shutil.rmtree(out, ignore_errors=True)
        tests = [f"{src}/*/tests", f"{src}/*/tests/*"]        # fnmatch patterns
        subprocess.run(["sphinx-apidoc", "-f", "-e", "-M", "-d", "4", "-o", str(out),
                        str(src), *_excluded(src, keep), *tests], check=True)
        (out / "modules.rst").unlink(missing_ok=True)      # the root page is <name>.rst
        for rst in out.glob("*.rst"):
            _links(rst)


if __name__ == "__main__":
    main()
