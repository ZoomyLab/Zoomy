#!/usr/bin/env python3
"""Write the MyST pages of the API reference ahead of ``jupyter-book build``.

    python docs/scripts/generate_apidoc.py            # always from scratch (clean)

The API reference is written by sphinx-autodoc2 from the docstrings, which are
Markdown (MyST).  Its settings are the ``autodoc2_*`` values of
``docs/book/_config.yml`` (packages, output directory, renderer, docstring
parser); this script renders with exactly those settings, so there is one place
to edit.  The pages land in ``docs/book/api/_autodoc2/<package>/`` and are never
committed (``.gitignore``).  ``_toc.yml`` lists them: the root page of each
package plus a glob of its modules.

Why before the build: ``jupyter-book build`` renders the same pages itself (the
sphinx extension finds identical files and leaves them), but only *warns* when a
package path is missing, which would publish an empty reference.  Here a missing
path, an empty package or too few pages exit 1.  Nothing is imported: autodoc2
reads the sources, so no solver or backend library has to be installed.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "docs" / "book"


def autodoc2_settings() -> dict:
    """``autodoc2_*`` values of ``_config.yml`` over the extension's defaults."""
    from autodoc2.config import CONFIG_PREFIX, Config

    given = yaml.safe_load((BOOK / "_config.yml").read_text())["sphinx"]["config"]
    values = {f"{CONFIG_PREFIX}{name}": field.metadata.get("sphinx_default", default)
              for name, default, field in Config().as_triple()}
    values.update({k: v for k, v in given.items() if k.startswith(CONFIG_PREFIX)})
    return values


def main() -> None:
    values = autodoc2_settings()
    for pkg in values["autodoc2_packages"]:
        path = (BOOK / pkg["path"]).resolve()
        if not (path / "__init__.py").is_file():
            raise SystemExit(f"API package {pkg['module']}: no {path}/__init__.py "
                             "(is the library submodule checked out?)")

    out = BOOK / values["autodoc2_output_dir"]
    shutil.rmtree(out, ignore_errors=True)

    from autodoc2.sphinx.extension import run_autodoc
    run_autodoc(SimpleNamespace(srcdir=str(BOOK), env=SimpleNamespace(), config=values))

    for pkg in values["autodoc2_packages"]:
        root = out / pkg["module"] / f"{pkg['module']}.md"
        pages = len(list(root.parent.glob("*.md"))) if root.is_file() else 0
        if pages < 2:
            raise SystemExit(f"API package {pkg['module']}: {pages} page(s) written, "
                             f"expected {root} and its modules")
        print(f"{pkg['module']}: {pages} pages -> {root.parent.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
