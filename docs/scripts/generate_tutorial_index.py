"""Tutorial notebooks -> docs/book/tutorials/index.md (a card grid with a tag filter) and
docs/book/developer/tutorials.md (how to add a notebook: the metadata, the tag vocabulary).

    python docs/scripts/generate_tutorial_index.py

Every ``tutorials/**/*.ipynb`` carries ``metadata.zoomy``::

    "zoomy": {"title": "...", "summary": "<one line>",
              "tags": ["type:tutorial", "dimension:1d", "physics:viscous", ...]}

Tags are ``category:value`` from :data:`TAGS`.  An unknown category or value, a
notebook without metadata, or a notebook without exactly one ``type`` is a build
error.  The filter categories (:data:`FILTERS`) feed the tag filter: a text field with
autocomplete above the grid and, right of it, a panel with one coloured chip (colour per
category, :data:`HUES`) for every tag that occurs.  ``_static/tutorial-filter.js`` shows a
card when it carries ALL selected tags (several per category allowed).  ``requires`` (:data:`HARNESS`) is read by
the test harness only (``tests/tutorials``: a notebook that needs more than ``cpu``
is collected and skipped) and is not a filter; it defaults to ``requires:cpu``.

The book renders every notebook that has a card: it is copied to
``docs/book/tutorials/ipynb/`` (git-ignored, regenerated here) and reached through
its card, not through ``_toc.yml`` (an orphan page).  A notebook under ``tutorials/pyodide/`` is
the JupyterLite deployment's content: its card links there instead.
"""
from __future__ import annotations

import difflib
import html
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "tutorials"
BOOK = ROOT / "docs" / "book" / "tutorials"
OUT = BOOK / "index.md"
HOWTO = ROOT / "docs" / "book" / "developer" / "tutorials.md"

# The filterable categories; the order is the order in the tag panel and on a card.
FILTERS = {
    "type": ("tutorial", "use-case"),
    "dimension": ("1d", "2d", "3d"),
    "physics": ("viscous", "inviscid", "hydrostatic", "non-hydrostatic",
                "compressible", "incompressible", "free-surface"),
    "component": ("model", "system-model", "numerical-system-model", "solver",
                  "printer", "mesh"),
    "method": ("fv", "lsq", "mood", "imex", "chorin"),
    "feature": ("ad", "mpi", "gpu", "coupling", "function-hook"),
    "backend": ("numpy", "jax", "foam", "amrex"),
}
# One colour (CSS hue) per category: chips, pills and card tags carry it as ``--zt-h``.
HUES = {"type": 265, "dimension": 205, "physics": 165, "component": 32,
        "method": 335, "feature": 85, "backend": 355}
# Metadata for the test harness, not a filter.
HARNESS = {"requires": ("cpu", "gpu", "mpi", "container")}
TAGS = {**FILTERS, **HARNESS}
REQUIRED = ("type",)        # exactly one value
DEFAULTS = {"requires": ("cpu",)}
KEYS = ("title", "summary", "tags")


@dataclass
class Notebook:
    """One notebook under ``tutorials/`` with its parsed ``metadata.zoomy``."""

    rel: str                                    # path under tutorials/, posix
    title: str = ""
    summary: str = ""
    tags: tuple = ()                            # effective: defaults applied, TAGS order
    errors: list = field(default_factory=list)

    @property
    def requires(self):
        return tuple(t.split(":", 1)[1] for t in self.tags if t.startswith("requires:"))

    @property
    def kind(self):
        return next((t.split(":", 1)[1] for t in self.tags if t.startswith("type:")), "")

    @property
    def runs_on_cpu(self):
        return set(self.requires) <= {"cpu"}


def _hint(word, options):
    near = difflib.get_close_matches(word, options, n=1)
    return f"; did you mean {near[0]!r}?" if near else f"; one of: {', '.join(options)}"


def parse(rel, meta):
    """Validate ``metadata.zoomy`` of one notebook; problems land in ``errors``."""
    nb = Notebook(rel)
    err = nb.errors
    if not isinstance(meta, dict):
        err.append(f"{rel}: no metadata.zoomy (title, summary, tags)")
        return nb
    for key in meta:
        if key not in KEYS:
            err.append(f"{rel}: unknown metadata.zoomy key {key!r}{_hint(key, KEYS)}")
    for key in ("title", "summary"):
        value = meta.get(key)
        if not isinstance(value, str) or not value.strip():
            err.append(f"{rel}: metadata.zoomy has no {key!r} text")
        elif "\n" in value.strip():
            err.append(f"{rel}: metadata.zoomy {key!r} is one line")
        else:
            setattr(nb, key, value.strip())
    tags = meta.get("tags")
    if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
        err.append(f"{rel}: metadata.zoomy 'tags' is a list of 'category:value' strings")
        tags = []
    have = set()
    for tag in tags:
        cat, _, val = tag.partition(":")
        if cat not in TAGS:
            err.append(f"{rel}: unknown tag {tag!r}: category {cat!r}{_hint(cat, tuple(TAGS))}")
        elif val not in TAGS[cat]:
            err.append(f"{rel}: unknown tag {tag!r}{_hint(val, TAGS[cat])}")
        else:
            have.add(tag)
    for cat in REQUIRED:
        n = sum(t.startswith(cat + ":") for t in have)
        if n != 1:
            err.append(f"{rel}: needs exactly one {cat}: tag ({' | '.join(TAGS[cat])}), has {n}")
    for cat, values in DEFAULTS.items():
        if not any(t.startswith(cat + ":") for t in have):
            have.update(f"{cat}:{v}" for v in values)
    order = [f"{cat}:{v}" for cat, values in TAGS.items() for v in values]
    nb.tags = tuple(t for t in order if t in have)
    return nb


def scan(src=SRC):
    """Every notebook below ``src``, parsed (errors are kept on the notebooks)."""
    found = []
    for path in sorted(Path(src).rglob("*.ipynb")):
        if ".ipynb_checkpoints" in path.parts:
            continue
        rel = path.relative_to(src).as_posix()
        try:
            meta = json.loads(path.read_text()).get("metadata", {}).get("zoomy")
        except ValueError as exc:
            nb = Notebook(rel)
            nb.errors.append(f"{rel}: not a notebook ({exc})")
            found.append(nb)
            continue
        found.append(parse(rel, meta))
    return found


def load(src=SRC, book=BOOK):
    """The notebooks of the index; any problem stops the build."""
    notebooks = scan(src)
    errors = [e for nb in notebooks for e in nb.errors]
    mirror_dir = Path(book) / "ipynb"
    for stray in sorted(Path(book).rglob("*.ipynb")):
        if mirror_dir not in stray.parents:
            errors.append(f"{stray}: notebooks live under tutorials/ "
                          "(docs/book/tutorials/ipynb/ is generated)")
    if errors:
        raise SystemExit("tutorial metadata:\n  " + "\n  ".join(errors))
    notebooks.sort(key=lambda nb: (FILTERS["type"].index(nb.kind), nb.title.lower(), nb.rel))
    return notebooks


def mirror(notebooks, src=SRC, book=BOOK):
    """Copy the notebooks the book renders into ``docs/book/tutorials/ipynb/``.

    The copies are ``orphan`` pages (notebook metadata ``orphan``, which myst-nb turns into
    front matter): built, reached through their card, in no toctree and without the
    "document isn't included in any toctree" warning.
    """
    dest = Path(book) / "ipynb"
    if dest.exists():
        shutil.rmtree(dest)
    for nb in notebooks:
        if not nb.rel.startswith("pyodide/"):
            target = dest / nb.rel
            target.parent.mkdir(parents=True, exist_ok=True)
            data = json.loads((Path(src) / nb.rel).read_text())
            data.setdefault("metadata", {})["orphan"] = True
            target.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


def link(rel):
    if rel.startswith("pyodide/"):
        return f"../jupyter-lite/_output/lab/index.html?path={rel.split('/', 1)[1]}", "url"
    return f"ipynb/{rel[:-len('.ipynb')]}", "doc"


def _bar_html():
    """The field above the grid: selected tags as pills, a text input, the autocomplete list."""
    return ['<div class="zt-box"><span class="zt-pills"></span>'
            '<input class="zt-text" type="text" autocomplete="off" spellcheck="false" '
            'placeholder="type tags, e.g. hydro, 1d" aria-label="Filter by tags" '
            'role="combobox" aria-expanded="false" aria-controls="zt-suggest">'
            '<ul class="zt-suggest" id="zt-suggest" role="listbox" hidden></ul></div>',
            '<button type="button" class="zt-clear">reset</button>',
            '<span class="zt-status" role="status" aria-live="polite"></span>']


def _panel_html(notebooks):
    """Every tag that occurs, grouped by category, one chip each (value text, category as title)."""
    out = []
    for cat, values in FILTERS.items():
        chips = []
        for v in values:
            n = sum(f"{cat}:{v}" in nb.tags for nb in notebooks)
            if n:
                chips.append(f'<button type="button" class="zt-chip" data-tag="{cat}:{v}" '
                             f'title="{cat}: {v}" aria-pressed="false">{v} <span class="zt-n">{n}</span></button>')
        if chips:
            out.append(f'<section class="zt-group" style="--zt-h:{HUES[cat]}"><div class="zt-gh">{cat}</div>'
                       f'<div class="zt-chips">{"".join(chips)}</div></section>')
    return out


def _card_tag(tag):
    """A tag on a card: the value in its category colour (category as title); ``requires`` plain."""
    cat, _, value = tag.partition(":")
    if cat not in HUES:
        return f'<span class="zt-tag">{html.escape(tag)}</span>'
    return f'<span class="zt-tag" style="--zt-h:{HUES[cat]}" title="{cat}: {value}">{html.escape(value)}</span>'


def render(notebooks):
    lines = [
        "# Tutorials and use cases",
        "",
        "<!-- generated by docs/scripts/generate_tutorial_index.py from the "
        "notebooks' metadata.zoomy; do not edit -->",
        "",
        "Executable notebooks from `tutorials/`.  Click tags in the panel or type them in the "
        "field (comma-separated, with completion): a notebook is shown when it carries "
        "all selected tags.",
        "",
        "::::::{div} zt-layout",
        ":::::{div} zt-bar",
    ]
    lines += _bar_html() + [":::::", "", ":::::{div} zt-main"]
    if notebooks:
        lines += ["::::{grid} 1 2 2 2", ":gutter: 3", ""]
    for nb in notebooks:
        target, kind = link(nb.rel)
        shown = [t for t in nb.tags if t.split(":", 1)[0] in FILTERS or t != "requires:cpu"]
        lines += [f":::{{grid-item-card}} {nb.title}",
                  ":class-card: zt-card",
                  f":link: {target}", f":link-type: {kind}", "",
                  f'<span class="zt-data" data-tags="{html.escape(" ".join(nb.tags))}"></span>'
                  f"{nb.summary}",
                  "",
                  f"`tutorials/{nb.rel}`",
                  "+++",
                  " ".join(f'{_card_tag(t)}' for t in shown),
                  ":::", ""]
    if notebooks:
        lines += ["::::", ""]
    else:
        lines += ["*No notebooks are published yet.*", ""]
    lines += ['<p class="zt-empty" hidden>No notebook carries every selected tag.</p>', ":::::", "",
              ":::::{div} zt-panel"] + _panel_html(notebooks) + [":::::", "::::::", ""]
    return "\n".join(lines)


def render_howto():
    """The developer page: the metadata of a notebook and the tag vocabulary (from :data:`TAGS`)."""
    lines = [
        "# Adding a tutorial",
        "",
        "<!-- generated by docs/scripts/generate_tutorial_index.py; do not edit -->",
        "",
        "Put the notebook under `tutorials/` and give it this notebook metadata, then run "
        "`python docs/scripts/generate_tutorial_index.py` (CI does):",
        "",
        "```json",
        '"zoomy": {"title": "...", "summary": "<one line>", '
        '"tags": ["type:tutorial", "dimension:1d", "physics:viscous"]}',
        "```",
        "",
        "Tags are `category:value`; an unknown tag or a missing `type` fails the build.  "
        "A notebook may carry several values of one category (except `type`).",
        "",
    ]
    for cat, values in FILTERS.items():
        lines.append(f"- `{cat}:` " + " / ".join(values) + (" (required)" if cat in REQUIRED else ""))
    lines += ["",
              "`requires:` " + " / ".join(HARNESS["requires"]) + " is no filter.  It tells the "
              "test harness what the notebook needs (default `requires:cpu`): "
              "`pytest tests/tutorials -m tutorial` executes every notebook that needs only "
              "the CPU and reports the others as skipped.",
              "",
              "The book renders every notebook that has a card, from a copy in "
              "`docs/book/tutorials/ipynb/` (git-ignored); a notebook under `tutorials/pyodide/` "
              "is the content of the JupyterLite deployment and its card links there.",
              ""]
    return "\n".join(lines)


def main():
    notebooks = load()
    mirror(notebooks)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(notebooks))
    HOWTO.parent.mkdir(parents=True, exist_ok=True)
    HOWTO.write_text(render_howto())
    print(f"wrote {OUT.relative_to(ROOT)} ({len(notebooks)} notebooks), {HOWTO.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
