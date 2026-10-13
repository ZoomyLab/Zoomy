"""Docstring handling of the API reference: MyST (Markdown), two small hooks.

The docstrings of the libraries are Markdown.  sphinx-autodoc2 hands each one
to the parser named in ``autodoc2_docstring_parser_regexes`` (``_config.yml``);
that is :class:`Parser` here, the stock MyST parser behind one preprocessing
step, :func:`sections_to_markdown`.  Markdown knows neither the Google/napoleon
sections nor the numpy ones; they would run into one paragraph (``Args:``) or
become headings (``Parameters`` over a line of dashes), so they are rewritten:

    Args:                                  **Args**
        cfl: CFL number.           ->
        mood: MOOD guard.                  - `cfl`: CFL number.
                                           - `mood`: MOOD guard.

    References:                            **References**
        Rusanov 1962, https://...  ->
                                           - Rusanov 1962, https://...

Fenced code blocks are left alone; everything else is read as MyST.  A heading
inside a docstring (Title over a line of dashes) becomes a bold label, because
the docstring sits inside a page whose own headings it must not disturb.

The second hook, :func:`toctree_to_links`, turns the ``toctree`` directives that
autodoc2 writes into its package pages into lists of links: the book's table of
contents is ``_toc.yml`` (sphinx-external-toc), which rejects in-page toctrees.

    python docs/book/_ext/zoomy_docstrings.py FILE.py    # show the converted docstrings
"""
from __future__ import annotations

import re

from myst_parser.parsers.sphinx_ import MystParser

# section title (lower case) -> how the block below it is written
ENTRIES = {"args", "arguments", "parameters", "params", "keyword args", "keyword arguments",
           "other parameters", "attributes", "raises", "warns", "returns", "return",
           "yields", "yield"}                          # `name (type): text` bullets
ITEMS = {"references", "see also"}                     # one bullet per line
TEXT = {"note", "notes", "warning", "warnings", "example", "examples", "usage", "todo"}
SECTIONS = ENTRIES | ITEMS | TEXT
_GOOGLE_TEXT = {"returns", "return", "yields", "yield"}    # Google style: free text, not entries

_HEADER = re.compile(r"^(?P<ind>[ ]*)(?P<title>[A-Za-z][A-Za-z ]*?):[ ]*$")
_RULE = re.compile(r"^[ ]*(-{3,}|={3,}|~{3,})[ ]*$")
_ENTRY = re.compile(r"^(?P<name>\*{0,2}[\w.]+(?:[ ]*,[ ]*\*{0,2}[\w.]+)*)"
                    r"(?:[ ]*\((?P<type>[^)]*)\))?[ ]*:(?:[ ]+(?P<text>.*))?$")
_NP_ENTRY = re.compile(r"^(?P<name>\*{0,2}[\w.]+(?:[ ]*,[ ]*\*{0,2}[\w.]+)*)[ ]*:[ ]*(?P<type>.*)$")
_FENCE = re.compile(r"^[ ]*(`{3,}|~{3,}|:{3,})")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _names(spec: str) -> str:
    return ", ".join(f"`{n.strip()}`" for n in spec.split(","))


def _groups(block: list[str]) -> list[list[str]] | None:
    """Lines at the block's base indentation start a group, deeper lines continue it."""
    base = min(_indent(l) for l in block if l.strip())
    groups: list[list[str]] = []
    for line in block:
        if line.strip() and _indent(line) == base:
            groups.append([line[base:]])
        elif groups:
            groups[-1].append(line)
        else:
            return None
    return groups


def _bullets(block: list[str], head_of) -> list[str] | None:
    """The block as bullets; ``head_of(first_line)`` returns the bullet head, or
    ``None`` if the block does not read as a list."""
    groups = _groups(block)
    if groups is None:
        return None
    out: list[str] = []
    for group in groups:
        head = head_of(group[0])
        if head is None:
            return None
        rest = [l for l in group[1:]]
        body = [l for l in rest if l.strip()]
        cont = min((_indent(l) for l in body), default=0)
        out.append("- " + head)
        out.extend("  " + l[cont:] if l.strip() else "" for l in rest)
    return out


def _google_head(line: str) -> str | None:
    m = _ENTRY.match(line)
    if m is None:
        return None
    return _names(m.group("name")) + (f" ({m.group('type')})" if m.group("type") else "") + (
        f": {m.group('text')}" if m.group("text") else "")


def _numpy_head(line: str) -> str:
    """``name : type`` (the description is on the lines below)."""
    m = _NP_ENTRY.match(line)
    if m is None:
        return line.rstrip(":")
    return _names(m.group("name")) + (f" ({m.group('type').strip()})" if m.group("type").strip() else "")


def _numpy_bullets(block: list[str]) -> list[str] | None:
    out = _bullets(block, _numpy_head)
    if out is None:
        return None
    # `- head` followed by the description: join as `- head: description`
    for k, line in enumerate(out):
        if line.startswith("- ") and k + 1 < len(out) and out[k + 1].startswith("  ") \
                and out[k + 1].strip() and not line.rstrip().endswith(":"):
            out[k] = line + ":"
    return out


def _body(block: list[str], title: str, kind: str, numpy: bool) -> list[str]:
    """The Markdown lines of a section body."""
    bullets = None
    if numpy and (title in ENTRIES or title in ITEMS):
        bullets = _numpy_bullets(block)
    elif not numpy and title in ENTRIES and title not in _GOOGLE_TEXT:
        bullets = _bullets(block, _google_head)
    elif not numpy and title in ITEMS:
        bullets = _bullets(block, lambda l: l)
    if bullets is None:
        base = min(_indent(l) for l in block if l.strip())
        bullets = [l[base:] if l.strip() else "" for l in block]
    return bullets


def _label(out: list[str], label: str, body: list[str]) -> None:
    if out and out[-1].strip():
        out.append("")
    out.extend([f"**{label}**", "", *body, ""])


def sections_to_markdown(doc: str) -> str:
    """Rewrite the napoleon (Google, numpy) sections and the headings of a docstring."""
    lines = doc.splitlines()
    out: list[str] = []
    fence = None
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _FENCE.match(line)
        if m:
            if fence is None:
                fence = m.group(1)[0]
            elif m.group(1)[0] == fence:
                fence = None
        if fence is not None or m:
            out.append(line)
            i += 1
            continue
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        # numpy style / any heading:  Title  over a line of dashes
        if line.strip() and not _RULE.match(line) and _RULE.match(nxt) \
                and _indent(nxt) == _indent(line):
            title = line.strip()
            if title.lower() in SECTIONS:
                j = i + 2
                while j < len(lines) and not (j + 1 < len(lines) and lines[j].strip()
                                              and not _RULE.match(lines[j])
                                              and _RULE.match(lines[j + 1])):
                    j += 1
                block = lines[i + 2:j]
                while block and not block[-1].strip():
                    block.pop()
                if any(l.strip() for l in block):
                    _label(out, title, _body(block, title.lower(), "numpy", True))
                    i = j
                    continue
            _label(out, title, [])
            out.pop()                                  # the heading alone: no blank body
            i += 2
            continue
        # Google style:  Args:  followed by an indented block
        h = _HEADER.match(line)
        if h and h.group("title").strip().lower() in SECTIONS:
            n = _indent(line)
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or _indent(lines[j]) > n):
                j += 1
            block = lines[i + 1:j]
            while block and not block[-1].strip():
                block.pop()
            if any(l.strip() for l in block):
                title = h.group("title").strip()
                _label(out, title, _body(block, title.lower(), "google", False))
                i = j
                continue
        out.append(line)
        i += 1
    return "\n".join(out) + ("\n" if doc.endswith("\n") else "")


class Parser(MystParser):
    """MyST, after :func:`sections_to_markdown`."""

    supported = ("md", "markdown", "myst")

    def parse(self, inputstring: str, document) -> None:
        super().parse(sections_to_markdown(inputstring), document)


_TOCTREE = re.compile(r"^(?P<fence>`{3,})\{toctree\}[^\n]*\n(?:[ ]*:[^\n]*\n)*\n?"
                      r"(?P<names>(?:[^\n`]+\n)*)(?P=fence)[ ]*$", re.M)


def toctree_to_links(app, docname: str, source: list[str]) -> None:
    """``source-read``: the toctrees of the autodoc2 pages become link lists."""
    if not docname.startswith("api/_autodoc2/"):
        return
    source[0] = _TOCTREE.sub(
        lambda m: "".join(f"- {{doc}}`{n.strip()}`\n" for n in m.group("names").splitlines()
                          if n.strip()),
        source[0])


def setup(app):
    app.connect("source-read", toctree_to_links)
    return {"parallel_read_safe": True, "parallel_write_safe": True}


if __name__ == "__main__":
    import ast
    import sys

    for path in sys.argv[1:]:
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                d = ast.get_docstring(node)
                if d and sections_to_markdown(d) != d:
                    print(f"--- {getattr(node, 'name', path)}\n{sections_to_markdown(d)}")
