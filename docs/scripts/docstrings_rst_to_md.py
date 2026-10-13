#!/usr/bin/env python3
"""Docstring markup: reStructuredText -> Markdown (MyST), in place or as a report.

    python docs/scripts/docstrings_rst_to_md.py library/zoomy_core/zoomy_core          # dry run: summary
    python docs/scripts/docstrings_rst_to_md.py PATH... --diff                         # + unified diff
    python docs/scripts/docstrings_rst_to_md.py PATH... --write                        # rewrite the files

The API reference reads every docstring as Markdown (MyST, sphinx-autodoc2); see
``docs/book/documentation.md``.  Old docstrings carry rST inline markup, which
Markdown shows as text.  This script rewrites it, *in docstrings only*: each
docstring is located with the ``ast`` module (module, class, function, method),
and only its source text is touched, code and comments stay byte for byte.

    ``x``                              ->  `x`
    :func:`x` :meth:`x` :class:`x` ... ->  `x`      (``~pkg.f`` -> `f`; ``t <target>`` -> `t`)
    :math:`x`                          ->  $x$
    `text <url>`_                      ->  [text](url)
    `text`_  with  ".. _text: url"     ->  [text](url)   (the target line is dropped)
    Text::  + indented block           ->  Text:  + a ``` fence around the block

Doctest lines (``>>>``), fenced blocks and literal blocks are not rewritten
inside.  What is left to do by hand (``.. directive::`` blocks, title underlines,
unresolved ``name_`` references) is counted in the summary.  A changed file is
parsed again and its code must be identical to the original (docstrings aside),
otherwise the file is not written.  The Sphinx-side hook for the napoleon
sections (``Args:``, ``Parameters`` over dashes) is ``docs/book/_ext/zoomy_docstrings.py``
and needs no source change.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import re
import sys
from collections import Counter
from pathlib import Path

ROLES = ("func|meth|class|mod|attr|data|obj|exc|const|ref|doc|term|envvar|program|"
         "command|file|samp|kbd|py:func|py:meth|py:class|py:mod|py:attr|py:data|py:obj|py:exc")
_ROLE = re.compile(rf":(?:{ROLES}):`(?P<body>[^`]+)`")
_MATH = re.compile(r":math:`(?P<body>[^`]+)`")
_DOUBLE = re.compile(r"(?<![`\w])``(?P<body>[^`\n](?:[^`]*[^`\n])?)``(?![`\w])")
_LINK_URL = re.compile(r"`(?P<text>[^`<]+?)\s*<(?P<url>[^`>\s]+)>`__?")
_LINK_NAME = re.compile(r"`(?P<name>[^`<>]+)`__?(?![\w`])")
_TARGET = re.compile(r"^[ \t]*\.\. _(?P<name>[^:`\n]+):[ \t]*(?P<url>\S+)[ \t]*$", re.M)
_DIRECTIVE = re.compile(r"^[ \t]*\.\. [\w-]+::", re.M)
_UNDERLINE = re.compile(r"^[ \t]*(\S.*)\n[ \t]*([-=~^]{3,})[ \t]*$", re.M)
_NAMEREF = re.compile(r"(?<![\w`])(?P<name>[A-Za-z][\w-]*)_(?![\w`])")


def _role(m: re.Match) -> str:
    body = m.group("body").strip()
    t = re.fullmatch(r"(?P<text>.+?)\s*<(?P<target>[^<>]+)>", body)
    if t:                                                       # `title <target>`
        body = t.group("text")
    elif body.startswith("~"):                                  # ~pkg.mod.name -> name
        body = body[1:].split(".")[-1]
    elif body.startswith("!"):
        body = body[1:]
    return f"`{body}`"


def _protected(lines: list[str], col: int = 0) -> list[bool]:
    """True for lines whose text must stay as written: fences, doctests, literal blocks."""
    def indent(k: int) -> int:
        return col if k == 0 else len(lines[k]) - len(lines[k].lstrip())

    prot = [False] * len(lines)
    fence = False
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith(("```", "~~~")):
            fence = not fence
            prot[i] = True
        elif fence or s.startswith((">>>", "...")):
            prot[i] = True
            if s.startswith(">>>"):                              # the output below it
                j = i + 1
                while j < len(lines) and lines[j].strip():
                    prot[j] = True
                    j += 1
                i = j
                continue
        elif s.endswith("::") and not s.startswith(".."):        # a literal block left as rST
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or indent(j) > indent(i)):
                prot[j] = True
                j += 1
            i = j
            continue
        i += 1
    return prot


def _literal_blocks(lines: list[str], prot: list[bool], stats: Counter, col: int) -> list[str]:
    """``Text::`` + an indented block -> ``Text:`` + a fenced block.

    ``col``: the column where the docstring literal starts; the first line has no
    indentation of its own (the quotes precede it), the paragraph sits at ``col``.
    The last line (the closing quotes) is never part of a block."""
    def indent(k: int) -> int:
        return col if k == 0 else len(lines[k]) - len(lines[k].lstrip())

    out: list[str] = []
    last = len(lines) - 1
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.rstrip()
        if prot[i] or i == last or not s.endswith("::") or s.lstrip().startswith(".."):
            out.append(line)
            i += 1
            continue
        ind = indent(i)
        j = i + 1
        while j < last and not lines[j].strip():
            j += 1
        if j >= last or j == i + 1 or indent(j) <= ind:
            out.append(line)                                    # no block below: leave it
            i += 1
            continue
        k = j
        while k < last and (not lines[k].strip() or indent(k) > ind):
            k += 1
        if k == last and lines[last].strip(" \t\"'") and indent(last) > ind:
            out.append(line)                                    # the closing quotes sit in the block
            i += 1
            continue
        while not lines[k - 1].strip():
            k -= 1
        head = s.strip()[:-2].rstrip() if i else s[:-2].rstrip()
        pad = " " * ind
        out.append(f"{head}:" if i == 0 else (f"{pad}{head}:" if head else ""))
        out.extend(lines[i + 1:j])
        out.append(f"{pad}```")
        out.extend(lines[j:k])
        out.append(f"{pad}```")
        stats["literal blocks fenced"] += 1
        i = k
    return out


def convert(text: str, stats: Counter | None = None, col: int = 0) -> str:
    """Rewrite the rST markup of one docstring (its source text, quotes included;
    ``col`` is the column where it starts)."""
    stats = stats if stats is not None else Counter()
    lines = text.split("\n")
    prot = _protected(lines, col)
    lines = _literal_blocks(lines, prot, stats, col)
    prot = _protected(lines, col)
    targets = {m.group("name").strip().lower(): m.group("url") for m in _TARGET.finditer(text)}
    used: set[str] = set()

    def sub(rx, fn, line, key):
        new, n = rx.subn(fn, line)
        stats[key] += n
        return new

    def link_name(m: re.Match) -> str:
        name = m.group("name").strip()
        url = targets.get(name.lower())
        if url is None:
            stats["unresolved `name`_ references"] += 1
            return m.group(0)
        used.add(name.lower())
        return f"[{name}]({url})"

    def link_url(m: re.Match) -> str:
        return f"[{m.group('text').strip()}]({m.group('url')})"

    out = []
    for line, p in zip(lines, prot):
        if not p and not _TARGET.match(line):
            line = sub(_ROLE, _role, line, "roles -> `x`")
            line = sub(_MATH, lambda m: f"${m.group('body').strip()}$", line, ":math: -> $x$")
            line = sub(_LINK_URL, link_url, line, "links `t <url>`_ -> [t](url)")
            line = sub(_LINK_NAME, link_name, line, "links `name`_ -> [name](url)")
            line = sub(_DOUBLE, lambda m: f"`{m.group('body')}`", line, "``x`` -> `x`")
        out.append(line)
    # drop the targets that became inline links (and the blank line they leave)
    result: list[str] = []
    for line in out:
        m = _TARGET.match(line)
        if m and m.group("name").strip().lower() in used:
            if result and not result[-1].strip():              # keep one blank line
                result.pop()
            continue
        result.append(line)
    text = "\n".join(result)
    stats["left: .. directive:: blocks"] += len(_DIRECTIVE.findall(text))
    stats["left: title underlines"] += len(_UNDERLINE.findall(text))
    return text


def docstring_spans(src: bytes):
    """(start, end, column) of every docstring literal, byte offsets, by the ``ast``."""
    tree = ast.parse(src)
    starts = [0]
    for line in src.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                c = body[0].value
                yield (starts[c.lineno - 1] + c.col_offset, starts[c.end_lineno - 1] + c.end_col_offset,
                       c.col_offset)


def _code_dump(src: bytes) -> str:
    """The code of a file without its docstrings."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                b[0].value.value = ""
    return ast.dump(tree)


def convert_source(src: bytes, stats: Counter) -> bytes:
    out, pos = [], 0
    for a, b, col in sorted(docstring_spans(src)):
        seg = src[a:b].decode("utf-8")
        new = convert(seg, stats, col)
        out.append(src[pos:a])
        out.append(new.encode("utf-8"))
        pos = b
        stats["docstrings changed"] += seg != new
        stats["docstrings"] += 1
    out.append(src[pos:])
    return b"".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", type=Path, help="python files or source trees")
    ap.add_argument("--write", action="store_true", help="rewrite the files (default: report only)")
    ap.add_argument("--diff", action="store_true", help="print the unified diff of every change")
    args = ap.parse_args(argv)

    files = sorted({f for p in args.paths for f in ([p] if p.is_file() else p.rglob("*.py"))
                    if "__pycache__" not in f.parts})
    stats: Counter = Counter()
    changed, refused = [], []
    for f in files:
        src = f.read_bytes()
        try:
            new = convert_source(src, stats)
        except (SyntaxError, UnicodeDecodeError) as e:
            refused.append(f"{f}: not parsed ({e.__class__.__name__})")
            continue
        if new == src:
            continue
        try:
            same_code = _code_dump(new) == _code_dump(src)
        except SyntaxError:
            same_code = False
        if not same_code:
            refused.append(f"{f}: the converted file does not parse to the same code; left alone")
            continue
        changed.append(f)
        if args.diff:
            sys.stdout.writelines(difflib.unified_diff(
                src.decode().splitlines(True), new.decode().splitlines(True), f"a/{f}", f"b/{f}", n=1))
        if args.write:
            f.write_bytes(new)

    print(f"files scanned {len(files)}, files {'rewritten' if args.write else 'to change'} "
          f"{len(changed)}, docstrings {stats['docstrings']} (changed {stats['docstrings changed']})")
    for key in sorted(k for k in stats if k not in ("docstrings", "docstrings changed")):
        print(f"  {key:<42}{stats[key]:>6}")
    for r in refused:
        print("  REFUSED " + r)
    if changed and not args.write:
        print("dry run: nothing written (use --write)")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
