"""Literature bibliography -> docs/book/references/literature.bib.

    python docs/scripts/collect_references.py            # scan + fetch BibTeX, write literature.bib
    python docs/scripts/collect_references.py --check    # offline: every cited DOI/URL is in literature.bib

The implementing code cites its references as ``Author Year, https://doi.org/<doi>``
(Python docstrings, C++ ``///`` comments, docs pages).  This script scans

* ``library/zoomy_{core,jax,foam,amrex}/**/*.{py,H,C,cpp}`` and
* ``docs/book/**/*.md`` (not ``_build``, not the generated ``api/_autodoc2``)

for ``https://doi.org/<doi>`` and for the non-DOI URLs of the verified set
``docs/book/references/literature_verified.tsv`` (key, DOI or URL, short
description; every row was resolved and checked by hand).

Default mode fetches BibTeX per DOI (``https://doi.org/<doi>`` with
``Accept: application/x-bibtex``, falling back to the Crossref transform API;
HTTP 429 is retried with backoff), normalises it and writes
``literature.bib`` sorted by key.  Keys are ``firstauthorYEARfirstword``; a key
that already exists in ``third_party.bib`` gets ``_lit`` appended, because
sphinxcontrib-bibtex needs unique keys across all configured bib files.  The
``keywords`` field lists where an entry is cited (module path or docs page), or
is ``uncited`` for a row of the verified set that no source cites (kept so it can
be pruned later).  A URL-only reference becomes ``@misc`` with ``howpublished``.

``--check`` needs no network: it exits 1 and lists every DOI/URL that is cited
in the sources but missing from ``literature.bib``, and every key shared with
``third_party.bib``.

Hand corrections that survive a re-run live in :data:`FIXES`.
"""
from __future__ import annotations

import argparse
import html
import os
import re
import ssl
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "library"
BOOK = ROOT / "docs" / "book"
REFS = BOOK / "references"
BIB = REFS / "literature.bib"
THIRD = REFS / "third_party.bib"
TSV = REFS / "literature_verified.tsv"

PACKAGES = ("zoomy_core", "zoomy_jax", "zoomy_foam", "zoomy_amrex")
SOURCE_SUFFIXES = {".py", ".H", ".C", ".cpp"}
SKIP_DIRS = {"__pycache__", "build", "_build", "node_modules", "venv"}
UA = "zoomy-collect-references/1.0 (https://github.com/ZoomyLab/Zoomy)"

# Metadata of the non-DOI references: the citation each project asks for.
URL_ENTRIES = {
    "https://github.com/jax-ml/jax": {
        "type": "misc",
        "author": (
            "Bradbury, James and Frostig, Roy and Hawkins, Peter and "
            "Johnson, Matthew James and Katariya, Yash and Leary, Chris and "
            "Maclaurin, Dougal and Necula, George and Paszke, Adam and "
            "Vander{P}las, Jake and Wanderman-{M}ilne, Skye and Zhang, Qiao"
        ),
        "title": "JAX: composable transformations of Python+NumPy programs",
        "year": "2018",
    },
}

# Corrections applied after fetching, keyed by lower-case DOI.  A value of None
# drops the field; "@type" changes the entry type.  Reproducible, so a re-run
# does not lose them.
FIXES: dict[str, dict[str, str | None]] = {
    # Crossref mangles a character in the author list
    "10.1137/s1064827503431090": {
        "author": "Audusse, Emmanuel and Bouchut, François and Bristeau, Marie-Odile "
                  "and Klein, Rupert and Perthame, Benoît"},
    # family name split wrongly ("Leer, Bram van")
    "10.1137/1025002": {
        "author": "Harten, Amiram and Lax, Peter D. and van Leer, Bram"},
    # the Crossref record lists one author and the online-first year
    "10.56021/9781421407944": {
        "author": "Golub, Gene H. and Van Loan, Charles F.", "year": "2013",
        "edition": "Fourth"},
    # print year of the volume, as cited in the code (Crossref has the online-first year)
    "10.4208/cicp.oa-2017-0263": {"year": "2019"},
    "10.1051/m2an/2010036": {"year": "2011"},
    "10.1002/fld.1880": {"year": "2009"},
    # chapter of a series volume, not a book; the title starts with the chapter number
    "10.1137/1.9781611971057.ch4": {
        "@type": "incollection", "title": "Algebraic multigrid",
        "editor": "McCormick, Stephen F.", "pages": "73--130"},
    "10.1016/s1570-8659(00)07005-8": {
        "@type": "incollection",
        "booktitle": "Handbook of Numerical Analysis, Volume 7: Techniques of "
                     "Scientific Computing (Part 3)",
        "editor": "Ciarlet, Philippe G. and Lions, Jacques-Louis"},
}

DOI_RE = re.compile(
    r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s<>\"'`\[\]{}|\\^]+)", re.I
)
KEEP_FIELDS = (
    "author", "editor", "title", "booktitle", "journal", "series", "volume",
    "number", "pages", "publisher", "address", "edition", "year", "isbn",
)
STOPWORDS = {"a", "an", "the", "on", "of", "for", "and", "in", "to", "with", "by", "from"}


# ----------------------------------------------------------------- scanning
def _clean_doi(raw: str) -> str:
    d = raw.rstrip(".,;:!?*'\"")
    while d.endswith(")") and d.count(")") > d.count("("):
        d = d[:-1].rstrip(".,;:!?*'\"")
    return d


def _walk_sources():
    for pkg in PACKAGES:
        for dirpath, dirs, files in os.walk(LIB / pkg):
            dirs[:] = sorted(
                d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")
            )
            for f in sorted(files):
                p = Path(dirpath) / f
                if p.suffix in SOURCE_SUFFIXES:
                    yield p
    for dirpath, dirs, files in os.walk(BOOK):
        rel = Path(dirpath).relative_to(BOOK)
        dirs[:] = sorted(
            d for d in dirs
            if d != "_build" and not d.startswith(".")
            and not (rel.parts[:1] == ("api",) and d == "_autodoc2")
        )
        for f in sorted(files):
            if f.endswith(".md"):
                yield Path(dirpath) / f


def _site(p: Path) -> str:
    """Where a citation sits: module path (Python), file path (C++), docs page."""
    try:
        rel = p.relative_to(BOOK)
        return "docs/" + rel.as_posix()
    except ValueError:
        pass
    rel = p.relative_to(LIB)
    if rel.suffix == ".py" and len(rel.parts) > 2 and rel.parts[0] == rel.parts[1]:
        mod = list(rel.parts[1:-1]) + [rel.stem]
        if mod[-1] == "__init__":
            mod.pop()
        return ".".join(mod)
    return rel.as_posix()


def read_verified() -> list[tuple[str, str, str]]:
    rows = []
    for i, line in enumerate(TSV.read_text(encoding="utf-8").splitlines()):
        if i == 0 or not line.strip():
            continue
        key, ident, short = (line.split("\t") + ["", ""])[:3]
        rows.append((key.strip(), ident.strip(), short.strip()))
    return rows


def _is_url(ident: str) -> bool:
    return not ident.startswith("10.")


def _url_re(url: str) -> re.Pattern:
    return re.compile(re.escape(url) + r"(?![\w\-]|\.\w|/\w)")


def scan(verified) -> dict[str, dict]:
    """Return ``{id_lower: {"id": str, "kind": "doi"|"url", "sites": set}}`` of what the sources cite."""
    urls = {ident: _url_re(ident) for _, ident, _ in verified if _is_url(ident)}
    found: dict[str, dict] = {}
    canon = {ident.lower(): ident for _, ident, _ in verified}
    for p in _walk_sources():
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        site = _site(p)
        for m in DOI_RE.finditer(text):
            doi = _clean_doi(m.group(1))
            rec = found.setdefault(
                doi.lower(), {"id": canon.get(doi.lower(), doi), "kind": "doi", "sites": set()}
            )
            rec["sites"].add(site)
        for url, rx in urls.items():
            if rx.search(text):
                rec = found.setdefault(url.lower(), {"id": url, "kind": "url", "sites": set()})
                rec["sites"].add(site)
    return found


# -------------------------------------------------------------- bib parsing
def _split_top(body: str) -> list[tuple[str, str]]:
    """Fields of one BibTeX entry body (after ``key,``) as ``[(name, value)]``."""
    fields, i, n = [], 0, len(body)
    while i < n:
        m = re.compile(r"\s*,?\s*([A-Za-z][\w\-:]*)\s*=\s*").match(body, i)
        if not m:
            break
        name, i = m.group(1).lower(), m.end()
        if i < n and body[i] == "{":
            depth, j = 0, i
            while j < n:
                depth += (body[j] == "{") - (body[j] == "}")
                j += 1
                if depth == 0:
                    break
            value, i = body[i + 1:j - 1], j
        elif i < n and body[i] == '"':
            j = body.index('"', i + 1)
            value, i = body[i + 1:j], j + 1
        else:
            m2 = re.compile(r"[^,}\s]+").match(body, i)
            value, i = m2.group(0), m2.end()
        fields.append((name, value))
    return fields


def parse_entry(text: str) -> tuple[str, str, dict[str, str]]:
    m = re.search(r"@\s*(\w+)\s*\{\s*([^,\s]*)\s*,", text)
    if not m:
        raise ValueError("no BibTeX entry")
    end = text.rstrip().rfind("}")
    return m.group(1).lower(), m.group(2), dict(_split_top(text[m.end():end]))


def parse_bib_file(path: Path) -> list[tuple[str, str, dict[str, str]]]:
    """All entries of a bib file (comments and ``@string`` skipped)."""
    text = path.read_text(encoding="utf-8")
    out, pos = [], 0
    for m in re.finditer(r"^@(\w+)\s*\{", text, re.M):
        if m.start() < pos:
            continue
        depth, j = 0, m.end() - 1
        while j < len(text):
            depth += (text[j] == "{") - (text[j] == "}")
            j += 1
            if depth == 0:
                break
        pos = j
        if m.group(1).lower() in {"comment", "string", "preamble"}:
            continue
        out.append(parse_entry(text[m.start():j]))
    return out


# ----------------------------------------------------------------- cleaning
def clean_names(s: str) -> str:
    """Author list: no ALL-CAPS names, initials spelled ``A. B.``."""
    out = []
    for name in re.split(r"\s+and\s+", s):
        if name.startswith("{"):  # corporate author
            out.append(name)
            continue
        fam, sep, given = name.partition(",")
        if fam.isupper() and len(fam) > 1:
            fam = fam.title()
        if given.isupper() and len(given.strip()) > 1:
            given = given.title()
        given = re.sub(r"\b([A-Z])\.(?=[A-Z])", r"\1. ", given)
        given = re.sub(r"\b([A-Z])(?=\s|$)", r"\1.", given)
        out.append(fam + sep + given)
    return " and ".join(out)


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFC", html.unescape(s))
    s = s.replace("\u2010", "-").replace("\u2011", "-")
    s = re.sub(r"<i>(.*?)</i>", r"\\emph{\1}", s, flags=re.S)
    s = re.sub(r"<b>(.*?)</b>", r"\\textbf{\1}", s, flags=re.S)
    s = re.sub(r"<sub>(.*?)</sub>", r"$_{\1}$", s, flags=re.S)
    s = re.sub(r"<sup>(.*?)</sup>", r"$^{\1}$", s, flags=re.S)
    s = re.sub(r"</?[A-Za-z][^>]*>", "", s)  # <scp>, MathML wrappers, ...
    s = s.replace("\u2013", "--").replace("\u2014", "---")
    s = re.sub(r"(?<!\\)&", r"\\&", s)
    s = re.sub(r"(?<!\\)%", r"\\%", s)
    s = re.sub(r"(?<!\\)#", r"\\#", s)
    return re.sub(r"\s+", " ", s).strip()


def _fold(s: str) -> str:
    s = re.sub(r"\\[`'^\"~=.uvHcdbtkr]\s*\{?([A-Za-z])\}?", r"\1", s)  # \'e -> e
    s = re.sub(r"[{}\\]", "", s)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def first_author_surname(author: str) -> str:
    first = re.split(r"\s+and\s+", author.strip())[0]
    first = first.strip()
    if first.startswith("{") and first.endswith("}"):  # corporate author
        return re.sub(r"[^a-z0-9]", "", _fold(first).lower())
    if "," in first:
        fam = first.split(",")[0]
    else:
        fam = first.split()[-1]
    return re.sub(r"[^a-z0-9]", "", _fold(fam).lower())


def title_word(title: str) -> str:
    for w in re.split(r"[^a-z0-9]+", _fold(title).lower()):
        if w and w not in STOPWORDS:
            return w
    return "untitled"


# ------------------------------------------------------------------ fetching
CA_BUNDLES = (
    "/etc/ssl/certs/ca-certificates.crt",  # Debian, Ubuntu
    "/etc/pki/tls/certs/ca-bundle.crt",  # Fedora, RHEL
    "/etc/ssl/cert.pem",  # macOS, Alpine
)


def _ssl_context() -> ssl.SSLContext:
    """Default context; a Python without a CA store (e.g. Homebrew) uses the system bundle."""
    ctx = ssl.create_default_context()
    if not (os.environ.get("SSL_CERT_FILE") or ctx.cert_store_stats()["x509_ca"]):
        for bundle in CA_BUNDLES:
            if os.path.exists(bundle):
                ctx.load_verify_locations(bundle)
                break
    return ctx


def http_get(url: str, accept: str, retries: int = 7) -> str | None:
    delay = 2.0
    ctx = _ssl_context()
    for _ in range(retries):
        req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=40, context=ctx) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:
                try:
                    wait = float(e.headers.get("Retry-After", ""))
                except ValueError:
                    wait = delay
                time.sleep(min(max(wait, delay), 120))
                delay *= 2
                continue
            return None
        except urllib.error.URLError as e:
            if isinstance(e.reason, ssl.SSLCertVerificationError):
                raise SystemExit(f"TLS verification failed for {url}: {e.reason}\n"
                                 "set SSL_CERT_FILE to a CA bundle") from e
            time.sleep(delay)
            delay *= 2
        except (TimeoutError, ConnectionError):
            time.sleep(delay)
            delay *= 2
    return None


def fetch_bibtex(doi: str) -> str | None:
    q = urllib.parse.quote(doi, safe="/()-._~:")
    for url in (
        f"https://doi.org/{q}",
        f"https://api.crossref.org/works/{q}/transform/application/x-bibtex",
    ):
        text = http_get(url, "application/x-bibtex")
        if text and text.lstrip().startswith("@"):
            try:
                _, _, fields = parse_entry(text)
            except ValueError:
                continue
            if "title" in fields and ("author" in fields or "editor" in fields):
                return text
        time.sleep(0.5)
    return None


def build_doi_entry(doi: str, text: str) -> tuple[str, dict[str, str]]:
    etype, _, raw = parse_entry(text)
    fields = {k: clean_text(raw[k]) for k in KEEP_FIELDS if k in raw and raw[k].strip()}
    for k in ("author", "editor"):
        if k in fields:
            fields[k] = clean_names(fields[k])
    if "title" in fields:
        fields["title"] = fields["title"].rstrip(". ")  # the style adds its own period
    if etype == "article":
        fields.pop("publisher", None)
        fields.pop("address", None)
    fields["doi"] = doi  # the style prints the DOI link; a url field would repeat it
    fix = FIXES.get(doi.lower(), {})
    for k, v in fix.items():
        if k == "@type":
            etype = v
        elif v is None:
            fields.pop(k, None)
        else:
            fields[k] = v
    return etype, fields


def build_url_entry(url: str) -> tuple[str, dict[str, str]]:
    meta = dict(URL_ENTRIES[url])
    etype = meta.pop("type")
    fields = {k: clean_text(v) if k != "author" else v for k, v in meta.items()}
    fields["howpublished"] = "\\url{%s}" % url
    return etype, fields


# ------------------------------------------------------------------- writing
def _field_text(name: str, value: str) -> str:
    if name == "title":
        value = "{" + value + "}"  # protect the case; the style would lower-case it
    return value


def render(entries: list[dict]) -> str:
    head = (
        "% Literature cited by the Zoomy code and docs.\n"
        "% GENERATED by docs/scripts/collect_references.py (python docs/scripts/collect_references.py);\n"
        "% do not edit by hand -- corrections go into FIXES in that script.\n"
        "% Rendered as https://zoomylab.github.io/Zoomy/literature.html\n"
        "%\n"
        "% - One entry per DOI (or URL) cited as 'Author Year, https://doi.org/<doi>'.\n"
        "% - keywords lists where the entry is cited (module path, C++ file, docs page);\n"
        "%   'uncited' marks a verified reference that no source cites yet.\n"
        "% - Keys: firstauthorYEARfirstword; '_lit' marks a key that third_party.bib\n"
        "%   already uses (sphinxcontrib-bibtex needs unique keys across bib files).\n\n"
    )
    out = []
    for e in sorted(entries, key=lambda e: e["key"]):
        lines = [f"@{e['type']}{{{e['key']},"]
        names = list(e["fields"])
        w = max(len(n) for n in names)
        for n in names:
            lines.append(f"  {n:<{w}} = {{{_field_text(n, e['fields'][n])}}},")
        lines[-1] = lines[-1].rstrip(",")
        lines.append("}")
        out.append("\n".join(lines))
    return head + "\n\n".join(out) + "\n"


def assign_keys(entries: list[dict], taken: set[str]) -> None:
    used = {k.lower() for k in taken}
    for e in sorted(entries, key=lambda e: e["ident"].lower()):
        f = e["fields"]
        who = f.get("author") or f.get("editor") or "anon"
        base = f"{first_author_surname(who)}{f.get('year', 'nd')}{title_word(f.get('title', ''))}"
        key = base
        if key.lower() in used:
            key = base + "_lit"
        n = 0
        while key.lower() in used:
            key = f"{base}{chr(ord('b') + n)}"
            n += 1
        used.add(key.lower())
        e["key"] = key


def _verify_against_tsv(entries, verified) -> list[str]:
    """Year and first-author surname of each entry must appear in the TSV short description."""
    short = {ident.lower(): s for _, ident, s in verified}
    bad = []
    for e in entries:
        s = short.get(e["ident"].lower())
        if s is None:
            continue
        f = e["fields"]
        year = f.get("year", "")
        said = re.search(r"\b(?:1[89]|20)\d\d\b", s)
        who = first_author_surname(f.get("author") or f.get("editor") or "")
        if said and said.group() != year:
            bad.append(f"{e['ident']}: year {year} not in '{s}'")
        if who and who not in re.sub(r"[^a-z0-9]", "", _fold(s).lower()):
            bad.append(f"{e['ident']}: author '{who}' not in '{s}'")
    return bad


# ---------------------------------------------------------------------- main
def _bib_ids(path: Path) -> set[str]:
    ids = set()
    for _, _, f in parse_bib_file(path):
        if "doi" in f:
            ids.add(f["doi"].strip().lower())
        hp = re.search(r"\\url\{([^}]*)\}", f.get("howpublished", ""))
        if hp:
            ids.add(hp.group(1).strip().lower())
    return ids


def _keys(path: Path) -> set[str]:
    return {k.lower() for _, k, _ in parse_bib_file(path)}


def check() -> int:
    verified = read_verified()
    cited = scan(verified)
    have = _bib_ids(BIB) if BIB.exists() else set()
    missing = {i: r for i, r in cited.items() if i not in have}
    clash = (_keys(BIB) & _keys(THIRD)) if BIB.exists() and THIRD.exists() else set()
    uncited = [ident for _, ident, _ in verified if ident.lower() not in cited]
    if uncited:
        print(f"info: {len(uncited)} verified reference(s) cited nowhere: " + ", ".join(uncited))
    if missing:
        print(f"{len(missing)} cited reference(s) missing from {BIB.relative_to(ROOT)} "
              "(run docs/scripts/collect_references.py):")
        for i, r in sorted(missing.items()):
            print(f"  {r['id']}   <- {', '.join(sorted(r['sites']))}")
    if clash:
        print("keys shared with third_party.bib (sphinxcontrib-bibtex needs unique keys): "
              + ", ".join(sorted(clash)))
    if missing or clash:
        return 1
    print(f"ok: {len(cited)} cited reference(s), all in {BIB.relative_to(ROOT)}")
    return 0


def collect() -> int:
    verified = read_verified()
    cited = scan(verified)
    entries: list[dict] = []
    failed: list[str] = []
    ids: dict[str, str] = {}  # lower -> canonical id
    for _, ident, _ in verified:
        ids[ident.lower()] = ident
    for low, rec in cited.items():
        ids.setdefault(low, rec["id"])
    for low, ident in sorted(ids.items()):
        if _is_url(ident):
            if ident not in URL_ENTRIES:
                failed.append(f"{ident}: no metadata in URL_ENTRIES")
                continue
            etype, fields = build_url_entry(ident)
        else:
            text = fetch_bibtex(ident)
            if text is None:
                failed.append(f"{ident}: no BibTeX from doi.org or Crossref")
                continue
            etype, fields = build_doi_entry(ident, text)
            time.sleep(0.2)
        sites = sorted(cited[low]["sites"]) if low in cited else []
        fields["keywords"] = ", ".join(sites) if sites else "uncited"
        entries.append({"type": etype, "fields": fields, "ident": ident, "cited": bool(sites)})
    if failed:
        print("FAILED, literature.bib not written:\n  " + "\n  ".join(failed), file=sys.stderr)
        return 1
    assign_keys(entries, _keys(THIRD) if THIRD.exists() else set())
    BIB.write_text(render(entries), encoding="utf-8")
    n_unc = sum(not e["cited"] for e in entries)
    print(f"wrote {BIB.relative_to(ROOT)}: {len(entries)} entries "
          f"({len(entries) - n_unc} cited, {n_unc} uncited)")
    if n_unc:
        print("uncited (kept, keywords = {uncited}): "
              + ", ".join(e["ident"] for e in entries if not e["cited"]))
    notv = [r["id"] for low, r in cited.items() if low not in {i.lower() for _, i, _ in verified}]
    if notv:
        print("cited but not in the verified set: " + ", ".join(notv))
    for line in _verify_against_tsv(entries, verified):
        print("MISMATCH " + line, file=sys.stderr)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="offline: exit 1 if a cited DOI/URL is missing from literature.bib")
    args = ap.parse_args(argv)
    return check() if args.check else collect()


if __name__ == "__main__":
    sys.exit(main())
