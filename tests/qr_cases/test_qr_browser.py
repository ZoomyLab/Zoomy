"""Every GUI session a thesis QR code opens runs in a real browser, in the notebook view, twice.

Headless Chromium (Playwright) opens each QR URL that loads a session (`?project=...`) with
`view=notebook` (added where the code does not carry it: the reader's "Open in Notebook Mode"),
appends one final code cell, clicks Run All, reads the verdict, and runs all cells again: the
second Run All is what a reader does after editing a cell, and what failed on the deployed site
(the open store stopped the solver rewriting its file). Everything goes through the DOM: the
GUI's own notebook, the Pyodide kernel it runs, its toolbar.

The final cell is one line (`session_runner.notebook_check_code`): it runs
`session_runner.notebook_check` in the kernel and prints `QR_CHECK <verdict>`. Pass = every cell
succeeded (the kernel's `sys.zoomy_cell_log`, and the cells' status in the DOM), and a session
that solves published a finite store with >= 2 snapshots up to its time_end
(`final_store_problem`; `qr_manifest.END_TIME`). Sessions whose solver card needs a backend
(Malpasset: jax, AMReX) cannot solve in the browser: their model, step, mesh and settings cells
must pass; the run and visualization cells are not judged.

GUI under test: `QR_GUI_BASE` (the site root, e.g. https://zoomylab.github.io/Zoomy for the
deployed one); else a local build served at http://127.0.0.1:<port>/Zoomy/ with the deploy's
layout: `QR_GUI_SITE` names a site built by `docs/scripts/build_gui_site.sh SITE`, else this
checkout's GUI is built into a temporary directory (npm, node, network).

    pytest tests/qr_cases -m qr_browser                       # local build
    QR_GUI_BASE=https://zoomylab.github.io/Zoomy pytest tests/qr_cases -m qr_browser
"""
import functools
import http.server
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qr_manifest as Q  # noqa: E402
import session_runner as R  # noqa: E402

#: per Run All, s (Pyodide boot and package loads included)
RUN_TIMEOUT = float(os.environ.get("QR_BROWSER_TIMEOUT", "900"))
#: the notebook view opening (GUI load, project fetch, session materialized)
OPEN_TIMEOUT = 240.0
#: a session's code cells, as the notebook shows them
CELLS = "li.theia-notebook-cell:has(.notebook-cell-status)"


def _skip_or_fail(reason):
    (pytest.fail if os.environ.get("QR_STRICT") else pytest.skip)(reason)


# --------------------------------------------------------------------------- targets

def _targets():
    """`[(key, page path + query relative to the site root, zip name, session index, session,
    view added)]`: every QR URL that opens a GUI session (not a `#/open` file link)."""
    out = []
    for t in Q.targets():
        parts = urlsplit(t.url)
        z = t.local_zip
        if parts.fragment or z is None or not z.exists():
            continue
        meta = Q.project_json(z) or {}
        sessions = meta.get("sessions") or []
        if not sessions:
            continue
        ids = [s.get("id") for s in sessions]
        idx = ids.index(meta["activeSession"]) if meta.get("activeSession") in ids else 0
        q = parse_qs(parts.query)
        added = q.get("view", [""])[0].lower() != "notebook"
        q["view"] = ["notebook"]
        rel = t.url.split("?", 1)[0][len(Q.SITE):] + "?" + urlencode(q, doseq=True, safe="/")
        out.append((t.key, rel, z.name, idx, sessions[idx], added))
    return out


TARGETS = _targets()


def _check_spec(zname, idx, session):
    """`(roles of the code cells, required cell indices, needs_store, time_end)`."""
    roles = [r for r, _ in Q.compose(session)]
    tag = Q.backend(session)
    local = tag in Q.BROWSER_LOCAL
    cut = len(roles) if local else roles.index("settings") + 1
    required = list(range(cut))
    solves = tag is not None and local
    return roles, required, solves, Q.time_end_spec(zname, idx, session) if solves else None


# --------------------------------------------------------------------------- GUI under test

class _Site(http.server.SimpleHTTPRequestHandler):
    """The site root served under /Zoomy/, as GitHub Pages serves it."""
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm",
                      ".json": "application/json"}

    def translate_path(self, path):
        path = path.split("?", 1)[0].split("#", 1)[0]
        if path.startswith("/Zoomy/"):
            path = path[len("/Zoomy"):]
        elif path == "/Zoomy":
            path = "/"
        return super().translate_path(path)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="session")
def gui_base(tmp_path_factory):
    """The site root URL (no trailing slash)."""
    if os.environ.get("QR_GUI_BASE"):
        yield os.environ["QR_GUI_BASE"].rstrip("/")
        return
    site = os.environ.get("QR_GUI_SITE")
    if not site:
        site = tmp_path_factory.mktemp("site")
        build = Q.ROOT / "docs" / "scripts" / "build_gui_site.sh"
        env = dict(os.environ, WHEEL_PYTHON=os.environ.get("WHEEL_PYTHON", sys.executable))
        p = subprocess.run(["bash", str(build), str(site)], env=env, capture_output=True, text=True)
        if p.returncode:
            _skip_or_fail(f"could not build the GUI ({build.name}): {p.stderr[-2000:]}")
    if not (Path(site) / "gui" / "index.html").exists():
        _skip_or_fail(f"no GUI build in {site} (docs/scripts/build_gui_site.sh SITE)")
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(_Site, directory=str(site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/Zoomy"
    server.shutdown()


@pytest.fixture(scope="session")
def browser():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        _skip_or_fail("playwright missing: pip install playwright; playwright install chromium")
        return
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


# --------------------------------------------------------------------------- the notebook

#: per code cell: its status icon (check / error / none), its output text
_CELL_STATE = """sel => Array.from(document.querySelectorAll(sel)).map(li => {
  const icon = li.querySelector('.notebook-cell-status-left .codicon');
  const out = li.querySelector('.zoomy-cell-output');
  const cls = icon ? icon.className : '';
  return {status: /codicon-check/.test(cls) ? 'ok' : /codicon-error/.test(cls) ? 'error'
                  : /spin|sync/.test(cls) ? 'running' : /clock/.test(cls) ? 'queued' : 'none',
          out: out ? out.innerText : ''};
})"""


def _cells(page):
    return page.evaluate(_CELL_STATE, CELLS)


def _toolbar(page, label):
    page.locator(".theia-notebook-main-toolbar-item", has_text=label).first.click()


def _open(page, url, n_cells):
    """Load the QR URL; wait for the notebook with the session's `n_cells` code cells."""
    page.goto(url)
    t0 = time.monotonic()
    while time.monotonic() - t0 < OPEN_TIMEOUT:
        if page.locator(CELLS).count() >= n_cells:
            return
        time.sleep(1)
    raise AssertionError(f"the notebook did not open with {n_cells} code cells in {OPEN_TIMEOUT:.0f} s "
                         f"(found {page.locator(CELLS).count()})")


def _append_check_cell(page, code, n_cells):
    """A code cell after the last one, holding `code` (one line: typed as is)."""
    page.locator(CELLS).last.click()
    _toolbar(page, "Code")
    page.wait_for_function(f"document.querySelectorAll({CELLS!r}).length > {n_cells}", timeout=30000)
    cell = page.locator(CELLS).last
    cell.locator(".monaco-editor").first.click()
    page.keyboard.insert_text(code)
    page.keyboard.press("Escape")


def _run_all(page, n_cells):
    """Clear the outputs, Run All, wait for the final cell's verdict; `(cells, verdict)`."""
    _toolbar(page, "Clear All Outputs")
    _toolbar(page, "Run All")
    t0 = time.monotonic()
    while time.monotonic() - t0 < RUN_TIMEOUT:
        cells = _cells(page)
        m = re.search(r"QR_CHECK (\{.*\})", cells[-1]["out"]) if len(cells) > n_cells else None
        if m:
            return cells, json.loads(m.group(1))
        if cells and cells[-1]["status"] == "error":
            return cells, {"ok": False, "problems": ["the check cell failed: " + cells[-1]["out"][-800:]]}
        time.sleep(2)
    return _cells(page), {"ok": False, "problems": [f"no verdict after {RUN_TIMEOUT:.0f} s"]}


def _judge(cells, verdict, roles, required):
    """Problems of one Run All: the verdict's and the required cells' DOM status."""
    problems = list(verdict.get("problems") or [])
    for i in required:
        c = cells[i] if i < len(cells) else {"status": "missing", "out": ""}
        if c["status"] != "ok":
            tail = [ln for ln in c["out"].strip().splitlines() if ln.strip()][-1:] or [""]
            problems.append(f"{roles[i]} cell (#{i + 1}) {c['status']}: {tail[0][:300]}")
    return problems


@pytest.mark.qr_browser
@pytest.mark.parametrize("key,rel,zname,idx,session,added", TARGETS,
                         ids=[t[0] + ("+view" if t[5] else "") for t in TARGETS])
def test_qr_notebook_runs_twice(key, rel, zname, idx, session, added, gui_base, browser, tmp_path):
    roles, required, solves, time_end = _check_spec(zname, idx, session)
    url = gui_base + "/" + rel
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000})
    page = ctx.new_page()
    console = []
    page.on("console", lambda m: console.append(f"[{m.type}] {m.text}"))
    report = {"key": key, "url": url, "session": f"{zname}#{idx}", "runs": []}
    try:
        _open(page, url, len(roles))
        n = page.locator(CELLS).count()
        assert n == len(roles), f"the notebook has {n} code cells, the session composes {roles}"
        _append_check_cell(page, R.notebook_check_code(n, required, solves, time_end), n)
        for run in (1, 2):
            t0 = time.monotonic()
            cells, verdict = _run_all(page, n)
            report["runs"].append({"run": run, "seconds": round(time.monotonic() - t0, 1),
                                   "problems": _judge(cells, verdict, roles, required),
                                   "verdict": verdict,
                                   "cells": [f"{r}={c['status']}" for r, c in zip(roles + ["check"], cells)]})
    finally:
        page.screenshot(path=str(tmp_path / "notebook.png"), full_page=True)
        (tmp_path / "console.log").write_text("\n".join(console))
        (tmp_path / "report.json").write_text(json.dumps(report, indent=1))
        ctx.close()
    lines = [f"{key} ({report['session']}{', view=notebook added' if added else ''}): {url}"]
    for r in report["runs"]:
        lines.append(f"  run {r['run']} ({r['seconds']} s): "
                     + ("PASS" if not r["problems"] else "FAIL: " + "; ".join(r["problems"])))
        lines.append("    " + ", ".join(r["cells"]))
    print("\n".join(lines))
    failed = [r for r in report["runs"] if r["problems"]]
    if failed:
        pytest.fail("\n".join(lines) + f"\n  screenshot / console: {tmp_path}", pytrace=False)
