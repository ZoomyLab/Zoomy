#!/usr/bin/env python3
"""Prepare the CI test-report page for the docs build.

Reads the Smart Tests artifacts laid out as::

    artifacts/test-reports/<tier>/<group>/<unit>/<timestamp>/{junit.xml,report.html}

    tier  : small | large
    group : zoomy | library | notebooks
    unit  : a backend (core/jax/amrex/foam) or a submodule (prepost/server) or
            a notebook backend.

and produces three things consumed by ``docs/book/ci-reports.md``:

* ``docs/_generated/test_report_summary.md`` — grouped pass/fail summary tables.
* ``docs/_generated/test_report_embeds.md``  — the per-unit pytest-html iframes.
* ``docs/book/_static/pytest-report-<tier>-<group>-<unit>.html`` — each unit's
  self-contained pytest-html report (or a stub when a run produced none).

The report structure (which groups and units exist) is defined ONCE here, in
``GROUPS`` — the summary, the embeds and the ``_static`` copies all derive from
it, so the page never drifts from what CI actually runs.
"""

from __future__ import annotations

import datetime as dt
import html
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "_generated"
SUMMARY_FILE = OUT_DIR / "test_report_summary.md"
EMBEDS_FILE = OUT_DIR / "test_report_embeds.md"
BOOK_STATIC = ROOT / "docs" / "book" / "_static"
REPORTS_ROOT = ROOT / "artifacts" / "test-reports"
TUTORIALS_SRC = ROOT / "tutorials"
TUTORIALS_MIRROR = ROOT / "docs" / "book" / "tutorials" / "ipynb"

TIERS: tuple[tuple[str, str], ...] = (
    ("small", "Small (every push)"),
    ("large", "Large / regression (weekly)"),
)

# group slug -> (heading, [(unit slug, unit label), ...]).
# MUST match the report units produced by .github/workflows/tests-report.yml.
GROUPS: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    ("zoomy", "Zoomy — superproject `tests/`", (
        ("core", "Core / NumPy"),
        ("jax", "JAX"),
        ("amrex", "AMReX"),
        ("foam", "OpenFOAM"),
    )),
    ("library", "Library — each submodule's own `tests/`", (
        ("core", "zoomy_core"),
        ("jax", "zoomy_jax"),
        ("amrex", "zoomy_amrex"),
        ("foam", "zoomy_foam"),
        ("prepost", "zoomy_prepost"),
        ("server", "zoomy_server"),
    )),
    ("notebooks", "Notebooks — published tutorials, executed", (
        ("core", "Core notebooks"),
        ("amrex", "AMReX notebooks"),
    )),
)


# --------------------------------------------------------------------------- #
# JUnit reading
# --------------------------------------------------------------------------- #
def _unit_dir(tier: str, group: str, unit: str) -> Path:
    return REPORTS_ROOT / tier / group / unit


def _junit_under(root: Path) -> list[Path]:
    return sorted(root.glob("**/junit.xml")) if root.is_dir() else []


def _latest_junit(tier: str, group: str, unit: str) -> Path | None:
    files = _junit_under(_unit_dir(tier, group, unit))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _parse_junit(path: Path) -> tuple[int, int, int, int, float]:
    root = ET.parse(path).getroot()
    suites = (root.findall("testsuite") if root.tag == "testsuites"
              else [root] if root.tag == "testsuite" else [])
    tests = failures = errors = skipped = 0
    duration = 0.0
    for s in suites:
        tests += int(float(s.attrib.get("tests", "0")))
        failures += int(float(s.attrib.get("failures", "0")))
        errors += int(float(s.attrib.get("errors", "0")))
        skipped += int(float(s.attrib.get("skipped", "0")))
        duration += float(s.attrib.get("time", "0"))
    return tests, failures, errors, skipped, duration


def _newest_mtime(files: list[Path]) -> str | None:
    if not files:
        return None
    try:
        m = max(p.stat().st_mtime for p in files)
        return dt.datetime.fromtimestamp(m, tz=dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _write_summary() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    all_junits = [
        j for tier, _ in TIERS for group, _, units in GROUPS for unit, _ in units
        for j in _junit_under(_unit_dir(tier, group, unit))
    ]
    wall = _newest_mtime(all_junits) or dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines += [f"_Latest run: {wall}_", ""]

    for tier, tier_label in TIERS:
        lines += [f"### {tier_label}", ""]
        for group, group_label, units in GROUPS:
            lines += [f"**{group_label}**", "",
                      "| Suite | Tests | Failures | Errors | Skipped | Duration (s) |",
                      "|---|--:|--:|--:|--:|--:|"]
            for unit, unit_label in units:
                j = _latest_junit(tier, group, unit)
                if j is None:
                    lines.append(f"| {unit_label} | — | — | — | — | — |")
                    continue
                t, f, e, sk, d = _parse_junit(j)
                flag = " ⚠️" if (f or e) else ""
                lines.append(f"| {unit_label} | {t} | {f}{flag} | {e} | {sk} | {d:.1f} |")
            lines.append("")
    SUMMARY_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Per-unit HTML embeds
# --------------------------------------------------------------------------- #
def _stub(tier: str, group: str, unit: str) -> str:
    t, g, u = (html.escape(x) for x in (tier, group, unit))
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>{t} · {g} · {u}</title></head>
<body style="font-family:system-ui,sans-serif;padding:1.5rem;max-width:48rem;color:#444;">
<p>No <code>report.html</code> for <strong>{t}</strong> · <strong>{g}</strong> · <strong>{u}</strong> yet.</p>
<p>Smart Tests writes it under <code>artifacts/test-reports/{t}/{g}/{u}/&lt;timestamp&gt;/</code>;
the small tier runs on every push, the large/regression tier weekly (Sun 03:00 UTC) or via a
manual Smart Tests run with large enabled.</p>
</body></html>
"""


def _copy_report(tier: str, group: str, unit: str) -> None:
    BOOK_STATIC.mkdir(parents=True, exist_ok=True)
    dest = BOOK_STATIC / f"pytest-report-{tier}-{group}-{unit}.html"
    reports = sorted(_unit_dir(tier, group, unit).glob("**/report.html")) \
        if _unit_dir(tier, group, unit).is_dir() else []
    if reports:
        shutil.copy2(max(reports, key=lambda p: p.stat().st_mtime), dest)
    else:
        dest.write_text(_stub(tier, group, unit), encoding="utf-8")


def _write_embeds() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for group, group_label, units in GROUPS:
        lines += [f"### {group_label}", ""]
        for unit, unit_label in units:
            lines += [f"#### {unit_label}", ""]
            for tier, tier_label in TIERS:
                _copy_report(tier, group, unit)
                src = f"_static/pytest-report-{tier}-{group}-{unit}.html"
                lines += [
                    f"*{tier_label}*",
                    "",
                    "```{eval-rst}",
                    ".. raw:: html",
                    "",
                    '   <div class="pytest-report-scroll">',
                    f'   <iframe src="{src}" title="{html.escape(unit_label)} — {tier}"></iframe>',
                    "   </div>",
                    "```",
                    "",
                ]
    EMBEDS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Tutorial notebook mirror (published set → book tree)
# --------------------------------------------------------------------------- #
# Keep in sync with tests/notebooks/test_tutorials.py::PUBLISHED — the same set
# is executed there (Notebooks report section) and mirrored onto the site.
PUBLISHED_TUTORIALS: tuple[str, ...] = (
    "swe/simple_numpy.ipynb",
    "sme/moments_2d.ipynb",
    "amrex/minimal.ipynb",
)


def _sync_tutorial_notebooks() -> None:
    if not TUTORIALS_SRC.is_dir():
        return
    if TUTORIALS_MIRROR.exists():
        shutil.rmtree(TUTORIALS_MIRROR)
    for rel in PUBLISHED_TUTORIALS:
        src = TUTORIALS_SRC / rel
        if not src.is_file():
            print(f"  WARNING: published tutorial missing: {src}")
            continue
        dest = TUTORIALS_MIRROR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def main() -> int:
    _sync_tutorial_notebooks()
    _write_embeds()      # also copies/stubs every _static report
    _write_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
