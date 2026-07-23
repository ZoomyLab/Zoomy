#!/usr/bin/env python3
"""Prepare docs build: JUnit summary, per-stack pytest-html embeds, tutorial notebook mirror."""

from __future__ import annotations

import datetime as dt
import html
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "_generated"
OUT_FILE = OUT_DIR / "test_report_summary.md"
BOOK_STATIC = ROOT / "docs" / "book" / "_static"
SMALL_REPORTS = ROOT / "artifacts" / "test-reports" / "small"
LARGE_REPORTS = ROOT / "artifacts" / "test-reports" / "large"
LEGACY_ROOT = ROOT / "artifacts" / "test-reports"
TUTORIALS_SRC = ROOT / "tutorials"
TUTORIALS_MIRROR = ROOT / "docs" / "book" / "tutorials" / "ipynb"

# slug -> book heading.
# MUST match the stack matrix in .github/workflows/tests-report.yml. CI tests
# four stacks (user ruling, 2026-07-22); dmplex / fenicsx / firedrake were
# dropped from CI scope, so listing them here only produced empty tables.
BACKENDS: tuple[tuple[str, str], ...] = (
    ("core", "Zoomy Core"),
    ("jax", "Zoomy JAX"),
    ("amrex", "AMReX"),
    ("foam", "OpenFOAM"),
)


def _junit_under(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(root.glob("**/junit.xml"))


def _parse_junit(path: Path) -> tuple[int, int, int, float, str | None]:
    root = ET.parse(path).getroot()
    if root.tag == "testsuites":
        suites = list(root.findall("testsuite"))
    elif root.tag == "testsuite":
        suites = [root]
    else:
        suites = []

    tests = failures = skipped = 0
    duration = 0.0
    timestamps: list[str] = []
    for s in suites:
        tests += int(float(s.attrib.get("tests", "0")))
        failures += int(float(s.attrib.get("failures", "0")))
        skipped += int(float(s.attrib.get("skipped", "0")))
        duration += float(s.attrib.get("time", "0"))
        ts = s.attrib.get("timestamp")
        if ts:
            timestamps.append(ts)
    ts_out = max(timestamps) if timestamps else None
    return tests, failures, skipped, duration, ts_out


def _newest_junit_mtime(files: list[Path]) -> str | None:
    if not files:
        return None
    try:
        m = max(p.stat().st_mtime for p in files)
        return dt.datetime.fromtimestamp(m, tz=dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    except OSError:
        return None


def _write_junit_table(lines: list[str], title: str, files: list[Path]) -> None:
    if title:
        lines.append(f"#### {title}")
        lines.append("")
    if not files:
        lines.append("_No JUnit files in this group._")
        lines.append("")
        return
    lines.append("| Report | Tests | Failures | Skipped | Duration (s) |")
    lines.append("|---|---:|---:|---:|---:|")
    for p in files:
        tests, failures, skipped, duration, _ts = _parse_junit(p)
        rel = p.relative_to(ROOT).as_posix()
        lines.append(
            f"| `{rel}` | {tests} | {failures} | {skipped} | {duration:.2f} |"
        )
    lines.append("")


def _latest_junit_xml_under(root: Path) -> Path | None:
    files = _junit_under(root)
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _write_summary_tables_by_tier(lines: list[str], tier_dir: Path, tier_title: str) -> None:
    """One subsection per tier: Generated (from newest junit.xml mtime on disk) + table."""
    lines.append(f"### {tier_title}")
    lines.append("")
    all_junits: list[Path] = []
    for slug, label in BACKENDS:
        all_junits.extend(_junit_under(tier_dir / slug))
    wall = _newest_junit_mtime(all_junits)
    if not wall:
        wall = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append(f"_Generated: {wall}_")
    lines.append("")
    lines.append("| Stack | Tests | Failures | Skipped | Duration (s) |")
    lines.append("|---|--:|--:|--:|--:|")
    for slug, label in BACKENDS:
        latest = _latest_junit_xml_under(tier_dir / slug)
        if latest is None:
            lines.append(f"| {label} | — | — | — | — |")
            continue
        tests, failures, skipped, duration, _ts = _parse_junit(latest)
        lines.append(f"| {label} | {tests} | {failures} | {skipped} | {duration:.2f} |")
    lines.append("")


def _legacy_junit_files() -> list[Path]:
    legacy: list[Path] = []
    if not LEGACY_ROOT.is_dir():
        return legacy
    for p in LEGACY_ROOT.glob("**/junit.xml"):
        for sub in (SMALL_REPORTS, LARGE_REPORTS):
            try:
                p.relative_to(sub)
            except ValueError:
                pass
            else:
                break
        else:
            legacy.append(p)
    return sorted(legacy)


def _write_junit_summary() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    has_modern = SMALL_REPORTS.is_dir() or LARGE_REPORTS.is_dir()
    legacy = _legacy_junit_files()

    if has_modern:
        _write_summary_tables_by_tier(
            lines, SMALL_REPORTS, "Small suite"
        )
        _write_summary_tables_by_tier(
            lines, LARGE_REPORTS, "Large / benchmark suite"
        )
    elif legacy:
        gen = _newest_junit_mtime(legacy)
        if not gen:
            gen = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines.append(f"_Generated: {gen}_")
        lines.append("")
        lines.append(
            "### Local / legacy layout (`artifacts/test-reports/` without `small/` or `large/`)"
        )
        lines.append("")
        _write_junit_table(lines, "", legacy)
    else:
        lines.append(
            f"_Generated: {dt.datetime.now(dt.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}_"
        )
        lines.append("")
        lines.append(
            "No JUnit report files found under `artifacts/test-reports/small/`, "
            "`artifacts/test-reports/large/`, or legacy `artifacts/test-reports/**/`."
        )

    OUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


# EXPLICIT allow-list of published notebooks — not a directory glob.
# The book sets only_build_toc_files: false, so ANYTHING mirrored here is built
# and shipped. A blanket rglob previously dragged in tutorials/legacy/ and every
# stale scratch notebook; one of them (swe/simple_numpy, old version) published a
# frozen `RuntimeError: SWASHES executable not found` traceback to the live site.
# Add a notebook here only once it executes clean — see tests/notebooks/
# smoke_notebooks.txt, which runs exactly this set.
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


def _stub(*, backend: str, tier: str) -> str:
    be = html.escape(backend)
    tier_esc = html.escape(tier)
    small_name = html.escape("test-reports-small-bundle")
    large_name = html.escape("test-reports-large-bundle")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>{tier_esc} — {be}</title></head>
<body style="font-family:system-ui,sans-serif;padding:1.5rem;max-width:48rem;">
<p>No <code>report.html</code> found for <strong>{tier_esc}</strong> / <strong>{be}</strong>.</p>
<p>CI stores reports under <code>artifacts/test-reports/{tier_esc}/{backend}/&lt;timestamp&gt;/</code>.
Smart Tests uploads <code>{small_name}</code> (small) and <code>{large_name}</code> (large); Render Webpage
downloads them before the book build.</p>
<p>Large jobs run on the weekly schedule or a manual Smart Tests run with large tests enabled.</p>
</body></html>
"""


def _latest_report_html(search_roots: list[Path]) -> Path | None:
    reports: list[Path] = []
    for root in search_roots:
        if root.is_dir():
            reports.extend(root.glob("**/report.html"))
    if not reports:
        return None
    return max(reports, key=lambda p: p.stat().st_mtime)


def _copy_pytest_html(dest_name: str, search_roots: list[Path], *, backend: str, tier: str) -> None:
    BOOK_STATIC.mkdir(parents=True, exist_ok=True)
    dest = BOOK_STATIC / dest_name
    latest = _latest_report_html(search_roots)
    if latest is None:
        dest.write_text(_stub(backend=backend, tier=tier), encoding="utf-8")
        return
    shutil.copy2(latest, dest)


def _copy_per_backend_reports() -> None:
    for slug, _label in BACKENDS:
        _copy_pytest_html(
            f"pytest-report-small-{slug}.html",
            [SMALL_REPORTS / slug],
            backend=slug,
            tier="small",
        )
        _copy_pytest_html(
            f"pytest-report-large-{slug}.html",
            [LARGE_REPORTS / slug],
            backend=slug,
            tier="large",
        )


def _legacy_single_report_fallback() -> None:
    """If only timestamped local dirs exist (no small/large split), surface one report under Core small."""
    latest = _latest_report_html([LEGACY_ROOT])
    if latest is None:
        return
    BOOK_STATIC.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, BOOK_STATIC / "pytest-report-small-core.html")
    for slug, _ in BACKENDS:
        if slug == "core":
            continue
        _copy_pytest_html(
            f"pytest-report-small-{slug}.html",
            [],
            backend=slug,
            tier="small",
        )
    for slug, _ in BACKENDS:
        _copy_pytest_html(
            f"pytest-report-large-{slug}.html",
            [],
            backend=slug,
            tier="large",
        )


def _copy_all_pytest_html() -> None:
    BOOK_STATIC.mkdir(parents=True, exist_ok=True)
    modern = SMALL_REPORTS.is_dir() or LARGE_REPORTS.is_dir()
    if modern:
        _copy_per_backend_reports()
        return
    if LEGACY_ROOT.is_dir() and _latest_report_html([LEGACY_ROOT]) is not None:
        _legacy_single_report_fallback()
        return
    _copy_per_backend_reports()


def main() -> int:
    _sync_tutorial_notebooks()
    _copy_all_pytest_html()
    return _write_junit_summary()


if __name__ == "__main__":
    raise SystemExit(main())
