#!/usr/bin/env python3
"""Prepare docs build: JUnit summary, latest pytest-html embed, tutorial notebook mirror."""

from __future__ import annotations

import datetime as dt
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "_generated"
OUT_FILE = OUT_DIR / "test_report_summary.md"
BOOK_STATIC = ROOT / "docs" / "book" / "_static"
PYTEST_REPORT_NAME = "pytest-report.html"
TUTORIALS_SRC = ROOT / "tutorials"
TUTORIALS_MIRROR = ROOT / "docs" / "book" / "tutorials" / "ipynb"


def _iter_junit_files() -> list[Path]:
    paths = []
    artifacts = ROOT / "artifacts" / "test-reports"
    if artifacts.exists():
        paths.extend(artifacts.glob("**/junit.xml"))
    return sorted(paths)


def _parse_junit(path: Path) -> tuple[int, int, int, float]:
    root = ET.parse(path).getroot()
    if root.tag == "testsuites":
        suites = list(root.findall("testsuite"))
    elif root.tag == "testsuite":
        suites = [root]
    else:
        suites = []

    tests = failures = skipped = 0
    duration = 0.0
    for s in suites:
        tests += int(float(s.attrib.get("tests", "0")))
        failures += int(float(s.attrib.get("failures", "0")))
        skipped += int(float(s.attrib.get("skipped", "0")))
        duration += float(s.attrib.get("time", "0"))
    return tests, failures, skipped, duration


def _write_junit_summary() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = _iter_junit_files()
    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = []
    lines.append(f"_Generated: {stamp}_")
    lines.append("")

    if not files:
        lines.append("No JUnit report files found under `artifacts/test-reports/**/junit.xml`.")
        OUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 0

    lines.append("| Report | Tests | Failures | Skipped | Duration (s) |")
    lines.append("|---|---:|---:|---:|---:|")
    for p in files:
        tests, failures, skipped, duration = _parse_junit(p)
        rel = p.relative_to(ROOT).as_posix()
        lines.append(
            f"| `{rel}` | {tests} | {failures} | {skipped} | {duration:.2f} |"
        )

    OUT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def _sync_tutorial_notebooks() -> None:
    if not TUTORIALS_SRC.is_dir():
        return
    if TUTORIALS_MIRROR.exists():
        shutil.rmtree(TUTORIALS_MIRROR)
    for ipynb in TUTORIALS_SRC.rglob("*.ipynb"):
        rel = ipynb.relative_to(TUTORIALS_SRC)
        dest = TUTORIALS_MIRROR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ipynb, dest)


def _copy_latest_pytest_html() -> None:
    BOOK_STATIC.mkdir(parents=True, exist_ok=True)
    dest = BOOK_STATIC / PYTEST_REPORT_NAME
    artifacts = ROOT / "artifacts" / "test-reports"
    reports = list(artifacts.glob("**/report.html")) if artifacts.is_dir() else []
    if not reports:
        stub = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Test report</title></head>
<body style="font-family:system-ui,sans-serif;padding:1.5rem;max-width:48rem;">
<p>No <code>report.html</code> found under <code>artifacts/test-reports/</code>.</p>
<p>Run the <em>tests-report</em> workflow (or <code>python tests/reporting/generate_test_report.py</code>)
and copy artifacts here, then re-run <code>python docs/scripts/generate_ci_test_report.py</code>.</p>
</body></html>
"""
        dest.write_text(stub, encoding="utf-8")
        return
    latest = max(reports, key=lambda p: p.stat().st_mtime)
    shutil.copy2(latest, dest)


def main() -> int:
    _sync_tutorial_notebooks()
    _copy_latest_pytest_html()
    return _write_junit_summary()


if __name__ == "__main__":
    raise SystemExit(main())
