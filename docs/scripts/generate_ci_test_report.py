#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "_generated"
OUT_FILE = OUT_DIR / "test_report_summary.md"


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


def main() -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
