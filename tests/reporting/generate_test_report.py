#!/usr/bin/env python3
"""
Generate HTML + JUnit test reports for CI or local use.

Examples:
  python tests/reporting/generate_test_report.py
  python tests/reporting/generate_test_report.py --markers "small or tutorial"
  python tests/reporting/generate_test_report.py --run-large --run-benchmark
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--markers", default="small or tutorial", help="Pytest marker expression.")
    p.add_argument("--output-dir", default="artifacts/test-reports", help="Output report directory.")
    p.add_argument("--run-large", action="store_true", help="Enable large tests.")
    p.add_argument("--run-benchmark", action="store_true", help="Enable benchmark tests.")
    p.add_argument(
        "--allow-no-tests",
        action="store_true",
        help="Treat pytest exit code 5 (no tests collected) as success.",
    )
    p.add_argument(
        "--ignore-pytest-exit-code",
        action="store_true",
        help="Exit 0 if junit.xml or report.html was written even when pytest failed (keeps CI artifacts).",
    )
    p.add_argument(
        "--pytest-paths",
        default="tests",
        help="Whitespace-separated roots for pytest (default: tests).",
    )
    p.add_argument("--extra-pytest-args", default="", help="Additional raw pytest args.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.output_dir) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("LOGURU_LEVEL", "WARNING")
    if args.run_large:
        env["ZOOMY_RUN_LARGE_TESTS"] = "1"
    if args.run_benchmark:
        env["ZOOMY_RUN_BENCHMARK_TESTS"] = "1"

    html_path = out_dir / "report.html"
    junit_path = out_dir / "junit.xml"
    have_pytest_html = True
    try:
        __import__("pytest_html")
    except Exception:
        have_pytest_html = False

    # Default: NO explicit paths — pytest.ini `testpaths` (tests + the
    # library/*/tests glob) governs collection, and the root conftest's
    # capability rule skips suites whose backend isn't installed here.
    path_args = [p for p in args.pytest_paths.split() if p.strip()]

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *path_args,
        "-m",
        args.markers,
        "--junitxml",
        str(junit_path),
        "-q",
        "--durations=25",
    ]
    if have_pytest_html:
        cmd.extend(["--html", str(html_path), "--self-contained-html"])
        css = Path(__file__).resolve().parent / "pytest_html_docs.css"
        if css.is_file():
            cmd.extend(["--css", str(css)])
    else:
        print(
            "pytest-html not installed; HTML report will be skipped. "
            "Install with: pip install pytest-html"
        )
    if args.extra_pytest_args.strip():
        cmd.extend(args.extra_pytest_args.split())

    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd, env=env)
    if rc == 5 and args.allow_no_tests:
        print("No tests collected; allowed by --allow-no-tests.")
        rc = 0
    if have_pytest_html:
        print(f"HTML report:  {html_path}")
    print(f"JUnit report: {junit_path}")
    # Size-rule audit (per user 2026-07-14): any test >5 min individually is
    # LARGE.  Flag small-suite violators so they get marked @pytest.mark.large.
    if junit_path.is_file() and not args.run_large:
        try:
            import xml.etree.ElementTree as _ET
            slow = [(tc.get("classname", ""), tc.get("name", ""), float(tc.get("time", 0) or 0))
                    for tc in _ET.parse(junit_path).getroot().iter("testcase")
                    if float(tc.get("time", 0) or 0) > 300.0]
            for cls, name, sec in slow:
                print(f"SIZE-RULE VIOLATION: {cls}::{name} took {sec:.0f}s (>300s) "
                      "in the small suite - mark it @pytest.mark.large.")
        except Exception as exc:
            print(f"size-rule audit skipped: {exc}")
    wrote_junit = junit_path.is_file()
    wrote_html = have_pytest_html and html_path.is_file()
    if args.ignore_pytest_exit_code and (wrote_junit or wrote_html):
        if rc != 0:
            print("Pytest failed but reports exist; exiting 0 (--ignore-pytest-exit-code).")
        return 0
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())

