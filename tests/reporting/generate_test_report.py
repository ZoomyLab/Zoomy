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

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-m",
        args.markers,
        "--junitxml",
        str(junit_path),
        "-q",
    ]
    if have_pytest_html:
        cmd.extend(["--html", str(html_path), "--self-contained-html"])
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
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())

