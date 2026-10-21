#!/usr/bin/env python
"""Fast-verify gate for ``zoomy_core`` — "is this change ok?".

ONE entry point (approved test-refactor spec 2026-07-19, final v3).

DEFAULT run = the T1 gate: ``pytest -m gate``, SCOPED to the area(s) inferred
from the files ``git diff`` reports as touched (model / systemmodel / nsm /
printer / solver).  If nothing maps (or ``--all-areas``), it falls back to the
FULL gate — never a silent skip.  Target: one area ~1-2 min, all areas < 5 min.

    micromamba run -n zoomy python scripts/verify.py                # scoped gate
    micromamba run -n zoomy python scripts/verify.py --all-areas    # full gate
    micromamba run -n zoomy python scripts/verify.py --tier feature # T2: all
                                                                    # small+large
                                                                    # +rederive
    micromamba run -n zoomy python scripts/verify.py -k sme -x      # extra args
                                                                    # pass through

Tiers:
  * ``gate``    (default)  — ``-m gate`` scoped by area: the 6 T1 model
    goldens + 7 structure/emit/solver goldens + 24 runtime smalls.
  * ``feature`` (``--tier feature``) — ALL small + large + rederive tests
    (the re-bless validator set; required green before a multi-family golden
    regen).  ``study``-parked tests stay excluded in every tier.

The gate assumes a WARM derivation cache for the non-golden tests (the shipped
``_prebuilt`` set + the REQ-163 ``sm_cache``); model GOLDENS always derive
no-cache by spec.  After a ``CACHE_VERSION`` bump regenerate first:

    python -m zoomy_core.systemmodel.build_prebuilt_cache
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent          # zoomy_core repo root
BUDGET_S = 120.0                                                # the 1-2 min acceptance

AREAS = ("model", "systemmodel", "nsm", "printer", "solver")

# source-package -> area(s); anything unmapped -> ALL areas (fallback).
_SRC_AREA = {
    "model": {"model"},
    "systemmodel": {"systemmodel"},
    "numerics": {"nsm"},
    "transformation": {"printer"},
    "fvm": {"solver"},
    "mesh": {"solver"},
    # the solver Procedure/Statement IR: consumed by BOTH walker families
    # (C-family printer -> 'printer', python ProcedureBuilder -> 'solver').
    "solver": {"printer", "solver"},
}
# tests/<subdir> -> area(s).  tests/transformation also hosts the runtime
# kernel contract (area 'solver'), so it maps to both.
_TEST_AREA = {
    "model": {"model"},
    "systemmodel": {"systemmodel"},
    "numerics": {"nsm"},
    "transformation": {"printer", "solver"},
    "fvm": {"solver"},
    "analysis": {"model"},
    "solver_ir": {"printer", "solver"},
}


def changed_areas() -> set[str] | None:
    """Map ``git diff`` (unstaged+staged+untracked) to gate areas.

    Returns None when the change set demands the FULL gate: nothing mapped,
    git unavailable, or a file outside the per-area map was touched
    (goldens, conftest, scripts, pyproject, misc source packages)."""
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=REPO, capture_output=True, text=True, check=True).stdout
        out += subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=REPO, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    areas: set[str] = set()
    for line in out.splitlines():
        f = line.strip()
        if not f:
            continue
        parts = pathlib.Path(f).parts
        if parts[0] == "zoomy_core" and len(parts) >= 2:
            mapped = _SRC_AREA.get(parts[1])
        elif parts[0] == "tests" and len(parts) >= 2:
            if parts[1] == "goldens":
                return None                       # infra touched -> full gate
            mapped = _TEST_AREA.get(parts[1])
            if mapped is None and parts[1].endswith(".py"):
                return None                       # tests/conftest.py etc.
        else:
            return None                           # scripts/, pyproject, docs …
        if mapped is None:
            return None                           # unmapped package -> full
        areas.update(mapped)
    return areas or None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Fast-verify gate: '-m gate' scoped to the touched areas "
                    "(fallback: all gate); --tier feature = all small+large"
                    "+rederive.  Prints wall-clock; non-zero exit on failure.")
    ap.add_argument("--tier", choices=["gate", "feature"], default="gate",
                    help="'gate' (default): the scoped T1 gate.  'feature': "
                         "the full T2 set — all small + large + rederive.")
    ap.add_argument("--all-areas", action="store_true",
                    help="gate tier only: skip the changed-file inference and "
                         "run the FULL gate.")
    ap.add_argument("--durations", type=int, default=10, metavar="N",
                    help="show the N slowest tests (0 = off; default 10).")
    ap.add_argument("pytest_args", nargs="*",
                    help="extra args forwarded verbatim to pytest.")
    args = ap.parse_args(argv)

    cmd = [sys.executable, "-m", "pytest", "tests/", "-q",
           "-p", "no:cacheprovider"]
    if args.tier == "feature":
        # T2: everything small + large + rederive (study stays parked).
        cmd += ["--run-large", "--run-rederive"]
    else:
        areas = None if args.all_areas else changed_areas()
        if areas is not None and set(areas) != set(AREAS):
            expr = "gate and (" + " or ".join(sorted(areas)) + ")"
            print(f"verify: scoped gate → areas {sorted(areas)}", flush=True)
        else:
            expr = "gate"
            print("verify: full gate (no narrower area mapping)", flush=True)
        cmd += ["-m", expr]
    if args.durations:
        cmd += [f"--durations={args.durations}"]
    cmd += args.pytest_args

    print(f"verify: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=REPO).returncode
    wall = time.time() - t0

    status = "PASS" if rc == 0 else "FAIL"
    budget = "within" if wall <= BUDGET_S else "OVER"
    print(f"\nverify: {status} in {wall:.1f}s wall-clock "
          f"({budget} the {BUDGET_S:.0f}s budget)", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
