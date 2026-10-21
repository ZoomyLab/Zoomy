#!/usr/bin/env python
"""Regenerate the checked-in golden snapshots (tests/goldens/*.txt).

ONE script regenerates all 25 goldens (model goldens derive NO-CACHE by
spec) plus the N02 time-canary baseline JSON; review = ``git diff``.

    micromamba run -n zoomy python scripts/regen_goldens.py            # all
    micromamba run -n zoomy python scripts/regen_goldens.py m03 n02    # subset
    micromamba run -n zoomy python scripts/regen_goldens.py --baseline # canary
                                                                       # baseline only

Re-bless protocol (approved spec §D): a regen touching MORE than one golden
family requires the rederive tier green first —

    python scripts/verify.py --tier feature      # includes -m rederive

— because a golden detects CHANGE, not WRONGNESS.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests" / "goldens"))

import goldenlib  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*",
                    help="golden names (prefix match, e.g. 'm03' or 'p'); "
                         "default: all")
    ap.add_argument("--baseline", action="store_true",
                    help="only (re)write the N02 time-canary baseline JSON")
    ap.add_argument("--list", action="store_true", help="list goldens and exit")
    args = ap.parse_args(argv)

    if args.list:
        for name, (_, family, tier) in goldenlib.GOLDENS.items():
            print(f"{name:28s} {family:12s} {tier}")
        return 0

    if args.baseline:
        data = goldenlib.write_time_baseline()
        print(f"baseline: {data}")
        return 0

    selected = list(goldenlib.GOLDENS)
    if args.names:
        selected = [n for n in selected
                    if any(n.startswith(p) for p in args.names)]
        if not selected:
            print(f"no golden matches {args.names}", file=sys.stderr)
            return 2
    families = set()
    # EVERY golden regenerates COLD, not just the model goldens.  A golden must
    # record the state of the SOURCE, never the state of whoever's build cache
    # happened to be warm when it was blessed.
    for name in selected:
        builder, family, _tier = goldenlib.GOLDENS[name]
        t0 = time.time()
        with goldenlib.no_cache():
            body = builder()
        path = goldenlib.write_golden(name, body)
        families.add(family)
        print(f"regen {name:28s} [{family}] {time.time() - t0:7.1f}s "
              f"-> {path.relative_to(REPO)}", flush=True)
    if selected == list(goldenlib.GOLDENS):
        data = goldenlib.write_time_baseline()
        print(f"baseline: {data}")
    if len(families) > 1:
        print("\nNOTE: this regen touched multiple golden families "
              f"({sorted(families)}) — the re-bless protocol requires the "
              "rederive tier green first: python scripts/verify.py --tier feature")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
