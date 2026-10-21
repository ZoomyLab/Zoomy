"""Tests and timings of the symbolic solvers -> two Markdown includes.

    python docs/scripts/generate_symsolver_results.py                 # render (docs build)
    MATRIX_STORE=path/to/store python docs/scripts/generate_symsolver_results.py --from-store

Inputs (``tests/results/symsolver/``):

* ``matrix_latest.json``: the rows of the cross-backend matrix
  (``docs/dev/symbolic_solver/matrix/matrix.py: measure()``), with the date and
  the library commits of the runs.  ``--from-store`` rewrites it from a matrix
  store; that needs the zoomy environment (the matrix imports the libraries).
* ``timings_latest.json``: the timing suite's latest record per case, backend
  and config (``tests/symsolver/timing_records.py``).

Outputs: ``docs/_generated/symsolver_matrix.md`` and
``docs/_generated/symsolver_timings.md``, included by
``docs/book/developer/verification.md`` and
``docs/book/developer/performance.md``.  The render step needs only the standard
library, so the docs build runs without a store and without the solvers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("ZOOMY_TIMING_RESULTS", ROOT / "tests" / "results" / "symsolver"))
OUT = ROOT / "docs" / "_generated"
MATRIX_DIR = ROOT / "docs" / "dev" / "symbolic_solver" / "matrix"
GITHUB = "https://github.com/ZoomyLab/Zoomy/blob/main"
README = f"{GITHUB}/docs/dev/symbolic_solver/matrix/README.md"
BACKENDS = ("jax", "foam", "amrex")
SCHEMA = 1


# ── store -> matrix_latest.json (zoomy environment) ─────────────────────────
def _verdict(value, th):
    if isinstance(value, str):
        return value                                   # "n/a", "unsupported"
    if th is None:
        return "info"
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "FAIL"
    return "PASS" if (value >= th[1] if th[0] == ">=" else value <= th[1]) else "FAIL"


def _git(path, *args):
    r = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def from_store():
    """Run ``matrix.measure()`` on ``MATRIX_STORE`` and write matrix_latest.json."""
    for p in (MATRIX_DIR, MATRIX_DIR.parent / "gates"):
        sys.path.insert(0, str(p))
    import numpy as np
    import matrix                                      # noqa: E402  (needs the libraries)
    from cases import CASES

    rows = []
    for case, check, b, c, v, th, note in matrix.measure():
        value = v if isinstance(v, str) else (None if v is None or math.isnan(v) else float(v))
        rows.append(dict(case=case, check=check, backend=b, config=c, value=value,
                         op=None if th is None else th[0],
                         threshold=None if th is None else float(th[1]),
                         source=None if th is None else th[2],
                         verdict=_verdict(v if isinstance(v, str) else float(v), th), note=note))
    libs, dates = defaultdict(set), []
    for f in matrix.STORE.glob("*/*/*/*.npz"):
        d = np.load(f, allow_pickle=True)
        if "lib" in d.files:
            for k, sha in json.loads(str(d["lib"])).items():
                libs[k].add(sha)
        dates.append(f.stat().st_mtime)
    run = (datetime.fromtimestamp(max(dates), timezone.utc).strftime("%Y-%m-%d")
           if dates else None)
    out = dict(schema=SCHEMA,
               generated=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               last_run=run, store=str(matrix.STORE),
               superrepo=_git(ROOT, "rev-parse", "--short", "HEAD"),
               libs={k: sorted(v) for k, v in sorted(libs.items())},
               cases={k: t for k, (t, _) in CASES.items()},
               configs=list(matrix.CONFIGS), rows=rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "matrix_latest.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


# ── matrix_latest.json -> symsolver_matrix.md ──────────────────────────────
def _case_key(case):
    return (case[0], int(case[1:]) if case[1:].isdigit() else 0)


def render_matrix():
    path = RESULTS / "matrix_latest.json"
    if not path.exists():
        return ("Not yet measured: no `tests/results/symsolver/matrix_latest.json`. "
                "Run the matrix and `generate_symsolver_results.py --from-store`.\n")
    data = json.loads(path.read_text())
    rows = data["rows"]
    count = defaultdict(lambda: defaultdict(int))     # (case, backend) -> verdict -> n
    for r in rows:
        count[(r["case"], r["backend"])][r["verdict"]] += 1
    total = defaultdict(int)
    for r in rows:
        total[r["verdict"]] += 1

    def cell(case, b):
        n = count.get((case, b))
        if not n:
            return "-"
        parts = []
        if n["PASS"]:
            parts.append(f"{n['PASS']} PASS")
        if n["FAIL"]:
            parts.append(f"**{n['FAIL']} FAIL**")
        if not parts and n["unsupported"]:
            parts.append("unsupported")
        if not parts and n["n/a"]:
            parts.append("n/a")
        if not parts and n["info"]:
            parts.append("info only")
        return ", ".join(parts)

    libs = ", ".join(f"{k} {'/'.join(v)}" for k, v in data.get("libs", {}).items())
    lines = [
        f"**{total['PASS']} PASS, {total['FAIL']} FAIL** "
        f"({total['info']} information rows, {total['n/a']} n/a, "
        f"{total['unsupported']} unsupported).  "
        f"Last run {data.get('last_run') or '?'}.  Library commits: {libs or '?'}.  "
        f"Rendered from `matrix_latest.json` of {data['generated'][:10]}.",
        "",
        "A cell counts the checks of a case on a backend over every config "
        "(serial, MPI, GPU).  Case setups and pass criteria: "
        f"[matrix README]({README}#cases-and-pass-criteria).  Every row: "
        f"[table.md]({GITHUB}/docs/dev/symbolic_solver/matrix/table.md).  `-`: no check "
        "of its own on that backend (JAX is the reference, or the runs are counted "
        "under C6, C7 and C9).",
        "",
        "| case | setup | " + " | ".join(BACKENDS) + " |",
        "|---|---|" + "---|" * len(BACKENDS),
    ]
    cases = data.get("cases", {})
    for case in sorted({r["case"] for r in rows} | set(cases), key=_case_key):
        title = cases.get(case, "").replace("|", "/")
        lines.append(f"| {case} | {title} | " + " | ".join(cell(case, b) for b in BACKENDS) + " |")
    fails = [r for r in rows if r["verdict"] == "FAIL"]
    lines += ["", "Failures:", ""]
    lines += ([f"- {r['case']} {r['check']}, {r['backend']} {r['config']}: {r['value']} "
               f"(need {r['op']} {r['threshold']:g}) {r['note']}" for r in fails] or ["- none"])
    return "\n".join(lines) + "\n"


# ── timings_latest.json -> symsolver_timings.md ────────────────────────────
def _timing_cases():
    """{key: title} of the registry (tests/symsolver/timing_cases.py), or {}."""
    sys.path.insert(0, str(ROOT / "tests" / "symsolver"))
    try:
        import timing_cases
        return {k: c.title for k, c in timing_cases.CASES.items()}
    except Exception:                                  # noqa: BLE001  (render without it)
        return {}


def render_timings():
    titles = _timing_cases()
    path = RESULTS / "timings_latest.json"
    recs = json.loads(path.read_text())["rows"] if path.exists() else []
    if not recs:
        lines = ["Not yet measured: no records in `tests/results/symsolver/timings_latest.json`.", ""]
        if titles:
            lines += ["| case | setup |", "|---|---|"]
            lines += [f"| `{k}` | {t} |" for k, t in titles.items()]
        return "\n".join(lines) + "\n"

    def f(x, fmt):
        return "-" if x is None else format(x, fmt)

    lines = ["Per case, backend and config: the march time per cell and step "
             "(µs, of the fastest of the repetitions), the min / median march "
             "time (s) of the repetitions, ms per step, and setup / compile "
             "time (s) of the first repetition.", "",
             "| case | backend | config | device | cells | steps | µs/cell/step "
             "| min / median (s) | ms/step | setup / compile (s) | status |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {k: i for i, k in enumerate(titles)}
    recs = sorted(recs, key=lambda r: (order.get(r["case"], 99), r["case"],
                                       BACKENDS.index(r["backend"]) if r["backend"] in BACKENDS else 9,
                                       r["config"]))
    for r in recs:
        dev = r.get("device") or ""
        if r.get("detail"):
            dev = f"{dev} {r['detail']}".strip()
        lines.append(
            f"| `{r['case']}` | {r['backend']} | {r['config']} | {dev or '-'} | "
            f"{r.get('cells', '-')} | {r.get('steps', '-')} | {f(r.get('us_cell_step'), '.3g')} | "
            f"{f(r.get('min_s'), '.3g')} / {f(r.get('median_s'), '.3g')} | "
            f"{f(r.get('ms_step'), '.3g')} | {f(r.get('setup_s'), '.3g')} / "
            f"{f(r.get('compile_s'), '.3g')} | {r.get('status', '?')} |")
    measured = {r["case"] for r in recs}
    missing = [k for k in titles if k not in measured]
    if missing:
        lines += ["", "Not yet measured: " + ", ".join(f"`{k}`" for k in missing) + "."]
    machines = {}
    for r in recs:
        m = (r.get("host"), r.get("cpu"), r.get("gpu"), r.get("cores_pinned"), r.get("cpus"),
             r.get("wave_speed"), json.dumps(r.get("libs"), sort_keys=True))
        machines.setdefault(m, []).append(r.get("date", "")[:10])
    lines += ["", "Machines (records grouped by host, hardware, pinning and library commits):", "",
              "| host | CPU | GPU | cores pinned (cpus) | wave speed | libraries | dates |",
              "|---|---|---|---|---|---|---|"]
    for (host, cpu, gpu, n, cpus, wave, libs), dates in machines.items():
        libs = ", ".join(f"{k} {v}" for k, v in json.loads(libs or "{}").items() if v)
        lines.append(f"| {host} | {cpu} | {gpu or '-'} | {n} ({cpus}) | {wave} | {libs} | "
                     f"{min(dates)} .. {max(dates)} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from-store", action="store_true",
                    help="rewrite matrix_latest.json from MATRIX_STORE first (zoomy env)")
    a = ap.parse_args()
    if a.from_store:
        from_store()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in (("symsolver_matrix.md", render_matrix()),
                       ("symsolver_timings.md", render_timings())):
        (OUT / name).write_text(text)
        print(f"wrote {OUT / name}")


if __name__ == "__main__":
    main()
