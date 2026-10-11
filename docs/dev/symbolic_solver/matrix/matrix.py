"""Cross-backend acceptance matrix of the symbolic solvers (README.md).

    python matrix.py run BACKEND CASE [CONFIG] [--force]   # runs -> store/
    python matrix.py table                                 # store -> table.md

BACKEND: jax foam amrex.  CASE: C1 C2 C3 C4 C5 C8 (C6 conservation, C7
parallel = serial and C9 GPU = CPU are evaluations of these runs).
CONFIG: serial (default) mpi2 mpi4 gpu1.  A run is skipped if its result is
in the store (--force reruns).
"""
import json
import os
import subprocess
import sys
import time as clock
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gates"))
import common as C  # noqa: E402  (x64)
from cases import CASES, prepare  # noqa: E402
from backends import BACKENDS  # noqa: E402

STORE = Path(os.environ.get("MATRIX_STORE", HERE / "store"))
WORK = Path(os.environ.get("MATRIX_WORK", HERE / "work"))


def path(backend, config, case, key):
    return STORE / backend / config / case / f"{key}.npz"


def load(backend, config, case, key):
    p = path(backend, config, case, key)
    return dict(np.load(p, allow_pickle=True)) if p.exists() else None


def provenance():
    """The library commits a run used (PYTHONPATH entries)."""
    out = {}
    for d in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if d:
            r = subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True)
            out[Path(d).name] = r.stdout.strip()
    return json.dumps(out)


def run(backend, case, config="serial", force=False):
    b = BACKENDS[backend]()
    if config not in b.configs:
        raise SystemExit(f"{backend}: no config {config} (has {b.configs})")
    cmd = None if os.environ.get("MATRIX_CHILD") else b.launch(config)
    if cmd:                                   # re-run under mpiexec / another env
        env = dict(os.environ, MATRIX_CHILD="1")
        argv = cmd + [__file__, "run", backend, case, config] + (["--force"] if force else [])
        raise SystemExit(subprocess.run(argv, env=env).returncode)
    for r in CASES[case][1]:
        if backend not in r.backends or config not in r.configs:
            continue
        p = path(backend, config, case, r.key)
        if p.exists() and not force:
            continue
        mesh, systems, Q0, names = prepare(r)
        t0 = clock.perf_counter()
        err = p.with_suffix(".err")
        try:
            out = b.run(r, mesh, systems, Q0, config, WORK / backend / config / case / r.key)
        except Exception as e:                # a backend failure is a result
            err.parent.mkdir(parents=True, exist_ok=True)
            msg = f"{type(e).__name__}: {e}"
            first = [ln for ln in msg.splitlines() if "error:" in ln][:3]
            err.write_text("\n".join(first + ["..."]) + "\n" + msg[-3000:])
            print(f"{backend} {config} {case} {r.key}: FAILED ({err})", flush=True)
            continue
        err.unlink(missing_ok=True)
        wall = clock.perf_counter() - t0
        if out is None:                       # non-root MPI rank
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        nc = Q0.shape[1]
        np.savez(p, t=out[0], Q=out[1], Q0=Q0, names=np.array(names), wall=wall, lib=provenance(),
                 xc=np.asarray(mesh.cell_centers)[:2, :nc],
                 vol=np.asarray(mesh.cell_volumes)[:nc])
        print(f"{backend} {config} {case} {r.key}: t={out[0]:.6g} [{wall:.1f}s]", flush=True)


# ── evaluation ──────────────────────────────────────────────────────────────
CONFIGS = ("serial", "mpi2", "mpi4", "gpu1")
CLOSED = ("C1", "C2", "C3", "C5")            # no boundary flux: mass exact
# check -> (op, threshold, source).  observed = ~2x the worst observed value.
from zoomy_core.solver.constants import ELLIPTIC_RTOL  # noqa: E402
TH = {
    "C8 parallel vs serial (max rel)": ("<=", 10 * ELLIPTIC_RTOL, "tolerance (10 x pressure rtol, over the steps)"),
    "C8 GPU vs CPU (max rel)": ("<=", 10 * ELLIPTIC_RTOL, "tolerance (10 x pressure rtol, over the steps)"),
    "C8 L1 vs JAX (rel)": ("<=", 3e-2, "observed (foam 1.6e-2)"),
    "C5 L1 vs JAX (rel)": ("<=", 3.5e-2, "observed (foam 1.7e-2)"),
    "sub O1 order 100->200": (">=", 0.75, "observed 0.86 (theory 1)"),
    "sub O2 order 100->200": (">=", 1.8, "theory 2 (observed 2.8)"),
    "trans O1 order 100->200": (">=", 0.75, "observed 0.86 (theory 1)"),
    "trans O2 order 100->200": (">=", 0.9, "observed 1.12 (sonic point)"),
    "sub O1 L1(h) n200": ("<=", 7.5e-3, "observed 3.7e-3"),
    "sub O2 L1(h) n200": ("<=", 1.1e-4, "observed 5.4e-5"),
    "trans O1 L1(h) n200": ("<=", 6e-2, "observed 3.1e-2"),
    "trans O2 L1(h) n200": ("<=", 3.1e-4, "observed 1.6e-4"),
    "sub O1 max|q/q_in - 1| n200": ("<=", 1.9e-2, "observed 9.5e-3"),
    "sub O2 max|q/q_in - 1| n200": ("<=", 1.7e-3, "observed 8.3e-4"),
    "trans O1 max|q/q_in - 1| n200": ("<=", 7e-2, "observed 3.5e-2"),
    "trans O2 max|q/q_in - 1| n200": ("<=", 9.5e-4, "observed 4.7e-4"),
    "O1 order 80->160": (">=", 0.75, "observed: JAX 0.82, pre-asymptotic (spec 0.9)"),
    "O2 order 80->160": (">=", 1.8, "spec (theory 2)"),
    "L1 order 40->80": (">=", 0.6, "spec"),
    "L1 vs JAX n80 (rel)": ("<=", 1.5e-3, "observed (foam 7.1e-4)"),
    "max|u| after 100 steps": ("<=", 1e-12, "spec (well-balanced: round-off)"),
    "h_min": (">=", 0.0, "theory (positivity)"),
    "mass change (rel)": ("<=", 1e-12, "tolerance (round-off)"),
    "parallel vs serial (max rel)": ("<=", 1e-12, "tolerance (round-off)"),
    "GPU vs CPU (max rel)": ("<=", 1e-12, "tolerance (round-off)"),
}


def _rows(d, skip_bed=True):
    names = [str(n) for n in d["names"]]
    return [i for i, n in enumerate(names) if not (skip_bed and n == "b")]


def relmax(a, b):
    return float(np.abs(a["Q"] - b["Q"]).max() / max(np.abs(b["Q"]).max(), 1e-300))


def l1rel(a, b):
    i = _rows(b)
    return float(np.abs(a["Q"][i] - b["Q"][i]).sum() / np.abs(b["Q"][i]).sum())


def h(d):
    return d["Q"][[str(n) for n in d["names"]].index("h")]


def mass(d):
    i = [str(n) for n in d["names"]].index("h")
    m0 = float((d["Q0"][i] * d["vol"]).sum())
    return abs(float((d["Q"][i] * d["vol"]).sum()) - m0) / abs(m0)


def restrict(fine, coarse):
    """The fine grid's h averaged onto the coarse grid's cells."""
    idx = np.zeros(fine["xc"].shape[1], dtype=int)
    cidx = np.zeros(coarse["xc"].shape[1], dtype=int)
    stride = 1
    for ax in range(2):
        u = np.unique(np.round(coarse["xc"][ax], 9))
        d = u[1] - u[0] if len(u) > 1 else 1.0
        f = lambda x: np.clip(np.floor((x - u[0] + d / 2) / d), 0, len(u) - 1).astype(int)  # noqa: E731
        idx += stride * f(fine["xc"][ax])
        cidx += stride * f(coarse["xc"][ax])
        stride *= len(u)
    avg = np.bincount(idx, h(fine), stride) / np.bincount(idx, None, stride)
    return avg[cidx]


def err_ref(d, ref):
    return float(np.abs(h(d) - restrict(ref, d)).mean())


def err_exact(d, variant):
    from cases import bernoulli
    return float(np.abs(h(d) - bernoulli(variant, d["xc"][0])).mean())


def order(e_coarse, e_fine, n_fine=None, n_ref=None):
    """Observed order between two grids (refinement 2).  Against a fine
    reference (n_ref) the errors are e_N = C (N^-p - n_ref^-p); p solves
    e_coarse / e_fine = (2^p - r) / (1 - r), r = (n_fine / n_ref)^p."""
    if not e_fine > 0:
        return float("inf")
    q = e_coarse / e_fine
    if n_ref is None:
        return float(np.log2(q))
    lo, hi = 0.0, 8.0
    for _ in range(100):
        p = 0.5 * (lo + hi)
        r = (n_fine / n_ref) ** p
        lo, hi = (p, hi) if (2 ** p - r) / (1 - r) < q else (lo, p)
    return p


def measure():
    """[(case, check, backend, config, value, note)] from the store."""
    out, B = [], tuple(BACKENDS)
    for e in sorted(STORE.glob("*/*/*/*.err")):     # runs that raised
        b, c, case = e.parts[-4:-1]
        first = e.read_text().splitlines()[0]
        out.append((case, f"{e.stem} runs", b, c, float("nan"), f"{e}: {first[:200]}"))
    get = lambda b, c, case, k: load(b, c, case, k)  # noqa: E731
    ref1 = get("jax", "mpi8", "C1", "ref_n320")
    ref2 = get("jax", "mpi8", "C2", "ref_n320")
    for b in B:
        s = lambda case, k: get(b, "serial", case, k)  # noqa: E731
        for o in (1, 2):                      # C1
            e = [s("C1", f"o{o}_n{n}") for n in (40, 80, 160)]
            if ref1 is not None and all(x is not None for x in e):
                es = [err_ref(x, ref1) for x in e]
                out.append(("C1", f"O{o} order 80->160", b, "serial", order(es[1], es[2]),
                            "L1(h) " + " / ".join(f"{x:.2e}" for x in es)
                            + (f"; corrected for the reference's error {order(es[1], es[2], 160, 320):.2f}"
                               if o == 2 else "")))
            d, j = s("C1", f"o{o}_n80"), get("jax", "serial", "C1", f"o{o}_n80")
            if b != "jax" and d is not None and j is not None:
                out.append(("C1", f"O{o} n80 max rel vs JAX (info)", b, "serial", relmax(d, j), ""))
        e = [s("C2", f"n{n}") for n in (40, 80)]  # C2
        if ref2 is not None and all(x is not None for x in e):
            es = [err_ref(x, ref2) for x in e]
            out.append(("C2", "L1 order 40->80", b, "serial", order(*es),
                        "L1(h) " + " / ".join(f"{x:.2e}" for x in es)
                        + f"; corrected for the reference's error {order(*es, 80, 320):.2f}"))
        if b != "jax" and e[1] is not None and s("C2", "n80") is not None:
            j = get("jax", "serial", "C2", "n80")
            if j is not None:
                out.append(("C2", "L1 vs JAX n80 (rel)", b, "serial", l1rel(e[1], j), ""))
        for c in CONFIGS:                       # C3
            for o in (1, 2):
                d = get(b, c, "C3", f"o{o}")
                if d is not None:
                    names = [str(n) for n in d["names"]]
                    q = d["Q"][[i for i, n in enumerate(names) if n.startswith("q")]]
                    out.append(("C3", f"O{o} max|u| after 100 steps", b, c,
                                float(np.abs(q / h(d)).max()), ""))
        for v in ("sub", "trans"):              # C4
            for o in (1, 2):
                e = [s("C4", f"{v}_o{o}_n{n}") for n in (50, 100, 200)]
                if all(x is not None for x in e):
                    es = [err_exact(x, v) for x in e]
                    out.append(("C4", f"{v} O{o} order 100->200", b, "serial",
                                order(es[1], es[2]), "L1(h) " + " / ".join(f"{x:.2e}" for x in es)))
                    out.append(("C4", f"{v} O{o} L1(h) n200", b, "serial", es[2], ""))
                    from cases import CHANNEL
                    q = e[2]["Q"][[str(n) for n in e[2]["names"]].index("q_x_0")]
                    out.append(("C4", f"{v} O{o} max|q/q_in - 1| n200", b, "serial",
                                float(np.abs(q / CHANNEL[v]["q"] - 1).max()),
                                "steady state: discharge = inflow (the open-domain mass balance)"))
                    qy = e[2]["Q"][[str(n) for n in e[2]["names"]].index("q_y_0")]
                    out.append(("C4", f"{v} O{o} max|q_y| n200 (info)", b, "serial",
                                float(np.abs(qy).max()), ""))
        for c in CONFIGS:                       # C5
            d = get(b, c, "C5", "n48")
            if d is not None:
                out.append(("C5", "h_min", b, c, float(h(d).min()), ""))
        d, j = s("C5", "n48"), get("jax", "serial", "C5", "n48")
        if b != "jax" and d is not None and j is not None:
            out.append(("C5", "L1 vs JAX (rel)", b, "serial", l1rel(d, j), ""))
        d, j = s("C8", "vam"), get("jax", "serial", "C8", "vam")
        if b != "jax" and d is not None and j is not None:
            out.append(("C8", "L1 vs JAX (rel)", b, "serial", l1rel(d, j), ""))
        for case in CLOSED:                     # C6
            for c in CONFIGS:
                ds = [get(b, c, case, r.key) for r in CASES[case][1]]
                ds = [x for x in ds if x is not None]
                if ds:
                    out.append(("C6", f"{case} mass change (rel)", b, c, max(map(mass, ds)),
                                f"max over {len(ds)} runs"))
        for case, key in (("C1", "o2_n80"), ("C5", "n48"), ("C8", "vam")):  # C7, C9
            ser = s(case, key)
            for c in CONFIGS[1:]:
                d = get(b, c, case, key)
                if ser is not None and d is not None:
                    chk = "GPU vs CPU (max rel)" if c == "gpu1" else "parallel vs serial (max rel)"
                    out.append(("C9" if c == "gpu1" else ("C8" if case == "C8" else "C7"),
                                f"{case} {chk}", b, c, relmax(d, ser), ""))
    return out


def threshold(check):
    if check.endswith(" runs"):
        return ("<=", 0.0, "the run completes")
    for k, v in TH.items():
        if check.endswith(k):
            return v
    return None


def table():
    rows = measure()
    lines = ["| case | check | backend | " + " | ".join(CONFIGS) + " | threshold (source) |",
             "|---|---|---|" + "---|" * len(CONFIGS) + "---|"]
    fails, seen = [], {}
    for case, check, b, c, v, note in rows:
        seen.setdefault((case, check, b), {})[c] = (v, note)
    for (case, check, b), cells in seen.items():
        th = threshold(check)
        txt = []
        for c in CONFIGS:
            if c not in cells:
                txt.append("-")
                continue
            v, note = cells[c]
            if th is None or th[1] is None:
                txt.append(f"{v:.2e}")
                continue
            ok = bool(v >= th[1] if th[0] == ">=" else v <= th[1])
            txt.append(f"{v:.2e} {'PASS' if ok else '**FAIL**'}")
            if not ok:
                fails.append(f"- {case} {check}, {b} {c}: {v:.3e} (need {th[0]} {th[1]:g}) {note}")
        thr = "-" if th is None or th[1] is None else f"{th[0]} {th[1]:g} ({th[2]})"
        lines.append(f"| {case} | {check} | {b} | " + " | ".join(txt) + f" | {thr} |")
    notes = [f"- {case} {check}, {b} {c}: {note}" for case, check, b, c, v, note in rows if note]
    text = "\n".join(lines + ["", "Failures:"] + (fails or ["- none"]) + ["", "Notes:"] + notes)
    (HERE / "table.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--force"]
    if a[0] == "run":
        run(*a[1:4], force="--force" in sys.argv)
    else:
        table()
