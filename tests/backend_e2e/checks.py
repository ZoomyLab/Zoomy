"""Checks on a result store: what the GUI reads (engine.open_hdf5 + the selected viewer), the
numbers (finite, snapshots up to time_end, mass), and a run of the same case on the host to
compare with."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from harness import ROOT

SESSION_RUNNER = ROOT / "tests" / "qr_cases" / "session_runner.py"
sys.path.insert(0, str(SESSION_RUNNER.parent))
from session_runner import check_final_store  # noqa: E402,F401  the QR suite's finished-run check


def _volumes(dim, verts, cells):
    """Cell sizes from /mesh: lengths (1-D), polygon areas (2-D); None for 3-D."""
    if dim == 1:
        x = verts[0]
        return np.abs(x[cells[1]] - x[cells[0]])
    if dim == 2:
        x, y = verts[0][cells], verts[1][cells]            # (k, n_cells)
        return 0.5 * np.abs(np.sum(x * np.roll(y, -1, 0) - np.roll(x, -1, 0) * y, axis=0))
    return None


def read_store(path) -> dict:
    """`{times, names, Q0, Q, vol, n_inner, values, nonfinite}` of a zoomy HDF5 store."""
    import h5py
    out = {"values": 0, "nonfinite": 0}

    def visit(_, obj):
        if isinstance(obj, h5py.Dataset) and obj.dtype.kind == "f":
            a = np.asarray(obj[()])
            out["values"] += int(a.size)
            out["nonfinite"] += int(a.size - np.isfinite(a).sum())

    with h5py.File(path, "r") as f:
        f.visititems(visit)
        m = f["mesh"]
        dim = int(np.asarray(m["dimension"])[()])
        n_inner = int(np.asarray(m["n_inner_cells"])[()]) if "n_inner_cells" in m else None
        verts = np.asarray(m["vertex_coordinates"])
        cells = np.asarray(m["cell_vertices"])
        if cells.shape[0] > cells.shape[1]:                 # (n_cells, k) layout
            cells = cells.T
        if verts.shape[0] > verts.shape[1]:
            verts = verts.T
        its = sorted((int(k.split("_")[1]), k) for k in f.get("fields", {}) if k.startswith("iteration_"))
        g = f["fields"] if its else None
        out["times"] = [float(np.asarray(g[k]["time"])) for _, k in its] if its else []
        out["Q0"] = np.asarray(g[its[0][1]]["Q"]) if its else None
        out["Q"] = np.asarray(g[its[-1][1]]["Q"]) if its else None
        names = g.attrs.get("names") if g is not None else None
        out["names"] = [n.decode() if isinstance(n, bytes) else str(n) for n in names] \
            if names is not None else None
    out["dim"], out["n_inner"] = dim, n_inner
    out["vol"] = _volumes(dim, verts, cells.astype(int))
    return out


def mass_change(s: dict, row) -> float:
    """Relative change of sum(Q[row] * vol) over the inner cells, first to last snapshot."""
    if isinstance(row, str):        # no /fields@names: the GUI's fallback, q1 = the depth
        i = s["names"].index(row) if s["names"] else {"h": 1}[row]
    else:
        i = int(row)
    n = s["n_inner"] or s["Q"].shape[1]
    v = s["vol"][:n]
    m0, m1 = float(np.sum(s["Q0"][i, :n] * v)), float(np.sum(s["Q"][i, :n] * v))
    return abs(m1 - m0) / max(abs(m0), 1e-300)


def compare(a: dict, b: dict) -> float:
    """max|a - b| / max|b| at the last snapshot, inner cells (the matrix's relmax)."""
    n = a["n_inner"] or a["Q"].shape[1]
    qa, qb = a["Q"][:, :n], b["Q"][:, :n]
    if qa.shape != qb.shape:
        return float("inf")
    return float(np.max(np.abs(qa - qb)) / max(float(np.max(np.abs(qb))), 1e-300))


def _env(lib: Path, extra=(), env=None) -> dict:
    path = [str(lib / p) for p in ("zoomy_core", "zoomy_plotting", *extra)]
    return dict(os.environ, PYTHONPATH=os.pathsep.join(path), PYTHONNOUSERSITE="1",
                JAX_PLATFORMS="cpu", MPLBACKEND="Agg", **(env or {}))


def run_cells(cells, lib: Path, work: Path, timeout: float, extra=(), env=None) -> dict:
    """The cells through the GUI's engine.py (tests/qr_cases/session_runner.py) in CPython."""
    work.mkdir(parents=True, exist_ok=True)
    spec, out = work / "case.json", work / "report.json"
    spec.write_text(json.dumps({"cells": cells, "needs_store": False}))
    p = subprocess.run([sys.executable, str(SESSION_RUNNER), str(spec),
                        str(lib / "zoomy_gui" / "engine.py"), str(out), str(work)],
                       env=_env(lib, extra, env), timeout=timeout, capture_output=True, text=True)
    if not out.exists():
        return {"ok": False, "error": f"session runner crashed ({p.returncode}): {p.stderr[-2000:]}",
                "cells": []}
    return json.loads(out.read_text())


def gui_view(h5: Path, viewer_snippet: str | None, lib: Path, work: Path) -> dict:
    """What the GUI does with a backend's store: `open_hdf5('/tmp/zoomy_sim/<job>.h5')`, then
    the selected viewer card's snippet with `time_step` = last, `field_name` = the default
    (renderVizCard in model-config-widget.ts)."""
    cells = [["open", f"open_hdf5({str(h5)!r})"]]
    if viewer_snippet:
        cells.append(["viewer", "time_step = store.n_snapshots - 1\nfield_name = next((n for n in "
                      "('h', 'height', 'q1') if n in store.field.keys()), None)\n" + viewer_snippet])
    return run_cells(cells, lib, work, timeout=300)


def find_store(work: Path) -> Path | None:
    """The store an in-process run left where the server collects it (simulation.h5)."""
    p = work / "simulation.h5"
    return p if p.exists() else None


def cell_report(rep: dict) -> str:
    return ", ".join(f"{c['role']}={c['status']} ({c['seconds']}s)" for c in rep.get("cells", []))
