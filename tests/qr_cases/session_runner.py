"""Execute composed session cells through the GUI's `engine.py`, then check the stores.

Shared by CPython (`python session_runner.py CASE.json ENGINE.py OUT.json`) and Pyodide
(`pyodide/run.mjs` imports it). Every cell goes through `engine.process_code`, i.e. one
`exec` in the worker's persistent scope, as a GUI run does. A store is every `*.h5` under the
working directory; every float dataset in it must be finite.
"""
import json
import os
import sys
import time

#: Matplotlib off-screen, as the notebook kernel sets it (pyodide-worker-source.ts).
os.environ.setdefault("MPLBACKEND", "AGG")


def load_engine(engine_path):
    """`process_code` of the GUI's engine.py, executed like the worker does."""
    ns = {"__name__": "zoomy_gui_engine"}
    with open(engine_path) as f:
        exec(compile(f.read(), engine_path, "exec"), ns)
    return ns["process_code"]


def run_cell(process_code, role, code):
    t0 = time.perf_counter()
    res = json.loads(process_code(code))
    out = res.get("output") or ""
    return {"role": role, "status": res.get("status"), "seconds": round(time.perf_counter() - t0, 2),
            "output_tail": out[-3000:]}


def check_stores(root="."):
    """`[{path, snapshots, t_last, values, nonfinite}]` for every HDF5 file under `root`."""
    import h5py
    import numpy as np

    found = []
    for d, _, files in os.walk(root):
        for fn in sorted(files):
            if not fn.endswith(".h5"):
                continue
            p = os.path.join(d, fn)
            info = {"path": os.path.relpath(p, root), "values": 0, "nonfinite": 0,
                    "snapshots": None, "t_last": None}

            def visit(name, obj, info=info):
                if isinstance(obj, h5py.Dataset) and obj.dtype.kind == "f":
                    a = np.asarray(obj[()])
                    info["values"] += int(a.size)
                    info["nonfinite"] += int(a.size - np.isfinite(a).sum())

            with h5py.File(p, "r") as f:
                f.visititems(visit)
                if "fields" in f:
                    keys = list(f["fields"].keys())
                    info["snapshots"] = len(keys)
                    times = [float(np.asarray(f["fields"][k]["time"])) for k in keys
                             if "time" in f["fields"][k]]
                    info["t_last"] = max(times) if times else None
            found.append(info)
    return found


def run_case(case, process_code, workdir):
    """Run `case` (`{"cells": [[role, code], ...], "needs_store": bool}`) in `workdir`."""
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    # zoomy_core's main directory: in the browser (no git, no __main__.__file__) it is the cwd.
    os.environ["ZOOMY_DIR"] = os.getcwd()
    report = {"cells": [], "ok": True, "error": None}
    for role, code in case["cells"]:
        r = run_cell(process_code, role, code)
        report["cells"].append(r)
        if r["status"] != "success":
            last = (r["output_tail"].strip().splitlines() or [""])[-1]
            report["ok"] = False
            report["error"] = f"{role} cell: {r['status']}: {last}"
            return report
    stores = check_stores(".")
    report["stores"] = stores
    if case.get("needs_store") and not any(s["snapshots"] for s in stores):
        report["ok"] = False
        report["error"] = "the run wrote no HDF5 store with /fields"
    bad = [s["path"] for s in stores if s["nonfinite"]]
    if bad:
        report["ok"] = False
        report["error"] = "non-finite values in " + ", ".join(bad)
    return report


def main(case_json, engine_path, out_json, workdir=None):
    with open(case_json) as f:
        case = json.load(f)
    blocked = case.get("blocked_modules") or []
    if blocked:
        sys.meta_path.insert(0, _Blocker(blocked))
    report = run_case(case, load_engine(engine_path), workdir or os.getcwd())
    report["modules_loaded"] = sorted(m for m in ("jax", "jaxlib", "zoomy_jax", "pyamg", "mpi4py",
                                                  "h5py", "matplotlib", "zoomy_plotting")
                                      if m in sys.modules)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=1)
    return report


class _Blocker:
    """Import hook: the browser has none of these packages (CPython env may)."""

    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.names:
            raise ModuleNotFoundError(f"No module named {name!r} (not available in the browser)",
                                      name=name)
        return None


if __name__ == "__main__":
    main(*sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)   # the report is written; JAX worker threads can stall a normal exit
