"""Execute composed session cells through the GUI's `engine.py`, then check the stores.

Shared by CPython (`python session_runner.py CASE.json ENGINE.py OUT.json`) and Pyodide
(`pyodide/run.mjs` imports it). Every cell goes through `engine.process_code`, i.e. one
`exec` in the worker's persistent scope, as a GUI run does. A store is every `*.h5` under the
working directory; every float dataset in it must be finite. A case that runs a simulation
must leave the `simulation.h5` the GUI and the server collect with its snapshots up to the
case's `time_end` (`check_final_store`, also the backend harness's check).
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


#: the store a run leaves for the GUI / the server to collect: the cases copy their solver's
#: output to it (`output/<name>.h5` is the solver's own, complete even when the copy is not)
ARTIFACT = "simulation.h5"
#: a run that marched has at least its initial and its final state
MIN_SNAPSHOTS = 2


def check_final_store(snapshots, t_last, time_end=None, min_snapshots=MIN_SNAPSHOTS):
    """Why a run's store is not a finished run, else None: fewer than `min_snapshots` snapshots,
    or, with a `time_end`, a last snapshot before it (relative tolerance 1e-6)."""
    if (snapshots or 0) < min_snapshots:
        return f"holds {snapshots or 0} snapshots, a finished run holds at least {min_snapshots}"
    if time_end is not None and (t_last is None or t_last < time_end - 1e-6 * max(1.0, abs(time_end))):
        return f"ends at t = {t_last}, before time_end = {time_end}"
    return None


def final_store_problem(stores, time_end=None):
    """Why the stores a run left (`check_stores`) are not a finished run's, else None. Only the
    `simulation.h5` counts: the solver's own output (`output/<name>.h5`) holding the snapshots does
    not make a copy taken before they were written a finished run."""
    final = next((s for s in stores if s["path"] == ARTIFACT), None)
    if final is None:
        return f"the run left no {ARTIFACT}"
    why = check_final_store(final["snapshots"], final["t_last"], time_end)
    return f"{ARTIFACT} {why}" if why else None


def end_time(spec):
    """The case's `time_end` in the store's time unit: a number, or an expression over the cells'
    names (the engine's scope: `settings`, the model cell's constants, ...); None: not known."""
    if isinstance(spec, str):
        return float(eval(spec, dict(sys._shallowflow_scope)))
    return spec


def zoomy_packages():
    """`{module: {version, file}}` of the zoomy_core / zoomy_plotting this run imported."""
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for mod, dist in (("zoomy_core", "zoomy_core"), ("zoomy_plotting", "zoomy-plotting")):
        m = sys.modules.get(mod)
        if m is not None:
            try:
                v = version(dist)
            except PackageNotFoundError:
                v = None
            out[mod] = {"version": v, "file": getattr(m, "__file__", None)}
    return out


def run_case(case, process_code, workdir):
    """Run `case` (`{"cells": [[role, code], ...], "needs_store": bool, "time_end": number |
    expression | None}`) in `workdir`. A `needs_store` case must leave a finished `simulation.h5`
    (`final_store_problem`)."""
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
    if case.get("needs_store"):
        try:
            report["time_end"] = end_time(case.get("time_end"))
        except Exception as e:     # an expression over names the cells did not define
            report["ok"] = False
            report["error"] = f"time_end {case.get('time_end')!r}: {type(e).__name__}: {e}"
            return report
        why = final_store_problem(stores, report["time_end"])
        if why:
            report["ok"], report["error"] = False, why
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
    report["packages"] = zoomy_packages()
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
