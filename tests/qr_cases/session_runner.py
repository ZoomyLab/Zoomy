"""Execute composed session cells through the GUI's `engine.py`, then check the stores.

Shared by CPython (`python session_runner.py CASE.json ENGINE.py OUT.json`) and Pyodide
(`pyodide/run.mjs` imports it). Every cell goes through `engine.process_code`, i.e. one
`exec` in the worker's persistent scope, as a GUI run does. A store is every `*.h5` under the
working directory and the run's own store; every float dataset in it must be finite. A case
that runs a simulation must leave its store as the GUI keeps it: the one the run opened
(`open_hdf5`, which publishes `store`; the GUI saves `store_source_path()`), with its snapshots
up to the case's `time_end` (`final_store_problem`; `check_final_store` is also the backend
harness's check). The browser tier (`test_qr_browser.py`) runs the same checks in the
notebook kernel (`notebook_check_code`).
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


def store_info(path):
    """`{path, snapshots, t_last, values, nonfinite}` of one HDF5 file: the float values it
    holds, how many are not finite, and its `fields/*/time` snapshots."""
    import h5py
    import numpy as np

    info = {"path": path, "values": 0, "nonfinite": 0, "snapshots": None, "t_last": None}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.dtype.kind == "f":
            a = np.asarray(obj[()])
            info["values"] += int(a.size)
            info["nonfinite"] += int(a.size - np.isfinite(a).sum())

    with h5py.File(path, "r") as f:
        f.visititems(visit)
        if "fields" in f:
            keys = list(f["fields"].keys())
            info["snapshots"] = len(keys)
            times = [float(np.asarray(f["fields"][k]["time"])) for k in keys
                     if "time" in f["fields"][k]]
            info["t_last"] = max(times) if times else None
    return info


def check_stores(root="."):
    """`store_info` of every HDF5 file under `root`, `path` relative to it."""
    found = []
    for d, _, files in os.walk(root):
        for fn in sorted(files):
            if fn.endswith(".h5"):
                p = os.path.join(d, fn)
                found.append(dict(store_info(p), path=os.path.relpath(p, root)))
    return found


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


def published_store_path(scope):
    """The file of the store a run published in `scope` (`open_hdf5`), else None: the store the
    GUI keeps of a run (`store_source_path()`)."""
    s = scope.get("store")
    return getattr(s, "source_path", None) if s is not None else None


def final_store_problem(store, time_end=None):
    """Why the run's store (`store_info` of the published store, None: the run published none)
    is not a finished run's, else None. The solver's other outputs do not count: a store the run
    did not open is not what the GUI shows or keeps."""
    if store is None:
        return "the run published no store (a solver card ends with open_hdf5(run_h5_path()))"
    why = check_final_store(store["snapshots"], store["t_last"], time_end)
    if why:
        return f"the run's store {store['path']} {why}"
    if store["nonfinite"]:
        return f"the run's store {store['path']} holds {store['nonfinite']} non-finite values"
    return None


def notebook_check(scope, n_before, required, needs_store, time_end):
    """The verdict of the QR browser tests' final notebook cell, run in the notebook kernel's
    namespace `scope`: `{"ok", "problems", "note", "store", "time_end"}`. The cells this check judges
    are the `n_before` cells the kernel ran last (`sys.zoomy_cell_log`: None or the error of
    every cell, in order); the `required` ones (indices among them) must have succeeded. A
    session that solves (`needs_store`) must have published a finished, finite store
    (`final_store_problem`), `time_end` a number or an expression over the notebook's names."""
    problems, note = [], None
    log = getattr(sys, "zoomy_cell_log", None)
    if log is None:      # a kernel before the log: the cells' own status (the DOM) is the record
        note = "the kernel keeps no cell log (sys.zoomy_cell_log): cells not judged here"
    else:
        mine = log[len(log) - n_before:] if n_before else []
        if len(mine) < n_before:
            problems.append(f"the kernel ran {len(mine)} cells, the notebook has {n_before} before this one")
        problems += [f"cell {i + 1}: {mine[i]}" for i in required if i < len(mine) and mine[i] is not None]
    store = None
    if needs_store:
        try:
            time_end = float(eval(time_end, dict(scope))) if isinstance(time_end, str) else time_end
        except Exception as e:     # an expression over names the cells did not define
            problems.append(f"time_end: {type(e).__name__}: {e}")
            time_end = None
        own = published_store_path(scope)
        store = store_info(own) if own and os.path.isfile(own) else None
        why = final_store_problem(store, time_end)
        if why:
            problems.append(why)
    return {"ok": not problems, "problems": problems, "note": note, "store": store,
            "time_end": time_end}


def notebook_check_code(n_before, required, needs_store, time_end):
    """One line of Python, the QR browser tests' final notebook cell: runs `notebook_check`
    (this module's own code, base64-encoded so it types into the cell editor as is) in the
    notebook's namespace, prints `QR_CHECK <verdict json>` and shows True or False."""
    import base64
    import inspect

    src = "\n\n".join(inspect.getsource(f) for f in (
        store_info, check_final_store, published_store_path, final_store_problem, notebook_check))
    src = (f"import json, os, sys\nMIN_SNAPSHOTS = {MIN_SNAPSHOTS}\n\n{src}\n\n"
           f"verdict = notebook_check(scope, {n_before!r}, {list(required)!r}, {bool(needs_store)!r}, "
           f"{time_end!r})\nprint('QR_CHECK ' + json.dumps(verdict))\nok = verdict['ok']\n")
    b64 = base64.b64encode(src.encode()).decode()
    return (f'_qr = {{"scope": globals()}}; exec(__import__("base64").b64decode("{b64}").decode(), _qr); '
            f'_qr["ok"]')


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
    expression | None}`) in `workdir`. A `needs_store` case must publish a finished store
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
    own = published_store_path(getattr(sys, "_shallowflow_scope", {}))
    run_store = store_info(own) if own and os.path.isfile(own) else None
    if run_store is not None and not any(os.path.samefile(s["path"], own) for s in stores):
        stores.append(run_store)    # the run's store outside the working directory
    report["stores"] = stores
    if case.get("needs_store"):
        try:
            report["time_end"] = end_time(case.get("time_end"))
        except Exception as e:     # an expression over names the cells did not define
            report["ok"] = False
            report["error"] = f"time_end {case.get('time_end')!r}: {type(e).__name__}: {e}"
            return report
        why = final_store_problem(run_store, report["time_end"])
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
