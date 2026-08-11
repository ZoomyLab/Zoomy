import os
import json
import uuid
import tempfile
from concurrent.futures import ProcessPoolExecutor

JOBS = {}
# Concurrency must be >= the number of participants in a coupled run: each
# preCICE participant is its own job that BLOCKS at the handshake until its
# peers join, so an N-participant coupling needs N free workers or the last
# participant queues behind ones already waiting for it -> deadlock. Generous
# default; override with ZOOMY_MAX_JOBS.
EXECUTOR = ProcessPoolExecutor(max_workers=int(os.environ.get("ZOOMY_MAX_JOBS", "8")))
JOBS_DIR = os.path.join(tempfile.gettempdir(), "zoomy_jobs")


def _write_progress(progress_file, iteration, time_val, dt, message=None):
    """Publish job progress, optionally with the solver's own latest log line.

    ``message`` is what the GUI shows: its status callback already renders
    ``s.message``, but nothing populated it, so a backend run was silent until
    it either finished or raised — and a compiled-then-coupled foam run takes
    minutes with nothing to look at. Forwarding the line makes a run
    observable while it happens rather than only in its post-mortem.
    """
    payload = {"iteration": iteration, "time": time_val, "dt": dt}
    if message:
        payload["message"] = message
    with open(progress_file, "w") as f:
        json.dump(payload, f)


def _run_job(adapter_cls_path, case_dir, output_dir, progress_file):
    try:
        import importlib
        import logging
        # Worker process: make the adapters' logger.info visible in the server
        # log (no-op when handlers were inherited via fork).
        logging.basicConfig(level=logging.INFO,
                            format="%(levelname)s: [%(name)s] %(message)s")
        mod_path, cls_name = adapter_cls_path.rsplit(".", 1)
        mod = importlib.import_module(mod_path)
        adapter = getattr(mod, cls_name)()

        def on_progress(iteration, time_val, dt, message=None):
            _write_progress(progress_file, iteration, time_val, dt, message)

        adapter.solve(case_dir, output_dir, on_progress)

        with open(progress_file, "w") as f:
            json.dump({"status": "complete"}, f)
    except Exception:
        import traceback
        with open(progress_file, "w") as f:
            json.dump({"status": "failed", "error": traceback.format_exc()}, f)


def submit(adapter, case_dir):
    job_id = str(uuid.uuid4())[:8]
    output_dir = os.path.join(JOBS_DIR, job_id)
    progress_file = os.path.join(output_dir, "progress.json")
    os.makedirs(output_dir, exist_ok=True)

    adapter_path = f"{adapter.__class__.__module__}.{adapter.__class__.__name__}"
    future = EXECUTOR.submit(_run_job, adapter_path, case_dir, output_dir, progress_file)
    JOBS[job_id] = {"future": future, "output_dir": output_dir, "progress_file": progress_file}
    return job_id


def _read_progress(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def get_status(job_id):
    if job_id not in JOBS:
        return None
    job = JOBS[job_id]
    progress = _read_progress(job["progress_file"])
    if progress and progress.get("status") == "failed":
        return {"job_id": job_id, "status": "failed", "error": progress.get("error")}
    if progress and progress.get("status") == "complete":
        return {"job_id": job_id, "status": "complete", "progress": progress}
    if job["future"].done():
        exc = job["future"].exception()
        if exc:
            return {"job_id": job_id, "status": "failed", "error": str(exc)}
        return {"job_id": job_id, "status": "complete", "progress": progress}
    return {"job_id": job_id, "status": "running", "progress": progress}


def get_hdf5_path(job_id):
    if job_id not in JOBS:
        return None
    path = os.path.join(JOBS[job_id]["output_dir"], "simulation.h5")
    return path if os.path.exists(path) else None


def list_artifacts(job_id):
    """Every collected output file in the job's output dir (name + size).

    The postprocess adapter drops the chain's products here — the
    normalized ``simulation.h5``, the ``to_vtk`` series, the ``lift3d``
    3-D store/series, and any ``*.png``/``*.gif``. ``progress.json`` (the
    server's own bookkeeping) is not an artifact.
    """
    if job_id not in JOBS:
        return []
    out_dir = JOBS[job_id]["output_dir"]
    items = []
    for fn in sorted(os.listdir(out_dir)):
        if fn == "progress.json":
            continue
        p = os.path.join(out_dir, fn)
        if os.path.isfile(p):
            items.append({"name": fn, "size": os.path.getsize(p)})
    return items


def get_artifact_path(job_id, name):
    """Absolute path of a named artifact in the job's output dir, or None.

    ``name`` is reduced to its basename so a client can never escape the
    output dir with ``../`` traversal."""
    if job_id not in JOBS:
        return None
    safe = os.path.basename(name)
    p = os.path.join(JOBS[job_id]["output_dir"], safe)
    return p if os.path.isfile(p) else None


def get_results(job_id, timeline=False):
    """Return simulation results as a JSON-serializable dict.

    Parses the job's HDF5 output file and returns either the final snapshot
    (timeline=False) or the full time history (timeline=True).

    Returns None if the job or its HDF5 file cannot be found.

    Structure::

        {
            "job_id": str,
            "time": float,                 # final snapshot time (non-timeline)
            "n_vars": int,
            "n_cells": int,
            "dim": int,                    # 1/2/3
            "coords": [[x,y,z], ...],      # cell centers
            "vertices": [[x,y,z], ...],    # mesh vertices (if available)
            "cells": [[v0,v1,...], ...],   # cell connectivity (if available)
            "Q": [[field0 values], ...],           # (n_vars, n_cells) — final state
            "Qaux": [[aux0 values], ...],          # optional
            # Only when timeline=True:
            "Q_timeline": [[[...], ...], ...],     # (n_snapshots, n_vars, n_cells)
            "Qaux_timeline": ...,
            "times": [t0, t1, ...],
            "n_snapshots": int,
        }
    """
    import numpy as np
    h5_path = get_hdf5_path(job_id)
    if not h5_path:
        return None

    try:
        from zoomy_core.misc import io as _io
    except ImportError:
        return {"error": "zoomy_core not available for result parsing"}

    result = {"job_id": job_id}

    try:
        if timeline:
            try:
                x, Q_tl, Qaux_tl, times = _io.load_timeline_of_fields_from_hdf5(h5_path)
            except Exception as e:
                return {"error": f"Failed to load timeline: {e}"}
            Q_tl = np.asarray(Q_tl)
            result["Q_timeline"] = Q_tl.tolist()
            result["Q"] = Q_tl[-1].tolist()
            if Qaux_tl is not None and len(Qaux_tl) > 0:
                Qaux_tl = np.asarray(Qaux_tl)
                result["Qaux_timeline"] = Qaux_tl.tolist()
                result["Qaux"] = Qaux_tl[-1].tolist()
            result["times"] = np.asarray(times).tolist()
            result["n_snapshots"] = len(times)
            result["time"] = float(times[-1]) if len(times) else 0.0
            result["coords"] = np.asarray(x).tolist() if x is not None else None
            result["n_vars"] = Q_tl.shape[1]
            result["n_cells"] = Q_tl.shape[2]
        else:
            Q, Qaux, t = _io.load_fields_from_hdf5(h5_path)
            Q = np.asarray(Q)
            result["Q"] = Q.tolist()
            if Qaux is not None:
                result["Qaux"] = np.asarray(Qaux).tolist()
            result["time"] = float(t) if t is not None else 0.0
            result["n_vars"] = Q.shape[0]
            result["n_cells"] = Q.shape[1]

        # Attempt to read mesh geometry from HDF5
        try:
            import h5py
            with h5py.File(h5_path, "r") as f:
                if "mesh" in f:
                    mg = f["mesh"]
                    if "vertices" in mg:
                        result["vertices"] = np.asarray(mg["vertices"][:]).tolist()
                    elif "nodes" in mg:
                        result["vertices"] = np.asarray(mg["nodes"][:]).tolist()
                    if "cells" in mg:
                        result["cells"] = np.asarray(mg["cells"][:]).tolist()
                    elif "connectivity" in mg:
                        result["cells"] = np.asarray(mg["connectivity"][:]).tolist()
                    if "dim" in mg.attrs:
                        result["dim"] = int(mg.attrs["dim"])
                    elif "coords" in result and result["coords"]:
                        c0 = result["coords"][0]
                        result["dim"] = len(c0) if hasattr(c0, "__len__") else 1
        except Exception:
            pass

        return result
    except Exception as e:
        return {"error": f"Failed to parse results: {e}"}


def list_jobs():
    return [get_status(jid) for jid in JOBS]


def cancel(job_id):
    if job_id not in JOBS:
        return False
    JOBS[job_id]["future"].cancel()
    del JOBS[job_id]
    return True
