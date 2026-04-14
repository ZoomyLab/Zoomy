import os
import json
import uuid
import tempfile
from concurrent.futures import ProcessPoolExecutor

JOBS = {}
EXECUTOR = ProcessPoolExecutor(max_workers=2)
JOBS_DIR = os.path.join(tempfile.gettempdir(), "zoomy_jobs")


def _write_progress(progress_file, iteration, time_val, dt):
    with open(progress_file, "w") as f:
        json.dump({"iteration": iteration, "time": time_val, "dt": dt}, f)


def _run_job(adapter_cls_path, case_dir, output_dir, progress_file):
    try:
        import importlib
        mod_path, cls_name = adapter_cls_path.rsplit(".", 1)
        mod = importlib.import_module(mod_path)
        adapter = getattr(mod, cls_name)()

        def on_progress(iteration, time_val, dt):
            _write_progress(progress_file, iteration, time_val, dt)

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


def list_jobs():
    return [get_status(jid) for jid in JOBS]


def cancel(job_id):
    if job_id not in JOBS:
        return False
    JOBS[job_id]["future"].cancel()
    del JOBS[job_id]
    return True
