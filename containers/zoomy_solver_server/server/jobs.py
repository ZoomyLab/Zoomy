import os
import uuid
import tempfile
from concurrent.futures import ProcessPoolExecutor

from server.runner import run_case, read_progress, load_result_fields


JOBS = {}
EXECUTOR = ProcessPoolExecutor(max_workers=2)
JOBS_DIR = os.path.join(tempfile.gettempdir(), "zoomy_jobs")


def _run_job(case_dict, output_dir, progress_file):
    try:
        run_case(case_dict, output_dir, progress_file)
    except Exception as e:
        import json, traceback
        with open(progress_file, "w") as f:
            json.dump({"status": "failed", "error": traceback.format_exc()}, f)


def submit(case_dict):
    job_id = str(uuid.uuid4())[:8]
    output_dir = os.path.join(JOBS_DIR, job_id)
    progress_file = os.path.join(output_dir, "progress.json")
    os.makedirs(output_dir, exist_ok=True)

    future = EXECUTOR.submit(_run_job, case_dict, output_dir, progress_file)
    JOBS[job_id] = {
        "future": future,
        "output_dir": output_dir,
        "progress_file": progress_file,
        "case": case_dict,
    }
    return job_id


def get_status(job_id):
    if job_id not in JOBS:
        return None

    job = JOBS[job_id]
    progress = read_progress(job["progress_file"])

    if progress and progress.get("status") == "failed":
        return {"job_id": job_id, "status": "failed", "error": progress.get("error")}

    if progress and progress.get("status") == "complete":
        return {"job_id": job_id, "status": "complete", "progress": progress}

    if job["future"].done():
        exc = job["future"].exception()
        if exc:
            return {"job_id": job_id, "status": "failed", "error": str(exc)}
        return {"job_id": job_id, "status": "complete", "progress": progress}

    return {
        "job_id": job_id,
        "status": "running",
        "progress": progress,
    }


def get_results(job_id):
    if job_id not in JOBS:
        return None
    return load_result_fields(JOBS[job_id]["output_dir"])


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
    job = JOBS[job_id]
    job["future"].cancel()
    del JOBS[job_id]
    return True
