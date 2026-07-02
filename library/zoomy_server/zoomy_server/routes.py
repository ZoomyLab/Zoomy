"""FastAPI routes for Zoomy Solver Server."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from zoomy_server import jobs
from zoomy_server.registry import build_registry

router = APIRouter(prefix="/api/v1")
_adapter = None
_session_dir = None
_predefined_json = None


def set_adapter(adapter):
    global _adapter
    _adapter = adapter


def set_session(session_dir=None, predefined_json=None):
    global _session_dir, _predefined_json
    _session_dir = session_dir
    _predefined_json = predefined_json


class JobRequest(BaseModel):
    case_dir: str


class CaseRequest(BaseModel):
    """A self-contained case submitted by a client (e.g. the browser GUI): the
    canonical composed ``.py`` (``zoomy_prepost.case`` format) plus an optional
    uploaded mesh (base64). The server materializes it into a case folder and
    runs it with the configured adapter — no pre-existing server-side folder."""
    case_py: str
    mesh_b64: Optional[str] = None
    mesh_name: Optional[str] = None


@router.get("/health")
def health():
    return {"status": "ok", "tag": _adapter.tag if _adapter else "unknown"}


@router.get("/registry")
def get_registry():
    """Return available models, solvers, and meshes from all sources.

    Merges: pre-defined cards → library discovery → catalog meshes → user session.
    The GUI calls this at startup to populate model/solver/mesh tabs.
    """
    return build_registry(
        session_dir=_session_dir,
        predefined_json=_predefined_json,
    )


@router.post("/jobs")
def create_job(req: JobRequest):
    if not _adapter:
        raise HTTPException(503, "No adapter configured")
    job_id = jobs.submit(_adapter, req.case_dir)
    return {"job_id": job_id}


@router.post("/cases")
def create_case_job(req: CaseRequest):
    """Ingest a self-contained composed case, materialize the adapter case
    folder (model.py, mesh.py, settings.json [+ uploaded mesh]) and run it.
    This is the endpoint the browser GUI uses — it uploads the composed .py
    rather than naming a server-side path."""
    if not _adapter:
        raise HTTPException(503, "No adapter configured")
    import base64
    import os
    import tempfile
    from zoomy_prepost import to_folder

    case_dir = tempfile.mkdtemp(prefix="zoomy_case_")
    try:
        to_folder(req.case_py, case_dir)
        if req.mesh_b64 and req.mesh_name:
            dest = os.path.join(case_dir, os.path.basename(req.mesh_name))
            with open(dest, "wb") as f:
                f.write(base64.b64decode(req.mesh_b64))
    except Exception as e:
        raise HTTPException(400, f"Invalid case: {e}")
    job_id = jobs.submit(_adapter, case_dir)
    return {"job_id": job_id}


@router.get("/jobs")
def list_all_jobs():
    return jobs.list_jobs()


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    status = jobs.get_status(job_id)
    if not status:
        raise HTTPException(404, "Job not found")
    return status


@router.get("/jobs/{job_id}/results/hdf5")
def download_hdf5(job_id: str):
    # Gate on job completion: simulation.h5 is created (mesh-only) at solve
    # start and fields are appended during the run, so serving it while the
    # job is still running would return an incomplete file (no /fields).
    status = jobs.get_status(job_id)
    if not status:
        raise HTTPException(404, "Job not found")
    if status["status"] == "failed":
        raise HTTPException(500, f"Job failed: {str(status.get('error', ''))[:1000]}")
    if status["status"] != "complete":
        raise HTTPException(425, "Job still running")  # 425 Too Early
    path = jobs.get_hdf5_path(job_id)
    if not path:
        raise HTTPException(404, "HDF5 not available")
    return FileResponse(path, media_type="application/x-hdf5", filename="simulation.h5")


@router.get("/jobs/{job_id}/results")
def get_job_results(job_id: str, timeline: bool = False):
    """Return simulation results as JSON (final snapshot, or full timeline).

    Query parameters:
        timeline: if true, include Q_timeline/Qaux_timeline/times arrays
                  for all snapshots (useful for slider-based visualization).
    """
    data = jobs.get_results(job_id, timeline=timeline)
    if data is None:
        raise HTTPException(404, "Results not available")
    if "error" in data and data.get("error"):
        raise HTTPException(500, data["error"])
    return data


@router.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    if jobs.cancel(job_id):
        return {"status": "cancelled"}
    raise HTTPException(404, "Job not found")
