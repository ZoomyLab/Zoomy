"""FastAPI routes for Zoomy Solver Server."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from zoomy_server import jobs
from zoomy_server.schemas import ZoomyCase

router = APIRouter(prefix="/api/v1")
_adapter = None


def set_adapter(adapter):
    global _adapter
    _adapter = adapter


@router.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.0",
        "tag": _adapter.tag if _adapter else "unknown",
    }


@router.post("/jobs")
def create_job(case: ZoomyCase):
    if not _adapter:
        raise HTTPException(503, "No adapter configured")
    job_id = jobs.submit(_adapter, case.model_dump())
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


@router.get("/jobs/{job_id}/results")
def get_results(job_id: str):
    results = jobs.get_results(job_id)
    if results is None:
        raise HTTPException(404, "Results not available")
    return results


@router.get("/jobs/{job_id}/results/hdf5")
def download_hdf5(job_id: str):
    path = jobs.get_hdf5_path(job_id)
    if not path:
        raise HTTPException(404, "HDF5 file not available")
    return FileResponse(path, media_type="application/x-hdf5", filename="simulation.h5")


@router.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    if jobs.cancel(job_id):
        return {"status": "cancelled"}
    raise HTTPException(404, "Job not found")


@router.get("/models")
def list_models():
    if not _adapter:
        return []
    return _adapter.list_models()
