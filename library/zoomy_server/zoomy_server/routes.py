"""FastAPI routes for Zoomy Solver Server."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from zoomy_server import jobs
from zoomy_server import results
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


class SaveResultRequest(BaseModel):
    """Save a completed job's HDF5 store into the named results shelf."""
    job_id: str
    name: str


class CaseRequest(BaseModel):
    """A self-contained case submitted by a client (e.g. the browser GUI): the
    canonical composed ``.py`` (``zoomy_prepost.case`` format) plus an optional
    uploaded mesh (base64). The server materializes it into a case folder and
    runs it with the configured adapter — no pre-existing server-side folder."""
    case_py: str
    mesh_b64: Optional[str] = None
    mesh_name: Optional[str] = None


class PostprocessRequest(BaseModel):
    """Run the post-processing chain on an EXISTING result store.

    The GUI/CLI uploads the just-finished run's ``simulation.h5`` (base64) plus
    the enabled chain steps; the server materializes a RESULTS folder (store +
    ``steps.json`` [+ ``model.py``]) and runs it through the connected
    ``postprocess`` adapter (see ``adapters/postprocess.py``). ``model_py`` is
    the composed case's model cell — only the ``lift3d`` step needs it (its
    ``interpolate_to_3d`` does the vertical lift)."""
    store_b64: str
    steps: list = []
    nz: int = 10
    model_py: Optional[str] = None


@router.get("/health")
def health():
    """Liveness + handshake. Beyond ``status``/``tag`` (kept for back-compat)
    this reports the adapter's ``capabilities()``: ``name`` and ``backends``
    (the solver tags this server can run) so the GUI learns who the backend is
    and which solvers it unlocks — instead of assuming a name."""
    if not _adapter:
        return {"status": "ok", "tag": "unknown", "backends": []}
    caps = _adapter.capabilities()
    tag = caps.get("tag", _adapter.tag)
    return {
        "status": "ok",
        "tag": tag,
        "name": caps.get("name"),
        "backends": caps.get("solvers", [tag]),
        "adapter": caps.get("adapter"),
    }


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


class CouplingRequest(BaseModel):
    coupling_id: str = "coupled"
    scheme: str = "parallel-explicit"
    end_time: Optional[float] = None
    sif: Optional[str] = None            # apptainer image (else ZOOMY_OPENFOAM_SIF)
    participants: list                   # [{name,type,template,binary}, ...]


#: post-solve OF -> zoomy HDF5: reuse zoomy_prepost.vtk_to_hdf5 on the foamToVTK
#: internal-field series (skip the raw t=0 dump, whose field set differs). Writes
#: <case>/simulation.h5, which _promote_foam_h5 then serves as the result store.
_VTK_TO_H5 = (
    "import sys, glob, os\n"
    "from zoomy_prepost import vtk_to_hdf5\n"
    "case = sys.argv[1]; name = os.path.basename(case)\n"
    "vtks = sorted(glob.glob(os.path.join(case, 'VTK', name + '_*.vtk')),\n"
    "              key=lambda p: int(p.rsplit('_', 1)[1].split('.')[0]))\n"
    "frames = vtks[1:] if len(vtks) > 1 else vtks\n"
    "if frames:\n"
    "    vtk_to_hdf5(frames, os.path.join(case, 'simulation.h5'))\n"
    "    print('wrote', os.path.join(case, 'simulation.h5'))\n"
    "else:\n"
    "    print('no VTK frames to convert')\n"
)


def _write_participant_runsh(case_dir, coupling_dir, binary, sif):
    """run.sh the foam adapter executes for this participant: launch its
    zoomyFoam/foamRun binary in-container (foamRun on PATH) or via apptainer,
    sharing the coupling folder so preCICE participants find each other; then
    foamToVTK -> zoomy_prepost.vtk_to_hdf5 to emit <case>/simulation.h5."""
    import os
    # solve + foamToVTK in the SAME OF context (in-container or via apptainer).
    inner = (f"source /opt/openfoam13/etc/bashrc; unset FOAM_SIGFPE FOAM_SETNAN; "
             f"'{binary}' -case '{case_dir}'; foamToVTK -case '{case_dir}'")
    lines = [
        "#!/bin/bash",
        "set +e; source /opt/openfoam13/etc/bashrc 2>/dev/null; set -e",
        "unset FOAM_SIGFPE FOAM_SETNAN",
        "if command -v foamRun >/dev/null 2>&1; then",
        f"  '{binary}' -case '{case_dir}'",
        f"  foamToVTK -case '{case_dir}'",
        "else",
        f'  apptainer exec --bind "{coupling_dir}" "{sif}" bash -lc "{inner}"',
        "fi",
        "# OF VTK -> zoomy HDF5 (host python has zoomy_prepost + meshio)",
        f'"${{ZOOMY_PY:-python3}}" -c "$VTK_TO_H5" \'{case_dir}\' || true',
    ]
    p = os.path.join(case_dir, "run.sh")
    with open(p, "w") as f:
        f.write("VTK_TO_H5=" + _shquote(_VTK_TO_H5) + "\n")
        f.write("\n".join(lines) + "\n")
    os.chmod(p, 0o755)


def _shquote(s):
    """Single-quote a string for safe embedding in the run.sh."""
    return "'" + s.replace("'", "'\\''") + "'"


@router.post("/couple")
def create_coupling_job(req: CouplingRequest):
    """Assemble + launch a preCICE coupling. build_coupled_bundle lays out the
    participant OF-cases + the shared generated precice-config under ONE coupling
    folder (= the exchange-directory); each participant is then submitted as its
    own foam job (they share the folder, so they find each other over sockets).
    Concurrency must be >= #participants (ZOOMY_MAX_JOBS). Returns the child
    job ids; poll /jobs/<id> + fetch /jobs/<id>/results/hdf5 per participant."""
    if not _adapter:
        raise HTTPException(503, "No adapter configured")
    import os
    import tempfile
    from zoomy_prepost.coupling import build_coupled_bundle

    coupling_dir = os.path.join(tempfile.mkdtemp(prefix="zoomy_couple_"), req.coupling_id)
    try:
        bundle = build_coupled_bundle(coupling_dir, req.participants,
                                      end_time=req.end_time, scheme=req.scheme)
    except Exception as e:
        raise HTTPException(400, f"Invalid coupling: {e}")
    sif = req.sif or os.environ.get("ZOOMY_OPENFOAM_SIF", "")
    out = []
    for p, (name, case_dir) in zip(req.participants, bundle["cases"]):
        _write_participant_runsh(case_dir, coupling_dir, p.get("binary", ""), sif)
        out.append({"name": name, "job": jobs.submit(_adapter, case_dir),
                    "case_dir": case_dir})
    return {"coupling_id": req.coupling_id, "dir": coupling_dir,
            "config": bundle["config"], "participants": out,
            "jobs": [o["job"] for o in out]}


@router.post("/postprocess")
def create_postprocess_job(req: PostprocessRequest):
    """Materialize a RESULTS folder from an uploaded store + enabled steps and
    run it through the connected ``postprocess`` adapter. This is the endpoint
    the GUI/CLI post-processing chain routes to when a postprocess backend is
    connected — the chain executes HERE (where ``zoomy_prepost`` lives), not in
    the browser. Artifacts are fetched back via ``GET /jobs/{id}/artifacts``."""
    if not _adapter:
        raise HTTPException(503, "No adapter configured")
    if getattr(_adapter, "tag", None) != "postprocess":
        raise HTTPException(
            409, "connected backend is not a postprocess adapter (tag=%r)"
            % getattr(_adapter, "tag", None))
    import base64
    import json
    import os
    import tempfile

    case_dir = tempfile.mkdtemp(prefix="zoomy_postproc_case_")
    try:
        with open(os.path.join(case_dir, "simulation.h5"), "wb") as f:
            f.write(base64.b64decode(req.store_b64))
        with open(os.path.join(case_dir, "steps.json"), "w") as f:
            json.dump({"steps": list(req.steps), "nz": int(req.nz)}, f)
        # lift3d resolves the model from model.py (module-level `model`).
        if req.model_py:
            with open(os.path.join(case_dir, "model.py"), "w") as f:
                f.write(req.model_py)
    except Exception as e:
        raise HTTPException(400, f"Invalid postprocess request: {e}")
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


@router.get("/jobs/{job_id}/artifacts")
def list_job_artifacts(job_id: str):
    """List every collected output file of a (complete) job. Used by the
    post-processing chain to discover the transformed products — the lifted
    3-D store, the VTK series, figures — before downloading them."""
    status = jobs.get_status(job_id)
    if not status:
        raise HTTPException(404, "Job not found")
    if status["status"] == "failed":
        raise HTTPException(500, f"Job failed: {str(status.get('error', ''))[:1000]}")
    if status["status"] != "complete":
        raise HTTPException(425, "Job still running")  # 425 Too Early
    return {"artifacts": jobs.list_artifacts(job_id)}


@router.get("/jobs/{job_id}/artifacts/{name}")
def download_job_artifact(job_id: str, name: str):
    """Download a single named artifact (h5 / vtu / pvd / png / gif) by name."""
    import os

    path = jobs.get_artifact_path(job_id, name)
    if not path:
        raise HTTPException(404, "Artifact not found")
    return FileResponse(path, filename=os.path.basename(path))


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


# ---------------------------------------------------------------------------
# Named result stores — the RESULTS SHELF. A completed job's simulation.h5
# lives in the ephemeral jobs dir (GC'd on cancel / restart); saving it here
# copies it into a persistent-ish results dir so it can be reopened by name
# from any later session or run. See results.py for the storage contract.
# ---------------------------------------------------------------------------

@router.post("/results")
def save_result(req: SaveResultRequest):
    """Copy a completed job's HDF5 into the results shelf under ``name``."""
    status = jobs.get_status(req.job_id)
    if not status:
        raise HTTPException(404, "Job not found")
    if status["status"] != "complete":
        raise HTTPException(425, "Job not complete")  # 425 Too Early
    src = jobs.get_hdf5_path(req.job_id)
    if not src:
        raise HTTPException(404, "HDF5 not available")
    try:
        return results.save(src, req.name)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/results")
def list_results():
    """List named results: ``[{name, size, created}, ...]`` (newest first)."""
    return results.list_results()


@router.get("/results/{name}/hdf5")
def download_result(name: str):
    path = results.get_path(name)
    if not path:
        raise HTTPException(404, "Result not found")
    return FileResponse(path, media_type="application/x-hdf5",
                        filename=results.slugify(name) + ".h5")


@router.delete("/results/{name}")
def delete_result(name: str):
    if results.delete(name):
        return {"status": "deleted"}
    raise HTTPException(404, "Result not found")
