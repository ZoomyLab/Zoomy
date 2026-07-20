"""#9 test_server_lifecycle — job lifecycle + the generic-runner contract +
the artifact path-traversal guard (absorbs test_routes and the relocated
thesis test_runner_case).

Three concerns:

* LIFECYCLE (mock adapter): submit -> running -> complete -> HDF5/artifact
  download -> cancel; unknown ids 404.
* TRAVERSAL GUARD: ``../`` artifact names never escape the job dir.
* GENERIC RUNNER (real NumpyAdapter): a composed case (SME(level=1) card
  model — the full derivation chain) is materialized with its own ``run.py``,
  POSTed via ``/cases`` (the GUI's endpoint), executed by the server AS the
  case's own solver code, and served back as ``/results/hdf5`` with
  fields + mesh; the server log carries "runner: executing case run.py" and
  the job's run.log captured the ## Run cell's own final print.
"""
from __future__ import annotations

import logging
import os
import time

import pytest
from fastapi.testclient import TestClient

from zoomy_server import routes as routes_module

pytestmark = [pytest.mark.small, pytest.mark.postprocessing, pytest.mark.gui]


def _poll_until(client, job_id, predicate, timeout_s=5.0, step_s=0.05):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        r = client.get(f"/api/v1/jobs/{job_id}")
        last = r
        if r.status_code == 200 and predicate(r.json()):
            return r
        time.sleep(step_s)
    raise AssertionError(
        f"job {job_id} never satisfied predicate within {timeout_s}s; "
        f"last: {last.status_code} {last.json() if last else 'n/a'}")


# ── lifecycle (mock adapter from conftest) ───────────────────────────────────

def test_job_lifecycle(client, case_dir):
    assert client.get("/api/v1/health").json()["status"] == "ok"
    assert client.get("/api/v1/jobs").json() == []

    # submit — schema-validated
    assert client.post("/api/v1/jobs", json={"foo": "bar"}).status_code == 422
    r = client.post("/api/v1/jobs", json={"case_dir": case_dir})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert job_id and isinstance(job_id, str)

    # runs to completion and is listed
    body = _poll_until(client, job_id,
                       lambda b: b.get("status") in ("complete", "failed")).json()
    assert body["status"] == "complete", body
    assert any(j.get("job_id") == job_id
               for j in client.get("/api/v1/jobs").json())

    # HDF5 download + artifact listing (progress.json is never an artifact)
    h5 = client.get(f"/api/v1/jobs/{job_id}/results/hdf5")
    assert h5.status_code == 200 and len(h5.content) > 0
    arts = client.get(f"/api/v1/jobs/{job_id}/artifacts").json()["artifacts"]
    names = [a["name"] for a in arts]
    assert "simulation.h5" in names and "progress.json" not in names
    assert all(a["size"] > 0 for a in arts)
    art = client.get(f"/api/v1/jobs/{job_id}/artifacts/simulation.h5")
    assert art.content == h5.content

    # cancel: first 200, second 404 (gone)
    assert client.delete(f"/api/v1/jobs/{job_id}").status_code == 200
    assert client.delete(f"/api/v1/jobs/{job_id}").status_code == 404

    # unknown ids 404 everywhere
    assert client.get("/api/v1/jobs/nope").status_code == 404
    assert client.get("/api/v1/jobs/nope/results/hdf5").status_code == 404
    assert client.get("/api/v1/jobs/nope/artifacts").status_code == 404


def test_artifact_path_traversal_guard(client, case_dir):
    from zoomy_server import jobs as jobs_module

    job_id = client.post("/api/v1/jobs",
                         json={"case_dir": case_dir}).json()["job_id"]
    _poll_until(client, job_id, lambda b: b.get("status") == "complete")

    # URL-encoded ../ escape -> reduced to basename -> not a job file -> 404
    r = client.get(f"/api/v1/jobs/{job_id}/artifacts/..%2f..%2fetc%2fpasswd")
    assert r.status_code == 404
    # unit-level: the resolver basenames the request
    assert jobs_module.get_artifact_path(job_id, "../../etc/passwd") is None
    p = jobs_module.get_artifact_path(job_id, "nested/../simulation.h5")
    assert p is not None and os.path.basename(p) == "simulation.h5"
    assert os.path.dirname(p) == jobs_module.JOBS[job_id]["output_dir"]


def test_named_results_shelf(client, case_dir, results_dir):
    """The named RESULTS SHELF (part of the absorbed test_routes surface):
    save-by-name slugs, lists, downloads the job's exact bytes, looks up by
    fancy name or slug, 404s/400s on bad input, and deletes for good."""
    job_id = client.post("/api/v1/jobs",
                         json={"case_dir": case_dir}).json()["job_id"]
    _poll_until(client, job_id, lambda b: b.get("status") == "complete")

    r = client.post("/api/v1/results", json={"job_id": job_id, "name": "Ref A"})
    assert r.status_code == 200, r.text
    entry = r.json()
    assert entry["name"] == "ref-a" and entry["size"] > 0 and "created" in entry
    assert any(e["name"] == "ref-a"
               for e in client.get("/api/v1/results").json())
    dl = client.get("/api/v1/results/ref-a/hdf5")
    assert dl.status_code == 200
    assert dl.content == client.get(
        f"/api/v1/jobs/{job_id}/results/hdf5").content

    # fancy name: fetchable by the original OR its slug
    client.post("/api/v1/results",
                json={"job_id": job_id, "name": "SWE Reference!"})
    assert client.get("/api/v1/results/SWE Reference!/hdf5").status_code == 200
    assert client.get("/api/v1/results/swe-reference/hdf5").status_code == 200

    # bad input: unknown job 404, unslugifiable name 400, unknown result 404
    assert client.post("/api/v1/results",
                       json={"job_id": "nope", "name": "x"}).status_code == 404
    assert client.post("/api/v1/results",
                       json={"job_id": job_id, "name": "!!!"}).status_code == 400
    assert client.get("/api/v1/results/does-not-exist/hdf5").status_code == 404

    # delete: gone for good, second delete 404
    assert client.delete("/api/v1/results/ref-a").json() == {"status": "deleted"}
    assert client.get("/api/v1/results/ref-a/hdf5").status_code == 404
    assert client.delete("/api/v1/results/ref-a").status_code == 404


# ── the generic-runner contract (real numpy adapter, composed case) ─────────

#: the SME(level=1) card model — same constructor as the committed SME1
#: fixture, so the warm derivation cache serves the run.py subprocess.
MODEL_CODE = """\
import numpy as np
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.systemmodel.system_model import SystemModel

model = SystemModel.from_model(SME(
    level=1,
    dimension=2,
    closures=[Newtonian(), NavierSlip(), StressFree()],
    boundary_conditions=[BC.Extrapolation(tag="left"), BC.Extrapolation(tag="right")],
    initial_conditions=IC.RP(high=lambda n: np.array([0.0, 2.0] + [0.0] * (n - 2)),
                             low=lambda n: np.array([0.0, 1.0] + [0.0] * (n - 2)),
                             jump_position_x=5.0),
))
"""

SPEC = {
    "meta": {"title": "Generic-runner test",
             "description": "SME(level=1) dam break via run.py"},
    "model": {"class_path": "zoomy_core.model.models.SME",
              "init": {"level": 1, "dimension": 2}, "code": MODEL_CODE},
    "mesh": {"spec": {"type": "create_1d", "domain": [0.0, 10.0],
                      "n_cells": 20}},
    "settings": {"time_end": 0.1, "cfl": 0.45, "output_snapshots": 2,
                 "numpy": {"reconstruction_order": 1}},
    "solver": {"tag": "numpy"},
}


@pytest.fixture
def numpy_client(clean_jobs, jobs_dir):
    """TestClient wired to the REAL NumpyAdapter (generic-runner path)."""
    from zoomy_server.adapters.numpy import NumpyAdapter
    from zoomy_server.server import ZoomyServer

    adapter = NumpyAdapter()
    routes_module.set_adapter(adapter)
    server = ZoomyServer(adapter)
    return TestClient(server.app)


def test_generic_runner_executes_case_run_py(numpy_client, jobs_dir, caplog):
    from zoomy_prepost import compose, parse

    caplog.set_level(logging.INFO, logger="zoomy.adapter")

    case_py = compose(SPEC)
    # the composed case DOES carry a run section (to_folder materializes it)
    assert "solver.solve(" in parse(case_py)["run"]["code"]

    # submit the composed .py via /cases — the GUI's endpoint; the server
    # materializes the folder itself (to_folder -> run.py present)
    r = numpy_client.post("/api/v1/cases", json={"case_py": case_py})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]

    body = _poll_until(numpy_client, job_id,
                       lambda b: b.get("status") in ("complete", "failed"),
                       timeout_s=300.0, step_s=0.5).json()
    assert body["status"] == "complete", body

    # served HDF5 has fields + mesh (the store the GUI viz consumes)
    h5 = numpy_client.get(f"/api/v1/jobs/{job_id}/results/hdf5")
    assert h5.status_code == 200
    p = os.path.join(jobs_dir, "served.h5")
    with open(p, "wb") as f:
        f.write(h5.content)
    import h5py
    with h5py.File(p) as f:
        assert "fields" in f and "mesh" in f
        assert len(f["fields"]) >= 2
        assert int(f["mesh/n_inner_cells"][()]) == 20

    # the case's OWN run.py ran (generic runner, not settings translation)
    assert any("runner: executing case run.py" in rec.message
               for rec in caplog.records)
    run_log = os.path.join(jobs_dir, job_id, "run.log")
    assert os.path.exists(run_log), run_log
    assert "done -> simulation.h5" in open(run_log).read()


def test_registry_and_postprocess_endpoints(client):
    """Fourth + fifth concern (verifier-flagged orphans from test_routes):
    the GUI's card-registry endpoint keeps its schema, and the postprocess
    endpoint fails LOUDLY (503/409) when no postprocess adapter is connected
    — the two live GUI surfaces the three main concerns don't touch."""
    reg = client.get("/api/v1/registry")
    assert reg.status_code == 200
    body = reg.json()
    # schema contract the GUI binds to: tab -> list of cards with unique ids
    assert isinstance(body, dict) and body, "registry must be a non-empty dict"
    ids = [c["id"] for cards in body.values() for c in cards]
    assert ids and len(ids) == len(set(ids)), "card ids must be globally unique"
    assert all("title" in c for cards in body.values() for c in cards)
    # postprocess error paths: with the test's numpy/mock adapter connected,
    # the endpoint must refuse with 409 (wrong adapter tag), and a garbage
    # body is a 422 regardless of adapter state
    assert client.post("/api/v1/postprocess", json={"nope": 1}).status_code == 422
    r = client.post("/api/v1/postprocess",
                    json={"store_b64": "", "steps": []})
    assert r.status_code in (409, 503), r.text
