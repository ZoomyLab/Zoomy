"""Route-level integration tests for the Zoomy server.

Exercises the HTTP surface the JS CLI (Phase 3) will call: health,
registry, job submit + status + HDF5 download, cancel, 404s. Every
test uses the TestClient from conftest with a ``MockSolverAdapter`` in
place of a real backend.
"""
import time

import pytest


def _poll_until(client, job_id, predicate, timeout_s=5.0):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        r = client.get(f"/api/v1/jobs/{job_id}")
        last = r
        if r.status_code == 200 and predicate(r.json()):
            return r
        time.sleep(0.05)
    raise AssertionError(
        f"job {job_id} never satisfied predicate within {timeout_s}s; "
        f"last response: {last.status_code} {last.json() if last else 'n/a'}"
    )


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_returns_ok_with_adapter_tag(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["tag"] == "mock"


# ---------------------------------------------------------------------------
# /jobs (POST)
# ---------------------------------------------------------------------------

def test_post_jobs_returns_job_id(client, case_dir):
    r = client.post("/api/v1/jobs", json={"case_dir": case_dir})
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body
    assert isinstance(body["job_id"], str)
    assert len(body["job_id"]) > 0


def test_post_jobs_rejects_garbage_body(client):
    # Missing the required case_dir field — Pydantic must respond 422.
    r = client.post("/api/v1/jobs", json={"foo": "bar"})
    assert r.status_code == 422


def test_post_jobs_rejects_wrong_type(client):
    # case_dir must be a string.
    r = client.post("/api/v1/jobs", json={"case_dir": 42})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# /jobs/{id} (GET)
# ---------------------------------------------------------------------------

def test_get_jobs_progression_to_complete(client, case_dir):
    job_id = client.post("/api/v1/jobs", json={"case_dir": case_dir}).json()["job_id"]
    # Within the poll window the status must eventually report complete.
    r = _poll_until(
        client, job_id,
        lambda body: body.get("status") in ("complete", "failed"),
        timeout_s=5.0,
    )
    body = r.json()
    assert body["status"] == "complete", body
    assert body["job_id"] == job_id


def test_get_unknown_job_returns_404(client):
    r = client.get("/api/v1/jobs/does-not-exist")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /jobs (GET, list)
# ---------------------------------------------------------------------------

def test_list_jobs_returns_array(client, case_dir):
    assert client.get("/api/v1/jobs").json() == []
    job_id = client.post("/api/v1/jobs", json={"case_dir": case_dir}).json()["job_id"]
    listed = client.get("/api/v1/jobs").json()
    assert isinstance(listed, list)
    assert any(j.get("job_id") == job_id for j in listed)


# ---------------------------------------------------------------------------
# /jobs/{id}/results/hdf5
# ---------------------------------------------------------------------------

def test_get_hdf5_after_complete(client, case_dir):
    job_id = client.post("/api/v1/jobs", json={"case_dir": case_dir}).json()["job_id"]
    _poll_until(client, job_id, lambda b: b.get("status") == "complete", timeout_s=5.0)

    r = client.get(f"/api/v1/jobs/{job_id}/results/hdf5")
    assert r.status_code == 200
    assert len(r.content) > 0
    # Should be served with the HDF5 media type.
    assert "hdf5" in r.headers.get("content-type", "").lower() or \
           "octet-stream" in r.headers.get("content-type", "").lower()


def test_get_hdf5_unknown_job_returns_404(client):
    r = client.get("/api/v1/jobs/nope/results/hdf5")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /jobs/{id} (DELETE — cancel)
# ---------------------------------------------------------------------------

def test_delete_job_returns_cancelled(client, case_dir):
    job_id = client.post("/api/v1/jobs", json={"case_dir": case_dir}).json()["job_id"]
    r = client.delete(f"/api/v1/jobs/{job_id}")
    assert r.status_code == 200
    assert r.json() == {"status": "cancelled"}


def test_delete_unknown_job_returns_404(client):
    r = client.delete("/api/v1/jobs/does-not-exist")
    assert r.status_code == 404


def test_delete_twice_second_is_404(client, case_dir):
    job_id = client.post("/api/v1/jobs", json={"case_dir": case_dir}).json()["job_id"]
    first = client.delete(f"/api/v1/jobs/{job_id}")
    assert first.status_code == 200
    # First DELETE removes from JOBS; second should be 404 (no longer known).
    second = client.delete(f"/api/v1/jobs/{job_id}")
    assert second.status_code == 404


# ---------------------------------------------------------------------------
# /results — the named RESULTS SHELF
# ---------------------------------------------------------------------------

def _run_to_complete(client, case_dir):
    job_id = client.post("/api/v1/jobs", json={"case_dir": case_dir}).json()["job_id"]
    _poll_until(client, job_id, lambda b: b.get("status") == "complete", timeout_s=5.0)
    return job_id


def test_save_result_then_list_and_download(client, case_dir, results_dir):
    job_id = _run_to_complete(client, case_dir)

    r = client.post("/api/v1/results", json={"job_id": job_id, "name": "Ref A"})
    assert r.status_code == 200, r.text
    entry = r.json()
    assert entry["name"] == "ref-a"          # slugged
    assert entry["size"] > 0
    assert "created" in entry

    listing = client.get("/api/v1/results").json()
    assert any(e["name"] == "ref-a" for e in listing)

    dl = client.get("/api/v1/results/ref-a/hdf5")
    assert dl.status_code == 200
    assert len(dl.content) > 0
    # Same bytes as the job's own HDF5 download.
    orig = client.get(f"/api/v1/jobs/{job_id}/results/hdf5")
    assert dl.content == orig.content


def test_save_result_slug_lookup_by_original_name(client, case_dir, results_dir):
    """A result saved under a fancy name is fetchable by its slug OR the
    original (both slugify to the same key)."""
    job_id = _run_to_complete(client, case_dir)
    client.post("/api/v1/results", json={"job_id": job_id, "name": "SWE Reference!"})
    assert client.get("/api/v1/results/SWE Reference!/hdf5").status_code == 200
    assert client.get("/api/v1/results/swe-reference/hdf5").status_code == 200


def test_save_result_unknown_job_404(client, results_dir):
    r = client.post("/api/v1/results", json={"job_id": "nope", "name": "x"})
    assert r.status_code == 404


def test_save_result_empty_name_400(client, case_dir, results_dir):
    job_id = _run_to_complete(client, case_dir)
    r = client.post("/api/v1/results", json={"job_id": job_id, "name": "!!!"})
    assert r.status_code == 400


def test_download_unknown_result_404(client, results_dir):
    assert client.get("/api/v1/results/does-not-exist/hdf5").status_code == 404


def test_delete_result(client, case_dir, results_dir):
    job_id = _run_to_complete(client, case_dir)
    client.post("/api/v1/results", json={"job_id": job_id, "name": "ref-a"})
    r = client.delete("/api/v1/results/ref-a")
    assert r.status_code == 200
    assert r.json() == {"status": "deleted"}
    # Now gone.
    assert client.get("/api/v1/results/ref-a/hdf5").status_code == 404
    assert client.delete("/api/v1/results/ref-a").status_code == 404


# ---------------------------------------------------------------------------
# /jobs/{id}/artifacts — chain output discovery + download
# ---------------------------------------------------------------------------

def test_list_artifacts_after_complete(client, case_dir):
    job_id = _run_to_complete(client, case_dir)
    r = client.get(f"/api/v1/jobs/{job_id}/artifacts")
    assert r.status_code == 200
    names = [a["name"] for a in r.json()["artifacts"]]
    assert "simulation.h5" in names
    # progress.json is bookkeeping, never an artifact.
    assert "progress.json" not in names
    assert all(a["size"] > 0 for a in r.json()["artifacts"])


def test_download_artifact_matches_hdf5(client, case_dir):
    job_id = _run_to_complete(client, case_dir)
    art = client.get(f"/api/v1/jobs/{job_id}/artifacts/simulation.h5")
    assert art.status_code == 200
    assert art.content == client.get(f"/api/v1/jobs/{job_id}/results/hdf5").content


def test_download_artifact_traversal_safe(client, case_dir):
    """A ``../`` name is reduced to its basename — no escape from the job dir."""
    job_id = _run_to_complete(client, case_dir)
    r = client.get(f"/api/v1/jobs/{job_id}/artifacts/..%2f..%2fetc%2fpasswd")
    assert r.status_code == 404


def test_list_artifacts_unknown_job_404(client):
    assert client.get("/api/v1/jobs/nope/artifacts").status_code == 404


# ---------------------------------------------------------------------------
# /postprocess — the chain-routing endpoint (only a postprocess adapter serves)
# ---------------------------------------------------------------------------

def test_postprocess_requires_postprocess_adapter(client):
    """The mock adapter (tag 'mock') is not a postprocess adapter -> 409."""
    import base64
    body = {"store_b64": base64.b64encode(b"\x89HDF\r\n\x1a\n").decode(),
            "steps": ["to_h5"]}
    r = client.post("/api/v1/postprocess", json=body)
    assert r.status_code == 409


def test_postprocess_rejects_garbage_body(client):
    # store_b64 is required — Pydantic responds 422.
    assert client.post("/api/v1/postprocess", json={"steps": []}).status_code == 422


# ---------------------------------------------------------------------------
# /registry
# ---------------------------------------------------------------------------

def test_registry_returns_known_schema(client):
    r = client.get("/api/v1/registry")
    assert r.status_code == 200
    body = r.json()
    # The registry is a merged dict of card manifests; it must at least be
    # a dict/list (not an error). The concrete keys depend on
    # build_registry's implementation; we assert structural validity only.
    assert isinstance(body, (dict, list))
