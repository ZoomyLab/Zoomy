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
