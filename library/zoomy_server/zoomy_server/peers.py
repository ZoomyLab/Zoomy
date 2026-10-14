"""A job's view of the other backends: a coupled case whose participants run in different images
(the VOF on the foam backend, the reduced model on the JAX backend) submits its other
participants to their backends' zoomy-server, as the GUI submits a case, and the participants
meet through preCICE in a directory every backend sees.

Deployment (environment of each server):

    ZOOMY_PEERS        the backends a job may submit to: a JSON object `{tag: url}`, or the path
                       of a JSON file holding it (read at every call: a backend that comes up
                       later registers by rewriting the file)
    ZOOMY_COUPLE_DIR   a directory every backend sees at this same path (apptainer: a bind at
                       the same path in each container; on one host without containment, any
                       shared directory); default `~/.cache/zoomy/couple`
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def peers() -> dict:
    """`{tag: url}` from `ZOOMY_PEERS` (a JSON object or the path of a file holding one)."""
    raw = os.environ.get("ZOOMY_PEERS", "").strip()
    if not raw:
        return {}
    if not raw.startswith("{"):
        p = Path(raw)
        raw = p.read_text() if p.exists() else "{}"
    return {str(k): str(v).rstrip("/") for k, v in json.loads(raw).items()}


def peer_url(tag: str) -> str:
    """The URL of the backend serving `tag` (a GUI solver tag: `jax`, `numpy`, `OpenFOAM`, ...)."""
    known = peers()
    for k, v in known.items():
        if k.lower() == tag.lower():
            return v
    raise RuntimeError(f"no {tag!r} backend known to this server: ZOOMY_PEERS names "
                       f"{sorted(known) or 'none'} (set it to {{\"{tag}\": \"http://host:port\"}})")


def couple_dir(name: str | None = None) -> Path:
    """A fresh directory under `ZOOMY_COUPLE_DIR` (every backend sees it at this path) for one
    coupled run's preCICE exchange and configuration."""
    root = Path(os.environ.get("ZOOMY_COUPLE_DIR")
                or Path.home() / ".cache" / "zoomy" / "couple")
    d = root / (name or uuid.uuid4().hex[:12])
    d.mkdir(parents=True, exist_ok=False)
    return d


class Peer:
    """The GUI's HTTP protocol against one backend (`zoomy_cli` HttpAdapter.submitCase)."""

    def __init__(self, url: str, poll_s: float = 2.0):
        self.url, self.poll_s = url.rstrip("/"), poll_s

    def _req(self, path, data=None, method=None, timeout=60):
        body = None if data is None else json.dumps(data).encode()
        req = urllib.request.Request(self.url + path, data=body, method=method,
                                     headers={"Content-Type": "application/json"} if body else {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    def submit(self, case_py: str) -> str:
        """POST a composed case (`/api/v1/cases`); returns the job id."""
        return json.loads(self._req("/api/v1/cases", {"case_py": case_py, "mesh_b64": None,
                                                      "mesh_name": None}))["job_id"]

    def status(self, job: str) -> dict:
        """The job's status (`running`, `complete`, `failed`, `cancelled`)."""
        return json.loads(self._req(f"/api/v1/jobs/{job}"))

    def wait(self, job: str, timeout: float | None = None, alive=None) -> dict:
        """Poll until the job ends; `alive()` False (a local partner died) cancels it."""
        t0 = time.monotonic()
        while True:
            try:
                s = self.status(job)
                if s.get("status") in ("complete", "failed", "cancelled"):
                    return s
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                pass
            if (alive is not None and not alive()) or (
                    timeout is not None and time.monotonic() - t0 > timeout):
                self.cancel(job)
                return {"status": "cancelled", "job_id": job}
            time.sleep(self.poll_s)

    def store(self, job: str, dest) -> Path:
        """The job's `simulation.h5`, written to `dest`."""
        dest = Path(dest)
        dest.write_bytes(self._req(f"/api/v1/jobs/{job}/results/hdf5", timeout=300))
        return dest

    def cancel(self, job: str):
        """Cancel the job (best effort)."""
        try:
            self._req(f"/api/v1/jobs/{job}", method="DELETE")
        except (OSError, urllib.error.URLError):
            pass


def compose(model: str, mesh: str, settings: dict, run: str, title: str = "participant") -> str:
    """A composed case (`model.py`, `mesh.py`, `settings.json`, `run.py` cells) as the GUI
    exports it (`zoomy_prepost.case.from_folder`)."""
    import tempfile
    from zoomy_prepost.case import from_folder
    with tempfile.TemporaryDirectory() as d:
        d = Path(d) / title
        d.mkdir()
        (d / "model.py").write_text(model)
        (d / "mesh.py").write_text(mesh)
        (d / "settings.json").write_text(json.dumps(settings, indent=2))
        (d / "run.py").write_text(run)
        return from_folder(str(d))


__all__ = ["Peer", "compose", "couple_dir", "peer_url", "peers"]
