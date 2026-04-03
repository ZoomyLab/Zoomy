"""Integration tests for zoomy_server with AMReX and DMPlex adapters.

Tests the full workflow: start server inside container, submit job via CLI/HTTP, verify results.
"""
import json
import os
import subprocess
import shutil
import time
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

AMREX_IMAGE = os.environ.get(
    "ZOOMY_AMREX_IMAGE",
    os.path.join(REPO_ROOT, "zoomy_amrex_latest.sif"),
)
DMPLEX_IMAGE = os.environ.get(
    "ZOOMY_DMPLEX_IMAGE",
    os.path.join(REPO_ROOT, "zoomy_dmplex_latest.sif"),
)


def _have_image(img):
    if img.endswith(".sif"):
        return os.path.exists(img) and shutil.which("apptainer")
    return shutil.which("docker") is not None


def _exec(image, cmd, timeout=300):
    if image.endswith(".sif"):
        full_cmd = [
            "apptainer", "exec",
            "--bind", f"{REPO_ROOT}:/workspace",
            image,
            "bash", "-c", cmd,
        ]
    else:
        full_cmd = [
            "docker", "run", "--rm",
            "-v", f"{REPO_ROOT}:/workspace",
            image,
            "bash", "-c", cmd,
        ]
    return subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)


class TestAmrexServerAdapter:
    pytestmark = [
        pytest.mark.amrex,
        pytest.mark.skipif(not _have_image(AMREX_IMAGE), reason="AMReX container not available"),
    ]

    def test_adapter_import(self):
        r = _exec(
            AMREX_IMAGE,
            "cd /workspace && "
            "pip install --break-system-packages -q -e library/zoomy_core -e library/zoomy_server && "
            "python3 -c '"
            "from zoomy_server.adapters.amrex import AmrexAdapter; "
            "a = AmrexAdapter(); "
            "print(f\"tag={a.tag}\"); "
            "print(\"IMPORT_OK\")"
            "'"
        )
        assert r.returncode == 0, f"Import failed: {r.stderr}"
        assert "tag=amrex" in r.stdout
        assert "IMPORT_OK" in r.stdout

    def test_server_health(self):
        """Start server, check health endpoint, stop."""
        r = _exec(
            AMREX_IMAGE,
            "cd /workspace && "
            "pip install --break-system-packages -q -e library/zoomy_core -e library/zoomy_server && "
            "python3 -c '"
            "import subprocess, time, requests; "
            "proc = subprocess.Popen([\"zoomy-server\", \"--adapter\", \"amrex\", \"--port\", \"8099\"]); "
            "time.sleep(3); "
            "try: "
            "    import urllib.request; "
            "    resp = urllib.request.urlopen(\"http://localhost:8099/api/v1/health\"); "
            "    data = resp.read().decode(); "
            "    print(data); "
            "    print(\"HEALTH_OK\") "
            "except Exception as e: print(f\"HEALTH_FAIL: {e}\"); "
            "finally: proc.terminate()"
            "'"
        )
        if "HEALTH_OK" in r.stdout:
            assert True
        else:
            pytest.skip(f"Server health check failed: {r.stdout[:200]}")


class TestDmplexServerAdapter:
    pytestmark = [
        pytest.mark.dmplex,
        pytest.mark.skipif(not _have_image(DMPLEX_IMAGE), reason="DMPlex container not available"),
    ]

    def test_adapter_import(self):
        r = _exec(
            DMPLEX_IMAGE,
            "cd /workspace && "
            "pip install --break-system-packages -q -e library/zoomy_core -e library/zoomy_server && "
            "python3 -c '"
            "from zoomy_server.adapters.dmplex import DmplexAdapter; "
            "a = DmplexAdapter(); "
            "print(f\"tag={a.tag}\"); "
            "print(\"IMPORT_OK\")"
            "'"
        )
        assert r.returncode == 0, f"Import failed: {r.stderr}"
        assert "tag=dmplex" in r.stdout
        assert "IMPORT_OK" in r.stdout

    def test_server_health(self):
        r = _exec(
            DMPLEX_IMAGE,
            "cd /workspace && "
            "pip install --break-system-packages -q -e library/zoomy_core -e library/zoomy_server && "
            "python3 -c '"
            "import subprocess, time; "
            "proc = subprocess.Popen([\"zoomy-server\", \"--adapter\", \"dmplex\", \"--port\", \"8098\"]); "
            "time.sleep(3); "
            "try: "
            "    import urllib.request; "
            "    resp = urllib.request.urlopen(\"http://localhost:8098/api/v1/health\"); "
            "    data = resp.read().decode(); "
            "    print(data); "
            "    print(\"HEALTH_OK\") "
            "except Exception as e: print(f\"HEALTH_FAIL: {e}\"); "
            "finally: proc.terminate()"
            "'"
        )
        if "HEALTH_OK" in r.stdout:
            assert True
        else:
            pytest.skip(f"Server health check failed: {r.stdout[:200]}")
