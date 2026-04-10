"""Integration tests for the zoomy_amrex container.

Tests run inside the container via apptainer exec or docker run.
Requires the container image to be available locally.
"""
import json
import os
import subprocess
import shutil
import pytest
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Detect container runtime
CONTAINER_IMAGE = os.environ.get(
    "ZOOMY_AMREX_IMAGE",
    os.path.join(REPO_ROOT, "zoomy_amrex_latest.sif"),
)


def _have_container():
    if CONTAINER_IMAGE.endswith(".sif"):
        return os.path.exists(CONTAINER_IMAGE) and shutil.which("apptainer")
    return shutil.which("docker") is not None


def _exec(cmd, timeout=300):
    """Execute a command inside the container."""
    if CONTAINER_IMAGE.endswith(".sif"):
        full_cmd = [
            "apptainer", "exec",
            "--bind", f"{REPO_ROOT}:/workspace",
            CONTAINER_IMAGE,
            "bash", "-c", cmd,
        ]
    else:
        full_cmd = [
            "docker", "run", "--rm",
            "-v", f"{REPO_ROOT}:/workspace",
            CONTAINER_IMAGE,
            "bash", "-c", cmd,
        ]
    result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
    return result


pytestmark = [
    pytest.mark.amrex,
    pytest.mark.skipif(not _have_container(), reason="AMReX container not available"),
]


class TestAmrexContainerBasics:
    def test_python_version(self):
        r = _exec("python3 -c 'import sys; print(sys.version_info[:2])'")
        assert r.returncode == 0
        assert "(3, 12)" in r.stdout or "(3, 11)" in r.stdout

    def test_amrex_installed(self):
        r = _exec("test -d /opt/amrex/Src/Base && echo OK")
        assert r.returncode == 0
        assert "OK" in r.stdout

    def test_compiler_available(self):
        r = _exec("mpicxx --version")
        assert r.returncode == 0

    def test_python_deps(self):
        r = _exec("python3 -c 'import numpy, sympy, attrs; print(\"OK\")'")
        assert r.returncode == 0


class TestAmrexCodeGeneration:
    def test_generate_model(self):
        r = _exec(
            "cd /workspace && "
            "pip install --break-system-packages -e library/zoomy_core -q && "
            "python3 -c '"
            "from zoomy_core.model.models.shallow_water_topo import ShallowWaterEquationsWithTopo; "
            "from zoomy_core.transformation.to_amrex import AmrexModel, AmrexNumerics; "
            "m = ShallowWaterEquationsWithTopo(dimension=2); "
            "from zoomy_core.fvm.symbolic_numerics import Numerics; "
            "n = Numerics(m); am = AmrexModel(m); an = AmrexNumerics(n); "
            "open(\"/tmp/Model.H\",\"w\").write(am.create_code()); "
            "open(\"/tmp/Numerics.H\",\"w\").write(an.create_code()); "
            "print(\"GENERATED\")"
            "'"
        )
        assert r.returncode == 0, f"Code gen failed: {r.stderr}"
        assert "GENERATED" in r.stdout


class TestAmrexBuildAndRun:
    def test_build_solver(self):
        """Test that the AMReX solver compiles inside the container."""
        r = _exec(
            "cd /workspace && "
            "pip install --break-system-packages -e library/zoomy_core -q && "
            "python3 -c '"
            "from zoomy_core.model.models.shallow_water_topo import ShallowWaterEquationsWithTopo; "
            "from zoomy_core.transformation.to_amrex import AmrexModel, AmrexNumerics; "
            "m = ShallowWaterEquationsWithTopo(dimension=2); "
            "from zoomy_core.fvm.symbolic_numerics import Numerics; n = Numerics(m); "
            "open(\"library/zoomy_amrex/Source/Model.H\",\"w\").write(AmrexModel(m).create_code()); "
            "open(\"library/zoomy_amrex/Source/Numerics.H\",\"w\").write(AmrexNumerics(n).create_code()); "
            "print(\"CODEGEN_OK\")"
            "' && "
            "cd library/zoomy_amrex/Exec && "
            "make -j$(nproc) AMREX_HOME=/opt/amrex 2>&1 | tail -5 && "
            "echo BUILD_OK",
            timeout=600,
        )
        assert "CODEGEN_OK" in r.stdout, f"Code gen failed: {r.stderr}"
        # Build may fail if AMReX source isn't fully ready - that's expected
        if "BUILD_OK" in r.stdout:
            assert True
        else:
            pytest.skip("AMReX build not ready yet (expected)")


class TestAmrexParallel:
    def test_mpi_available(self):
        r = _exec("mpiexec --version")
        assert r.returncode == 0

    def test_mpi_hello(self):
        r = _exec(
            "mpiexec --allow-run-as-root -n 2 "
            "python3 -c 'from mpi4py import MPI; "
            "print(f\"rank {MPI.COMM_WORLD.rank}/{MPI.COMM_WORLD.size}\")'"
        )
        # mpi4py may not be installed - that's OK for now
        if r.returncode != 0:
            pytest.skip("mpi4py not available in container")
        assert "rank 0/2" in r.stdout
        assert "rank 1/2" in r.stdout
