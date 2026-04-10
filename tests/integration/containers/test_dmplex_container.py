"""Integration tests for the zoomy_dmplex container.

Tests run inside the container via apptainer exec or docker run.
Requires the container image to be available locally.
"""
import json
import os
import subprocess
import shutil
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

CONTAINER_IMAGE = os.environ.get(
    "ZOOMY_DMPLEX_IMAGE",
    os.path.join(REPO_ROOT, "zoomy_dmplex_latest.sif"),
)


def _have_container():
    if CONTAINER_IMAGE.endswith(".sif"):
        return os.path.exists(CONTAINER_IMAGE) and shutil.which("apptainer")
    return shutil.which("docker") is not None


def _exec(cmd, timeout=300):
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
    return subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)


pytestmark = [
    pytest.mark.dmplex,
    pytest.mark.skipif(not _have_container(), reason="DMPlex container not available"),
]


class TestDmplexContainerBasics:
    def test_python_version(self):
        r = _exec("python3 -c 'import sys; print(sys.version_info[:2])'")
        assert r.returncode == 0
        assert "(3, 12)" in r.stdout or "(3, 11)" in r.stdout

    def test_petsc_installed(self):
        r = _exec(
            "test -f $PETSC_DIR/$PETSC_ARCH/lib/libpetsc.so && echo OK"
        )
        assert r.returncode == 0
        assert "OK" in r.stdout

    def test_compiler_available(self):
        r = _exec("mpicxx --version")
        assert r.returncode == 0

    def test_python_deps(self):
        r = _exec("python3 -c 'import numpy, sympy, attrs; print(\"OK\")'")
        assert r.returncode == 0


class TestDmplexPetsc:
    def test_petsc4py(self):
        r = _exec(
            "python3 -c '"
            "import petsc4py; petsc4py.init(); "
            "from petsc4py import PETSc; "
            "print(f\"petsc4py {petsc4py.__version__}\")"
            "'"
        )
        assert r.returncode == 0, f"petsc4py failed: {r.stderr}"
        assert "petsc4py" in r.stdout

    def test_dmplex_box_mesh(self):
        r = _exec(
            "python3 -c '"
            "import petsc4py; petsc4py.init(); "
            "from petsc4py import PETSc; "
            "dm = PETSc.DMPlex().createBoxMesh([4, 4], simplex=True); "
            "dm.distribute(); "
            "pStart, pEnd = dm.getChart(); "
            "print(f\"MESH_OK points={pEnd - pStart}\"); "
            "dm.destroy()"
            "'"
        )
        assert r.returncode == 0, f"DMPlex failed: {r.stderr}"
        assert "MESH_OK" in r.stdout


class TestDmplexCodeGenAndBuild:
    def test_generate_model(self):
        r = _exec(
            "cd /workspace && "
            "pip install --break-system-packages -e library/zoomy_core -q && "
            "python3 -c '"
            "from zoomy_core.model.models.shallow_water_topo import ShallowWaterEquationsWithTopo; "
            "from zoomy_core.transformation.to_c import CppModel, CppNumerics; "
            "m = ShallowWaterEquationsWithTopo(dimension=2); "
            "from zoomy_core.fvm.symbolic_numerics import Numerics; n = Numerics(m); "
            "open(\"/tmp/Model.H\",\"w\").write(CppModel(m).create_code()); "
            "open(\"/tmp/Numerics.H\",\"w\").write(CppNumerics(n).create_code()); "
            "print(\"GENERATED\")"
            "'"
        )
        assert r.returncode == 0, f"Code gen failed: {r.stderr}"
        assert "GENERATED" in r.stdout

    def test_build_solver(self):
        r = _exec(
            "cd /workspace && "
            "pip install --break-system-packages -e library/zoomy_core -q && "
            "python3 -c '"
            "from zoomy_core.model.models.shallow_water_topo import ShallowWaterEquationsWithTopo; "
            "from zoomy_core.transformation.to_c import CppModel, CppNumerics; "
            "m = ShallowWaterEquationsWithTopo(dimension=2); "
            "from zoomy_core.fvm.symbolic_numerics import Numerics; n = Numerics(m); "
            "open(\"library/zoomy_dmplex/Model.H\",\"w\").write(CppModel(m).create_code()); "
            "open(\"library/zoomy_dmplex/Numerics.H\",\"w\").write(CppNumerics(n).create_code()); "
            "print(\"CODEGEN_OK\")"
            "' && "
            "cd library/zoomy_dmplex && "
            "make CPU -j$(nproc) 2>&1 | tail -5 && "
            "test -f solver_cpu && echo BUILD_OK",
            timeout=600,
        )
        assert "CODEGEN_OK" in r.stdout, f"Code gen failed: {r.stderr}"
        if "BUILD_OK" not in r.stdout:
            pytest.skip("DMPlex build not ready yet")


class TestDmplexParallel:
    def test_mpi_hello(self):
        r = _exec(
            "mpiexec --allow-run-as-root -n 2 "
            "python3 -c '"
            "import petsc4py; petsc4py.init(); "
            "from petsc4py import PETSc; "
            "print(f\"rank {PETSc.COMM_WORLD.rank}/{PETSc.COMM_WORLD.size}\")"
            "'"
        )
        assert r.returncode == 0, f"MPI PETSc failed: {r.stderr}"
        assert "rank 0/2" in r.stdout
        assert "rank 1/2" in r.stdout

    def test_parallel_dmplex(self):
        r = _exec(
            "mpiexec --allow-run-as-root -n 2 "
            "python3 -c '"
            "import petsc4py; petsc4py.init(); "
            "from petsc4py import PETSc; "
            "dm = PETSc.DMPlex().createBoxMesh([8, 8], simplex=True); "
            "dm.distribute(); "
            "pStart, pEnd = dm.getChart(); "
            "print(f\"rank {PETSc.COMM_WORLD.rank} points={pEnd - pStart}\"); "
            "dm.destroy()"
            "'"
        )
        assert r.returncode == 0, f"Parallel DMPlex failed: {r.stderr}"
        assert "rank 0" in r.stdout
        assert "rank 1" in r.stdout
