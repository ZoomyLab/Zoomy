import json
import logging
import os
import shutil
import subprocess
import tempfile

from zoomy_server.adapter import SolverAdapter

logger = logging.getLogger("zoomy.adapter.amrex")

AMREX_HOME = os.environ.get("AMREX_HOME", "/opt/amrex")


class AmrexAdapter(SolverAdapter):
    tag = "amrex"

    def solve(self, case, output_dir, on_progress):
        from zoomy_server._registry import resolve_model
        from zoomy_core.transformation.to_amrex import AmrexModel, AmrexNumerics
        from zoomy_core.fvm.symbolic_numerics import Numerics

        model_spec = case["model"]
        solver_spec = case.get("solver", {})
        mesh_spec = case.get("mesh", {})

        # Resolve and instantiate the model
        cls = resolve_model(model_spec["class_path"])
        model = cls(**model_spec.get("init", {}))

        # Create a build directory
        build_dir = tempfile.mkdtemp(prefix="zoomy_amrex_")
        src_dir = os.path.join(build_dir, "Source")
        exec_dir = os.path.join(build_dir, "Exec")
        os.makedirs(src_dir)
        os.makedirs(exec_dir)

        try:
            # Generate Model.H and Numerics.H
            logger.info("Generating AMReX code...")
            numerics = Numerics(model)
            am = AmrexModel(model)
            an = AmrexNumerics(numerics)
            with open(os.path.join(src_dir, "Model.H"), "w") as f:
                f.write(am.create_code())
            with open(os.path.join(src_dir, "Numerics.H"), "w") as f:
                f.write(an.create_code())

            # Copy solver source files from zoomy_amrex
            zoomy_amrex_src = self._find_zoomy_amrex_source()
            for f in ["main.cpp", "make_rhs.H", "init_solution.cpp",
                       "write_plotfiles.cpp", "plotfile_utils.H",
                       "plotfile_utils.cpp", "constants.H", "Make.package"]:
                src = os.path.join(zoomy_amrex_src, f)
                if os.path.exists(src):
                    shutil.copy2(src, src_dir)

            # Write GNUmakefile
            dim = model_spec.get("init", {}).get("dimension", 2)
            self._write_makefile(exec_dir, src_dir, dim)

            # Write inputs file
            self._write_inputs(exec_dir, mesh_spec, solver_spec, output_dir)

            # Build
            logger.info("Building AMReX solver...")
            result = subprocess.run(
                ["make", f"-j{os.cpu_count() or 2}"],
                cwd=exec_dir, capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                logger.error(f"Build failed:\n{result.stderr}")
                raise RuntimeError(f"AMReX build failed: {result.stderr[-500:]}")

            # Find the executable
            exe = self._find_executable(exec_dir)
            if not exe:
                raise RuntimeError("No AMReX executable found after build")

            # Run
            logger.info(f"Running AMReX solver: {exe}")
            proc = subprocess.Popen(
                [exe, "inputs"],
                cwd=exec_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )

            time_end = solver_spec.get("time_end", 1.0)
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("Step"):
                    parts = line.split()
                    try:
                        iteration = int(parts[1])
                        time_val = float(parts[parts.index("time:") + 1].rstrip("s"))
                        dt_val = float(parts[parts.index("dt:") + 1])
                        on_progress(iteration, time_val, dt_val)
                    except (ValueError, IndexError):
                        pass
                logger.debug(line)

            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"AMReX solver exited with code {proc.returncode}")

            on_progress(-1, time_end, 0.0)

        finally:
            shutil.rmtree(build_dir, ignore_errors=True)

    def list_models(self):
        try:
            from zoomy_server._registry import scan_models
            return scan_models()
        except Exception:
            return []

    def _find_zoomy_amrex_source(self):
        candidates = [
            os.path.join(os.environ.get("ZOOMY_ROOT", ""), "library/zoomy_amrex/Source"),
            "/workspace/library/zoomy_amrex/Source",
            os.path.join(os.path.dirname(__file__), "../../../../zoomy_amrex/Source"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                return c
        raise FileNotFoundError("Cannot find zoomy_amrex/Source directory")

    def _write_makefile(self, exec_dir, src_dir, dim):
        content = f"""AMREX_HOME ?= {AMREX_HOME}
DEBUG        = FALSE
USE_MPI      = TRUE
USE_OMP      = FALSE
COMP         = gnu
DIM          = {dim}

include $(AMREX_HOME)/Tools/GNUMake/Make.defs
include {src_dir}/Make.package
VPATH_LOCATIONS  += {src_dir}
INCLUDE_LOCATIONS += {src_dir}
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Tools/GNUMake/Make.rules
"""
        with open(os.path.join(exec_dir, "GNUmakefile"), "w") as f:
            f.write(content)

    def _write_inputs(self, exec_dir, mesh_spec, solver_spec, output_dir):
        nx = mesh_spec.get("nx", mesh_spec.get("n_cells", 100))
        ny = mesh_spec.get("ny", 1)
        domain = mesh_spec.get("domain", [0.0, 1.0])
        x0 = domain[0] if len(domain) >= 2 else 0.0
        x1 = domain[1] if len(domain) >= 2 else 1.0
        y0 = mesh_spec.get("y_min", 0.0)
        y1 = mesh_spec.get("y_max", 1.0)

        content = f"""geometry.n_cell_x = {nx}
geometry.n_cell_y = {ny}
geometry.phy_bb_x0 = {x0}
geometry.phy_bb_x1 = {x1}
geometry.phy_bb_y0 = {y0}
geometry.phy_bb_y1 = {y1}

solver.time_end = {solver_spec.get('time_end', 1.0)}
solver.cfl = {solver_spec.get('cfl', 0.5)}
solver.spatial_order = {solver_spec.get('spatial_order', 1)}

output.plot_dt_interval = {solver_spec.get('time_end', 1.0) / max(solver_spec.get('output_snapshots', 10), 1)}
output.identifier = 0
"""
        with open(os.path.join(exec_dir, "inputs"), "w") as f:
            f.write(content)

    def _find_executable(self, exec_dir):
        for f in os.listdir(exec_dir):
            path = os.path.join(exec_dir, f)
            if os.path.isfile(path) and os.access(path, os.X_OK) and f.startswith("main"):
                return path
        return None
