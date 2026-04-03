import json
import logging
import os
import shutil
import subprocess
import tempfile

from zoomy_server.adapter import SolverAdapter

logger = logging.getLogger("zoomy.adapter.dmplex")

PETSC_DIR = os.environ.get("PETSC_DIR", "/opt/petsc")
PETSC_ARCH = os.environ.get("PETSC_ARCH", "arch-container-opt")


class DmplexAdapter(SolverAdapter):
    tag = "dmplex"

    def solve(self, case, output_dir, on_progress):
        from zoomy_server._registry import resolve_model
        from zoomy_core.transformation.to_c import CppModel, CppNumerics
        from zoomy_core.fvm.symbolic_numerics import Numerics

        model_spec = case["model"]
        solver_spec = case.get("solver", {})
        mesh_spec = case.get("mesh", {})

        # Resolve and instantiate the model
        cls = resolve_model(model_spec["class_path"])
        model = cls(**model_spec.get("init", {}))

        # Create a build directory
        build_dir = tempfile.mkdtemp(prefix="zoomy_dmplex_")

        try:
            # Generate Model.H and Numerics.H
            logger.info("Generating DMPlex code...")
            numerics = Numerics(model)
            cm = CppModel(model)
            cn = CppNumerics(numerics)
            with open(os.path.join(build_dir, "Model.H"), "w") as f:
                f.write(cm.create_code())
            with open(os.path.join(build_dir, "Numerics.H"), "w") as f:
                f.write(cn.create_code())

            # Copy solver source files from zoomy_dmplex
            zoomy_dmplex_src = self._find_zoomy_dmplex_source()
            for f in os.listdir(zoomy_dmplex_src):
                if f.endswith((".cpp", ".hpp", ".H", ".h")):
                    src = os.path.join(zoomy_dmplex_src, f)
                    dst = os.path.join(build_dir, f)
                    # Don't overwrite generated Model.H / Numerics.H
                    if f not in ("Model.H", "Numerics.H"):
                        shutil.copy2(src, dst)

            # Copy Makefile
            makefile_src = os.path.join(zoomy_dmplex_src, "Makefile")
            if os.path.exists(makefile_src):
                shutil.copy2(makefile_src, build_dir)

            # Write settings.json
            mesh_path = mesh_spec.get("mesh_path", "")
            self._write_settings(build_dir, solver_spec, mesh_path, output_dir)

            # Build (CPU only)
            logger.info("Building DMPlex solver...")
            env = os.environ.copy()
            env["PETSC_DIR"] = PETSC_DIR
            env["PETSC_ARCH"] = PETSC_ARCH
            result = subprocess.run(
                ["make", "CPU", f"-j{os.cpu_count() or 2}"],
                cwd=build_dir, capture_output=True, text=True, timeout=300, env=env
            )
            if result.returncode != 0:
                logger.error(f"Build failed:\n{result.stderr}")
                raise RuntimeError(f"DMPlex build failed: {result.stderr[-500:]}")

            # Run
            n_procs = case.get("n_procs", 1)
            exe = os.path.join(build_dir, "solver_cpu")
            if not os.path.exists(exe):
                raise RuntimeError("solver_cpu not found after build")

            logger.info(f"Running DMPlex solver with {n_procs} process(es)")
            cmd = ["mpiexec", "-n", str(n_procs), "--allow-run-as-root",
                   exe] if n_procs > 1 else [exe]

            proc = subprocess.Popen(
                cmd, cwd=build_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )

            time_end = solver_spec.get("time_end", 1.0)
            for line in proc.stdout:
                line = line.strip()
                # Parse progress from DMPlex output (format: "[TS] Step X, t = Y, dt = Z")
                if "Step" in line and "t =" in line:
                    try:
                        parts = line.split()
                        step_idx = next(i for i, p in enumerate(parts) if p == "Step") + 1
                        iteration = int(parts[step_idx].rstrip(","))
                        t_idx = next(i for i, p in enumerate(parts) if p == "t") + 2
                        time_val = float(parts[t_idx].rstrip(","))
                        dt_idx = next(i for i, p in enumerate(parts) if p == "dt") + 2
                        dt_val = float(parts[dt_idx].rstrip(","))
                        on_progress(iteration, time_val, dt_val)
                    except (ValueError, IndexError, StopIteration):
                        pass
                logger.debug(line)

            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"DMPlex solver exited with code {proc.returncode}")

            on_progress(-1, time_end, 0.0)

        finally:
            shutil.rmtree(build_dir, ignore_errors=True)

    def list_models(self):
        try:
            from zoomy_server._registry import scan_models
            return scan_models()
        except Exception:
            return []

    def _find_zoomy_dmplex_source(self):
        candidates = [
            os.path.join(os.environ.get("ZOOMY_ROOT", ""), "library/zoomy_dmplex"),
            "/workspace/library/zoomy_dmplex",
            os.path.join(os.path.dirname(__file__), "../../../../zoomy_dmplex"),
        ]
        for c in candidates:
            if os.path.isdir(c) and os.path.exists(os.path.join(c, "main.cpp")):
                return c
        raise FileNotFoundError("Cannot find zoomy_dmplex source directory")

    def _write_settings(self, build_dir, solver_spec, mesh_path, output_dir):
        settings = {
            "name": "zoomy_server_run",
            "io": {
                "directory": output_dir,
                "filename": "simulation",
                "snapshots": solver_spec.get("output_snapshots", 10),
                "snapshot_logic": "interpolate",
                "clean_directory": True,
                "mesh_path": mesh_path,
                "write_3d": False,
            },
            "solver": {
                "t_end": solver_spec.get("time_end", 1.0),
                "cfl": solver_spec.get("cfl", 0.5),
                "reconstruction_order": solver_spec.get("spatial_order", 1),
            },
        }
        with open(os.path.join(build_dir, "settings.json"), "w") as f:
            json.dump(settings, f, indent=4)
