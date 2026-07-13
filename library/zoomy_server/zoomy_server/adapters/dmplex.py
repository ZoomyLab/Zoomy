"""DMPlex solver adapter — C++ codegen + PETSc build + MPI run.

Case folder: model.py, numerics.py, mesh.py, settings.json

Server workflow:
  1. Execute mesh.py → mesh.msh
  2. Import model.py, numerics.py
  3. Transform → Model.H, Numerics.H
  4. Compile with PETSc
  5. Run with MPI
"""

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

    def solve(self, case_dir, output_dir, on_progress):
        # Generic runner: a composed case carries its own run.py (the ## Run
        # section) — execute THAT instead of translating settings into solver
        # calls. Legacy case folders (no run.py) fall through unchanged.
        # (Inert until the dmplex backend wrapper class lands in run.py form.)
        if os.path.exists(os.path.join(case_dir, "run.py")):
            self.run_mesh_script(case_dir)
            self.run_case_script(case_dir, output_dir, on_progress)
            return

        from zoomy_core.transformation.to_c import CppModel, CppNumerics

        # 1. Execute mesh.py (preprocessing)
        self.run_mesh_script(case_dir)

        # 2. Load settings
        settings = self.load_settings(case_dir)

        # 3. Import model and numerics from case folder
        model = self.resolve_model(case_dir)
        numerics = self.resolve_numerics(case_dir, model)

        # 4. Create build directory
        build_dir = tempfile.mkdtemp(prefix="zoomy_dmplex_")

        try:
            # 5. Transform model + numerics → C++
            logger.info("Generating C++ code...")
            with open(os.path.join(build_dir, "Model.H"), "w") as f:
                f.write(CppModel(model).create_code())
            with open(os.path.join(build_dir, "Numerics.H"), "w") as f:
                f.write(CppNumerics(numerics).create_code())

            # 6. Copy solver source files (including subdirectories like nlohmann/)
            src_dir = self._find_solver_source()
            for item in os.listdir(src_dir):
                src_path = os.path.join(src_dir, item)
                dst_path = os.path.join(build_dir, item)
                if item in ("Model.H", "Numerics.H", "obj_cpu", "obj_gpu", "solver_cpu", "solver_gpu"):
                    continue  # skip generated/build artifacts
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                elif item.endswith((".cpp", ".hpp", ".H", ".h", "Makefile")):
                    shutil.copy2(src_path, dst_path)

            # 7. Copy mesh file
            mesh_file = settings.get("mesh", "mesh.msh")
            mesh_src = os.path.join(case_dir, mesh_file)
            if os.path.exists(mesh_src):
                shutil.copy2(mesh_src, build_dir)

            # 8. Write DMPlex settings.json
            self._write_dmplex_settings(build_dir, settings, mesh_file, output_dir)

            # 9. Compile
            logger.info("Building solver...")
            env = os.environ.copy()
            env["PETSC_DIR"] = PETSC_DIR
            env["PETSC_ARCH"] = PETSC_ARCH
            result = subprocess.run(
                ["make", "CPU", f"-j{os.cpu_count() or 2}"],
                cwd=build_dir, capture_output=True, text=True, timeout=300, env=env,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Build failed: {result.stderr[-500:]}")

            # 10. Run
            n_procs = settings.get("n_procs", 1)
            exe = os.path.join(build_dir, "solver_cpu")
            cmd = (["mpiexec", "-n", str(n_procs), "--allow-run-as-root", exe]
                   if n_procs > 1 else [exe])
            cmd += ["-settings", "settings.json"]

            logger.info(f"Running with {n_procs} process(es)")
            proc = subprocess.Popen(
                cmd, cwd=build_dir,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            time_end = settings.get("time_end", 1.0)
            out_lines = []
            for line in proc.stdout:
                s = line.strip()
                out_lines.append(s)
                self._parse_progress(s, on_progress)
                logger.debug(s)
            proc.wait()
            if proc.returncode != 0:
                tail = "\n".join(out_lines[-30:])
                raise RuntimeError(
                    f"Solver exited with code {proc.returncode}\n"
                    f"--- solver output tail ---\n{tail}"
                )

            # Copy any stray top-level outputs (older solver builds wrote here)
            for f in os.listdir(build_dir):
                if f.endswith((".vtu", ".pvd", ".h5", ".vtk")):
                    shutil.copy2(os.path.join(build_dir, f), output_dir)

            # The solver writes simulation-NNN.vtu + simulation.vtu.series into
            # io.directory (= output_dir). Convert to the shared zoomy HDF5 so
            # the server serves the same simulation.h5 as every other backend.
            self._convert_vtu_series(output_dir)

            on_progress(-1, time_end, 0.0)

        finally:
            shutil.rmtree(build_dir, ignore_errors=True)

    @staticmethod
    def _convert_vtu_series(output_dir):
        """simulation.vtu.series (ParaView .series JSON) -> simulation.h5 via
        zoomy_prepost. Synthesizes a .pvd so the converter gets real times."""
        series = os.path.join(output_dir, "simulation.vtu.series")
        if os.path.exists(os.path.join(output_dir, "simulation.h5")):
            return
        if not os.path.exists(series):
            logger.warning("dmplex: no simulation.vtu.series in %s — nothing to convert", output_dir)
            return
        try:
            from zoomy_prepost import vtk_to_hdf5
            with open(series) as f:
                entries = json.load(f).get("files", [])
            pvd = os.path.join(output_dir, "simulation.pvd")
            with open(pvd, "w") as f:
                f.write('<?xml version="1.0"?>\n<VTKFile type="Collection"><Collection>\n')
                for e in entries:
                    f.write(f'<DataSet timestep="{e["time"]}" file="{e["name"]}"/>\n')
                f.write("</Collection></VTKFile>\n")
            vtk_to_hdf5(pvd, os.path.join(output_dir, "simulation.h5"))
            logger.info("dmplex: converted %d snapshots -> simulation.h5", len(entries))
        except Exception as e:
            logger.error("dmplex: VTU->HDF5 conversion failed: %s", e)

    def _find_solver_source(self):
        candidates = [
            os.path.join(os.environ.get("ZOOMY_ROOT", ""), "library/zoomy_dmplex"),
            "/workspace/library/zoomy_dmplex",
            os.path.join(os.path.dirname(__file__), "../../../../zoomy_dmplex"),
        ]
        for c in candidates:
            if os.path.isdir(c) and os.path.exists(os.path.join(c, "main.cpp")):
                return c
        raise FileNotFoundError("Cannot find zoomy_dmplex source directory")

    def _write_dmplex_settings(self, build_dir, settings, mesh_file, output_dir):
        # Keys must match what the C++ Settings.hpp actually READS:
        # io.directory (not output_directory), io.filename, io.snapshots
        # (not write_interval). Wrong keys silently fall back to the defaults
        # ("output"/"sol"/10) and the results never land in output_dir.
        dmplex_settings = {
            "io": {
                "mesh_path": mesh_file,
                "directory": output_dir,
                "filename": "simulation",
                "snapshots": settings.get("output_snapshots", 10),
            },
            "solver": {
                "t_end": settings.get("time_end", 1.0),
                "CFL": settings.get("cfl", 0.5),
                "min_dt": settings.get("min_dt", 1e-8),
                "reconstruction_order": settings.get("reconstruction_order", 2),
                "limiter": settings.get("limiter", "venkatakrishnan"),
            },
        }
        with open(os.path.join(build_dir, "settings.json"), "w") as f:
            json.dump(dmplex_settings, f, indent=2)

    @staticmethod
    def _parse_progress(line, on_progress):
        if "Step" in line and "dt" in line:
            try:
                parts = line.split()
                step = int(parts[parts.index("Step") + 1].strip(","))
                time_val = float(parts[parts.index("Time") + 1].strip(","))
                dt_val = float(parts[parts.index("dt") + 1].strip(","))
                on_progress(step, time_val, dt_val)
            except (ValueError, IndexError):
                pass
