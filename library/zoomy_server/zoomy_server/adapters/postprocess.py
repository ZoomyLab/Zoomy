"""Post-processing adapter — pipeline steps + visualization on existing results.

Case folder (NOT a solver case — a RESULTS folder):
  simulation.h5 (zoomy store)  OR  a VTK series (.pvd / .vtu.series / .vtu / .vtk)
  steps.json    (optional)  {"steps": ["lift3d", "to_vtk", ...], "nz": 10}
  model.py      (needed by the lift3d step — the composed-case model)
  visualize.py  (optional)  runs after the steps with cwd = the case copy;
                            reads simulation.h5, writes *.png / *.gif

Workflow:
  1. Copy the case folder to a scratch dir (visualize.py writes next to its
     inputs; the submitted folder stays untouched).
  2. Normalize results to simulation.h5 (zoomy_prepost.vtk_to_hdf5 if the
     folder carries VTK instead).
  3. Run the requested steps (zoomy_prepost.steps).
  4. Execute visualize.py, if present.
  5. Collect simulation.h5 + step outputs (+ *.png/*.gif) into output_dir.
"""

import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile

from zoomy_server.adapter import SolverAdapter

logger = logging.getLogger("zoomy.adapter.postprocess")

#: extensions collected into output_dir at the end
_COLLECT = (".h5", ".pvd", ".vtu", ".vtk", ".vtk.series", ".png", ".gif")


class PostprocessAdapter(SolverAdapter):
    tag = "postprocess"

    def solve(self, case_dir, output_dir, on_progress):
        work = tempfile.mkdtemp(prefix="zoomy_postproc_")
        try:
            shutil.copytree(case_dir, work, dirs_exist_ok=True)

            h5 = self._normalize_results(work)
            steps, nz = self._load_steps(work)

            n_total = len(steps) + 1  # +1 for visualize/collect
            for i, step in enumerate(steps):
                logger.info("postprocess: step %d/%d: %s", i + 1, len(steps), step)
                self._run_step(step, work, h5, nz)
                on_progress(i + 1, float(i + 1), float(n_total))

            self._run_visualize(work)
            self._collect(work, output_dir)
            on_progress(-1, float(n_total), 0.0)
        finally:
            shutil.rmtree(work, ignore_errors=True)

    # ── 1. results normalization ─────────────────────────────────────
    @staticmethod
    def _normalize_results(work):
        """Ensure ``work/simulation.h5`` exists; convert a VTK series if
        that is what the case carries.  Returns the h5 path."""
        h5 = os.path.join(work, "simulation.h5")
        if os.path.exists(h5):
            return h5
        from zoomy_prepost import vtk_to_hdf5

        source = None
        series = sorted(glob.glob(os.path.join(work, "*.vtu.series")))
        pvds = sorted(glob.glob(os.path.join(work, "*.pvd")))
        if series:
            # ParaView .series JSON -> synthesize a .pvd (same trick as the
            # dmplex adapter) so the converter gets real times.
            with open(series[0]) as f:
                entries = json.load(f).get("files", [])
            source = os.path.join(work, "_series.pvd")
            with open(source, "w") as f:
                f.write('<?xml version="1.0"?>\n<VTKFile type="Collection"><Collection>\n')
                for e in entries:
                    f.write(f'<DataSet timestep="{e["time"]}" file="{e["name"]}"/>\n')
                f.write("</Collection></VTKFile>\n")
        elif pvds:
            source = pvds[0]
        else:
            frames = sorted(glob.glob(os.path.join(work, "*.vtu"))
                            + glob.glob(os.path.join(work, "*.vtk")))
            if frames:
                source = frames
        if source is None:
            raise RuntimeError(
                "postprocess case has no results: expected simulation.h5 or a "
                "VTK series (.pvd / .vtu.series / .vtu / .vtk) in the case folder")
        vtk_to_hdf5(source, h5)
        logger.info("postprocess: normalized VTK results -> simulation.h5")
        return h5

    # ── 2. steps ─────────────────────────────────────────────────────
    @staticmethod
    def _load_steps(work):
        path = os.path.join(work, "steps.json")
        if not os.path.exists(path):
            return [], 10
        try:
            with open(path) as f:
                spec = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"invalid steps.json: {e}")
        return list(spec.get("steps", [])), int(spec.get("nz", 10))

    @staticmethod
    def _run_step(step, work, h5, nz):
        from zoomy_prepost import hdf5_to_vtk, lift3d, vtk_to_hdf5

        name = str(step).replace("-", "_")
        if name == "to_h5":
            return  # results are already normalized to simulation.h5
        if name == "to_vtk":
            hdf5_to_vtk(h5, work, basename="simulation")
            return
        if name == "lift3d":
            if not os.path.exists(os.path.join(work, "model.py")):
                raise RuntimeError(
                    "step 'lift3d' needs the case's model.py (the model's "
                    "interpolate_to_3d does the lifting) — none in the case folder")
            pvd = lift3d(work, h5, os.path.join(work, "simulation_3d"), Nz=nz)
            # Also emit the lifted result as a zoomy store so consumers that
            # speak the store format (zoomy_plotting.read_hdf5 / the GUI viz
            # cards + results shelf) can open the 3-D lift directly, not only
            # the VTK series. Same conversion the VTK-input normalization uses.
            vtk_to_hdf5(pvd, os.path.join(work, "simulation_3d.h5"))
            return
        raise RuntimeError(
            f"unknown postprocess step {step!r} (known: to_h5, to_vtk, lift3d)")

    # ── 3. visualization ─────────────────────────────────────────────
    @staticmethod
    def _run_visualize(work):
        viz = os.path.join(work, "visualize.py")
        if not os.path.exists(viz):
            return
        logger.info("postprocess: running visualize.py")
        env = os.environ.copy()
        env.setdefault("MPLBACKEND", "Agg")  # no display on the server
        result = subprocess.run(
            [sys.executable, "visualize.py"],
            cwd=work, capture_output=True, text=True, timeout=600, env=env,
        )
        if result.returncode != 0:
            tail = ((result.stdout or "") + "\n" + (result.stderr or ""))[-1000:]
            raise RuntimeError(
                f"visualize.py exited with code {result.returncode}\n"
                f"--- visualize output tail ---\n{tail}"
            )

    # ── 4. collect outputs ───────────────────────────────────────────
    @staticmethod
    def _collect(work, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        n = 0
        for f in sorted(os.listdir(work)):
            if f.endswith(_COLLECT) and f != "_series.pvd":
                shutil.copy2(os.path.join(work, f), output_dir)
                n += 1
        logger.info("postprocess: collected %d output files -> %s", n, output_dir)
