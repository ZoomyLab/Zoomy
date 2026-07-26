"""Base solver adapter for the Zoomy server.

Each backend (NumPy, JAX, DMPlex, AMReX, Firedrake) implements a subclass
that knows how to run a case folder.

A case folder contains:
  model.py, numerics.py, mesh.py, settings.json [, run.py]

When the folder carries a ``run.py`` (the composed case's ``## Run`` section,
materialized by ``zoomy_prepost.case.to_folder``) the adapters execute the
case's OWN solver code via :meth:`run_case_script` instead of translating
settings into solver calls (the generic runner).
"""

import json
import logging
import os
import re
import shutil
import sys
import importlib
import subprocess

logger = logging.getLogger("zoomy.adapter")

#: progress lines recognized in case-script output: the zoomy python solvers
#: ("iteration: N, time: T, dt: D, ...") and the C++ solvers ("Step N, Time T, dt D").
_PROGRESS_RES = (
    re.compile(r"iteration:\s*(\d+).*?time:\s*([0-9.eE+-]+).*?dt:\s*([0-9.eE+-]+)"),
    re.compile(r"Step\s+(\d+).*?Time\s+([0-9.eE+-]+).*?dt\s+([0-9.eE+-]+)"),
)


class SolverAdapter:
    """Base class for solver backends."""

    tag = "base"

    def solve(self, case_dir, output_dir, on_progress):
        """Run a simulation from a case folder.

        Parameters
        ----------
        case_dir : str
            Path to the case folder (model.py, numerics.py, mesh.py, settings.json).
        output_dir : str
            Path where output files should be written.
        on_progress : callable(iteration, time, dt)
            Progress callback.
        """
        raise NotImplementedError

    def list_models(self):
        return []

    def capabilities(self):
        """Handshake payload returned by ``GET /health``: who this backend is
        and what it can run. The GUI queries it at connect time — it never
        assumes a backend's identity — and enables the solver tags in
        ``solvers`` (defaults to this adapter's own tag). An adapter that serves
        more than one solver tag overrides this to widen ``solvers``."""
        cls = type(self)
        return {
            "tag": self.tag,
            "name": cls.__name__.replace("Adapter", "") + " backend",
            "solvers": [self.tag],
            "adapter": cls.__module__ + "." + cls.__name__,
        }

    # ── Shared helpers ───────────────────────────────────────────────

    #: per-backend settings branches recognized in settings.json
    BACKEND_TAGS = ("numpy", "jax", "amrex", "dmplex", "firedrake", "foam")

    def load_settings(self, case_dir):
        """Load settings.json and resolve the TWO-LEVEL settings shape.

        ``settings`` = general keys valid for every backend (time_end, cfl,
        output_snapshots, mesh, ...) + optional per-backend branches::

            {"time_end": 0.6, "cfl": 0.45, "backend": "jax",
             "numpy": {"reconstruction_order": 1, "limiter": "minmod"},
             "jax":   {"reconstruction_order": 2}}

        The RECEIVING adapter merges its own branch over the general keys and
        drops the other backends' branches — so one case file carries valid,
        possibly different, solver-specific options (limiters, schemes, ...)
        for several backends at once. ``backend`` is informational (which
        backend the case was composed for); submission target decides.
        """
        path = os.path.join(case_dir, "settings.json")
        with open(path) as f:
            raw = json.load(f)
        eff = {k: v for k, v in raw.items() if k not in self.BACKEND_TAGS}
        tag = getattr(self, "tag", None)
        branch = raw.get(tag)
        if isinstance(branch, dict):
            eff.update(branch)
        return eff

    @staticmethod
    def run_mesh_script(case_dir):
        """Execute mesh.py in the case folder (preprocessing)."""
        mesh_py = os.path.join(case_dir, "mesh.py")
        if os.path.exists(mesh_py):
            subprocess.run(
                [sys.executable, mesh_py],
                cwd=case_dir,
                check=True,
            )

    # ── Generic runner (run-py-first) ────────────────────────────────

    def run_case_script(self, case_dir, output_dir, on_progress):
        """Execute the case's OWN ``run.py`` (the composed case's ``## Run``
        section) and collect its artifacts into ``output_dir``.

        The case folder is copied to ``output_dir/case`` first (isolation:
        the script runs with the copy as cwd, so a shared case folder is
        never mutated by outputs), then ``[sys.executable, "run.py"]`` runs
        there with stdout streamed for progress lines. Artifacts
        (simulation.h5, images, VTK) are copied into ``output_dir``; when the
        script produced VTK but no simulation.h5, the VTK output is converted
        via ``zoomy_prepost.vtk_to_hdf5``. The full captured output is written
        to ``output_dir/run.log``.
        """
        logger.info("runner: executing case run.py")
        run_dir = os.path.join(output_dir, "case")
        if os.path.abspath(case_dir) != os.path.abspath(run_dir):
            shutil.copytree(case_dir, run_dir, dirs_exist_ok=True)

        proc = subprocess.Popen(
            [sys.executable, "run.py"], cwd=run_dir,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        out_lines = []
        for line in proc.stdout:
            s = line.rstrip()
            out_lines.append(s)
            self._parse_progress_line(s, on_progress)
            logger.debug(s)
        proc.wait()
        with open(os.path.join(output_dir, "run.log"), "w") as f:
            f.write("\n".join(out_lines) + "\n")
        if proc.returncode != 0:
            tail = "\n".join(out_lines[-30:])
            raise RuntimeError(
                f"run.py exited with code {proc.returncode}\n"
                f"--- run.py output tail ---\n{tail}"
            )
        logger.info("runner: run.py finished: %s", out_lines[-1] if out_lines else "")

        self._collect_artifacts(run_dir, output_dir)
        time_end = 0.0
        try:
            time_end = float(self.load_settings(case_dir).get("time_end", 0.0))
        except Exception:
            pass
        on_progress(-1, time_end, 0.0)

    @staticmethod
    def _parse_progress_line(line, on_progress):
        for rx in _PROGRESS_RES:
            m = rx.search(line)
            if m:
                try:
                    on_progress(int(m.group(1)), float(m.group(2)), float(m.group(3)))
                except (ValueError, TypeError):
                    pass
                return

    def _collect_artifacts(self, run_dir, output_dir):
        """Copy the results run.py produced into output_dir: simulation.h5,
        images, VTK (mesh.h5/mesh.msh are inputs, not results)."""
        for f in sorted(os.listdir(run_dir)):
            src = os.path.join(run_dir, f)
            if not os.path.isfile(src):
                continue
            if f == "simulation.h5" or f.endswith(
                (".png", ".gif", ".vtu", ".pvd", ".vtk", ".vtu.series")
            ):
                shutil.copy2(src, output_dir)
        if not os.path.exists(os.path.join(output_dir, "simulation.h5")):
            self._convert_vtk_outputs(output_dir)

    @staticmethod
    def _convert_vtk_outputs(output_dir):
        """VTK output (a ``.pvd``, or a ParaView ``.vtu.series`` a ``.pvd`` is
        synthesized from — see the dmplex adapter) -> simulation.h5 via
        zoomy_prepost, so the server serves the same HDF5 for every backend."""
        names = sorted(os.listdir(output_dir))
        pvds = [f for f in names if f.endswith(".pvd")]
        series = [f for f in names if f.endswith(".vtu.series")]
        try:
            from zoomy_prepost import vtk_to_hdf5
            if pvds:
                pvd = os.path.join(output_dir, pvds[0])
            elif series:
                with open(os.path.join(output_dir, series[0])) as f:
                    entries = json.load(f).get("files", [])
                pvd = os.path.join(output_dir, series[0][: -len(".vtu.series")] + ".pvd")
                with open(pvd, "w") as f:
                    f.write('<?xml version="1.0"?>\n<VTKFile type="Collection"><Collection>\n')
                    for e in entries:
                        f.write(f'<DataSet timestep="{e["time"]}" file="{e["name"]}"/>\n')
                    f.write("</Collection></VTKFile>\n")
            else:
                logger.warning(
                    "runner: no simulation.h5 and no VTK output to convert in %s", output_dir)
                return
            vtk_to_hdf5(pvd, os.path.join(output_dir, "simulation.h5"))
            logger.info("runner: converted %s -> simulation.h5", os.path.basename(pvd))
        except Exception as e:
            logger.error("runner: VTK->HDF5 conversion failed: %s", e)

    @staticmethod
    def import_from_case(case_dir, module_name):
        """Import a Python module from the case folder.

        Parameters
        ----------
        module_name : str
            'model' or 'numerics' (without .py)

        Returns
        -------
        module
        """
        filepath = os.path.join(case_dir, f"{module_name}.py")
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def resolve_model(case_dir):
        """Import model.py and return the Model instance.

        Prefers an explicit module-level ``model`` instance (the composed case
        format does ``model = SME(level=2)``, and any case that instantiates its
        model with arguments/IC/BC) over scanning for a Model subclass — otherwise
        a bare ``SME()`` constructed from the imported class would drop those args.
        Falls back to the first Model subclass (subclass-style cases that bake
        their IC/BC and are meant to be constructed with no arguments).
        """
        mod = SolverAdapter.import_from_case(case_dir, "model")
        from zoomy_core.model.basemodel import Model
        if isinstance(getattr(mod, "model", None), Model):
            return mod.model
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, type) and issubclass(obj, Model) and obj is not Model:
                return obj()
        if hasattr(mod, "model"):
            return mod.model
        raise RuntimeError(f"No Model subclass found in {case_dir}/model.py")

    @staticmethod
    def resolve_numerics(case_dir, model):
        """Import numerics.py and return the Numerics instance; if the case has
        no numerics.py, fall back to the default Riemann solver so ONE shared
        case runs on every backend (numpy/jax don't ship a numerics.py)."""
        import os
        from zoomy_core.fvm.riemann_solvers import NonconservativeRusanov
        if not os.path.exists(os.path.join(case_dir, "numerics.py")):
            return NonconservativeRusanov(model)
        mod = SolverAdapter.import_from_case(case_dir, "numerics")
        # Find a Numerics subclass or a factory function
        from zoomy_core.model.basefunction import SymbolicRegistrar
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, type) and issubclass(obj, SymbolicRegistrar) and obj is not SymbolicRegistrar:
                return obj(model)
        # Look for module-level 'numerics' variable or function
        if hasattr(mod, 'numerics'):
            n = mod.numerics
            return n(model) if callable(n) else n
        # Default: NonconservativeRusanov
        from zoomy_core.fvm.riemann_solvers import NonconservativeRusanov
        return NonconservativeRusanov(model)
