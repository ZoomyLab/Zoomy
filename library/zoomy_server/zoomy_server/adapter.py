"""Base solver adapter for the Zoomy server.

Each backend (NumPy, JAX, DMPlex, AMReX, Firedrake) implements a subclass
that knows how to run a case folder.

A case folder contains:
  model.py, numerics.py, mesh.py, settings.json
"""

import json
import os
import sys
import importlib
import subprocess


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
