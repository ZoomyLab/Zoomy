"""DUNE-FEM solver adapter — the shared FOLDER case format.

Matches numpy/jax/amrex/dmplex: the server hands us a ``case_dir`` containing
``run.py`` (plus optional ``mesh.py`` / ``settings.json``) and we execute it.

There is deliberately no translation layer here. A dune case writes its own
UFL weak form and drives its own time loop — that is the whole point of the
backend — so the adapter's job is only to run the case and collect artifacts,
exactly as the generic runner does for every other backend.
"""
import logging
import os
import sys

from zoomy_server.adapter import SolverAdapter

logger = logging.getLogger("zoomy.adapter.dune")


class DuneAdapter(SolverAdapter):
    tag = "dune"

    def solve(self, case_dir, output_dir, on_progress):
        run_py = os.path.join(case_dir, "run.py")
        if not os.path.exists(run_py):
            raise FileNotFoundError(
                f"dune adapter: no run.py in {case_dir}. A dune case IS its "
                "run.py (UFL weak form + time loop); there is no settings-to-"
                "solver translation for this backend."
            )
        self.run_mesh_script(case_dir)
        self.run_case_script(case_dir, output_dir, on_progress, entry=[sys.executable, "run.py"])
