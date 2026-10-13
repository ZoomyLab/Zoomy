"""AMReX solver adapter — the shared FOLDER case format on the symbolic AMReX solver.

Matches numpy/jax: the server hands us a ``case_dir`` containing ``model.py`` /
``mesh.py`` / ``settings.json`` [/ ``run.py``], and we run it.  The settings runner
prints the case's solver template to C++ (``zoomy_amrex.symsolver``), builds it
(cached) and runs it on the mesh's uniform Cartesian grid; inside the AMReX image
(zoomy-server's amrex backend) both happen natively.  ``ZOOMY_AMREX_CUDA=1`` (the GPU
image) selects the CUDA build; ``ranks`` in the settings the MPI ranks.
"""
import logging
import os

from zoomy_server.adapter import SolverAdapter

logger = logging.getLogger("zoomy.adapter.amrex")


class AmrexAdapter(SolverAdapter):
    tag = "amrex"

    def solve(self, case_dir, output_dir, on_progress):
        # Generic runner: a composed case carries its own run.py (the ## Run
        # section) — execute THAT instead of translating settings into solver
        # calls.
        if os.path.exists(os.path.join(case_dir, "run.py")):
            self.run_mesh_script(case_dir)
            self.run_case_script(case_dir, output_dir, on_progress)
            return

        from zoomy_amrex.symsolver import SymbolicChorinSolverAmrex, SymbolicHyperbolicSolverAmrex
        from zoomy_server.symbolic import solve_cxx
        self.run_mesh_script(case_dir)
        settings = self.load_settings(case_dir)
        logger.info("amrex: running the case on the symbolic AMReX solver")
        solve_cxx(self, case_dir, output_dir, on_progress,
                  SymbolicHyperbolicSolverAmrex, SymbolicChorinSolverAmrex,
                  ranks=int(settings.get("ranks", 1)),
                  gpu=os.environ.get("ZOOMY_AMREX_CUDA") == "1")

    def list_models(self):
        try:
            from zoomy_server._registry import scan_models
            return scan_models()
        except Exception:
            return []
