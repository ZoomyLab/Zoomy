"""NumPy solver adapter — runs the symbolic NumPy solver in-process.

Case folder: model.py, mesh.py, settings.json [, run.py]
"""

import os
import logging

from zoomy_server.adapter import SolverAdapter

logger = logging.getLogger("zoomy.adapter.numpy")


class NumpyAdapter(SolverAdapter):
    tag = "numpy"

    def solve(self, case_dir, output_dir, on_progress):
        # Generic runner: a composed case carries its own run.py (the ## Run
        # section) — execute THAT instead of translating settings into solver
        # calls.
        if os.path.exists(os.path.join(case_dir, "run.py")):
            self.run_case_script(case_dir, output_dir, on_progress)
            return

        # Settings runner: the case on SymbolicHyperbolicSolverNumpy (Chorin for a split)
        from zoomy_core.symsolver.numpy_backend import (SymbolicChorinSolverNumpy,
                                                        SymbolicHyperbolicSolverNumpy)
        from zoomy_server.symbolic import solve_in_process
        self.run_mesh_script(case_dir)
        solve_in_process(self, case_dir, output_dir, on_progress,
                         SymbolicHyperbolicSolverNumpy, SymbolicChorinSolverNumpy)
