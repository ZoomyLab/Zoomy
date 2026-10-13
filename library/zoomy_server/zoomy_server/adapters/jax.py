"""JAX solver adapter — runs the symbolic JAX solver in-process.

Same folder case format as numpy (model.py, mesh.py, settings.json [, run.py]).
JAX falls back to CPU when no GPU is present.
"""

import os
import logging

from zoomy_server.adapter import SolverAdapter

logger = logging.getLogger("zoomy.adapter.jax")


class JaxAdapter(SolverAdapter):
    tag = "jax"

    def solve(self, case_dir, output_dir, on_progress):
        # Generic runner: a composed case carries its own run.py (the ## Run
        # section) — execute THAT instead of translating settings into solver
        # calls.
        if os.path.exists(os.path.join(case_dir, "run.py")):
            self.run_mesh_script(case_dir)
            self.run_case_script(case_dir, output_dir, on_progress)
            return

        # Settings runner: the case on SymbolicHyperbolicSolverJax (Chorin for a split)
        import jax  # noqa: F401  (initialises the backend; CPU fallback if no GPU)
        from zoomy_jax.symsolver.solver import (SymbolicChorinSolverJax,
                                                SymbolicHyperbolicSolverJax)
        from zoomy_server.symbolic import solve_in_process
        self.run_mesh_script(case_dir)
        solve_in_process(self, case_dir, output_dir, on_progress,
                         SymbolicHyperbolicSolverJax, SymbolicChorinSolverJax)
