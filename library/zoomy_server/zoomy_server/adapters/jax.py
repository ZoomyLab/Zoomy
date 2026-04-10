"""JAX solver adapter — runs JIT-compiled FVM solver."""

import os
import logging

from zoomy_server.adapter import SolverAdapter

logger = logging.getLogger("zoomy.adapter.jax")


class JaxAdapter(SolverAdapter):
    tag = "jax"

    def solve(self, case_dir, output_dir, on_progress):
        import jax
        from zoomy_core.mesh.base_mesh import BaseMesh
        from zoomy_core.mesh.fvm_mesh import FVMMesh
        from zoomy_core.mesh.lsq_mesh import LSQMesh
        from zoomy_jax.fvm.solver_jax import HyperbolicSolver
        from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax
        import zoomy_core.fvm.timestepping as timestepping

        # 1. Execute mesh.py
        self.run_mesh_script(case_dir)

        # 2. Load settings
        settings = self.load_settings(case_dir)

        # 3. Load mesh
        mesh_file = os.path.join(case_dir, settings.get("mesh", "mesh.h5"))
        if mesh_file.endswith(".h5"):
            mesh = LSQMesh.from_hdf5(mesh_file)
        else:
            mesh = LSQMesh.from_fvm(FVMMesh.from_base(BaseMesh.from_msh(mesh_file)))

        # 4. Import model
        model = self.resolve_model(case_dir)

        # 5. Pick solver
        solver_type = settings.get("solver_type", "hyperbolic")
        has_source = any(
            e != 0 for row in model.source().tolist()
            for e in (row if isinstance(row, list) else [row])
        ) if hasattr(model, 'source') else False

        if solver_type == "imex" or has_source:
            try:
                solver = IMEXSourceSolverJax(
                    time_end=settings.get("time_end", 0.1),
                    compute_dt=timestepping.adaptive(CFL=settings.get("cfl", 0.45)),
                    min_dt=settings.get("min_dt", 1e-6),
                    reconstruction_order=settings.get("reconstruction_order", 1),
                    limiter=settings.get("limiter", "venkatakrishnan"),
                )
            except Exception:
                logger.info("IMEX not available, using Hyperbolic")
                solver = HyperbolicSolver(
                    time_end=settings.get("time_end", 0.1),
                    compute_dt=timestepping.adaptive(CFL=settings.get("cfl", 0.45)),
                    reconstruction_order=settings.get("reconstruction_order", 1),
                    limiter=settings.get("limiter", "venkatakrishnan"),
                )
        else:
            solver = HyperbolicSolver(
                time_end=settings.get("time_end", 0.1),
                compute_dt=timestepping.adaptive(CFL=settings.get("cfl", 0.45)),
                reconstruction_order=settings.get("reconstruction_order", 1),
                limiter=settings.get("limiter", "venkatakrishnan"),
            )

        # 6. Run
        Q, Qaux = solver.solve(mesh, model, write_output=False)
        on_progress(-1, settings.get("time_end", 0.1), 0.0)
