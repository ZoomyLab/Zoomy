"""JAX solver adapter — runs the JIT-compiled FVM solver.

Same folder case format as numpy/dmplex (model.py, mesh.py, settings.json).
JAX falls back to CPU when no GPU is present.
"""

import os
import logging

from zoomy_server.adapter import SolverAdapter

logger = logging.getLogger("zoomy.adapter.jax")


class JaxAdapter(SolverAdapter):
    tag = "jax"

    def solve(self, case_dir, output_dir, on_progress):
        import jax  # noqa: F401  (initialises the backend; CPU fallback if no GPU)
        from zoomy_core.mesh.base_mesh import BaseMesh
        from zoomy_core.mesh.fvm_mesh import FVMMesh
        from zoomy_core.mesh.lsq_mesh import LSQMesh
        from zoomy_jax.fvm.solver_jax import HyperbolicSolver
        from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax
        from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
        from zoomy_core.misc.misc import Settings
        import zoomy_core.fvm.timestepping as timestepping

        # 1. Mesh preprocessing + settings
        self.run_mesh_script(case_dir)
        settings = self.load_settings(case_dir)

        # 2. Load mesh
        mesh_file = os.path.join(case_dir, settings.get("mesh", "mesh.h5"))
        if mesh_file.endswith(".h5"):
            mesh = LSQMesh.from_hdf5(mesh_file)
        else:
            mesh = LSQMesh.from_fvm(FVMMesh.from_base(BaseMesh.from_msh(mesh_file)))

        # 3. Model
        model = self.resolve_model(case_dir)

        # 4. Output settings — write the result HDF5 into output_dir so the
        #    server can serve it (the old adapter used write_output=False).
        solver_settings = Settings.default()
        solver_settings.output.directory = output_dir
        solver_settings.output.filename = "simulation"
        solver_settings.output.snapshots = settings.get("output_snapshots", 10)
        solver_settings.output.clean_directory = True

        # 5. Reconstruction order / limiter live on the NumericalSystemModel now
        #    (the solver ctors no longer take them). Auto-promotes the raw Model,
        #    carrying its baked initial/boundary conditions.
        nsm = NumericalSystemModel.from_system_model(
            model,
            reconstruction=ReconstructionSpec(
                order=settings.get("reconstruction_order", 1),
                limiter=settings.get("limiter", "venkatakrishnan"),
            ),
        )

        # 6. Pick solver
        solver_type = settings.get("solver_type", "hyperbolic")
        has_source = any(
            e != 0 for row in model.source().tolist()
            for e in (row if isinstance(row, list) else [row])
        ) if hasattr(model, "source") else False

        common = dict(
            settings=solver_settings,
            time_end=settings.get("time_end", 0.1),
            compute_dt=timestepping.adaptive(CFL=settings.get("cfl", 0.45)),
        )
        if solver_type == "imex" or has_source:
            try:
                solver = IMEXSourceSolverJax(min_dt=settings.get("min_dt", 1e-6), **common)
            except Exception:
                logger.info("IMEX not available, using Hyperbolic")
                solver = HyperbolicSolver(**common)
        else:
            solver = HyperbolicSolver(**common)

        # 7. Run
        Q, Qaux = solver.solve(mesh, nsm, write_output=True)
        on_progress(-1, settings.get("time_end", 0.1), 0.0)
