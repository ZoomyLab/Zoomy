"""NumPy solver adapter — runs FVM solvers in-process.

Case folder: model.py, numerics.py, mesh.py, settings.json
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
        # calls. Legacy case folders (no run.py) fall through unchanged.
        if os.path.exists(os.path.join(case_dir, "run.py")):
            self.run_mesh_script(case_dir)
            self.run_case_script(case_dir, output_dir, on_progress)
            return

        from zoomy_core.mesh.lsq_mesh import LSQMesh
        from zoomy_core.mesh.base_mesh import BaseMesh
        from zoomy_core.mesh.fvm_mesh import FVMMesh
        from zoomy_core.fvm.solver_numpy import HyperbolicSolver, FreeSurfaceFlowSolver
        from zoomy_core.fvm.solver_imex_numpy import IMEXSolver, FSFIMEXSolver
        from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
        from zoomy_core.misc.misc import Settings
        import zoomy_core.fvm.timestepping as timestepping

        # 1. Execute mesh.py (preprocessing)
        self.run_mesh_script(case_dir)

        # 2. Load settings
        settings = self.load_settings(case_dir)

        # 3. Load mesh (.h5 or .msh)
        mesh_file = os.path.join(case_dir, settings.get("mesh", "mesh.h5"))
        if mesh_file.endswith(".h5"):
            mesh = LSQMesh.from_hdf5(mesh_file)
        else:
            base = BaseMesh.from_msh(mesh_file)
            mesh = LSQMesh.from_fvm(FVMMesh.from_base(base))

        # 4. Import model
        model = self.resolve_model(case_dir)

        # 5. Configure solver
        solver_settings = Settings.default()
        solver_settings.output.directory = output_dir
        solver_settings.output.filename = "simulation"
        solver_settings.output.snapshots = settings.get("output_snapshots", 10)
        solver_settings.output.clean_directory = True

        has_free_surface = "h" in model.variables.keys() and "b" in model.variables.keys()
        solver_type = settings.get("solver_type", "imex")

        if solver_type == "hyperbolic":
            SolverClass = FreeSurfaceFlowSolver if has_free_surface else HyperbolicSolver
        else:
            SolverClass = FSFIMEXSolver if has_free_surface else IMEXSolver

        # Reconstruction order / limiter live on the NumericalSystemModel now
        # (the solver constructors no longer take them — see
        # HyperbolicSolver docstring). Build the NSM here so the case's
        # ``reconstruction_order`` / ``limiter`` settings take effect; the raw
        # Model is auto-promoted via SystemModel.from_model, carrying its
        # baked initial/boundary conditions onto ``nsm.sm``.
        nsm = NumericalSystemModel.from_system_model(
            model,
            reconstruction=ReconstructionSpec(
                order=settings.get("reconstruction_order", 1),
                limiter=settings.get("limiter", "venkatakrishnan"),
            ),
        )

        solver = SolverClass(
            settings=solver_settings,
            time_end=settings.get("time_end", 0.1),
            compute_dt=timestepping.adaptive(CFL=settings.get("cfl", 0.45)),
            min_dt=settings.get("min_dt", 1e-6),
        )

        # 6. Run
        Q, Qaux = solver.solve(mesh, nsm, write_output=True)
        on_progress(-1, settings.get("time_end", 0.1), 0.0)
