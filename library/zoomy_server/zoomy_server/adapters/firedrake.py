"""Firedrake solver adapter — in-process FEM, shared folder case format.

`FiredrakeHyperbolicSolver.solve(mesh, model)` accepts the resolved folder-format
Model directly (via SystemModel.from_model), like numpy/jax. Firedrake writes VTK
(`simulation.pvd`), so we convert it to the shared HDF5 with `zoomy_prepost` — no
custom HDF5 writer. Needs a gmsh `.msh` mesh (named physical groups = BC tags).
"""
import os
import logging

from zoomy_server.adapter import SolverAdapter

logger = logging.getLogger("zoomy.adapter.firedrake")


class FiredrakeAdapter(SolverAdapter):
    tag = "firedrake"

    def solve(self, case_dir, output_dir, on_progress):
        # Generic runner: a composed case carries its own run.py (the ## Run
        # section) — execute THAT instead of translating settings into solver
        # calls. Legacy case folders (no run.py) fall through unchanged.
        if os.path.exists(os.path.join(case_dir, "run.py")):
            self.run_mesh_script(case_dir)
            self.run_case_script(case_dir, output_dir, on_progress)
            return

        from zoomy_firedrake.firedrake_solver import FiredrakeHyperbolicSolver
        from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
        from zoomy_core.misc.misc import Settings
        from zoomy_prepost import vtk_to_hdf5

        # mesh.py should emit a gmsh .msh (firedrake reads gmsh, not the LSQ .h5)
        self.run_mesh_script(case_dir)
        settings = self.load_settings(case_dir)
        model = self.resolve_model(case_dir)
        mesh_file = os.path.join(case_dir, settings.get("mesh", "mesh.msh"))

        s = Settings.default()
        s.output.directory = output_dir            # absolute -> VTK lands here
        s.output.filename = "simulation"
        s.output.snapshots = settings.get("output_snapshots", 10)

        solver = FiredrakeHyperbolicSolver(
            CFL=settings.get("cfl", 0.9),
            time_end=settings.get("time_end", 0.1),
            dg_degree=max(0, settings.get("reconstruction_order", 1) - 1),
            riemann_solver_cls=PositiveNonconservativeRusanov,
            settings=s,
        )
        solver.solve(mesh_file, model)

        # Firedrake output is VTK -> convert to the shared HDF5 the server serves.
        pvd = os.path.join(output_dir, "simulation.pvd")
        if os.path.exists(pvd):
            vtk_to_hdf5(pvd, os.path.join(output_dir, "simulation.h5"))
        else:
            logger.warning("no simulation.pvd produced at %s", output_dir)
        on_progress(-1, settings.get("time_end", 0.1), 0.0)
