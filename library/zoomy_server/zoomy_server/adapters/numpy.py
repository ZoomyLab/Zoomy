import os
import logging

from zoomy_server.adapter import SolverAdapter

logger = logging.getLogger("zoomy.adapter.numpy")


class NumpyAdapter(SolverAdapter):
    tag = "numpy"

    def solve(self, case, output_dir, on_progress):
        from zoomy_core.mesh.mesh import Mesh
        from zoomy_core.misc.misc import Settings
        from zoomy_core.fvm.solver_numpy import HyperbolicSolver
        import zoomy_core.fvm.timestepping as timestepping
        import zoomy_core.fvm.flux as fvmflux

        mesh = self._build_mesh(case["mesh"])
        model = self._build_model(case["model"])

        settings = Settings.default()
        settings.output.directory = output_dir
        settings.output.filename = "simulation"
        settings.output.snapshots = case.get("solver", {}).get("output_snapshots", 10)
        settings.output.clean_directory = True

        solver = HyperbolicSolver(
            settings=settings,
            time_end=case.get("solver", {}).get("time_end", 0.1),
            compute_dt=timestepping.adaptive(CFL=case.get("solver", {}).get("cfl", 0.45)),
            flux=fvmflux.Rusanov(),
        )

        Q, Qaux = solver.solve(mesh, model)
        on_progress(-1, case.get("solver", {}).get("time_end", 0.1), 0.0)

    def list_models(self):
        try:
            from zoomy_server._registry import scan_models
            return scan_models()
        except Exception:
            return []

    def _build_mesh(self, mesh_spec):
        from zoomy_core.mesh.mesh import Mesh
        if mesh_spec["type"] == "create_1d":
            return Mesh.create_1d(tuple(mesh_spec["domain"]), mesh_spec["n_cells"])
        elif mesh_spec["type"] == "create_2d":
            return Mesh.create_2d(
                [mesh_spec.get("x_min", 0), mesh_spec.get("x_max", 1)],
                [mesh_spec.get("y_min", 0), mesh_spec.get("y_max", 1)],
                mesh_spec.get("nx", 50), mesh_spec.get("ny", 50),
            )
        raise ValueError(f"Unknown mesh type: {mesh_spec['type']}")

    def _build_model(self, model_spec):
        from zoomy_server._registry import resolve_model
        cls = resolve_model(model_spec["class_path"])
        model = cls(**model_spec.get("init", {}))
        if model_spec.get("parameters"):
            for k, v in model_spec["parameters"].items():
                idx = list(model.parameters.keys()).index(k)
                model.parameter_values[idx] = v
        return model
