import os
import logging
import numpy as np

from zoomy_server.adapter import SolverAdapter

logger = logging.getLogger("zoomy.adapter.numpy")


class NumpyAdapter(SolverAdapter):
    tag = "numpy"

    def solve(self, case, output_dir, on_progress):
        from zoomy_core.mesh.lsq_mesh import LSQMesh as Mesh
        from zoomy_core.misc.misc import Settings, Zstruct
        from zoomy_core.model.legacy.numerical_model import NumericalModel
        from zoomy_core.fvm.generated_model_solver import GeneratedModelSolver
        import zoomy_core.fvm.timestepping as timestepping
        import zoomy_core.model.boundary_conditions as BC
        import zoomy_core.model.initial_conditions as IC

        mesh = self._build_mesh(case["mesh"])
        raw_model = self._build_model(case["model"])

        # Wrap with NumericalModel for regularization (1/h → 1/(h+eps), wet/dry)
        nv = raw_model.n_variables
        bcs = BC.BoundaryConditions([
            BC.Extrapolation(tag="left"),
            BC.Extrapolation(tag="right"),
        ])

        ic_spec = case.get("initial_conditions")
        if ic_spec and ic_spec.get("type") == "dam_break":
            h_left = ic_spec.get("h_left", 2.0)
            h_right = ic_spec.get("h_right", 1.0)
            x_jump = ic_spec.get("x_jump", 0.0)
            def ic_func(x, _nv=nv, _hl=h_left, _hr=h_right, _xj=x_jump):
                Q = np.zeros(_nv)
                Q[1] = _hl if float(x[0]) < _xj else _hr
                return Q
            ic = IC.UserFunction(ic_func)
        else:
            # Default: uniform h=1
            def ic_default(n, _nv=nv):
                Q = np.zeros(_nv)
                Q[1] = 1.0
                return Q
            ic = IC.Constant(constants=ic_default)

        model = NumericalModel(raw_model,
                               boundary_conditions=bcs,
                               initial_conditions=ic)

        settings = Settings.default()
        settings.output.directory = output_dir
        settings.output.filename = "simulation"
        settings.output.snapshots = case.get("solver", {}).get("output_snapshots", 10)
        settings.output.clean_directory = True

        solver_spec = case.get("solver", {})
        solver = GeneratedModelSolver(
            settings=settings,
            time_end=solver_spec.get("time_end", 0.1),
            compute_dt=timestepping.adaptive(CFL=solver_spec.get("cfl", 0.45)),
            min_dt=solver_spec.get("min_dt", 1e-6),
        )

        Q, Qaux = solver.solve(mesh, model)
        on_progress(-1, solver_spec.get("time_end", 0.1), 0.0)

    def list_models(self):
        try:
            from zoomy_server._registry import scan_models
            return scan_models()
        except Exception:
            return []

    def _build_mesh(self, mesh_spec):
        from zoomy_core.mesh.lsq_mesh import LSQMesh as Mesh
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
