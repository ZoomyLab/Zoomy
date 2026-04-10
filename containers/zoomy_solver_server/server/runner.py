import os
import json
import logging
import numpy as np

from zoomy_core.mesh.mesh import Mesh
from zoomy_core.misc.misc import Zstruct, Settings
import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.misc.io as io

from server.registry import resolve_model_class
from server.main import SERVER_TAG

logger = logging.getLogger("zoomy.runner")


def _get_solver_class():
    if SERVER_TAG == "jax":
        try:
            from zoomy_jax.fvm.solver_jax import HyperbolicSolver
            import zoomy_jax.fvm.flux as fvmflux
            logger.info("Using JAX HyperbolicSolver")
            return HyperbolicSolver, fvmflux.Rusanov()
        except ImportError:
            logger.warning("zoomy_jax not available, falling back to numpy")

    from zoomy_core.fvm.solver_numpy import HyperbolicSolver
    import zoomy_core.fvm.flux as fvmflux
    return HyperbolicSolver, fvmflux.Rusanov()


def build_mesh(mesh_spec):
    if mesh_spec["type"] == "create_1d":
        return Mesh.create_1d(tuple(mesh_spec["domain"]), mesh_spec["n_cells"])
    elif mesh_spec["type"] == "create_2d":
        return Mesh.create_2d(
            [mesh_spec.get("x_min", 0), mesh_spec.get("x_max", 1)],
            [mesh_spec.get("y_min", 0), mesh_spec.get("y_max", 1)],
            mesh_spec.get("nx", 50),
            mesh_spec.get("ny", 50),
        )
    raise ValueError(f"Unknown mesh type: {mesh_spec['type']}")


def build_model(model_spec):
    cls = resolve_model_class(model_spec["class_path"])
    model = cls(**model_spec.get("init", {}))
    if model_spec.get("parameters"):
        for k, v in model_spec["parameters"].items():
            idx = list(model.parameters.keys()).index(k)
            model.parameter_values[idx] = v
    return model


def run_case(case_dict, output_dir, progress_file):
    os.makedirs(output_dir, exist_ok=True)

    mesh = build_mesh(case_dict["mesh"])
    model = build_model(case_dict["model"])

    SolverClass, flux = _get_solver_class()

    solver_spec = case_dict.get("solver", {})
    settings = Settings.default()
    settings.output.directory = output_dir
    settings.output.filename = "simulation"
    settings.output.snapshots = solver_spec.get("output_snapshots", 10)
    settings.output.clean_directory = True

    solver = SolverClass(
        settings=settings,
        time_end=solver_spec.get("time_end", 0.1),
        compute_dt=timestepping.adaptive(CFL=solver_spec.get("cfl", 0.45)),
        flux=flux,
    )

    Q, Qaux = solver.solve(mesh, model)

    write_progress(progress_file, {
        "status": "complete",
        "iteration": -1,
        "time": solver_spec.get("time_end", 0.1),
        "time_end": solver_spec.get("time_end", 0.1),
    })

    return output_dir


def write_progress(progress_file, data):
    with open(progress_file, "w") as f:
        json.dump(data, f)


def read_progress(progress_file):
    try:
        with open(progress_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_result_fields(output_dir):
    h5_path = os.path.join(output_dir, "simulation.h5")
    if not os.path.exists(h5_path):
        return None
    try:
        data = io.load_fields_from_hdf5(h5_path)
        result = {}
        for key in data:
            val = data[key]
            if isinstance(val, np.ndarray):
                result[key] = val.tolist()
            else:
                result[key] = val
        return result
    except Exception as e:
        return {"error": str(e)}
