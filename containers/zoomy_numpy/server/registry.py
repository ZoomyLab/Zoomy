import importlib
import pkgutil
import logging

logger = logging.getLogger("zoomy.registry")

ALLOWED_MODELS = {}
ALLOWED_SOLVERS = {}


def _scan_submodules(package_name, target_base):
    results = {}
    try:
        pkg = importlib.import_module(package_name)
    except Exception as e:
        logger.error("Cannot import %s: %s", package_name, e)
        return results

    for importer, modname, ispkg in pkgutil.walk_packages(pkg.__path__, prefix=package_name + "."):
        try:
            mod = importlib.import_module(modname)
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if isinstance(obj, type) and issubclass(obj, target_base) and obj is not target_base:
                    class_path = f"{modname}.{attr_name}"
                    results[class_path] = obj
        except Exception as e:
            logger.debug("Skipping %s: %s", modname, e)
            continue
    return results


def init_registry():
    global ALLOWED_MODELS, ALLOWED_SOLVERS

    try:
        from zoomy_core.model.basemodel import Model
        ALLOWED_MODELS = _scan_submodules("zoomy_core.model.models", Model)
        logger.info("Registered %d model classes", len(ALLOWED_MODELS))
    except ImportError as e:
        logger.error("Cannot import Model base: %s", e)

    try:
        from zoomy_core.fvm.solver_numpy import Solver
        ALLOWED_SOLVERS = _scan_submodules("zoomy_core.fvm", Solver)
        try:
            jax_solvers = _scan_submodules("zoomy_jax.fvm", Solver)
            ALLOWED_SOLVERS.update(jax_solvers)
        except Exception:
            pass
        logger.info("Registered %d solver classes", len(ALLOWED_SOLVERS))
    except ImportError as e:
        logger.error("Cannot import Solver base: %s", e)


def resolve_model_class(class_path):
    if class_path in ALLOWED_MODELS:
        return ALLOWED_MODELS[class_path]
    class_name = class_path.rsplit(".", 1)[-1]
    for key, cls in ALLOWED_MODELS.items():
        if key.endswith("." + class_name):
            return cls
    available = [k.rsplit(".", 1)[-1] for k in ALLOWED_MODELS]
    raise ValueError(f"Unknown model class: {class_path}. Available: {available}")


def resolve_solver_class(class_path):
    if class_path in ALLOWED_SOLVERS:
        return ALLOWED_SOLVERS[class_path]
    raise ValueError(f"Unknown solver class: {class_path}")


def list_models():
    return [
        {"class_path": k, "name": v.__name__}
        for k, v in ALLOWED_MODELS.items()
    ]
