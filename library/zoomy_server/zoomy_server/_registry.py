import importlib
import pkgutil
import logging

logger = logging.getLogger("zoomy.registry")

_MODELS = {}


def scan_models():
    global _MODELS
    if _MODELS:
        return [{"class_path": k, "name": v.__name__} for k, v in _MODELS.items()]

    try:
        from zoomy_core.model.basemodel import Model
        pkg = importlib.import_module("zoomy_core.model.models")
        for _, modname, _ in pkgutil.walk_packages(pkg.__path__, prefix="zoomy_core.model.models."):
            try:
                mod = importlib.import_module(modname)
                for attr in dir(mod):
                    obj = getattr(mod, attr)
                    if isinstance(obj, type) and issubclass(obj, Model) and obj is not Model:
                        _MODELS[f"{modname}.{attr}"] = obj
            except Exception:
                continue
        logger.info("Registered %d model classes", len(_MODELS))
    except ImportError as e:
        logger.error("Cannot scan models: %s", e)

    return [{"class_path": k, "name": v.__name__} for k, v in _MODELS.items()]


def resolve_model(class_path):
    if not _MODELS:
        scan_models()
    if class_path in _MODELS:
        return _MODELS[class_path]
    class_name = class_path.rsplit(".", 1)[-1]
    for key, cls in _MODELS.items():
        if key.endswith("." + class_name):
            return cls
    available = [k.rsplit(".", 1)[-1] for k in _MODELS]
    raise ValueError(f"Unknown model: {class_path}. Available: {available}")
