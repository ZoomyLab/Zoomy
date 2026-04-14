"""Model/solver/mesh auto-discovery and card registry.

Three sources of cards:
1. Library — auto-discovered from zoomy_core by scanning for subclasses
2. User session — auto-discovered from a session directory (models/, meshes/)
3. Pre-defined — cards.json for non-Python backends (AMReX, DMPlex, etc.)

For Python classes, only the class path is needed — parameters are
extracted at runtime via param introspection.
"""

import importlib
import inspect
import json
import os
import pkgutil
from pathlib import Path


def _is_concrete(cls):
    """Check if a class is concrete (not a base/abstract class)."""
    # Skip classes with 'projectable = False' (intermediate derivation steps)
    if hasattr(cls, 'projectable') and cls.projectable is False:
        return False
    return True


def discover_models(package_name="zoomy_core.model.models"):
    """Scan a package for Model subclasses. Returns list of card dicts."""
    try:
        from zoomy_core.model.basemodel import Model
    except ImportError:
        return []

    pkg = importlib.import_module(package_name)
    cards = []

    for importer, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
        if modname.startswith("_") or modname in ("legacy", "basisfunctions",
                "symbolic_integrator", "ins_generator", "model_derivation",
                "derived_system", "zeta_projection", "vam_zeta_projection",
                "vam_derivation"):
            continue
        try:
            mod = importlib.import_module(f"{package_name}.{modname}")
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if (issubclass(obj, Model)
                        and obj is not Model
                        and obj.__module__ == mod.__name__
                        and _is_concrete(obj)):
                    class_path = f"{mod.__name__}.{name}"
                    cards.append({
                        "id": f"lib-{modname}-{name.lower()}",
                        "title": name,
                        "class": class_path,
                        "source": "library",
                    })
        except Exception:
            pass

    return cards


def discover_solvers():
    """Scan solver modules for Solver subclasses. Returns list of card dicts."""
    try:
        from zoomy_core.fvm.solver_numpy import Solver
    except ImportError:
        return []

    modules = [
        "zoomy_core.fvm.solver_numpy",
        "zoomy_core.fvm.solver_imex_numpy",
    ]
    # Optional: splitting solver
    try:
        importlib.import_module("zoomy_core.fvm.solver_splitting_numpy")
        modules.append("zoomy_core.fvm.solver_splitting_numpy")
    except ImportError:
        pass

    cards = []
    for mod_name in modules:
        try:
            mod = importlib.import_module(mod_name)
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if (issubclass(obj, Solver)
                        and obj is not Solver
                        and obj.__module__ == mod.__name__):
                    class_path = f"{mod.__name__}.{name}"
                    cards.append({
                        "id": f"lib-{name.lower()}",
                        "title": name,
                        "class": class_path,
                        "source": "library",
                        "requires_tag": "numpy",
                    })
        except Exception:
            pass

    return cards


def discover_user_models(session_dir):
    """Scan a session directory for user-defined model .py files.

    Looks for ``session_dir/models/*.py``. Each file should define
    a Model subclass. Returns list of card dicts.
    """
    models_dir = Path(session_dir) / "models"
    if not models_dir.is_dir():
        return []

    cards = []
    for py_file in sorted(models_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        cards.append({
            "id": f"user-{py_file.stem}",
            "title": py_file.stem.replace("_", " ").title(),
            "file": str(py_file),
            "source": "user",
        })
    return cards


def discover_user_meshes(session_dir):
    """Scan session_dir/meshes/ for mesh files (.msh, .h5, .py)."""
    meshes_dir = Path(session_dir) / "meshes"
    if not meshes_dir.is_dir():
        return []

    cards = []
    for mesh_file in sorted(meshes_dir.iterdir()):
        if mesh_file.name.startswith("_"):
            continue
        suffix = mesh_file.suffix
        if suffix in (".msh", ".h5", ".py"):
            cards.append({
                "id": f"user-mesh-{mesh_file.stem}",
                "title": mesh_file.stem.replace("_", " ").title(),
                "file": str(mesh_file),
                "source": "user",
                "type": suffix.lstrip("."),
            })
    return cards


def discover_catalog_meshes():
    """Discover meshes from the MeshCatalog (GitHub Pages index).

    Returns list of card dicts. Uses cached index if available.
    """
    try:
        from zoomy_core.mesh.mesh_catalog import MeshCatalog
        catalog = MeshCatalog(auto_fetch=True)
        cards = []
        for name in catalog.names():
            entry = catalog.get(name)
            # Pretty title from mesh name (e.g. "basic_shapes__quad_2d" → "Quad 2D")
            parts = name.split("__")
            title = parts[-1].replace("_", " ").title() if len(parts) > 1 else name.replace("_", " ").title()
            category = parts[0].replace("_", " ").title() if len(parts) > 1 else ""
            cards.append({
                "id": f"catalog-{name}",
                "title": title,
                "source": "catalog",
                "category": category,
                "mesh_name": name,
                "mesh_sizes": entry.sizes or ["default"],
                "mesh_types": entry.types or [],
                "description": f"{category}: {title}" if category else title,
            })
        return cards
    except Exception:
        return []


def load_predefined_cards(cards_json_path):
    """Load pre-defined cards from a JSON file.

    These are fully specified cards for non-Python backends
    or models that are only available in Docker containers.
    """
    if not os.path.isfile(cards_json_path):
        return {}
    with open(cards_json_path) as f:
        return json.load(f)


def build_registry(session_dir=None, predefined_json=None):
    """Build the complete card registry from all sources.

    Merges: pre-defined cards → library discovery → catalog meshes → user session.
    Order within mesh tab: builtin create → user meshes → catalog meshes.

    Returns the full tabs structure for the GUI.
    """
    # 1. Library discovery
    lib_models = discover_models()
    lib_solvers = discover_solvers()

    # 2. User session discovery
    user_models = discover_user_models(session_dir) if session_dir else []
    user_meshes = discover_user_meshes(session_dir) if session_dir else []

    # 3. Mesh catalog (remote index)
    catalog_meshes = discover_catalog_meshes()

    # 4. Pre-defined cards
    predefined = load_predefined_cards(predefined_json) if predefined_json else {}
    predefined_tabs = {t["id"]: t for t in predefined.get("tabs", [])}

    # --- Models: pre-defined → library → user ---
    model_cards = list(predefined_tabs.get("model", {}).get("cards", []))
    predefined_classes = {c.get("class") for c in model_cards}
    for card in lib_models:
        if card["class"] not in predefined_classes:
            model_cards.append(card)
    model_cards.extend(user_models)

    # --- Solvers: pre-defined → library ---
    solver_cards = list(predefined_tabs.get("solver", {}).get("cards", []))
    predefined_solver_classes = {c.get("class") for c in solver_cards if c.get("class")}
    for card in lib_solvers:
        if card["class"] not in predefined_solver_classes:
            solver_cards.append(card)

    # --- Meshes: builtin create (from pre-defined) → user → catalog ---
    mesh_tab_def = predefined_tabs.get("mesh", {})
    builtin_meshes = [c for c in mesh_tab_def.get("cards", []) if c.get("source") == "builtin"]
    other_predefined = [c for c in mesh_tab_def.get("cards", []) if c.get("source") != "builtin"]
    mesh_cards = builtin_meshes + user_meshes + catalog_meshes + other_predefined

    # Build final structure
    tabs = [
        predefined_tabs.get("dashboard", {"id": "dashboard", "title": "Dashboard", "type": "dashboard"}),
        {"id": "model", "title": "Model", "type": "cards", "cardType": "model", "cards": model_cards},
        {
            "id": "mesh", "title": "Mesh", "type": "cards", "cardType": "mesh",
            "layout": mesh_tab_def.get("layout", "grid"),
            "columns": mesh_tab_def.get("columns", 2),
            "cards": mesh_cards,
        },
        {"id": "solver", "title": "Solver", "type": "cards", "cardType": "solver", "cards": solver_cards},
    ]

    # Keep visualization tab from pre-defined if present
    if "visualization" in predefined_tabs:
        tabs.append(predefined_tabs["visualization"])

    return {"tabs": tabs}


# --- Manifest cache ---

MANIFEST_FILENAME = "registry_manifest.json"


def save_manifest(registry, path):
    """Save registry to a JSON manifest file for instant GUI loading."""
    with open(path, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def load_manifest(path):
    """Load a cached registry manifest. Returns None if not found."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def refresh_manifest(session_dir=None, predefined_json=None, manifest_dir=None):
    """Build the registry and save as a manifest file.

    Call this on server startup or via a CLI command to keep the
    manifest up-to-date. The GUI loads the manifest for instant
    startup, then background-refreshes from the live endpoint.
    """
    registry = build_registry(session_dir, predefined_json)
    if manifest_dir is None:
        manifest_dir = session_dir or "."
    path = os.path.join(manifest_dir, MANIFEST_FILENAME)
    save_manifest(registry, path)
    return path
