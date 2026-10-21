import zipfile
import json
import time
import re
import inspect
import numpy as np


def _safe(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9_-]", "_", s)).strip("_").lower()


def _extract_model(model, session_folder):
    cls = model.__class__
    class_path = cls.__module__ + "." + cls.__name__
    title = cls.__name__

    init = {}
    for pname, p in cls.param.objects("existing").items():
        if pname == "name":
            continue
        val = getattr(model, pname, p.default)
        if isinstance(val, (int, float, str, bool)):
            init[pname] = val

    # Parameter values live directly in model.parameters (a Zstruct of floats)
    if hasattr(model, "parameters") and hasattr(model.parameters, "as_dict"):
        params = {k: float(v) for k, v in model.parameters.as_dict(recursive=False).items()}
    else:
        params = {}

    base_module = cls.__module__.rsplit(".", 1)[0] if "." in cls.__module__ else cls.__module__
    code = "from {} import {}\n\nclass UserModel({}):\n    pass".format(
        cls.__module__, cls.__name__, cls.__name__
    )

    try:
        src = inspect.getsource(cls)
        if "<stdin>" not in inspect.getfile(cls) and "zoomy_core" not in inspect.getfile(cls):
            code = src
    except (TypeError, OSError):
        pass

    card_id = "card-" + _safe(title)
    folder = session_folder + "/model/" + _safe(title)

    return {
        "id": card_id, "tab": "model", "title": title,
        "description": "Model: " + title,
        "params": params, "class_path": class_path, "init": init,
        "_folder": folder, "_code": code
    }


def _extract_mesh(mesh, session_folder):
    init = {}
    title = "Mesh"

    if hasattr(mesh, "n_cells"):
        n = int(mesh.n_cells)
        if hasattr(mesh, "cell_centers"):
            cc = mesh.cell_centers
            if cc.ndim == 2 and cc.shape[0] == 1:
                x = cc[0]
                init = {"type": "create_1d", "x_min": float(x.min()), "x_max": float(x.max()), "n_cells": n}
                title = "1D Mesh ({} cells)".format(n)
            elif cc.ndim == 2 and cc.shape[0] >= 2:
                init = {"type": "create_2d",
                        "x_min": float(cc[0].min()), "x_max": float(cc[0].max()),
                        "y_min": float(cc[1].min()), "y_max": float(cc[1].max()),
                        "n_cells": n}
                title = "2D Mesh ({} cells)".format(n)
        if not init:
            init = {"n_cells": n}
            title = "Mesh ({} cells)".format(n)

    card_id = "card-" + _safe(title)
    folder = session_folder + "/mesh/" + _safe(title)

    return {
        "id": card_id, "tab": "mesh", "title": title,
        "description": "Mesh configuration",
        "params": init, "init": init,
        "_folder": folder, "_code": None
    }


def _extract_solver(solver, session_folder):
    cls = solver.__class__
    title = cls.__name__

    params = {}
    if hasattr(solver, "time_end"):
        params["time_end"] = float(solver.time_end)
    if hasattr(solver, "min_dt"):
        params["min_dt"] = float(solver.min_dt)
    if hasattr(solver, "settings") and hasattr(solver.settings, "output"):
        out = solver.settings.output
        if hasattr(out, "snapshots"):
            params["output_snapshots"] = int(out.snapshots)

    code = "from {} import {}\nimport zoomy_core.fvm.timestepping as timestepping\nimport zoomy_core.fvm.flux as fvmflux\n\nsolver = {}(\n    time_end={},\n    compute_dt=timestepping.adaptive(CFL=0.45),\n    flux=fvmflux.Rusanov(),\n)".format(
        cls.__module__, cls.__name__, cls.__name__,
        params.get("time_end", 0.1)
    )

    card_id = "card-" + _safe(title)
    folder = session_folder + "/solver/" + _safe(title)

    return {
        "id": card_id, "tab": "solver", "title": title,
        "description": "Solver: " + title,
        "params": params,
        "_folder": folder, "_code": code
    }


def _extract_vis(visualization, session_folder, subtab="matplotlib"):
    title = "Custom Visualization"
    code = visualization if isinstance(visualization, str) else ""

    card_id = "card-" + _safe(title)
    folder = session_folder + "/visualization/" + subtab + "/" + _safe(title)

    return {
        "id": card_id, "tab": "visualization", "subtab": subtab,
        "title": title, "description": "Custom visualization script",
        "params": {},
        "_folder": folder, "_code": code
    }


def save_session(name, model=None, mesh=None, solver=None, visualization=None, path=None):
    path = path or (_safe(name) + ".zip")
    session_id = "s-" + str(int(time.time() * 1000))
    session_folder = _safe(name)

    selections = {}
    cards = []

    if model is not None:
        card = _extract_model(model, session_folder)
        cards.append(card)
        selections["model"] = card["id"]

    if mesh is not None:
        card = _extract_mesh(mesh, session_folder)
        cards.append(card)
        selections["mesh"] = card["id"]

    if solver is not None:
        card = _extract_solver(solver, session_folder)
        cards.append(card)
        selections["solver"] = card["id"]

    if visualization is not None:
        card = _extract_vis(visualization, session_folder)
        cards.append(card)
        selections["visualization"] = card["id"]

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", json.dumps({
            "version": "1.0",
            "sessions": [{"id": session_id, "title": name, "description": "Simulation session."}],
            "activeSession": session_id,
            "selections": selections
        }, indent=2))

        for card in cards:
            folder = card.pop("_folder")
            code = card.pop("_code", None)
            zf.writestr(folder + "/card.json", json.dumps(card, indent=2))
            if code:
                zf.writestr(folder + "/code.py", code)

    return path
