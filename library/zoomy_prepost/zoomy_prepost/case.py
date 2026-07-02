"""Zoomy case interchange format.

A *case* is a single jupytext "percent" ``.py`` where each section is a cell
tagged with ``# %% zoomy={"role": ...}`` metadata carrying the structured card
data. This one file is simultaneously:

  * a runnable Python script,
  * a notebook (``.py`` <-> ``.ipynb`` via jupytext) that opens in a backend
    container's Jupyter kernel,
  * a lossless projection of the GUI cards (the ``zoomy`` cell metadata).

Roles (cell order): ``meta``, ``model``, ``mesh``, ``settings``, ``solver``.

The server runs a case by ``to_folder`` -> the adapter's case folder
(``model.py``, ``mesh.py``, ``settings.json``), keeping the existing per-backend
adapters unchanged.

    spec = {
      "meta":     {"title": str, "description": str},
      "model":    {"class_path": "pkg.mod.Class", "init": {...}},
      "mesh":     {"spec": {"type": "create_1d"|"create_2d", ...}}  # or {"code": "..."}
                  # optionally {"file": "mesh.msh"} for an uploaded mesh
      "settings": {"time_end": ..., "cfl": ..., ...},
      "solver":   {"tag": "numpy"|"jax"|..., "params": {...}},
    }
"""
from __future__ import annotations

import json
import os

_FMT = "py:percent"
ROLES = ("meta", "model", "mesh", "settings", "solver")


# --------------------------------------------------------------------------- #
# code generation
# --------------------------------------------------------------------------- #
def _model_code(model):
    mod, cls = model["class_path"].rsplit(".", 1)
    kw = ", ".join(f"{k}={v!r}" for k, v in (model.get("init") or {}).items())
    return f"from {mod} import {cls}\n\nmodel = {cls}({kw})"


def _mesh_code(mesh):
    """Generate mesh.py source that writes BOTH mesh.h5 (numpy/jax) and mesh.msh
    (dmplex/firedrake), or references an uploaded mesh file."""
    if mesh.get("code"):
        return mesh["code"]
    if mesh.get("file"):
        # An uploaded mesh is written into the case folder by the server; the
        # mesh cell just records which file to use (adapters read settings["mesh"]).
        return f'# uploaded mesh: {mesh["file"]}\nmesh_file = {mesh["file"]!r}'
    spec = mesh.get("spec", {}) or {}
    t = spec.get("type", "create_1d")
    if t == "create_1d":
        dom = spec.get("domain", [0.0, 1.0])
        n = spec.get("n_cells", 100)
        return (
            "from zoomy_core.mesh import BaseMesh\n"
            "from zoomy_prepost import mesh_to_gmsh\n\n"
            f"mesh = BaseMesh.create_1d(domain=({dom[0]}, {dom[1]}), n_inner_cells={n})\n"
            'mesh.write_to_hdf5("mesh.h5")\n'
            'mesh_to_gmsh(mesh, "mesh.msh", boundary_group=None)\n'
        )
    if t == "create_2d":
        dom = spec.get("domain", [0.0, 1.0, 0.0, 1.0])
        nx = spec.get("nx", spec.get("n_cells", 32))
        ny = spec.get("ny", spec.get("n_cells", 32))
        return (
            "from zoomy_core.mesh import BaseMesh\n"
            "from zoomy_prepost import mesh_to_gmsh\n\n"
            f"mesh = BaseMesh.create_2d(domain=({dom[0]}, {dom[1]}, {dom[2]}, {dom[3]}), "
            f"n_inner_cells_x={nx}, n_inner_cells_y={ny})\n"
            'mesh.write_to_hdf5("mesh.h5")\n'
            'mesh_to_gmsh(mesh, "mesh.msh", boundary_group=None)\n'
        )
    raise ValueError(f"unknown mesh spec type {t!r}")


def _settings_dict(spec):
    s = dict(spec.get("settings", {}) or {})
    # if an uploaded mesh is used, point the adapters at it
    mesh = spec.get("mesh", {}) or {}
    if mesh.get("file"):
        s.setdefault("mesh", mesh["file"])
    return s


# --------------------------------------------------------------------------- #
# compose / parse
# --------------------------------------------------------------------------- #
def compose(spec) -> str:
    """spec dict -> canonical jupytext percent ``.py`` string."""
    import jupytext
    import nbformat

    nb = nbformat.v4.new_notebook()
    cells = []

    meta = spec.get("meta", {}) or {}
    md = f"# {meta.get('title', 'Zoomy case')}\n\n{meta.get('description', '')}".rstrip()
    mc = nbformat.v4.new_markdown_cell(md)
    mc.metadata = {"zoomy": {"role": "meta", **meta}}
    cells.append(mc)

    model = spec["model"]
    c = nbformat.v4.new_code_cell(_model_code(model))
    c.metadata = {"zoomy": {"role": "model", "class_path": model["class_path"],
                            "init": model.get("init", {})}}
    cells.append(c)

    mesh = spec.get("mesh", {}) or {}
    c = nbformat.v4.new_code_cell(_mesh_code(mesh))
    zmesh = {"role": "mesh"}
    zmesh.update({k: mesh[k] for k in ("spec", "file") if k in mesh})
    c.metadata = {"zoomy": zmesh}
    cells.append(c)

    settings = _settings_dict(spec)
    c = nbformat.v4.new_code_cell(f"settings = {json.dumps(settings, indent=2)}")
    c.metadata = {"zoomy": {"role": "settings", "settings": settings}}
    cells.append(c)

    solver = spec.get("solver", {}) or {}
    tag = solver.get("tag", "numpy")
    c = nbformat.v4.new_code_cell(
        f"# solver backend: {tag!r} (applied by the zoomy-server adapter on submit)\n"
        f"solver_tag = {tag!r}"
    )
    c.metadata = {"zoomy": {"role": "solver", "tag": tag, "params": solver.get("params", {})}}
    cells.append(c)

    nb.cells = cells
    return jupytext.writes(nb, fmt=_FMT)


def parse(py: str) -> dict:
    """canonical percent ``.py`` -> spec dict (reads the ``zoomy`` cell metadata)."""
    import jupytext

    nb = jupytext.reads(py, fmt=_FMT)
    spec = {}
    for cell in nb.cells:
        z = cell.metadata.get("zoomy")
        if not z:
            continue
        role = z.get("role")
        if role == "meta":
            spec["meta"] = {k: v for k, v in z.items() if k != "role"}
        elif role == "model":
            spec["model"] = {"class_path": z.get("class_path"), "init": z.get("init", {})}
        elif role == "mesh":
            m = {k: z[k] for k in ("spec", "file") if k in z}
            m["code"] = cell.source
            spec["mesh"] = m
        elif role == "settings":
            spec["settings"] = z.get("settings", {})
        elif role == "solver":
            spec["solver"] = {"tag": z.get("tag", "numpy"), "params": z.get("params", {})}
    return spec


# --------------------------------------------------------------------------- #
# materialize the adapter case folder
# --------------------------------------------------------------------------- #
def to_folder(py: str, dest_dir: str) -> str:
    """canonical percent ``.py`` -> {model.py, mesh.py, settings.json} in dest_dir
    (the folder format the zoomy-server adapters consume). Returns dest_dir."""
    import jupytext

    nb = jupytext.reads(py, fmt=_FMT)
    os.makedirs(dest_dir, exist_ok=True)
    settings = {}
    for cell in nb.cells:
        z = cell.metadata.get("zoomy") or {}
        role = z.get("role")
        if role in ("model", "mesh", "numerics"):
            with open(os.path.join(dest_dir, f"{role}.py"), "w") as f:
                f.write(cell.source.rstrip() + "\n")
        elif role == "settings":
            settings = z.get("settings", {})
    with open(os.path.join(dest_dir, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2)
    return dest_dir


# --------------------------------------------------------------------------- #
# notebook conversion (jupytext)
# --------------------------------------------------------------------------- #
def to_notebook(py: str) -> str:
    """percent ``.py`` string -> ``.ipynb`` JSON string."""
    import jupytext
    import nbformat
    return nbformat.writes(jupytext.reads(py, fmt=_FMT))


def from_notebook(ipynb: str) -> str:
    """``.ipynb`` JSON string -> canonical percent ``.py`` string."""
    import jupytext
    import nbformat
    return jupytext.writes(nbformat.reads(ipynb, as_version=4), fmt=_FMT)


def from_folder(case_dir: str) -> str:
    """An existing adapter case folder (model.py, mesh.py, settings.json) ->
    canonical percent ``.py``. Lets any runnable case be exported as a single
    file / notebook and re-ingested unchanged."""
    import jupytext
    import nbformat

    nb = nbformat.v4.new_notebook()
    cells = []
    for role, fname in (("model", "model.py"), ("mesh", "mesh.py"), ("numerics", "numerics.py")):
        path = os.path.join(case_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                c = nbformat.v4.new_code_cell(f.read().rstrip())
            c.metadata = {"zoomy": {"role": role}}
            cells.append(c)
    sp = os.path.join(case_dir, "settings.json")
    settings = json.load(open(sp)) if os.path.exists(sp) else {}
    c = nbformat.v4.new_code_cell(f"settings = {json.dumps(settings, indent=2)}")
    c.metadata = {"zoomy": {"role": "settings", "settings": settings}}
    cells.append(c)
    nb.cells = cells
    return jupytext.writes(nb, fmt=_FMT)
