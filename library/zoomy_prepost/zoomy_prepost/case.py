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
      "settings": {...},          # see the TWO-LEVEL shape below
      "solver":   {"tag": "numpy"|"jax"|...},   # -> settings["backend"]
      "visualization": {"code": "..."},          # optional
    }

Settings are TWO-LEVEL: general keys valid for every backend at the top
(time_end, cfl, output_snapshots, mesh, ...) plus optional per-backend
branches with solver-specific options (limiters, schemes, ...)::

    settings = {"time_end": 0.6, "cfl": 0.45, "backend": "jax",
                "numpy": {"reconstruction_order": 1, "limiter": "minmod"},
                "jax":   {"reconstruction_order": 2}}

One case file can carry branches for several backends; the RECEIVING
zoomy-server adapter merges its own branch over the general keys and drops
the rest (SolverAdapter.load_settings). "backend" marks which backend the
case was composed for — informational; the submission target decides.
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
# compose / parse — MARKDOWN HEADINGS are the primary structure
# --------------------------------------------------------------------------- #
# Sections are delimited by markdown-heading cells (## Model, ## Mesh,
# ## Settings, ## Solver, ## Visualization) which survive jupytext AND human
# editing in Jupyter; the ``zoomy`` cell metadata is kept as a machine hint
# and parse-fallback for hand-built files without headings.
_SECTIONS = ("model", "mesh", "settings", "solver", "visualization", "numerics")

import re as _re

# NOTE alternation order: "solver settings" must match BEFORE "solver" —
# "## Solver settings" is the SETTINGS section (option A: one merged section;
# bare "## Solver" only appears in legacy files and maps to the old solver
# section for backward-compatible parsing).
_HEADING_RE = _re.compile(
    r"^\s*#{1,3}\s*(model|mesh|solver\s+settings|settings|solver|visuali[sz]ation|numerics|run)\b",
    _re.IGNORECASE | _re.MULTILINE,
)


def _section_of_markdown(source: str):
    """The section a markdown cell opens (via its heading), or None."""
    m = _HEADING_RE.search(source or "")
    if not m:
        return None
    s = " ".join(m.group(1).lower().split())
    if s.startswith("visuali"):
        return "visualization"
    if s == "solver settings":
        return "settings"
    return s


def compose(spec) -> str:
    """spec dict -> canonical jupytext percent ``.py`` string (heading-based)."""
    import jupytext
    import nbformat

    nb = nbformat.v4.new_notebook()
    cells = []

    def md(text, zoomy):
        c = nbformat.v4.new_markdown_cell(text)
        c.metadata = {"zoomy": zoomy}
        return c

    def code(source, zoomy):
        c = nbformat.v4.new_code_cell(source)
        c.metadata = {"zoomy": zoomy}
        return c

    meta = spec.get("meta", {}) or {}
    title = meta.get("title", "Zoomy case")
    head = f"# {title}\n\n{meta.get('description', '')}".rstrip()
    if meta.get("gui_url"):
        head += f"\n\n[← Back to the Zoomy GUI]({meta['gui_url']})"
    cells.append(md(head, {"role": "meta", **{k: v for k, v in meta.items()}}))

    model = spec["model"]
    cells.append(md("## Model", {"role": "heading", "section": "model"}))
    cells.append(code(model.get("code") or _model_code(model),
                      {"role": "model", "class_path": model.get("class_path"),
                       "init": model.get("init", {})}))

    mesh = spec.get("mesh", {}) or {}
    cells.append(md("## Mesh", {"role": "heading", "section": "mesh"}))
    zmesh = {"role": "mesh"}
    zmesh.update({k: mesh[k] for k in ("spec", "file") if k in mesh})
    cells.append(code(_mesh_code(mesh), zmesh))

    # Option A: ONE merged "Solver settings" section — the settings ARE the
    # solver settings; the backend tag is just an informational entry in it.
    settings = _settings_dict(spec)
    solver = spec.get("solver", {}) or {}
    if solver.get("tag"):
        settings.setdefault("backend", solver["tag"])
    cells.append(md("## Solver settings", {"role": "heading", "section": "settings"}))
    cells.append(code(f"settings = {json.dumps(settings, indent=2)}",
                      {"role": "settings", "settings": settings}))

    # Run: the case is FULLY RUNNABLE end-to-end — `python case.py` (or
    # Run-All in Jupyter) solves in-process wherever zoomy_core is importable
    # (JupyterLite, a backend container's JupyterLab, any local env).
    run = spec.get("run", {}) or {}
    cells.append(md("## Run", {"role": "heading", "section": "run"}))
    cells.append(code(run.get("code") or _run_code(), {"role": "run"}))

    viz = spec.get("visualization", {}) or {}
    if viz.get("code"):
        cells.append(md("## Visualization", {"role": "heading", "section": "visualization"}))
        cells.append(code(viz["code"], {"role": "visualization"}))

    nb.cells = cells
    return jupytext.writes(nb, fmt=_FMT)


def _run_code():
    """The generated in-process runner: resolves the effective settings
    (general keys + the numpy branch) and solves with the numpy solver.
    Uses only `model`, `mesh`, `settings` from the sections above."""
    return '''\
# Runs IN-PROCESS with the numpy solver (no server needed) — works in
# JupyterLite, a backend container's JupyterLab, or any env with zoomy_core.
# For other backends, submit this file via the Zoomy GUI / zoomy-server.
import os

from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
from zoomy_core.fvm.solver_numpy import FreeSurfaceFlowSolver
import zoomy_core.fvm.timestepping as ts
from zoomy_core.misc.misc import Zstruct

# effective settings = general keys + this backend's branch
_eff = {k: v for k, v in settings.items() if not isinstance(v, dict)}
_eff.update(settings.get("numpy", {}))

nsm = NumericalSystemModel.from_system_model(
    model,
    reconstruction=ReconstructionSpec(
        order=_eff.get("reconstruction_order", 1),
        limiter=_eff.get("limiter", "venkatakrishnan")))
solver = FreeSurfaceFlowSolver(
    time_end=_eff.get("time_end", 0.1),
    compute_dt=ts.adaptive(CFL=_eff.get("cfl", 0.45)),
    settings=Zstruct(output=Zstruct(
        directory=os.getcwd(), filename="simulation",
        snapshots=_eff.get("output_snapshots", 10), clean_directory=False)))

mesh.write_to_hdf5("simulation.h5")   # mesh first; the solver appends /fields
Q, Qaux = solver.solve(mesh, nsm, write_output=True)
print("done -> simulation.h5")'''


def parse(py: str) -> dict:
    """canonical percent ``.py`` -> spec dict.

    HEADINGS FIRST: markdown cells opening a known section (## Model, ## Mesh,
    ## Settings, ## Solver, ## Visualization) claim the code cells that follow
    until the next heading — robust to hand editing in Jupyter. The ``zoomy``
    cell metadata is used as fallback/extra hints (class_path, init, settings
    dict, solver tag) when present.
    """
    import jupytext

    nb = jupytext.reads(py, fmt=_FMT)
    sources = {}   # section -> [code sources]
    hints = {}     # section -> metadata dict
    meta = {}
    current = None
    for cell in nb.cells:
        z = cell.metadata.get("zoomy") or {}
        if cell.cell_type == "markdown":
            sec = _section_of_markdown(cell.source)
            if sec:
                current = sec
            elif z.get("role") == "meta" or current is None:
                # leading title cell (or explicit meta metadata)
                if z.get("role") == "meta":
                    meta = {k: v for k, v in z.items() if k != "role"}
                elif not meta:
                    first = (cell.source or "").lstrip().splitlines() or [""]
                    if first[0].startswith("#"):
                        meta = {"title": first[0].lstrip("# ").strip()}
            continue
        if cell.cell_type != "code":
            continue
        sec = current or ({"model": "model", "mesh": "mesh", "settings": "settings",
                           "solver": "solver", "visualization": "visualization",
                           "numerics": "numerics"}.get(z.get("role")))
        if not sec:
            continue
        sources.setdefault(sec, []).append(cell.source)
        if z:
            hints.setdefault(sec, {}).update({k: v for k, v in z.items() if k != "role"})

    def joined(sec):
        return "\n\n".join(s.rstrip() for s in sources.get(sec, [])).strip()

    spec = {}
    if meta:
        spec["meta"] = meta
    if "model" in sources:
        h = hints.get("model", {})
        spec["model"] = {"code": joined("model"),
                         "class_path": h.get("class_path"), "init": h.get("init", {})}
    if "mesh" in sources:
        h = hints.get("mesh", {})
        m = {"code": joined("mesh")}
        m.update({k: h[k] for k in ("spec", "file") if k in h})
        spec["mesh"] = m
    if "settings" in sources or "settings" in hints:
        h = hints.get("settings", {})
        settings = h.get("settings")
        if settings is None:
            settings = _settings_from_code(joined("settings"))
        spec["settings"] = settings or {}
    # backend tag (option A): settings["backend"]; legacy fallback = a bare
    # "## Solver" section / solver_tag line / solver metadata from old files.
    tag = (spec.get("settings") or {}).get("backend")
    if not tag and ("solver" in sources or "solver" in hints):
        h = hints.get("solver", {})
        tag = h.get("tag") or _tag_from_code(joined("solver"))
    if tag:
        spec["solver"] = {"tag": tag,
                          "params": hints.get("solver", {}).get("params", {})}
    if "visualization" in sources:
        spec["visualization"] = {"code": joined("visualization")}
    if "numerics" in sources:
        spec["numerics"] = {"code": joined("numerics")}
    if "run" in sources:
        spec["run"] = {"code": joined("run")}   # kept for re-export fidelity
    return spec


def _settings_from_code(source: str):
    """Recover the settings dict from a ``settings = {...}`` code cell."""
    m = _re.search(r"settings\s*=\s*(\{.*\})", source or "", _re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        try:
            import ast
            return ast.literal_eval(m.group(1))
        except Exception:
            return None


def _tag_from_code(source: str):
    """Recover the backend tag from a ``solver_tag = "..."`` line."""
    m = _re.search(r"solver_tag\s*=\s*['\"]([\w-]+)['\"]", source or "")
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# materialize the adapter case folder
# --------------------------------------------------------------------------- #
def to_folder(py: str, dest_dir: str) -> str:
    """canonical percent ``.py`` -> {model.py, mesh.py, settings.json
    [, numerics.py, visualize.py]} in dest_dir (the folder format the
    zoomy-server adapters consume; visualize.py is the optional reproducible
    viz — adapters ignore it). Heading-based via parse(). Returns dest_dir."""
    spec = parse(py)
    os.makedirs(dest_dir, exist_ok=True)
    # NOTE: the "run" section is deliberately NOT materialized — the server
    # adapters invoke the solver themselves; the runner only lives in the
    # single-file/notebook form.
    files = {
        "model.py": (spec.get("model") or {}).get("code"),
        "mesh.py": (spec.get("mesh") or {}).get("code"),
        "numerics.py": (spec.get("numerics") or {}).get("code"),
        "visualize.py": (spec.get("visualization") or {}).get("code"),
    }
    for name, src in files.items():
        if src:
            with open(os.path.join(dest_dir, name), "w") as f:
                f.write(src.rstrip() + "\n")
    with open(os.path.join(dest_dir, "settings.json"), "w") as f:
        json.dump(spec.get("settings", {}), f, indent=2)
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
    """An existing adapter case folder (model.py, mesh.py, settings.json
    [, numerics.py, visualize.py]) -> canonical heading-based percent ``.py``.
    Lets any runnable case be exported as a single file / notebook and
    re-ingested unchanged (to_folder(from_folder(d)) round-trips)."""
    import jupytext
    import nbformat

    nb = nbformat.v4.new_notebook()
    cells = []

    def heading(section, title):
        c = nbformat.v4.new_markdown_cell(f"## {title}")
        c.metadata = {"zoomy": {"role": "heading", "section": section}}
        return c

    title = nbformat.v4.new_markdown_cell(f"# {os.path.basename(os.path.abspath(case_dir))}")
    title.metadata = {"zoomy": {"role": "meta", "title": os.path.basename(os.path.abspath(case_dir))}}
    cells.append(title)

    for section, fname, htitle in (
        ("model", "model.py", "Model"),
        ("mesh", "mesh.py", "Mesh"),
    ):
        path = os.path.join(case_dir, fname)
        if os.path.exists(path):
            cells.append(heading(section, htitle))
            with open(path) as f:
                c = nbformat.v4.new_code_cell(f.read().rstrip())
            c.metadata = {"zoomy": {"role": section}}
            cells.append(c)

    sp = os.path.join(case_dir, "settings.json")
    settings = json.load(open(sp)) if os.path.exists(sp) else {}
    cells.append(heading("settings", "Solver settings"))
    c = nbformat.v4.new_code_cell(f"settings = {json.dumps(settings, indent=2)}")
    c.metadata = {"zoomy": {"role": "settings", "settings": settings}}
    cells.append(c)

    for section, fname, htitle in (
        ("numerics", "numerics.py", "Numerics"),
        ("visualization", "visualize.py", "Visualization"),
    ):
        path = os.path.join(case_dir, fname)
        if os.path.exists(path):
            cells.append(heading(section, htitle))
            with open(path) as f:
                c = nbformat.v4.new_code_cell(f.read().rstrip())
            c.metadata = {"zoomy": {"role": section}}
            cells.append(c)

    nb.cells = cells
    return jupytext.writes(nb, fmt=_FMT)
