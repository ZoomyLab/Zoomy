#!/usr/bin/env python3
"""Regenerate gui/demo/coupling-examples.zip — the loadable coupled-simulation
examples (SME<->SME, SME<->VOF, SME<->VOF<->SME).

Each child is a COMPLETE, runnable preCICE participant `case.py`: a real SME (or
VOF) model built with initial + boundary conditions (including the preCICE Coupled
BC whose mesh_name matches the sibling ../precice-config.xml), a real mesh cell
(`BaseMesh.create_1d/2d(...); mesh.write_to_hdf5(...)`), and settings pinning the
OpenFOAM backend + the coupled solver. The parent folder carries coupling.yml +
the generated precice-config.xml.

Run:  python apps/theia-preview/gui/demo/make_coupling_examples.py
(regenerates the zip in place). Requires the repo's zoomy_prepost on the path.
"""
import os, sys, json, shutil, zipfile, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))   # .../Zoomy-tut
sys.path.insert(0, os.path.join(REPO, "library", "zoomy_prepost"))
from zoomy_prepost.coupling import make_coupled_precice_config

ZIP = os.path.join(HERE, "coupling-examples.zip")

# ---------------------------------------------------------------- cell builders
RUN_NOTE = (
    '# %% [markdown] zoomy={"role":"heading","section":"run"}\n# ## Run\n\n'
    '# %% zoomy={"role":"run"}\n'
    "# COUPLED PARTICIPANT — do not run standalone. Launch the whole coupling from\n"
    '# the parent folder\'s "Run coupled" (OpenFOAM backend + preCICE): every\n'
    "# participant is started together, shares the coupling folder (the preCICE\n"
    "# exchange-directory) and they find each other over sockets. In the GUI, running\n"
    "# this case alone is gated on a connected OpenFOAM backend (the selected solver\n"
    "# is zoomyFoam / incompressibleVOF, not numpy) — it is NOT a local numpy run.\n")


def header(title, desc):
    return (f'# %% [markdown] zoomy={{"role":"meta","title":{json.dumps(title)},'
            f'"description":{json.dumps(desc)}}}\n# # {title}\n#\n# {desc}\n\n')


def sme_model_cell():
    # PLAIN SME — card-representable (class_path + init select the SME card), so it
    # survives the GUI load->save round-trip unchanged. Deliberately NO hand-written
    # preCICE `Coupled` BC: the GUI regenerates the model from the card on save/submit
    # (wall/wall), which would DROP such a BC. The coupling is instead injected into the
    # participant's OF case from the preserved precice-config.xml + coupling.yml at run
    # time (see zoomy_prepost.coupling.inject_precice_controlDict).
    meta = {"role": "model", "class_path": "zoomy_core.model.models.SME",
            "init": {"level": 2, "dimension": 2}}
    return ('# %% [markdown] zoomy={"role":"heading","section":"model"}\n# ## Model\n\n'
            f'# %% zoomy={json.dumps(meta)}\n'
            "import numpy as np\n"
            "from zoomy_core.model.models import SME\n"
            "import zoomy_core.model.boundary_conditions as BC\n"
            "import zoomy_core.model.initial_conditions as IC\n"
            "from zoomy_core.systemmodel import SystemModel\n"
            "\n"
            "# Plain SME arm (wall boundaries). The coupling interface (the 'coupled'\n"
            "# patch) gets its preCICE BC injected from the coupling config at run time,\n"
            "# NOT from this model — a round-tripped card model is always wall/wall.\n"
            "# A dam-break Riemann state (h: 2.0 -> 1.0 at x=5) gives the arm something to move.\n"
            "model = SystemModel.from_model(SME(\n"
            "    level=2, dimension=2,\n"
            "    boundary_conditions=BC.BoundaryConditions([\n"
            '        BC.FromModel(tag="outer", definition="wall"),\n'
            '        BC.FromModel(tag="coupled", definition="wall")]),\n'
            "    initial_conditions=IC.RP(\n"
            "        high=lambda n: np.array([0.0, 2.0] + [0.0] * (n - 2)),\n"
            "        low=lambda n: np.array([0.0, 1.0] + [0.0] * (n - 2)),\n"
            "        jump_position_x=5.0)))\n\n")


def vof_model_cell():
    # vof-openfoam card has class=None; carry an explicit card id so the GUI can
    # still select it (applySpec falls back to spec.model.card when class_path is null).
    # No Coupled BC here either — coupling is injected from the config at run time.
    meta = {"role": "model", "class_path": None, "card": "vof-openfoam", "init": {}}
    return ('# %% [markdown] zoomy={"role":"heading","section":"model"}\n# ## Model\n\n'
            f'# %% zoomy={json.dumps(meta)}\n'
            "# Incompressible Volume-of-Fluid participant (interFoam / incompressibleVoF).\n"
            "# The VOF field + solver are configured on the OpenFOAM backend. Its coupling\n"
            "# interface(s) to the SME neighbour(s) are injected into the OF case from the\n"
            "# coupling config at run time (mesh/patch recorded in ../coupling.yml), so no\n"
            "# preCICE BC is declared here (a round-tripped card model carries none).\n"
            "vof_participant = True  # placeholder: VOF spec lives in the OpenFOAM case\n\n")


def mesh_cell_1d():
    meta = {"role": "mesh", "spec": {"x_min": 0, "x_max": 10, "n_cells": 40}}
    return ('# %% [markdown] zoomy={"role":"heading","section":"mesh"}\n# ## Mesh\n\n'
            f'# %% zoomy={json.dumps(meta)}\n'
            "from zoomy_core.mesh import BaseMesh\n"
            "\n"
            "mesh = BaseMesh.create_1d(domain=(0, 10), n_inner_cells=40)\n"
            'mesh.write_to_hdf5("mesh.h5")\n\n')


def mesh_cell_2d():
    meta = {"role": "mesh",
            "spec": {"x_min": 0, "x_max": 1.5, "y_min": 0, "y_max": 0.4, "nx": 30, "ny": 10}}
    return ('# %% [markdown] zoomy={"role":"heading","section":"mesh"}\n# ## Mesh\n\n'
            f'# %% zoomy={json.dumps(meta)}\n'
            "from zoomy_core.mesh import BaseMesh\n"
            "\n"
            "mesh = BaseMesh.create_2d((0, 1.5, 0, 0.4), nx=30, ny=10)\n"
            'mesh.write_to_hdf5("mesh.h5")\n\n')


def settings_cell(solver_id):
    st = {"backend": "OpenFOAM", "solver_id": solver_id}
    return ('# %% [markdown] zoomy={"role":"heading","section":"settings"}\n'
            "# ## Solver settings\n\n"
            f'# %% zoomy={{"role":"settings","settings":{json.dumps(st)}}}\n'
            f"settings = {json.dumps(st, indent=2)}\n\n")


def sme_child(title, desc):
    return (header(title, desc) + sme_model_cell() + mesh_cell_1d()
            + settings_cell("solver-foam") + RUN_NOTE)


def vof_child(title, desc):
    return (header(title, desc) + vof_model_cell() + mesh_cell_2d()
            + settings_cell("solver-foam-vof") + RUN_NOTE)


# ---------------------------------------------------------------- assembly
def build(cases_dir, name, kids):   # kids: [(child, type, py_text)]
    import re
    from zoomy_prepost.coupling import participant_coupling_spec
    cdir = os.path.join(cases_dir, name)
    for cn, _t, py in kids:
        d = os.path.join(cdir, cn)
        os.makedirs(d)
        with open(os.path.join(d, "case.py"), "w") as fh:
            fh.write(py)
    parts = [{"name": cn, "type": t} for cn, t, _py in kids]
    cfg = make_coupled_precice_config(parts, exchange_directory=".", max_time=4.0)
    with open(os.path.join(cdir, "precice-config.xml"), "w") as fh:
        fh.write(cfg)
    # Enriched coupling.yml: record each participant's coupling INTERFACE (the
    # mesh it provides + the 'coupled' patch + the exchanged profile) — set here at
    # couple-time so the injection is explicit and survives a GUI load->save. The
    # coupling comes from THIS, not from any model BC.
    specs = participant_coupling_spec(parts)
    ym = ["# Zoomy coupling manifest", f"coupling_id: {name}",
          "scheme: parallel-explicit", f"canonical_output: {kids[0][0]}",
          "participants:"]
    for (cn, t, _py), s in zip(kids, specs):
        ym += [f"  - name: {cn}", f"    type: {t}",
               f"    mesh: {s['mesh']}", "    patch: coupled",
               f"    write: [{', '.join(s['write'])}]",
               f"    read: [{', '.join(s['read'])}]"]
    with open(os.path.join(cdir, "coupling.yml"), "w") as fh:
        fh.write("\n".join(ym) + "\n")
    # sanity: every participant's provide-mesh must be a mesh the config declares
    meshes = set(re.findall(r'<mesh name="([^"]+)"', cfg))
    for s in specs:
        assert s["mesh"] in meshes, f"{name}/{s['name']}: mesh {s['mesh']!r} not in config {sorted(meshes)}"
    print(f"  {name}: {[cn for cn, *_ in kids]}  meshes={sorted(meshes)}  interfaces={[(s['name'], s['mesh']) for s in specs]}")


def main():
    root = tempfile.mkdtemp(prefix="coupling-examples-")
    cases = os.path.join(root, "cases")
    os.makedirs(cases)
    print("regenerating coupled example children:")

    # SME<->SME: sme_in provides Mesh0, sme_out provides Mesh1 (from the config).
    build(cases, "example_sme_sme", [
        ("sme_in",  "sme", sme_child("sme_in (SME participant)",  "Coupled SME arm (inflow).")),
        ("sme_out", "sme", sme_child("sme_out (SME participant)", "Coupled SME arm (outflow).")),
    ])
    # SME<->VOF: sme provides SweMesh, vof provides VofInletMesh.
    build(cases, "example_sme_vof", [
        ("sme", "sme", sme_child("sme (SME participant)", "Coupled SME arm.")),
        ("vof", "vof", vof_child("vof (VOF participant)", "Coupled incompressible VOF.")),
    ])
    # SME<->VOF<->SME: sme_in->Mesh0, vof->Mesh1, sme_out->Mesh2; the middle VOF
    # reads BOTH arms (Mesh0 + Mesh2) — recorded in coupling.yml + injected.
    build(cases, "example_sme_vof_sme", [
        ("sme_in",  "sme", sme_child("sme_in (SME inflow)",  "Coupled SME inflow arm.")),
        ("vof",     "vof", vof_child("vof (VOF confluence)", "Coupled VOF, two SME interfaces.")),
        ("sme_out", "sme", sme_child("sme_out (SME outflow)", "Coupled SME outflow arm.")),
    ])

    with open(os.path.join(root, "project.json"), "w") as fh:
        json.dump({"version": 1, "cases": []}, fh)

    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _dirs, files in os.walk(root):
            for f in sorted(files):
                fp = os.path.join(base, f)
                z.write(fp, os.path.relpath(fp, root))
    shutil.rmtree(root, ignore_errors=True)
    print("zip ->", ZIP)


if __name__ == "__main__":
    main()
