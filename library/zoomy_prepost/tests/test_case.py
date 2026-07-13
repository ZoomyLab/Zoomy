"""Round-trip tests for the zoomy case interchange format (compose/parse/to_folder/notebook)."""
import json
import os
import tempfile

from zoomy_prepost import compose, parse, to_folder, to_notebook, from_notebook

SPEC = {
    "meta": {"title": "SWE dam break", "description": "test case"},
    "model": {"class_path": "zoomy_core.model.models.SME", "init": {"level": 2, "dimension": 2}},
    "mesh": {"spec": {"type": "create_2d", "domain": [0, 10, 0, 10], "nx": 32, "ny": 32}},
    "settings": {"time_end": 0.6, "cfl": 0.45, "output_snapshots": 10},
    "solver": {"tag": "jax", "params": {"reconstruction_order": 1}},
}


def test_compose_parse_roundtrip():
    py = compose(SPEC)
    assert "# %%" in py and "zoomy=" in py
    spec2 = parse(py)
    # v2: parse returns the section code too (headings-first); hints preserved
    assert spec2["model"]["class_path"] == "zoomy_core.model.models.SME"
    assert spec2["model"]["init"] == {"level": 2, "dimension": 2}
    assert "model = SME(" in spec2["model"]["code"]
    assert spec2["settings"]["time_end"] == 0.6
    assert spec2["solver"]["tag"] == "jax"
    assert spec2["mesh"]["spec"]["type"] == "create_2d"
    print("OK compose/parse round-trip")


def test_to_folder():
    py = compose(SPEC)
    d = tempfile.mkdtemp()
    to_folder(py, d)
    assert os.path.exists(os.path.join(d, "model.py"))
    assert os.path.exists(os.path.join(d, "mesh.py"))
    s = json.load(open(os.path.join(d, "settings.json")))
    assert s["time_end"] == 0.6 and s["cfl"] == 0.45
    model_src = open(os.path.join(d, "model.py")).read()
    assert "model = SME(level=2, dimension=2)" in model_src
    mesh_src = open(os.path.join(d, "mesh.py")).read()
    assert "create_2d" in mesh_src and "mesh.h5" in mesh_src and "mesh.msh" in mesh_src
    # the run section IS materialized: run.py = the case's own runner (the
    # server's generic runner executes exactly this file)
    run_src = open(os.path.join(d, "run.py")).read()
    assert "solver.solve(" in run_src and "simulation.h5" in run_src
    # standalone form: loads model/mesh/settings from the sibling files
    assert 'exec(open("model.py").read())' in run_src
    print(f"OK to_folder -> {sorted(os.listdir(d))}")


def test_notebook_roundtrip():
    py = compose(SPEC)
    ipynb = to_notebook(py)
    assert '"cells"' in ipynb
    py2 = from_notebook(ipynb)
    assert parse(py2)["model"] == parse(py)["model"]
    assert parse(py2)["settings"] == parse(py)["settings"]
    print("OK .py <-> .ipynb round-trip")


def test_headings_structure_and_viz():
    """v2: markdown headings are the structure; Visualization is optional."""
    spec = dict(SPEC)
    spec["visualization"] = {"code": "import matplotlib.pyplot as plt\nplt.plot([1, 2])"}
    py = compose(spec)
    # option A: one merged "Solver settings" section (no separate "## Solver")
    for h in ("## Model", "## Mesh", "## Solver settings", "## Visualization"):
        assert h in py, h
    assert "solver_tag" not in py
    assert '"backend": "jax"' in py
    s2 = parse(py)
    assert s2["visualization"]["code"].startswith("import matplotlib")
    # viz is ALWAYS attached: no card selected -> the generated default plot
    py0 = compose(SPEC)
    assert "## Visualization" in py0
    assert "MatplotlibPlotter" in parse(py0)["visualization"]["code"]
    print("OK headings + always-attached visualization (default fallback)")


def test_parse_headings_only_no_metadata():
    """A hand-written notebook with ONLY headings (no zoomy metadata) parses."""
    py = '''# %% [markdown]
# # My hand case

# %% [markdown]
# ## Model

# %%
model = 42

# %% [markdown]
# ## Settings

# %%
settings = {"time_end": 0.25, "cfl": 0.5}

# %% [markdown]
# ## Solver

# %%
solver_tag = "jax"
'''
    s = parse(py)
    assert s["model"]["code"] == "model = 42"
    assert s["settings"] == {"time_end": 0.25, "cfl": 0.5}
    assert s["solver"]["tag"] == "jax"
    assert s["meta"]["title"] == "My hand case"
    print("OK heading-only parse (no metadata)")


def test_to_folder_writes_visualize():
    spec = dict(SPEC)
    spec["visualization"] = {"code": "print('viz')"}
    d = tempfile.mkdtemp()
    to_folder(compose(spec), d)
    assert open(os.path.join(d, "visualize.py")).read().strip() == "print('viz')"
    # and from_folder picks it back up (run.py too, as the run section)
    from zoomy_prepost import from_folder
    s2 = parse(from_folder(d))
    assert s2["visualization"]["code"] == "print('viz')"
    assert s2["run"]["code"] == parse(compose(spec))["run"]["code"]
    print("OK visualize.py + run.py folder round-trip")


if __name__ == "__main__":
    test_compose_parse_roundtrip()
    test_to_folder()
    test_notebook_roundtrip()
    print("\n=== composed .py ===")
    print(compose(SPEC))
