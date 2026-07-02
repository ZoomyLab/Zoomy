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
    assert spec2["model"] == {"class_path": "zoomy_core.model.models.SME",
                              "init": {"level": 2, "dimension": 2}}
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
    print(f"OK to_folder -> {sorted(os.listdir(d))}")


def test_notebook_roundtrip():
    py = compose(SPEC)
    ipynb = to_notebook(py)
    assert '"cells"' in ipynb
    py2 = from_notebook(ipynb)
    assert parse(py2)["model"] == parse(py)["model"]
    assert parse(py2)["settings"] == parse(py)["settings"]
    print("OK .py <-> .ipynb round-trip")


if __name__ == "__main__":
    test_compose_parse_roundtrip()
    test_to_folder()
    test_notebook_roundtrip()
    print("\n=== composed .py ===")
    print(compose(SPEC))
