"""`zoomy_server.case_run`: a composed case folder runs as the GUI runs the case (its sections in
one namespace, in order, not as `__main__`); a folder that only ships a run.py runs it as a script.
"""
import json
import os
import subprocess
import sys

from zoomy_server.case_run import entry, is_case


def _folder(tmp_path, files):
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return tmp_path


def _run(folder):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    return subprocess.run(entry(str(folder)), cwd=folder, env=env, capture_output=True, text=True)


def test_sections_share_one_namespace(tmp_path):
    d = _folder(tmp_path, {
        "model.py": "wave_length = 3.0\nnsm = 'step'\nif __name__ == '__main__':\n    raise SystemExit(9)\n",
        "mesh.py": "n_cells = int(10 * wave_length)\n",
        "settings.json": json.dumps({"time_end": 0.5}),
        "run.py": "import json\njson.dump({'n': n_cells, 'nsm': nsm, 't': settings['time_end'],\n"
                  "           'name': __name__, 'zoomy_dir': __import__('os').environ['ZOOMY_DIR']},\n"
                  "          open('out.json', 'w'))\ndisplay('no output on a server')\n"})
    assert is_case(str(d))
    p = _run(d)
    assert p.returncode == 0, p.stderr
    out = json.loads((d / "out.json").read_text())
    assert out == {"n": 30, "nsm": "step", "t": 0.5, "name": "zoomy_case",
                   "zoomy_dir": os.path.abspath(d)}


def test_runner_only_folder_runs_as_a_script(tmp_path):
    d = _folder(tmp_path, {"run.py": "if __name__ == '__main__':\n    open('ran', 'w').write('x')\n"})
    assert not is_case(str(d)) and entry(str(d))[-1] == "run.py"
    assert _run(d).returncode == 0 and (d / "ran").exists()


def test_run_h5_path_is_the_jobs_own_store(tmp_path):
    """Two jobs' solver cards write their stores to their own folders, not one shared file."""
    paths = []
    for job in ("a", "b"):
        d = tmp_path / job
        d.mkdir()
        _folder(d, {"mesh.py": "pass\n",
                    "run.py": "import json\njson.dump(run_h5_path(), open('path.json', 'w'))\n"})
        assert _run(d).returncode == 0
        paths.append(json.loads((d / "path.json").read_text()))
    assert paths == [str(tmp_path / j / "simulation.h5") for j in ("a", "b")]
