"""Run a composed case folder the way the GUI runs the case.

The GUI's Run executes the case's cells in ONE namespace, in order: the model (with the
case's own steps), the mesh, the solver settings, the run. `zoomy_prepost.case.to_folder`
writes them as model.py (model + steps), mesh.py, settings.json and run.py; each is a cell, not
a standalone script (a mesh cell may use the model cell's names, a run cell the steps' `nsm`).
Here they run in that order in one namespace, with the GUI runtime's helpers (engine.py):
`display` (no output on a server), `typed_kwargs`, `run_h5_path`, where a solver card writes
the run's store (the GUI: the tab's `/tmp/zoomy_sim/sim.h5`; here the job's own
`simulation.h5`, so concurrent jobs share no file), and `open_hdf5`, which publishes the store
in the GUI and here makes it the job's `simulation.h5`. As in the GUI, `__name__` is not
`"__main__"`.

    python -m zoomy_server.case_run          # in the case folder
"""
import json
import os
import shutil
import sys

#: the case's sections, in the order the GUI runs its cells
SECTIONS = ("model.py", "mesh.py", "settings.json", "run.py")
#: the store the server serves for a job
ARTIFACT = "simulation.h5"


def is_case(folder):
    """A composed case (model / mesh sections next to run.py), not a folder that only ships its
    own runner (a coupling folder's run.py)."""
    return any(os.path.exists(os.path.join(folder, f)) for f in ("model.py", "mesh.py"))


def entry(folder):
    """The command that runs `folder`'s run.py: the composed case as the GUI runs it, else the
    runner script itself."""
    if is_case(folder):
        return [sys.executable, "-m", "zoomy_server.case_run"]
    return [sys.executable, "run.py"]


def namespace():
    """The cells' namespace with the GUI runtime's helpers."""
    from zoomy_core.misc.options import typed_kwargs

    ns = {"__name__": "zoomy_case", "store": None}

    def display(*_a, **_k):
        pass

    def run_h5_path():
        """Where a solver card writes the run's store: the job's simulation.h5 (the case
        folder is the job's directory)."""
        return os.path.abspath(ARTIFACT)

    def open_hdf5(path):
        """The run's store: the job's simulation.h5 (published as `store` when zoomy_plotting is
        installed, as the GUI does)."""
        if os.path.abspath(path) != os.path.abspath(ARTIFACT):
            shutil.copyfile(path, ARTIFACT)
        try:
            import zoomy_plotting as zp
        except ImportError:
            return None
        ns["store"] = zp.read_hdf5(ARTIFACT)
        return ns["store"]

    ns.update(display=display, run_h5_path=run_h5_path, open_hdf5=open_hdf5,
              typed_kwargs=typed_kwargs)
    return ns


def run(folder="."):
    # zoomy_core resolves relative outputs against its main directory: the case folder, as the
    # browser's cwd is (no git root, no __main__ script there)
    os.environ["ZOOMY_DIR"] = os.path.abspath(folder)
    ns = namespace()
    for name in SECTIONS:
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            src = f.read()
        if name == "settings.json":
            ns["settings"] = json.loads(src)
        else:
            exec(compile(src, path, "exec"), ns)


if __name__ == "__main__":
    run()
