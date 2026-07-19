#!/usr/bin/env python
"""Prove the GENERIC RUNNER path: a composed case whose materialized folder
carries run.py (the ## Run section) is executed by the server AS the case's
own solver code — not translated into adapter solver calls.

    python test_runner_case.py [port]        # default 8086

Flow: compose (SME model card + mesh-create-1d card + settings) -> to_folder
(assert run.py materialized) -> start `python -m zoomy_server.cli --adapter
numpy` -> POST /cases -> poll -> GET /results/hdf5 (verify fields+mesh) ->
assert the server log shows "runner: executing case run.py" and the job's
run.log captured the ## Run cell's own "done -> simulation.h5" print.

Companion of test_apptainer_server.sh, which proves the LEGACY path (a case
folder without run.py still runs through the settings-translation adapters).
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CARDS = os.path.join(HERE, "../../../library/zoomy_gui/cards")
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8086
BASE = f"http://127.0.0.1:{PORT}/api/v1"
LOG = os.path.join(tempfile.gettempdir(), f"zserve_runner_{PORT}.log")


def http(method, url, payload=None, timeout=8):
    req = urllib.request.Request(url, method=method)
    data = None
    if payload is not None:
        req.add_header("content-type", "application/json")
        data = json.dumps(payload).encode()
    with urllib.request.urlopen(req, data=data, timeout=timeout) as r:
        return r.read()


def main():
    # -- compose the case from the GUI card templates --------------------
    sme = json.load(open(os.path.join(CARDS, "models/default.json")))[0]
    mesh_card = [c for c in json.load(open(os.path.join(CARDS, "meshes/default.json")))
                 if c["id"] == "mesh-create-1d"][0]
    spec = {
        "meta": {"title": "Generic-runner test", "description": "SME dam break via run.py"},
        "model": {"class_path": sme["class"], "init": sme["init"], "code": sme["template"]},
        "mesh": {"code": mesh_card["template"].format(x_min=0, x_max=10, n_cells=100)},
        "settings": {"time_end": 0.1, "cfl": 0.45, "output_snapshots": 5,
                     "numpy": {"reconstruction_order": 1}},
        "solver": {"tag": "numpy"},
    }
    from zoomy_prepost import compose, to_folder
    case_py = compose(spec)
    d = tempfile.mkdtemp(prefix="zoomy_runner_case_")
    to_folder(case_py, d)
    assert os.path.exists(os.path.join(d, "run.py")), "run.py NOT materialized"
    print(f"OK composed case materialized with run.py -> {sorted(os.listdir(d))}")

    # -- start the host server -------------------------------------------
    subprocess.run(["fuser", "-k", f"{PORT}/tcp"], capture_output=True)
    time.sleep(1)
    srv = subprocess.Popen(
        [sys.executable, "-m", "zoomy_server.cli", "--adapter", "numpy",
         "--port", str(PORT)],
        stdout=open(LOG, "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(60):
            try:
                print("health:", http("GET", f"{BASE}/health").decode())
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError(f"server never became healthy; log: {LOG}")

        # -- submit the composed .py via /cases (the GUI's endpoint) ------
        job = json.loads(http("POST", f"{BASE}/cases", {"case_py": case_py}))["job_id"]
        print("job_id =", job)

        status = None
        for _ in range(120):
            status = json.loads(http("GET", f"{BASE}/jobs/{job}"))
            if status["status"] in ("complete", "failed"):
                break
            time.sleep(1)
        assert status and status["status"] == "complete", f"job did not complete: {status}"
        print("job complete")

        # -- verify the HDF5 result ---------------------------------------
        h5 = http("GET", f"{BASE}/jobs/{job}/results/hdf5", timeout=30)
        h5_path = os.path.join(tempfile.gettempdir(), f"runner_{PORT}.h5")
        open(h5_path, "wb").write(h5)
        import h5py
        with h5py.File(h5_path) as f:
            groups = sorted(f.keys())
            assert "fields" in groups and "mesh" in groups, groups
            n_snap = len(f["fields"])
        print(f"OK hdf5 has fields ({n_snap} snapshots) + mesh: {groups}")

        # -- prove the run.py path executed -------------------------------
        log = open(LOG).read()
        assert "runner: executing case run.py" in log, \
            f"generic-runner log line missing — see {LOG}"
        print("OK server log shows: runner: executing case run.py")

        run_log = os.path.join(tempfile.gettempdir(), "zoomy_jobs", job, "run.log")
        captured = open(run_log).read()
        assert "done -> simulation.h5" in captured, \
            f"## Run cell's own print missing from {run_log}"
        print("OK run.log captured the ## Run cell's 'done -> simulation.h5'")

        print(f"\nRUNNER LOOP OK: the case's own run.py was executed by the "
              f"'numpy' server on port {PORT} and produced the served HDF5.")
    finally:
        srv.terminate()
        subprocess.run(["fuser", "-k", f"{PORT}/tcp"], capture_output=True)


if __name__ == "__main__":
    main()
