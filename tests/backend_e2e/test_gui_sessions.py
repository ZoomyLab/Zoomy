"""Every GUI session / solver card through its backend container, as the GUI runs it.

The case.py is composed by the GUI's own code (compose.mjs), submitted to the backend's
zoomy-server over the GUI's HTTP protocol (harness.GuiClient), the returned store is opened by
the GUI's reader and the selected viewer card (checks.gui_view), and its numbers are checked:
every float finite, at least two snapshots, the time_end reached, the conserved quantity within
EXPECT. In dev mode the
case also runs on the host (HOST_RUN) and the two stores are compared. compose.mjs also says
where the GUI's Run sends the case: numpy to the browser, every other tag to its backend.
"""
import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import checks as C  # noqa: E402
import harness as H  # noqa: E402
import sessions as S  # noqa: E402

TIMEOUT = float(os.environ.get("E2E_JOB_TIMEOUT", "840"))
#: same store in-process and in the container: max relative difference (round-off; JAX and
#: XLA versions of host and image may differ)
SAME = float(os.environ.get("E2E_SAME_TOL", "1e-9"))
#: the same case on the host, through engine.py: NumPy / JAX in-process; AMReX / foam with the
#: host Python driving the C++ build and run through apptainer, in the same image
HOST_RUN = {"numpy": (), "jax": ("zoomy_jax",), "amrex": ("zoomy_amrex",),
            "openfoam": ("zoomy_foam",)}
#: the store the GUI keeps of a browser run (writeRunOutputs): the one the run opened
#: (a solver card's open_hdf5), else the simulation.h5 it left
KEEP_STORE = ("import os, shutil\n_p = store_source_path()\n"
              "if _p and not os.path.exists('simulation.h5'):\n"
              "    shutil.copyfile(_p, 'simulation.h5')")


def _host_env(backend, container, work):
    """Environment of a host run: the C++ backends' image and scratch build caches."""
    if backend == "amrex":
        d = work / "images"
        d.mkdir(parents=True, exist_ok=True)
        (d / "zoomy_amrex.sif").symlink_to(container.image)
        return {"ZOOMY_AMREX_CONTAINERS": str(d), "ZOOMY_AMREX_SYMCACHE": str(work / "amrex_cache"),
                "ZOOMY_AMREX_BUILD_CPUS": H.cpus(), "ZOOMY_AMREX_RUN_CPUS": H.cpus(),
                "ZOOMY_AMREX_IN_IMAGE": "0"}
    if backend == "openfoam":
        return {"ZOOMY_OF_SIF": container.image, "ZOOMY_FOAM_BINCACHE": str(work / "foam_cache"),
                "ZOOMY_FOAM_IN_IMAGE": "0"}
    return {}


def _params():
    out = []
    for name in H.selected_backends():
        b = H.BACKENDS[name]
        for case in sorted(S.for_backend(b.adapter), key=lambda c: not c.qr_wired):
            if os.environ.get("E2E_QR_ONLY") == "1" and not case.qr_wired:
                continue
            marks = [pytest.mark.backend_e2e]
            out.append(pytest.param(name, case, marks=marks, id=f"{name}-{case.id}"))
    return out


def _fail(msg, row, report):
    row["result"], row["reason"] = "FAIL", msg
    report.append(row)
    pytest.fail(msg, pytrace=False)


@pytest.mark.parametrize("backend,case", _params())
def test_gui_case(backend, case, servers, lib, report, tmp_path, request):
    row = {"test": request.node.name, "backend": backend, "case": case.id, "qr": list(case.qr),
           "time_end": case.time_end}
    if case.skip:
        row["result"], row["reason"] = "SKIP", case.skip
        report.append(row)
        pytest.skip(case.skip)
    url = servers(backend)
    for peer in S.PEERS.get(case.id, ()):      # the backends its job submits participants to
        servers(peer)
    composed = S.compose([case], lib)[0]
    case_py, spec = composed["casePy"], composed["spec"]
    (tmp_path / "case.py").write_text(case_py)
    row["gui_route"] = composed["route"]       # where the GUI's Run sends it
    settings = dict(spec.get("settings") or {}, **(spec.get("solver") or {}).get("params", {}))

    # the backend run, exactly the GUI's protocol
    t0 = time.monotonic()
    try:
        res = H.GuiClient(url).submit_case(case_py, timeout=TIMEOUT)
    except H.JobFailed as e:
        row["seconds"] = round(time.monotonic() - t0, 1)
        tail = [ln for ln in str(e).strip().splitlines() if ln.strip()][-12:]
        if "CalledProcessError" in str(e):        # mesh.py: its output is in the server log
            tail += ["--- server log ---", *servers.info[backend][4].tail(8).splitlines()]
        _fail("backend job failed:\n  " + "\n  ".join(tail), row, report)
    except Exception as e:                     # e.g. HTTP 404: the job left no simulation.h5
        row["seconds"] = round(time.monotonic() - t0, 1)
        _fail(f"{type(e).__name__}: {e}", row, report)
    row["seconds"] = res["seconds"]
    h5 = tmp_path / "zoomy_sim" / f"{res['job_id']}.h5"   # where the GUI writes it
    h5.parent.mkdir()
    h5.write_bytes(res["hdf5"])

    # the numbers
    s = C.read_store(h5)
    problems = []
    if s["nonfinite"]:
        problems.append(f"{s['nonfinite']} of {s['values']} floats non-finite")
    t_end = None if case.id in S.TIME_UNITS else settings.get("time_end")
    if case.id in S.TIME_UNITS:
        row["time_unit"] = S.TIME_UNITS[case.id]
    if t_end is not None and s["times"] and abs(s["times"][-1] - float(t_end)) > 1e-9 * max(1.0, float(t_end)):
        problems.append(f"t_last {s['times'][-1]!r} != time_end {t_end!r}")
    row.update(snapshots=len(s["times"]), t_last=s["times"][-1] if s["times"] else None,
               names=s["names"])
    # a finished run: the QR suite's check (>= 2 snapshots, the last at time_end); a case with no
    # time_end (the replayed coupling, the Bingham analytics summary) is held to the snapshots
    why = C.check_final_store(row["snapshots"], row["t_last"], S.END_TIME.get(case.id, t_end),
                              min_snapshots=1 if case.id in S.SUMMARY_STORE else 2)
    if why:
        problems.append(f"the store {why}")
    exp = S.EXPECT.get(case.id)
    if exp and s["times"]:
        try:
            dm = C.mass_change(s, exp["row"])
        except (ValueError, IndexError, KeyError):
            problems.append(f"no row {exp['row']!r} in the store (names {s['names']})")
        else:
            row["mass_change"] = dm
            if dm > exp["tol"]:
                problems.append(f"rel. change of {exp['row']} {dm:.3e} > {exp['tol']:.0e}")

    # what the GUI then does with it: engine.open_hdf5 + the selected viewer card
    viewer = next((c for c in json.loads((lib / "zoomy_gui/cards/visualizations/default.json").read_text())
                   if c["id"] == (spec.get("visualization") or {}).get("card")), None)
    snippet = (lib / "zoomy_gui" / viewer["snippet"]).read_text() if viewer and viewer.get("snippet") else None
    gui = C.gui_view(h5, snippet, lib, tmp_path / "gui")
    row["gui"] = C.cell_report(gui)
    if not gui["ok"]:
        problems.append(f"GUI side: {gui['error']}")

    # the same case on the host (dev mode: the same library)
    container = servers.info[backend][4]
    if case.id in S.NO_HOST_RUN:
        row["host_run"] = f"none: {S.NO_HOST_RUN[case.id]}"
    elif H.MODE == "dev" and backend in HOST_RUN and (backend in ("numpy", "jax") or container.rt == "apptainer"):
        cells = [(r, c) for r, c in S.cells(case_py) if r not in ("visualization", "postproc")]
        cells.append(("outputs", KEEP_STORE))
        t1 = time.monotonic()
        rep = C.run_cells(cells, lib, tmp_path / "host", TIMEOUT, extra=HOST_RUN[backend],
                          env=_host_env(backend, container, tmp_path))
        row["host_seconds"] = round(time.monotonic() - t1, 1)
        own = C.find_store(tmp_path / "host")
        if not rep["ok"] or own is None:
            row["host_run"] = f"no comparison: {rep.get('error') or 'no simulation.h5'}"
        else:
            d = C.compare(C.read_store(own), s)
            row["host_run"] = d
            if d > SAME:
                problems.append(f"host run vs container: max rel diff {d:.3e} > {SAME:.0e}")

    want = "browser" if S.canon(case.tag) == "numpy" else "backend"
    if composed["route"] != want:
        problems.append(f"the GUI's Run sends it to the {composed['route']}, not the {want} "
                        "(runAssembly, zoomy_cli cards.mjs solverRoute)")
    if problems:
        _fail("; ".join(problems), row, report)
    row["result"] = "PASS"
    report.append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "names"}, default=str))
