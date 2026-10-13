"""Every GUI session a thesis QR code opens runs, in CPython and in headless Pyodide.

CPython: the composed case cells through the GUI's `engine.py`; browser-local sessions with
jax blocked. Pyodide: the same cells in Pyodide 0.29.3 under node, booted as
`pyodide-worker.js` boots. Both run the zoomy_core / zoomy_plotting wheels the deploy serves
(`docs/scripts/build_gui_wheels.py`): `QR_WHEELS` names the directory (CI: built once and
pip-installed, so CPython imports them too), else the wheels are built here for Pyodide and
CPython reads the checkout. A JAX session adds `zoomy_jax` from the checkout. Pass = every
cell succeeds and every float in every HDF5 store the run wrote is finite. The sessions
carry no reference outputs to compare against.
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qr_manifest as Q  # noqa: E402

HERE = Q.HERE
ENGINE = Q.GUI / "engine.py"
RUN_MJS = HERE / "pyodide" / "run.mjs"
BUILD_WHEELS = Q.ROOT / "docs" / "scripts" / "build_gui_wheels.py"
TIMEOUT = int(os.environ.get("QR_TIMEOUT", "900"))
#: Solver tags the browser runs itself; any other tag is submitted to a backend.
BROWSER_LOCAL = (None, "numpy")
NOT_IN_BROWSER = ["jax", "jaxlib", "zoomy_jax"]
#: Time-limited variants, set through the session's own `time_end` knob (GUI Parameters
#: panel); model, mesh and numerics are unchanged.
TIME_LIMITS = {("malpasset-session.zip", 0): 2.0,    # Malpasset JAX: 2000 s -> 2 s
               ("malpasset-session.zip", 1): 2.0}    # Malpasset AMReX: 2000 s -> 2 s


def _amrex_sif():
    """The AMReX image `zoomy_amrex.symsolver` builds in (its own lookup)."""
    own = Q.ROOT / "containers" / "zoomy_amrex"
    d = Path(os.environ.get("ZOOMY_AMREX_CONTAINERS") or (
        own if (own / "zoomy_amrex.sif").exists() else Path.home() / "git/Zoomy/containers/zoomy_amrex"))
    return d / "zoomy_amrex.sif"


#: Backends this machine cannot provide: the cells before Solver settings run, the solve is
#: skipped.  AMReX runs inside its image (AMREX_HOME) or through apptainer and the image.
UNAVAILABLE = ({} if os.environ.get("AMREX_HOME") or (shutil.which("apptainer") and _amrex_sif().exists())
               else {"amrex": "the solve needs AMReX: its backend image (AMREX_HOME) or apptainer "
                              "+ zoomy_amrex.sif"})


def _skip_or_fail(reason):
    """Missing tooling skips locally; with `QR_STRICT=1` (CI) it fails."""
    (pytest.fail if os.environ.get("QR_STRICT") else pytest.skip)(reason)


def _params(jax_mark):
    out = []
    for zname, idx, sess in Q.sessions():
        marks = [pytest.mark.qr_jax] if jax_mark and Q.backend(sess) == "jax" else []
        out.append(pytest.param(zname, idx, sess, marks=marks,
                                id=f"{zname.removesuffix('-session.zip')}-{idx}"))
    return out


SESSIONS = _params(jax_mark=False)


def _case(zname, idx, session):
    """`(backend tag, case spec)` for the runners."""
    s = json.loads(json.dumps(session))
    if (zname, idx) in TIME_LIMITS:
        ov = s["cardOverrides"][s["selections"]["solver"]]
        ov.setdefault("params", {})["time_end"] = TIME_LIMITS[(zname, idx)]
    tag = Q.backend(s)
    cells = Q.compose(s)
    if tag in UNAVAILABLE:
        cells = cells[:[r for r, _ in cells].index("settings")]
    return tag, {"cells": cells, "needs_store": tag is not None and tag not in UNAVAILABLE,
                 "blocked_modules": NOT_IN_BROWSER if tag in BROWSER_LOCAL else []}


def _verdict(rep, label):
    cells = ", ".join(f"{c['role']}={c['status']} ({c['seconds']}s)" for c in rep["cells"])
    msg = f"{label}: {rep['error']}\n  cells: {cells}"
    if not rep["ok"]:
        last = rep["cells"][-1]["output_tail"] if rep["cells"] else ""
        pytest.fail(msg + "\n  output tail:\n" + last[-2000:], pytrace=False)
    print(msg, "\n  stores:", rep.get("stores"))


# --------------------------------------------------------------------------- static

@pytest.mark.qr
def test_manifest_targets_resolve():
    """Every QR URL is on the site; every linked zip is in the repo and holds the opened path."""
    ts = Q.targets()
    assert ts, f"no targets in {Q.manifest_path()}"
    for t in ts:
        assert t.url.startswith(Q.SITE), t
        if t.project:
            assert t.project.startswith("gui/projects/"), t
            assert t.local_zip.exists(), f"{t.key}: {t.local_zip} missing"
            names = zipfile.ZipFile(t.local_zip).namelist()
            assert t.path is None or t.path in names, f"{t.key}: {t.path} not in {t.local_zip.name}"
            assert t.path is not None or "project.json" in names, f"{t.key}: no project.json"
    assert Q.sessions(), "no GUI sessions behind the QR codes"


@pytest.mark.qr
def test_manifest_snapshot_is_current():
    """The vendored snapshot equals the thesis manifest (when a thesis checkout is present)."""
    src = Path(os.environ["QR_MANIFEST"]) if os.environ.get("QR_MANIFEST") else Q.THESIS_MANIFEST
    if not src.exists():
        pytest.skip("no thesis checkout: the snapshot is the manifest")
    assert Q.load_manifest(src) == Q.load_manifest(Q.SNAPSHOT), \
        f"refresh {Q.SNAPSHOT.relative_to(Q.ROOT)} from {src}"


@pytest.mark.qr
def test_deploy_ships_projects_and_worker_version():
    """render-webpage.yml deploys projects/ next to the worker and the wheels the worker
    installs, built by the script the QR CI installs from; the harness pins its Pyodide."""
    deploy = (Q.ROOT / ".github/workflows/render-webpage.yml").read_text()
    loop = re.search(r"for a in ([^;]+);", deploy)
    assert loop and "projects" in loop.group(1).split(), "deploy no longer copies projects/"
    build = "docs/scripts/build_gui_wheels.py"
    assert f"{build} docs/book/_build/html/gui/gui/wheels" in deploy, "deploy serves no wheels"
    assert build in (Q.ROOT / ".github/workflows/qr-cases.yml").read_text()
    worker = (Q.GUI / "pyodide-worker.js").read_text()
    pinned = json.loads((HERE / "pyodide/package.json").read_text())["dependencies"]["pyodide"]
    assert f"/pyodide/v{pinned}/" in worker, f"worker Pyodide != harness pyodide@{pinned}"


# --------------------------------------------------------------------------- CPython

def _wheel_versions(wheels):
    """`{module: version}` of the wheels in `wheels` (`index.json`)."""
    index = json.loads((Path(wheels) / "index.json").read_text())
    return {n.replace("-", "_"): e["wheel"].split("-")[1] for n, e in index.items()}


def _check_packages(rep, wheels, where):
    """The run imported zoomy_core / zoomy_plotting from the wheels, not the checkout."""
    for mod, version in _wheel_versions(wheels).items():
        got = rep["packages"].get(mod)
        if got is None:
            continue
        assert got["version"] == version, f"{where}: {mod} {got} is not the wheel {version}"
        assert not str(got["file"]).startswith(str(Q.ROOT / "library")), \
            f"{where}: {mod} imported from the checkout: {got['file']}"


@pytest.mark.qr
@pytest.mark.parametrize("zname,idx,session", _params(jax_mark=True))
def test_session_cpython(zname, idx, session, tmp_path):
    tag, case = _case(zname, idx, session)
    installed = os.environ.get("QR_WHEELS")
    path = [] if installed else [str(Q.ROOT / "library/zoomy_core"),
                                 str(Q.ROOT / "library/zoomy_plotting")]
    if tag == "jax":
        if importlib.util.find_spec("jax") is None:
            _skip_or_fail("jax not installed (the JAX session needs it)")
        path.append(str(Q.ROOT / "library/zoomy_jax"))
    if tag == "amrex":
        path.append(str(Q.ROOT / "library/zoomy_amrex"))
    spec = tmp_path / "case.json"
    spec.write_text(json.dumps(case))
    path += [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(path), JAX_PLATFORMS="cpu")
    out = tmp_path / "report.json"
    subprocess.run([sys.executable, str(HERE / "session_runner.py"), str(spec), str(ENGINE),
                    str(out), str(tmp_path / "work")], env=env, timeout=TIMEOUT, check=False)
    assert out.exists(), "session runner wrote no report (crashed)"
    rep = json.loads(out.read_text())
    _verdict(rep, f"CPython {zname}#{idx} [{tag}]")
    if installed:
        _check_packages(rep, installed, "CPython")
    if tag in BROWSER_LOCAL:
        assert not set(rep["modules_loaded"]) & set(NOT_IN_BROWSER)
    if tag in UNAVAILABLE:
        pytest.skip(f"model/mesh cells pass; {UNAVAILABLE[tag]}")


@pytest.mark.qr
def test_c1_bump_cpython(tmp_path):
    """C1 bump on SymbolicHyperbolicSolverNumpy (t = 0.5) in CPython, jax never imported."""
    env = dict(os.environ)
    if not os.environ.get("QR_WHEELS"):
        env["PYTHONPATH"] = str(Q.ROOT / "library/zoomy_core")
    code = "import json, c1_bump; print(json.dumps(c1_bump.main()))"
    p = subprocess.run([sys.executable, "-c", code], cwd=HERE, env=env, capture_output=True,
                       text=True, timeout=TIMEOUT, check=True)
    _check_c1(json.loads(p.stdout.strip().splitlines()[-1]))


def _check_c1(rep):
    m = rep["march"]
    assert "error" not in m, m.get("error")
    assert m["t"] == pytest.approx(0.5, abs=1e-12)
    assert m["finite"] and m["moved"] > 1e-3
    assert m["mass_rel_err"] <= 1e-12, m
    bad = set(rep["imported_optional"]) & {"jax", "jaxlib", "pyamg", "mpi4py", "petsc4py"}
    assert not bad, f"NumPy path imported optional add-ons: {bad}"
    assert not rep["import_failures"], rep["import_failures"]   # every zoomy_core module imports
    print(f"C1 bump [{rep['platform']}]: {m}; optional imported {rep['imported_optional']}, "
          f"absent {rep['absent_optional']}; import failures: {rep['import_failures']}")


# --------------------------------------------------------------------------- Pyodide

@pytest.fixture(scope="session")
def node_pyodide():
    node = os.environ.get("QR_NODE") or shutil.which("node")
    pyo = Path(os.environ.get("QR_PYODIDE_DIR") or HERE / "pyodide/node_modules/pyodide")
    if not node or not (pyo / "pyodide.mjs").exists():
        _skip_or_fail("node + pyodide npm package missing: npm ci --prefix tests/qr_cases/pyodide")
    return node, pyo


@pytest.fixture(scope="session")
def wheels(tmp_path_factory):
    """The deploy's wheel directory: `QR_WHEELS`, else built here from the checkout."""
    if os.environ.get("QR_WHEELS"):
        return Path(os.environ["QR_WHEELS"])
    out = tmp_path_factory.mktemp("wheels")
    subprocess.run([sys.executable, str(BUILD_WHEELS), str(out)], check=True, timeout=TIMEOUT,
                   capture_output=True)
    return out


def _run_node(node_pyodide, wheels, tmp_path, *args):
    node, pyo = node_pyodide
    out = tmp_path / "report.json"
    env = dict(os.environ, QR_PYODIDE_DIR=str(pyo))
    p = subprocess.run([node, str(RUN_MJS), "--wheels", str(wheels), *args, "--out", str(out)],
                       env=env, capture_output=True, text=True, timeout=TIMEOUT)
    if not out.exists():
        pytest.fail(f"Pyodide run crashed (exit {p.returncode}):\n{p.stderr[-4000:]}", pytrace=False)
    return json.loads(out.read_text())


@pytest.mark.qr
@pytest.mark.parametrize("zname,idx,session", SESSIONS)
def test_session_pyodide(zname, idx, session, tmp_path, node_pyodide, wheels):
    tag, case = _case(zname, idx, session)
    if tag not in BROWSER_LOCAL:
        pytest.skip(f"solver card needs a {tag!r} backend: the GUI submits this case, "
                    "the browser does not run it")
    spec = tmp_path / "case.json"
    spec.write_text(json.dumps(case))
    rep = _run_node(node_pyodide, wheels, tmp_path, "--case", str(spec))
    _verdict(rep, f"Pyodide {rep.get('pyodide')} {zname}#{idx} ({rep.get('wall_seconds')}s)")
    _check_packages(rep, wheels, "Pyodide")


@pytest.mark.qr
def test_c1_bump_pyodide(tmp_path, node_pyodide, wheels):
    """C1 bump on SymbolicHyperbolicSolverNumpy (t = 0.5) in Pyodide 0.29.3."""
    _check_c1(_run_node(node_pyodide, wheels, tmp_path, "--script", "c1_bump.py"))
