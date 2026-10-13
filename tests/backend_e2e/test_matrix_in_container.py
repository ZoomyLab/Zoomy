"""A subset of the cross-backend matrix INSIDE each backend container, against the host.

matrix_driver.py runs each case's SUBSET runs with matrix.py's own `run` inside the image (its
Python; foam / AMReX: its OpenFOAM / AMReX toolchain) into a container store, and on the host
into a host store (`E2E_MATRIX_HOST_STORE`: an existing store instead, e.g. the checkout's
docs/dev/symbolic_solver/matrix/store). Every stored array (t, Q, Q0, xc, vol; custom runs:
the metrics) must agree to E2E_MATRIX_TOL; the timing / provenance keys are ignored.
"""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

#: case -> runs (a backend runs those its cases.py Run lists it for)
SUBSET = {"C1": ("o1_n20", "o2_n20"),                          # SWE bump, O1 / O2
          "C8": ("vam",),                                      # VAM(1) Chorin, pressure solve
          "C28": ("quad_expl_dir0", "quad_impl_neu",           # heat equation BCs (foam, AMReX,
                  "orders_quad_expl"),                         # numpy) / orders (JAX)
          # built without jax (C8, C28 are not: gates 7 / 21 / 41 / 43 import it)
          "C32": ("noslip_n20",),                              # implicit diffusion, no-slip walls
          "C33": ("imex_k0",)}                                 # implicit source, SME(2)
TOL = float(os.environ.get("E2E_MATRIX_TOL", "1e-9"))          # the matrix's pressure tolerance
TIMEOUT = float(os.environ.get("E2E_MATRIX_TIMEOUT", "840"))
IGNORE = {"wall", "lib"}


def _params():
    if os.environ.get("E2E_QR_ONLY") == "1":
        return []
    return [pytest.param(b, c, marks=pytest.mark.backend_e2e, id=f"{b}-{c}")
            for b in H.selected_backends() for c in SUBSET]


def _result(log: Path) -> dict:
    lines = [ln for ln in log.read_text(errors="replace").splitlines() if ln.startswith("E2E_RESULT ")]
    return json.loads(lines[-1].split(" ", 1)[1]) if lines else {}


def _rel(a, b) -> float:
    """max|a - b| / max|b| (matrix.py's relmax) of two stored values, recursing into metrics."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return math.inf
        return max((_rel(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return math.inf
        return max((_rel(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, str) or isinstance(b, str):
        return 0.0 if a == b else math.inf
    if a is None or b is None or isinstance(a, bool):
        return 0.0 if a == b else math.inf
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return math.inf
    if a.size == 0:
        return 0.0
    if np.isnan(a).any() or np.isnan(b).any():
        return 0.0 if np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(
            a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=0) else math.inf
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))   # matrix relmax


def compare(pa: Path, pb: Path) -> dict:
    """`{stored key: rel diff}` of two matrix.py npz runs, IGNORE dropped."""
    a, b = dict(np.load(pa, allow_pickle=True)), dict(np.load(pb, allow_pickle=True))
    out = {}
    for k in sorted((set(a) | set(b)) - IGNORE):
        if k not in a or k not in b:
            out[k] = math.inf
        elif k == "metrics":
            out[k] = _rel(json.loads(str(a[k])), json.loads(str(b[k])))
        elif a[k].dtype.kind in "US":
            out[k] = 0.0 if np.array_equal(a[k], b[k]) else math.inf
        else:
            out[k] = _rel(a[k], b[k])
    return out


def _host(backend, image, snap, case, keys, base) -> tuple[Path, dict, float]:
    """The same runs on the host (this Python, the snapshot library; foam / AMReX through
    apptainer with the same image)."""
    store = base / "host" / "store"
    for d in ("amrex_cache", "foam_cache", "run"):
        (base / "host" / d).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONNOUSERSITE="1", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join(str(snap / "library" / p) for p in
                                          ("zoomy_core", "zoomy_jax", "zoomy_foam", "zoomy_amrex")),
               ZOOMY_AMREX_SYMCACHE=str(base / "host" / "amrex_cache"),
               ZOOMY_FOAM_BINCACHE=str(base / "host" / "foam_cache"),
               ZOOMY_AMREX_BUILD_CPUS=H.cpus(), ZOOMY_AMREX_RUN_CPUS=H.cpus())
    if backend.name == "openfoam":
        env["ZOOMY_OF_SIF"] = image
    if backend.name == "amrex":
        d = base / "host" / "images"
        d.mkdir(parents=True, exist_ok=True)
        link = d / "zoomy_amrex.sif"
        if not link.exists():
            link.symlink_to(image)
        env["ZOOMY_AMREX_CONTAINERS"] = str(d)
    log = base / f"host-{case}.log"
    t0 = time.monotonic()
    with open(log, "w") as f:
        subprocess.run([sys.executable, str(H.HERE / "matrix_driver.py"), str(snap), str(store),
                        str(base / "host" / "run"), backend.matrix, case, keys],
                       env=env, stdout=f, stderr=subprocess.STDOUT, timeout=TIMEOUT)
    return store, _result(log), time.monotonic() - t0


@pytest.mark.parametrize("backend,case", _params())
def test_matrix_in_container(backend, case, snap, report, request):
    b = H.BACKENDS[backend]
    img, how = H.image(b)
    if img is None:
        pytest.skip(how)
    if (b.name in ("openfoam", "amrex") and H.runtime() != "apptainer"
            and not os.environ.get("E2E_MATRIX_HOST_STORE")):
        pytest.skip("the host reference runs foam / AMReX through apptainer "
                    "(set E2E_MATRIX_HOST_STORE to compare with an existing store)")
    base = H.WORK / f"{H.CHECKOUT.name}-{H.MODE}" / backend / "matrix"
    base.mkdir(parents=True, exist_ok=True)
    keys = ",".join(SUBSET[case])
    row = {"test": request.node.name, "backend": backend, "case": case, "image": how}

    c = H.Container(b, img, snap, base / "container", dev=H.MODE == "dev")
    log = base / f"container-{case}.log"
    t0 = time.monotonic()
    c.run(["python3", "/e2e/harness/matrix_driver.py", "/zoomy", "/e2e/work/store", "/e2e/work/run",
           b.matrix, case, keys, "--in-container"], TIMEOUT, log)
    row["container_seconds"] = round(time.monotonic() - t0, 1)
    got = _result(log)
    row["container"] = got
    if not got:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-25:])
        row["result"] = "FAIL"
        report.append(row)
        pytest.fail(f"no result from the container run:\n{tail}", pytrace=False)

    if os.environ.get("E2E_MATRIX_HOST_STORE"):
        hstore, hres = Path(os.environ["E2E_MATRIX_HOST_STORE"]), None
    elif "ok" not in got.values():                    # nothing to compare with
        hstore, hres = base / "host" / "store", None
    else:
        hstore, hres, secs = _host(b, img, snap, case, keys, base)
        row["host_seconds"], row["host"] = round(secs, 1), hres
    cstore = base / "container" / "store"
    problems, diffs = [], {}
    for k, status in got.items():
        if status == "n/a":
            continue
        if status != "ok":
            if "No module named 'jax'" in status:
                status = "the case builder imports jax, which this image does not ship"
            problems.append(f"container {k}: {status[:400]}")
            continue
        hp = hstore / b.matrix / "serial" / case / f"{k}.npz"
        if not hp.exists():
            problems.append(f"host {k}: {(hres or {}).get(k, 'not in ' + str(hstore))}")
            continue
        d = compare(cstore / b.matrix / "serial" / case / f"{k}.npz", hp)
        diffs[k] = max(d.values()) if d else 0.0
        bad = {kk: v for kk, v in d.items() if v > TOL}
        if bad:
            problems.append(f"{k}: container vs host {bad} > {TOL:.0e}")
    row["max_rel_diff"] = diffs
    if not diffs and not problems:
        problems.append("no run of this case for this backend")
    row["result"] = "FAIL" if problems else "PASS"
    report.append(row)
    print(json.dumps(row, default=str))
    if problems:
        pytest.fail("; ".join(problems), pytrace=False)
