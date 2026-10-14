"""A subset of the cross-backend matrix INSIDE each backend container, against the host.

matrix_driver.py runs each case's SUBSET runs with matrix.py's own `run` inside the image (its
Python; foam / AMReX: its OpenFOAM / AMReX toolchain) into a container store, and on the host
into a host store (`E2E_MATRIX_HOST_STORE`: an existing store instead, e.g. the checkout's
docs/dev/symbolic_solver/matrix/store). Every stored array (t, Q, Q0, xc, vol) must agree to
the matrix's tolerance for two runs of the case (`tol`); of a custom run's metrics the values
(`VALUES`: J, the derivatives, an error), not the round-off measures; the timing / provenance
keys are ignored. Then the matrix's own checks of the case (`matrix_driver.py --checks`: the
table's thresholds, C6 mass, C7 parallel = serial) must pass on the container's runs.
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

#: case -> runs, serial (a backend runs those its cases.py Run lists it for)
SUBSET = {"C1": ("o1_n20", "o2_n20"),                          # SWE bump, walls, O1 / O2
          "C4": ("sub_o2_n25", "trans_o1_n25"),                # channel: Dirichlet, Wall,
                                                               # InflowOutflow, Extrapolation
          "C8": ("vam",),                                      # VAM(1) Chorin, pressure solve
          "C26": ("visc",),                                    # implicit diffusion, Stress.full
          "C28": ("quad_expl_dir0", "quad_impl_neu",           # heat equation BCs (every backend)
                  "orders_quad_expl"),                         # orders (JAX, numpy)
          "C31": ("swe_ssp2_k0", "vam_ssp2_k0"),               # IMEX-SSP2: SWE, VAM(1) + Manning
          "C32": ("noslip_n20",),                              # implicit diffusion, no-slip walls
          "C33": ("imex_k0",),                                 # implicit source, SME(2)
          # the image-dependent paths, once each (the backends cases.py lists them for)
          "C5": ("n48",),                                      # wet/dry, O2 eta + MOOD
          "C14": ("o1_n20", "o2_n20"),                         # IMEX, implicit Manning source
          "C15": ("hyperbolic_fwd", "hyperbolic_rev",          # AD (JAX): jvp vs FD, reverse
                  "imex_fwd", "imex_rev"),
          "C17": ("cover_o2", "box", "lake", "wave16", "wave32"),   # AMR (AMReX, gate 37)
          "C22": ("o1_dt_fwd", "o1_dt_rev", "o2_dt_fwd", "o2_dt_rev"),   # AD, Euler 3-D (JAX)
          "C23": ("o1_n8", "o2_n8"),                           # Euler 3-D
          "C35": ("c1_full", "c1_half", "c1_restart", "c1_typo")}   # restart (h5 / foam / AMReX)
C35_C1 = SUBSET["C35"]
#: cases that list runs for some backends only (cases.py `backends`): the others are not
#: parametrized (a case that runs nothing on a backend fails, so a change there shows here)
ONLY = {"C15": ("jax",), "C22": ("jax",), "C17": ("amrex",)}
#: distributed runs: backend -> (config, case, runs), launched with the image's mpiexec (JAX:
#: matrix.py; foam / AMReX: their drivers). A marching run is checked as matrix C7 (= the
#: serial run, both in the image); every run by the matrix's checks of its case (C35: the
#: restart bitwise on the ranks; C17: AMR on 2 ranks = 1 rank)
DISTRIBUTED = {"jax": [("mpi2", "C1", ("o2_n40",)), ("mpi2", "C35", C35_C1)],
               "openfoam": [("mpi2", "C1", ("o2_n40",)), ("mpi2", "C35", C35_C1)],
               "amrex": [("mpi2", "C1", ("o2_n40",)), ("mpi2", "C35", C35_C1),
                         ("mpi2", "C17", ("box",))]}
#: custom runs: the metrics that are values (= host to `tol`); the others are measures at
#: round-off (a mass drift, coupled - single) or noise (|FD - jvp|, seconds, messages naming
#: paths), judged by the matrix's checks
VALUES = {"C15": ("J", "fwd", "rev"), "C22": ("J", "fwd", "rev"), "C17": ("L1",), "C35": ("t",)}
#: custom cases whose distributed checks read the serial run (C17: vs the 1-rank run)
SERIAL_REF = ("C17",)
RO = 1e-12                                                     # matrix.py RO
TIMEOUT = float(os.environ.get("E2E_MATRIX_TIMEOUT", "840"))
IGNORE = {"wall", "lib"}


def tol(case: str, key: str) -> float:
    """Container = host to the matrix's tolerance for two runs of a case that may differ in
    round-off (the image's numpy / JAX vs the host's): round-off, or the case's implicit-solve
    tolerance (matrix.py NUMPY_TOL, PRESS, TH["C33 xb"]). `E2E_MATRIX_TOL` overrides."""
    if os.environ.get("E2E_MATRIX_TOL"):
        return float(os.environ["E2E_MATRIX_TOL"])
    if case == "C28":
        return RO if "_expl" in key else 1e-9                  # gate 43: Newton 1e-10
    if case == "C31":
        return 1e-9 if key.startswith("vam") else 1e-8         # PRESS; C14's Newton
    return {"C8": 1e-9, "C14": 1e-8, "C26": 1e-8, "C32": 1e-8, "C33": 1e-10}.get(case, RO)


def _params():
    if os.environ.get("E2E_QR_ONLY") == "1":
        return []
    out = []
    for b in H.selected_backends():
        out += [pytest.param(b, "serial", c, SUBSET[c], marks=pytest.mark.backend_e2e,
                             id=f"{b}-{c}-serial") for c in SUBSET if b in ONLY.get(c, (b,))]
        out += [pytest.param(b, cfg, c, keys, marks=pytest.mark.backend_e2e, id=f"{b}-{c}-{cfg}")
                for cfg, c, keys in DISTRIBUTED.get(b, ())]
    return out


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


def compare(pa: Path, pb: Path, values=None) -> dict:
    """`{stored key: rel diff}` of two matrix.py npz runs, IGNORE dropped; of the metrics the
    keys `values` (None: every one)."""
    a, b = dict(np.load(pa, allow_pickle=True)), dict(np.load(pb, allow_pickle=True))
    out = {}
    for k in sorted((set(a) | set(b)) - IGNORE):
        if k not in a or k not in b:
            out[k] = math.inf
        elif k == "metrics":
            ma, mb = json.loads(str(a[k])), json.loads(str(b[k]))
            for m in (sorted(set(ma) | set(mb)) if values is None else values):
                if m in ma or m in mb:
                    out[f"metrics.{m}"] = (_rel(ma[m], mb[m]) if m in ma and m in mb
                                           else math.inf)
        elif a[k].dtype.kind in "US":
            out[k] = 0.0 if np.array_equal(a[k], b[k]) else math.inf
        else:
            out[k] = _rel(a[k], b[k])
    return out


def _host(backend, image, snap, case, keys, base, config) -> tuple[Path, dict, float]:
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
    log = base / f"host-{case}-{config}.log"
    t0 = time.monotonic()
    with open(log, "w") as f:
        subprocess.run([sys.executable, str(H.HERE / "matrix_driver.py"), str(snap), str(store),
                        str(base / "host" / "run"), backend.matrix, case, keys, config],
                       env=env, stdout=f, stderr=subprocess.STDOUT, timeout=TIMEOUT)
    return store, _result(log), time.monotonic() - t0


def matrix_checks(snap, store, backend, case) -> list[dict]:
    """The matrix's checks of `case` on `backend` over the runs in `store` (on the host)."""
    env = dict(os.environ, PYTHONNOUSERSITE="1", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join(str(snap / "library" / p) for p in
                                          ("zoomy_core", "zoomy_jax", "zoomy_foam", "zoomy_amrex")))
    r = subprocess.run([sys.executable, str(H.HERE / "matrix_driver.py"), str(snap), str(store),
                        "--checks", backend.matrix, case], env=env, capture_output=True,
                       text=True, timeout=600)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("E2E_CHECKS ")]
    if not lines:
        raise RuntimeError(f"matrix checks failed:\n{(r.stdout + r.stderr)[-3000:]}")
    return json.loads(lines[-1].split(" ", 1)[1])


@pytest.mark.parametrize("backend,config,case,runs", _params())
def test_matrix_in_container(backend, config, case, runs, snap, report, request):
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
    keys = ",".join(runs)
    row = {"test": request.node.name, "backend": backend, "case": case, "config": config,
           "image": how}

    c = H.Container(b, img, snap, base / "container", dev=H.MODE == "dev")

    def in_image(cfg):
        log = base / f"container-{case}-{cfg}.log"
        c.run(["python3", "/e2e/harness/matrix_driver.py", "/zoomy", "/e2e/work/store",
               "/e2e/work/run", b.matrix, case, keys, cfg, "--in-container"], TIMEOUT, log)
        return log, _result(log)

    t0 = time.monotonic()
    custom = case in VALUES
    serial = None                                     # the ranks' reference
    if config != "serial" and (not custom or case in SERIAL_REF):
        serial = in_image("serial")[1]
    log, got = in_image(config)
    row["container_seconds"] = round(time.monotonic() - t0, 1)
    row["container"] = got
    if not got:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-25:])
        row["result"] = "FAIL"
        report.append(row)
        pytest.fail(f"no result from the container run:\n{tail}", pytrace=False)

    cstore = base / "container" / "store"
    ref, what = "serial", "container vs host"
    if config != "serial" and custom:                 # the matrix's checks alone
        hstore, hres = None, None
    elif config != "serial":                          # matrix C7: the ranks = serial, in the image
        hstore, hres, what = cstore, serial, f"{config} vs serial (container)"
    elif os.environ.get("E2E_MATRIX_HOST_STORE"):
        hstore, hres = Path(os.environ["E2E_MATRIX_HOST_STORE"]), None
    elif "ok" not in got.values():                    # nothing to compare with
        hstore, hres = base / "host" / "store", None
    else:
        hstore, hres, secs = _host(b, img, snap, case, keys, base, config)
        row["host_seconds"], row["host"] = round(secs, 1), hres
    problems, diffs = [], {}
    for k, status in got.items():
        if status == "n/a":
            continue
        if status != "ok":
            problems.append(f"container {k}: {status[:400]}")
            continue
        if hstore is None:
            diffs[k] = None
            continue
        hp = hstore / b.matrix / ref / case / f"{k}.npz"
        if not hp.exists():
            problems.append(f"reference {k}: {(hres or {}).get(k, 'not in ' + str(hstore))}")
            continue
        d = compare(cstore / b.matrix / config / case / f"{k}.npz", hp, VALUES.get(case))
        diffs[k] = max(d.values()) if d else 0.0
        bad = {kk: v for kk, v in d.items() if v > tol(case, k)}
        if bad:
            problems.append(f"{k}: {what} {bad} > {tol(case, k):.0e}")
    row["max_rel_diff"] = diffs
    if not diffs and not problems:
        problems.append("no run of this case for this backend")
    if diffs:
        checks = [x for x in matrix_checks(snap, cstore, b, case) if x["config"] == config]
        row["checks"] = {f"{x['check']}": x["value"] for x in checks}
        problems += [f"matrix check {x['check']} ({x['config']}): {x['value']:.3e}, need "
                     f"{x['op']} {x['bound']:g}" for x in checks if not x["ok"]]
        if custom and not checks:
            problems.append("no matrix check of the custom runs")
    row["result"] = "FAIL" if problems else "PASS"
    report.append(row)
    print(json.dumps(row, default=str))
    if problems:
        pytest.fail("; ".join(problems), pytrace=False)
