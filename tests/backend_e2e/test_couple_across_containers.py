"""preCICE coupling ACROSS backend containers (matrix C16): each participant of gate 3's split dam
break in its own image, two `apptainer exec` processes on the host network, the m2n:sockets
handshake through a shared exchange directory (bound at the same path in both). The coupled state
must be the single-domain march (on the host, the same fixed step) to round-off and conserve mass:
matrix C16 (gate 26), 1e-14.

Participants: JAX `gates/participant.py` (`participant_mpi.py` on ranks), foam / AMReX
`gates/couple_side.py` (the driver builds and runs natively in its image). What this adds to the
host matrix: the images' own preCICE (versions must match: preCICE refuses to couple different
ones), MPI and toolchains, and two separate containers finding each other.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

#: matrix C16 pair -> (Left, Right) backends
PAIRS = {"jax_jax": ("jax", "jax"), "jax_foam": ("jax", "openfoam"),
         "foam_foam": ("openfoam", "openfoam"), "amrex_amrex": ("amrex", "amrex"),
         "amrex_jax": ("amrex", "jax")}
#: config -> ranks per participant; mpi2 for the pairs whose gates run ranks on both sides
#: (gates 26, 28, 38 aa)
RANKS = {"serial": 1, "mpi2": 2}
DISTRIBUTED = ("jax_jax", "foam_foam", "amrex_amrex")
TH = 1e-14                                         # matrix TH["couple"]
TIMEOUT = float(os.environ.get("E2E_MATRIX_TIMEOUT", "840"))
GATES = "/zoomy/docs/dev/symbolic_solver/gates"
XCHG = "/e2e/couple"                               # the exchange directory, in both containers


def _params():
    if os.environ.get("E2E_QR_ONLY") == "1":
        return []
    sel = set(H.selected_backends())
    return [pytest.param(pair, cfg, marks=pytest.mark.backend_e2e, id=f"C16-{pair}-{cfg}")
            for cfg in RANKS for pair, sides in PAIRS.items()
            if set(sides) <= sel and (cfg == "serial" or pair in DISTRIBUTED)]


def side_cmd(backend, name, ranks):
    """The participant `name` in its image; it reads and writes in the exchange directory."""
    if backend == "jax":
        script = "participant_mpi.py" if ranks > 1 else "participant.py"
        argv = ((f"mpiexec -n {ranks} " if ranks > 1 else "")
                + f"python3 {GATES}/{script} {name} 1 {name}.npy window"
                + (" 1d" if ranks > 1 else ""))
        return ["bash", "-c", f"cd {XCHG} && exec {argv}"]
    return ["python3", f"{GATES}/couple_side.py", {"openfoam": "foam"}.get(backend, backend),
            name, XCHG, str(ranks)]


def _host(snap, *args) -> str:
    """`gates/couple_side.py ARGS` on the host (the snapshot library), its output."""
    lib = snap / "library"
    env = dict(os.environ, PYTHONNOUSERSITE="1", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join([str(snap / "docs/dev/symbolic_solver/gates")]
                                          + [str(lib / p) for p in ("zoomy_core", "zoomy_jax")]))
    r = subprocess.run([sys.executable, str(snap / "docs/dev/symbolic_solver/gates/couple_side.py"),
                        *map(str, args)], env=env, capture_output=True, text=True, timeout=600)
    if r.returncode:
        raise RuntimeError(f"couple_side.py {args[0]} on the host failed:\n{(r.stdout + r.stderr)[-3000:]}")
    return r.stdout


def _tail(log: Path, n=25) -> str:
    return "\n".join(log.read_text(errors="replace").splitlines()[-n:]) if log.exists() else ""


@pytest.mark.parametrize("pair,config", _params())
def test_couple_across_containers(pair, config, snap, report, request):
    if H.runtime() != "apptainer":
        pytest.skip("two containers on the host network: apptainer (docker would need "
                    "--network host for both, not wired)")
    sides = PAIRS[pair]
    imgs = {}
    for b in sides:
        img, how = H.image(H.BACKENDS[b])
        if img is None:
            pytest.skip(how)
        imgs[b] = img
    ranks = RANKS[config]
    root = H.WORK / f"{H.CHECKOUT.name}-{H.MODE}" / "couple"
    base = root / f"{pair}-{config}"
    xchg = base / "exchange"
    shutil.rmtree(xchg, ignore_errors=True)
    xchg.mkdir(parents=True)
    _host(snap, "config", xchg, XCHG)            # exchange-directory = the containers' path
    row = {"test": request.node.name, "pair": pair, "config": config,
           "images": {b: imgs[b] for b in sides}}

    procs = []
    t0 = time.monotonic()
    for name, b in zip(("Left", "Right"), sides):
        # a work dir (build caches) per backend and side, kept between the pairs
        c = H.Container(H.BACKENDS[b], imgs[b], snap, root / f"work-{b}-{name}",
                        dev=H.MODE == "dev", shared=((xchg, XCHG),))
        log = base / f"{name}-{b}.log"
        procs.append((name, c, c.start(side_cmd(b, name, ranks), log), log))
    try:                                         # one participant failing leaves its peer waiting
        while (any(p.poll() is None for _, _, p, _ in procs)
               and time.monotonic() - t0 < TIMEOUT):
            if any(p.poll() not in (None, 0) for _, _, p, _ in procs):
                break
            time.sleep(1)
    finally:
        for _, c, p, _ in procs:
            c.kill(p)
    row["seconds"] = round(time.monotonic() - t0, 1)
    rcs = {name: p.returncode for name, _, p, _ in procs}
    row["returncodes"] = rcs
    if any(rcs.values()) or not all((xchg / f"{n}.npy").exists() for n in ("Left", "Right")):
        row["result"] = "FAIL"
        report.append(row)
        pytest.fail(f"{pair} {config}: participants {rcs} after {row['seconds']} s\n"
                    + "\n".join(f"--- {name} ({b}): {log}\n{_tail(log)}"
                                for (name, _, _, log), b in zip(procs, sides)), pytrace=False)

    out = _host(snap, "check", xchg)
    m = json.loads([ln for ln in out.splitlines() if ln.startswith("COUPLE_CHECK ")][-1].split(" ", 1)[1])
    row.update(m)
    bad = {k: v for k, v in m.items() if not v <= TH}
    row["result"] = "FAIL" if bad else "PASS"
    report.append(row)
    print(json.dumps(row, default=str))
    if bad:
        pytest.fail(f"{pair} {config}: {bad} > {TH:.0e} (coupled across containers vs single domain)",
                    pytrace=False)
