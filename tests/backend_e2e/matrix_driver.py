"""A subset of the cross-backend matrix (docs/dev/symbolic_solver/matrix), run by matrix.py's
own `run` (same store format), here: inside a backend container or on the host.

    python matrix_driver.py ROOT STORE WORK BACKEND CASE KEY[,KEY...] [--in-container]

ROOT holds docs/dev/symbolic_solver/{matrix,gates} of the tested checkout. In a container the
foam / AMReX drivers build natively (`ZOOMY_{FOAM,AMREX}_IN_IMAGE=1`); for older drivers that
always call `apptainer exec <image>`, the apptainer shim (shim/) stands in for it. An image
without jax gets a stub for gates/common.py's `jax.config.update("jax_enable_x64", True)`,
the only jax it touches; one without mpi4py a one-rank stub while gate 21 is imported.
Last line: `E2E_RESULT {key: ok | n/a | unsupported | err: ...}`.
"""
import json
import os
import sys
import types
from pathlib import Path


def _with_mpi_stub():
    """Gate 21 (C8's case builder) does `comm = MPI.COMM_WORLD` at import, for its MPI runs;
    without mpi4py (zoomy_jax image) it is imported once against a one-rank stub, which is
    removed again, so a serial run that did reach for MPI would still fail."""
    try:
        import mpi4py  # noqa: F401
        return
    except ImportError:
        pass
    world = types.SimpleNamespace(Get_rank=lambda: 0, Get_size=lambda: 1, rank=0, size=1)
    stub = types.ModuleType("mpi4py")
    stub.MPI = types.SimpleNamespace(COMM_WORLD=world)
    sys.modules["mpi4py"] = stub
    try:
        import gate21_parallel_chorin  # noqa: F401
    except ImportError:
        pass
    finally:
        del sys.modules["mpi4py"]


def main(root, store, work, backend, case, keys, in_container=False):
    os.environ["MATRIX_STORE"], os.environ["MATRIX_WORK"] = store, work
    m = Path(root) / "docs/dev/symbolic_solver/matrix"
    sys.path[:0] = [str(m), str(m.parent / "gates")]
    if in_container:
        os.environ["PATH"] = f"{Path(__file__).resolve().parent / 'shim'}:{os.environ['PATH']}"
        fake = Path(work) / "images"           # the drivers stat the image they would run
        fake.mkdir(parents=True, exist_ok=True)
        for n in ("zoomy_openfoam.sif", "zoomy_amrex.sif"):
            (fake / n).touch()
        os.environ["ZOOMY_OF_SIF"] = str(fake / "zoomy_openfoam.sif")
        os.environ["ZOOMY_AMREX_CONTAINERS"] = str(fake)
        os.environ.setdefault("ZOOMY_FOAM_IN_IMAGE", "1")    # drivers that build natively
        os.environ.setdefault("ZOOMY_AMREX_IN_IMAGE", "1")
    try:
        import jax  # noqa: F401
    except ImportError:
        stub = types.ModuleType("jax")
        stub.config = types.SimpleNamespace(update=lambda *a, **k: None)
        sys.modules["jax"] = stub
        import common  # noqa: F401
        del sys.modules["jax"]
    import cases                    # before any gate: gate 7 imports a module `cases` too
    if "jax" in sys.modules:
        _with_mpi_stub()
    import matrix

    desc, runs = cases.CASES[case]
    want = keys.split(",")
    cases.CASES[case] = (desc, [r for r in runs if r.key in want])
    prov = Path(root) / "PROVENANCE.json"           # the snapshot's (harness.snapshot)
    matrix.provenance = lambda: prov.read_text() if prov.exists() else "{}"
    matrix.run(backend, case, "serial", force=True)
    out = {}
    for k in want:
        p = matrix.path(backend, "serial", case, k)
        r = next((r for r in runs if r.key == k), None)
        if r is None or backend not in r.backends or "serial" not in r.configs:
            out[k] = "n/a"
        elif p.exists():
            out[k] = "ok"
        elif p.with_suffix(".unsupported").exists():
            out[k] = "unsupported"
        elif p.with_suffix(".err").exists():
            out[k] = "err: " + p.with_suffix(".err").read_text()[-600:]
        else:
            out[k] = "missing"
    print("E2E_RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--in-container"]
    main(*a, in_container="--in-container" in sys.argv)
    sys.stdout.flush()
    os._exit(0)          # JAX worker threads can stall a normal exit
