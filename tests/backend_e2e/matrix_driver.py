"""A subset of the cross-backend matrix (docs/dev/symbolic_solver/matrix), run by matrix.py's
own `run` (same store format), here: inside a backend container or on the host.

    python matrix_driver.py ROOT STORE WORK BACKEND CASE KEY[,KEY...] [CONFIG] [--in-container]
    python matrix_driver.py ROOT STORE --checks BACKEND CASE

ROOT holds docs/dev/symbolic_solver/{matrix,gates} of the tested checkout; CONFIG serial
(default), mpiN (matrix.py launches the ranks with the image's mpiexec) or gpu1 / gpu2 (the
container started with the GPUs; JAX: the image's Python with JAX_PLATFORMS=cuda instead of
the host's zoomy-cuda env, gpu2 one rank per GPU through `zoomy_jax.gpurank`; AMReX: the
CUDA build, as on the host). The case builders
import no backend, so every image builds every case. In a container the foam / AMReX drivers
build natively (`ZOOMY_{FOAM,AMREX}_IN_IMAGE=1`); for older drivers that always call
`apptainer exec <image>`, the apptainer shim (shim/) stands in for it.
Last line: `E2E_RESULT {key: ok | n/a | unsupported | err: ... | unknown: ...}`.

`--checks`: the matrix's own checks (`matrix.measure`, the thresholds of the table) of CASE on
BACKEND over the runs in STORE, and the checks naming the case (C6 mass, C7 parallel = serial);
a check whose inputs are not in STORE is not made (e.g. the cross-backend ones need JAX's run).
Last line: `E2E_CHECKS [{check, config, value, op, bound, ok, note}, ...]`.
"""
import json
import os
import sys
from pathlib import Path


def main(root, store, work, backend, case, keys, config="serial", in_container=False):
    os.environ["MATRIX_STORE"], os.environ["MATRIX_WORK"] = store, work
    m = Path(root) / "docs/dev/symbolic_solver/matrix"
    sys.path[:0] = [str(m), str(m.parent / "gates")]
    prov = Path(root) / "PROVENANCE.json"           # the snapshot's (harness.snapshot)
    if prov.exists():
        os.environ["MATRIX_PROVENANCE"] = str(prov)
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
        import backends
        host_launch = backends.Jax.launch

        def launch(self, config):            # the image's Python, not the host's zoomy-cuda env
            if not config.startswith("gpu"):
                return host_launch(self, config)
            gpu = ["env", "JAX_PLATFORMS=cuda", "XLA_PYTHON_CLIENT_PREALLOCATE=false"]
            n = int(config[3:])
            return gpu + ([sys.executable] if n == 1 else
                          ["mpiexec", "-n", str(n), sys.executable, "-m", "zoomy_jax.gpurank"])
        backends.Jax.launch = launch
    import cases
    import matrix

    desc, runs = cases.CASES[case]
    want = keys.split(",")
    cases.CASES[case] = (desc, [r for r in runs if r.key in want])
    matrix.run(backend, case, config, force=True)
    out = {}
    for k in want:
        p = matrix.path(backend, config, case, k)
        r = next((r for r in runs if r.key == k), None)
        if r is None:
            out[k] = f"unknown: no run {k!r} in cases.py's {case}"
        elif backend not in r.backends or config not in r.configs:
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


def checks(root, store, backend, case):
    os.environ["MATRIX_STORE"] = store
    m = Path(root) / "docs/dev/symbolic_solver/matrix"
    sys.path[:0] = [str(m), str(m.parent / "gates")]
    import matrix

    rows = []
    for c, check, b, cfg, v, th, note in matrix.measure():
        if b != backend or th is None or isinstance(v, str):
            continue
        if c != case and not check.startswith(case + " "):
            continue
        v = float(v)
        ok = bool(v >= th[1] if th[0] == ">=" else v <= th[1])
        rows.append(dict(check=f"{c} {check}" if c != case else check, config=cfg, value=v,
                         op=th[0], bound=th[1], ok=ok, note=note))
    print("E2E_CHECKS " + json.dumps(rows), flush=True)


if __name__ == "__main__":
    if "--checks" in sys.argv:
        a = [x for x in sys.argv[1:] if x != "--checks"]
        checks(*a)
    else:
        a = [x for x in sys.argv[1:] if x != "--in-container"]
        main(*a, in_container="--in-container" in sys.argv)
    sys.stdout.flush()
    os._exit(0)          # JAX worker threads can stall a normal exit
