"""Reference data: one .npz per test, one timings.json.

    pytest                       # compare
    pytest --overwrite-results   # rewrite the references it touches
"""
import json, os, pathlib, numpy as np

DIR = pathlib.Path(__file__).parent / "refs"
DIAG = DIR / "diagnostics"
TIMES = DIR / "timings.json"
SLOWER_OK = 1.5           # a test may get 50% slower; faster ratchets down.  Was 1.25 (user
                          # ruling 2026-07-21: 10% sits below this shared box's noise floor,
                          # 23% spread measured on identical code); on the box of 2026-10-14
                          # (8 pinned cores) of five full runs of identical code one was clean
                          # and four each failed a different gate: vam_first_order 19.7 -> 27.0 s,
                          # bc_wall +37%, ritter_dry_o2_small +35%, bc_periodic +132%
JIT_SLACK = 2.0           # s, on top of that for a test whose reference is below JIT_FLOOR:
JIT_FLOOR = 10.0          # its wall time is mostly XLA compile, which does not scale with the
                          # reference (the persistent compile cache hits or misses; measured on
                          # identical code: bc_periodic 1.5 s and 3.6 s, ritter_dry_o2_small
                          # 3.3 s and 4.5 s, a cold single-file run 3x)


def dump(name, **arrays):
    """Write a test's own measured arrays UNCONDITIONALLY, to
    ``refs/diagnostics/<name>.npz``.

    Not a golden and never compared — call it BEFORE the asserts.  A
    convergence test that trips its order floor dies before ``check()`` and
    destroys the very error vectors a reader needs in order to judge whether
    the floor or the scheme is at fault: the whole deliverable of this tier is
    "every fitted rate WITH its error vector", and a bare
    ``AssertionError: rate 0.736`` cannot supply that.  Measured cost is a few
    kB per test.

    ``refs/diagnostics/`` is gitignored: it is regenerated on every run and
    would otherwise churn the repo on each march.
    """
    DIAG.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DIAG / f"{name}.npz", **arrays)
    print(f"[diag] wrote diagnostics/{name}.npz")


def check(name, overwrite=False, **arrays):
    """Compare the given arrays against refs/<name>.npz, or write it."""
    p = DIR / f"{name}.npz"
    if overwrite or not p.exists():
        DIR.mkdir(exist_ok=True)
        np.savez_compressed(p, **arrays)
        print(f"[refs] wrote {p.name}")
        return
    ref = np.load(p)
    for k, v in arrays.items():
        assert np.allclose(v, ref[k]), \
            f"{name}.{k}: max|diff| {np.abs(v - ref[k]).max():.3e}"


def check_time(name, seconds, overwrite=False):
    """Fail if more than `SLOWER_OK` slower than the recorded time (plus `JIT_SLACK` for a
    reference below `JIT_FLOOR`); lower the record if faster.  Where `CI` is set (GitHub does) the
    time is printed and not judged: a hosted runner is not the machine the records come from, and
    timing regressions belong to the manual tier."""
    db = json.loads(TIMES.read_text()) if TIMES.exists() else {}
    ref = db.get(name)
    print(f"[time] {name}: {seconds:.2f} s (ref {ref})")
    if os.environ.get("CI") and not overwrite:
        return
    if overwrite or ref is None or seconds < ref:
        db[name] = round(seconds, 3)
        DIR.mkdir(exist_ok=True)
        TIMES.write_text(json.dumps(dict(sorted(db.items())), indent=1))
        return
    limit = ref * SLOWER_OK + (JIT_SLACK if ref < JIT_FLOOR else 0.0)
    assert seconds <= limit, \
        f"{name}: {seconds:.2f}s vs {ref:.2f}s (+{100*(seconds/ref-1):.0f}%, limit {limit:.2f}s)"
