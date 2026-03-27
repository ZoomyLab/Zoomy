"""
Benchmark matrix: SWE + GN × small/medium × NumPy-f64 × JAX {cpu,gpu}×{f32,f64}.

Each JAX configuration runs in a subprocess for env-var isolation
(JAX_PLATFORM_NAME, ZOOMY_JAX_ENABLE_X64).
90 s timeout per subprocess; on timeout the runtime is marked ">90 s".
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.mesh.mesh as petscMesh
from zoomy_core.fvm.solver_imex_numpy import IMEXSourceSolver

# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
from tutorials.swe.simple_swe_v2 import make_model as make_swe
from tutorials.swe.gn_classical_linear_analysis_v2 import ClassicalGreenNaghdi1D


CASES = [
    dict(name="swe_small",  model="swe", n_cells=300,  time_end=0.25, cfl=0.9),
    dict(name="swe_medium", model="swe", n_cells=1200, time_end=6.0,  cfl=0.9),
    dict(name="gn_small",   model="gn",  n_cells=160,  time_end=0.8,  cfl=0.5),
    dict(name="gn_medium",  model="gn",  n_cells=320,  time_end=16.0, cfl=0.5),
]

JAX_CONFIGS = [
    ("cpu", "float32", "0"),
    ("cpu", "float64", "1"),
    ("gpu", "float32", "0"),
    ("gpu", "float64", "1"),
]

TIMEOUT_S = 90

# ---------------------------------------------------------------------------
# NumPy reference runner (in-process)
# ---------------------------------------------------------------------------

def run_numpy(case):
    mesh = petscMesh.Mesh.create_1d(
        domain=(0.0, 10.0), n_inner_cells=case["n_cells"], lsq_degree=2)
    if case["model"] == "swe":
        model = make_swe()
    else:
        model = ClassicalGreenNaghdi1D()
    solver = IMEXSourceSolver(
        time_end=case["time_end"],
        compute_dt=timestepping.adaptive(CFL=case["cfl"]))
    object.__setattr__(solver, "source_mode", "auto")
    object.__setattr__(solver, "jv_backend", "analytic")
    object.__setattr__(solver, "implicit_maxiter", 6)
    object.__setattr__(solver, "gmres_maxiter", 30)
    t0 = time.perf_counter()
    Q, _ = solver.solve(mesh, model, write_output=False)
    t1 = time.perf_counter()
    s = solver.last_stats
    n = mesh.n_inner_cells
    return {
        "Q": np.asarray(Q[:, :n], dtype=np.float64),
        "steps": int(s.n_steps),
        "wall_s": round(t1 - t0, 3),
        "source_mode": s.source_mode,
    }

# ---------------------------------------------------------------------------
# JAX subprocess runner (code sent as string)
# ---------------------------------------------------------------------------

_JAX_RUNNER = r"""
import json, os, sys, time
import numpy as np

os.chdir(os.environ["ZOOMY_CWD"])

import zoomy_core.fvm.timestepping as timestepping
import zoomy_core.mesh.mesh as petscMesh
from tutorials.swe.simple_swe_v2 import make_model as make_swe
from tutorials.swe.gn_classical_linear_analysis_v2 import ClassicalGreenNaghdi1D
from zoomy_jax.fvm.solver_imex_jax import IMEXSourceSolverJax
import jax

case = json.loads(os.environ["ZOOMY_CASE"])
out_npy = os.environ["ZOOMY_OUT_NPY"]
meta_path = os.environ["ZOOMY_OUT_JSON"]

mesh = petscMesh.Mesh.create_1d(
    domain=(0.0, 10.0), n_inner_cells=case["n_cells"], lsq_degree=2)
if case["model"] == "swe":
    model = make_swe()
else:
    model = ClassicalGreenNaghdi1D()
solver = IMEXSourceSolverJax(
    time_end=case["time_end"],
    compute_dt=timestepping.adaptive(CFL=case["cfl"]))
object.__setattr__(solver, "source_mode", "auto")
object.__setattr__(solver, "jv_backend", "ad")
object.__setattr__(solver, "implicit_maxiter", 6)
object.__setattr__(solver, "gmres_maxiter", 30)

t0 = time.perf_counter()
Q, _ = solver.solve(mesh, model, write_output=False)
t1 = time.perf_counter()

s = solver.last_stats
n = mesh.n_inner_cells
Qnp = np.asarray(Q[:, :n], dtype=np.float64)
np.save(out_npy, Qnp)
meta = dict(
    steps=int(s.n_steps),
    wall_s=round(t1 - t0, 3),
    compile_s=round(s.compile_time_s, 3),
    run_s=round(s.runtime_only_s, 3),
    source_mode=s.source_mode,
    dtype=str(np.asarray(Q).dtype),
    backend=str(jax.default_backend()),
)
with open(meta_path, "w") as f:
    json.dump(meta, f)
print(json.dumps(meta))
"""

def run_jax_subprocess(case, device, precision, x64_flag, tmp_dir):
    out_npy = os.path.join(tmp_dir, f"{case['name']}_{device}_{precision}.npy")
    out_json = os.path.join(tmp_dir, f"{case['name']}_{device}_{precision}.json")
    env = os.environ.copy()
    env.update({
        "LOGURU_LEVEL": "WARNING",
        "JAX_PLATFORM_NAME": device,
        "ZOOMY_JAX_ENABLE_X64": x64_flag,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "ZOOMY_CASE": json.dumps(case),
        "ZOOMY_OUT_NPY": out_npy,
        "ZOOMY_OUT_JSON": out_json,
        "ZOOMY_CWD": os.getcwd(),
    })
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _JAX_RUNNER],
            env=env, capture_output=True, text=True,
            timeout=TIMEOUT_S)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout)[-500:]
            return {"error": err}
        Q = np.load(out_npy)
        with open(out_json) as f:
            meta = json.load(f)
        meta["Q"] = Q
        return meta
    except subprocess.TimeoutExpired:
        return {"error": f"timeout ({TIMEOUT_S}s)"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _l2_linf(A, B):
    d = B - A
    return float(np.sqrt(np.mean(d * d))), float(np.max(np.abs(d)))


def main():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="zoomy_bench_matrix_")

    rows = []
    for case in CASES:
        print(f"\n{'='*70}")
        print(f"Case: {case['name']}  (n_cells={case['n_cells']}, "
              f"time_end={case['time_end']})")
        print(f"{'='*70}")

        # NumPy reference
        print("  Running NumPy f64 ... ", end="", flush=True)
        np_ref = run_numpy(case)
        print(f"done ({np_ref['wall_s']:.1f}s, {np_ref['steps']} steps)")
        rows.append(dict(
            case=case["name"], engine="numpy", device="cpu",
            precision="float64", steps=np_ref["steps"],
            wall_s=np_ref["wall_s"], compile_s=0, run_s=np_ref["wall_s"],
            dtype="float64", l2=0.0, linf=0.0,
            source_mode=np_ref["source_mode"], error="",
        ))

        # JAX configs
        for device, precision, x64 in JAX_CONFIGS:
            tag = f"jax-{device}-{precision}"
            print(f"  Running {tag} ... ", end="", flush=True)
            res = run_jax_subprocess(case, device, precision, x64, tmp)
            if "error" in res and "Q" not in res:
                print(f"FAILED: {res['error'][:120]}")
                rows.append(dict(
                    case=case["name"], engine="jax", device=device,
                    precision=precision, steps=-1,
                    wall_s=-1, compile_s=-1, run_s=-1,
                    dtype="?", l2=-1, linf=-1,
                    source_mode="?", error=res["error"][:200],
                ))
                continue
            l2, linf = _l2_linf(np_ref["Q"], res["Q"])
            print(f"done ({res['wall_s']:.1f}s, "
                  f"compile={res['compile_s']:.1f}s, "
                  f"run={res['run_s']:.3f}s, "
                  f"L2={l2:.2e})")
            rows.append(dict(
                case=case["name"], engine="jax", device=device,
                precision=precision, steps=res["steps"],
                wall_s=res["wall_s"], compile_s=res["compile_s"],
                run_s=res["run_s"], dtype=res.get("dtype", "?"),
                l2=l2, linf=linf,
                source_mode=res.get("source_mode", "?"),
                error="",
            ))

    # ── pretty table ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 110)
    print(f"{'Case':<14} {'Engine':<8} {'Device':<6} {'Prec':<8} "
          f"{'Steps':>6} {'Compile':>8} {'Run':>8} {'Total':>8} "
          f"{'L2 vs NP':>10} {'Linf vs NP':>12}")
    print("-" * 110)
    for r in rows:
        if r["wall_s"] < 0:
            print(f"{r['case']:<14} {r['engine']:<8} {r['device']:<6} "
                  f"{r['precision']:<8} {'ERR':>6} {'-':>8} {'-':>8} "
                  f"{'-':>8} {'-':>10} {'-':>12}  {r['error'][:60]}")
        else:
            print(f"{r['case']:<14} {r['engine']:<8} {r['device']:<6} "
                  f"{r['precision']:<8} {r['steps']:>6} "
                  f"{r['compile_s']:>7.2f}s {r['run_s']:>7.3f}s "
                  f"{r['wall_s']:>7.2f}s "
                  f"{r['l2']:>10.2e} {r['linf']:>12.2e}")
    print("=" * 110)

    # Save raw data
    out_json = os.path.join(tmp, "results.json")
    with open(out_json, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "Q"} for r in rows], f, indent=2)
    print(f"\nRaw results saved to {out_json}")


if __name__ == "__main__":
    main()
