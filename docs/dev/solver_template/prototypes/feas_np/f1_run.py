"""F1 driver: reference numpy solver vs single-source body (numpy / jax.jit) vs jax solver.
usage: f1_run.py {1d|2d} N NSTEP [mem_steps] [parts]"""
import sys, time, json, tracemalloc, gc, numpy as np
from common import *
import jax, jax.numpy as jnp
import zoomy_core.model.initial_conditions as IC
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.fvm.solver_numpy import HyperbolicSolver as NPSolver
from zoomy_jax.fvm.solver_jax import HyperbolicSolver as JXSolver
import shared_body as SB, vk_bug

case, N, NSTEP = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
MEM = int(sys.argv[4]) if len(sys.argv) > 4 else 3
PARTS = sys.argv[5].split(",") if len(sys.argv) > 5 else ["ref", "snp", "sjx", "jxs", "flags"]
dim = 1 if case.startswith("1d") else 2
BED = case.endswith("bed")
tags = ("left", "right") if dim == 1 else ("left", "right", "bottom", "top")
model = SWE(dimension=dim, boundary_conditions=BC.BoundaryConditions([BC.Wall(tag=t) for t in tags]))
sm = SystemModel.from_model(model)
def ic(x):
    out = np.zeros(dim + 2)
    if dim == 1:
        out[0] = 0.3 * np.exp(-((x[0] - 0.7) / 0.08) ** 2) if BED else 0.0
        out[1] = (2.0 if x[0] < 0.5 else 1.0) - out[0]
    else: out[1] = 2.0 if (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2 < 0.3 ** 2 else 1.0
    return out
sm.initial_conditions = IC.UserFunction(function=ic)
sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
nsm = NumericalSystemModel.from_system_model(sm, reconstruction=ReconstructionSpec(order=2, limiter="venkatakrishnan"))
mk = (lambda: LSQMesh.create_1d((0.0, 1.0), N)) if dim == 1 else (lambda: LSQMesh.create_2d((0.0, 1.0, 0.0, 1.0), N, N))
R = {"case": case, "N": N, "nstep": NSTEP, "nc": N ** dim}

def new_ref():
    s = NPSolver(time_end=1e9, compute_dt=timestepping.adaptive(CFL=0.4))
    s.setup_simulation(mk(), nsm, write_output=False); return s
t0 = time.perf_counter(); ref = new_ref(); R["ref_setup_s"] = time.perf_counter() - t0
Q0, A0, p = ref._sim_Q.copy(), ref._sim_Qaux.copy(), ref._sim_parameters
dt = float(ref.compute_dt(Q0, A0, p, ref._sim_face_inradius, ref._sim_compute_max_abs_eigenvalue))
R["dt"] = dt
print(f"[{case} N={N}] nc={R['nc']} dt={dt:.3e} setup {R['ref_setup_s']:.1f}s", flush=True)

def timed(fn, n, sync=lambda x: x):
    ts = []
    out = None
    for i in range(n):
        t0 = time.perf_counter(); out = fn(i); sync(out); ts.append(time.perf_counter() - t0)
    return out, ts
def peak(fn, n):
    gc.collect(); tracemalloc.start(); tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    fn(n); cur, pk = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return (pk - base) / 2 ** 20

# ---------------- reference numpy solver
if "ref" in PARTS:
    st = {"t": 0.0}
    def ref_step(i):
        ref._sim_time = st["t"]; ref.step(dt); st["t"] += dt
    _, ts = timed(ref_step, NSTEP)
    Qref = np.array(ref._sim_Q); np.save(f"f1_{case}{N}_ref.npy", Qref)
    R["ref_ms_med"] = 1e3 * float(np.median(ts))
    r2 = new_ref(); st2 = {"t": 0.0}
    def ref_mem(n):
        for _ in range(n): r2._sim_time = st2["t"]; r2.step(dt); st2["t"] += dt
    R["ref_peak_MiB"] = peak(ref_mem, MEM)
    print(f"  ref numpy solver: {R['ref_ms_med']:.1f} ms/step, peak {R['ref_peak_MiB']:.1f} MiB", flush=True)
else:
    Qref = np.load(f"f1_{case}{N}_ref.npy")

# ---------------- shared body, numpy
G = SB.Geo(ref); Kn = SB.NumpyKernels(ref)
# one-flux-evaluation bitwise check against the reference flux operator
fr = ref._sim_flux_operator(dt, 0.0, Q0, A0, p, np.zeros_like(Q0))
for scat in ("add.at", "bincount"):
    step_n, flux_n = SB.make_step(np, SB.NumpyOps(scat), G, Kn, p)
    fs = flux_n(0.0, Q0, A0)
    R[f"flux1_bitwise_{scat}"] = bool(np.array_equal(fs, fr)); R[f"flux1_maxdiff_{scat}"] = float(np.abs(fs - fr).max())
print(f"  1 flux eval vs reference: add.at bitwise={R['flux1_bitwise_add.at']} ({R['flux1_maxdiff_add.at']:.1e}); "
      f"bincount bitwise={R['flux1_bitwise_bincount']} ({R['flux1_maxdiff_bincount']:.1e})", flush=True)
if "snp" in PARTS:
    for scat in ("add.at", "bincount"):
        step_n, _ = SB.make_step(np, SB.NumpyOps(scat), G, Kn, p)
        st = {"Q": Q0.copy(), "A": A0.copy(), "t": 0.0}
        def s_step(i):
            st["Q"], st["A"] = step_n(st["Q"], st["A"], st["t"], dt); st["t"] += dt
        _, ts = timed(s_step, NSTEP)
        R[f"snp_{scat}_ms_med"] = 1e3 * float(np.median(ts))
        R[f"snp_{scat}_maxdQ_vs_ref"] = float(np.abs(st["Q"] - Qref).max())
        R[f"snp_{scat}_bitwise_vs_ref"] = bool(np.array_equal(st["Q"], Qref))
        if scat == "add.at": np.save(f"f1_{case}{N}_snp.npy", st["Q"])
        st2 = {"Q": Q0.copy(), "A": A0.copy(), "t": 0.0}
        def s_mem(n):
            for _ in range(n): st2["Q"], st2["A"] = step_n(st2["Q"], st2["A"], st2["t"], dt); st2["t"] += dt
        R[f"snp_{scat}_peak_MiB"] = peak(s_mem, MEM)
        print(f"  shared numpy [{scat}]: {R[f'snp_{scat}_ms_med']:.1f} ms/step, max|dQ| vs ref {R[f'snp_{scat}_maxdQ_vs_ref']:.2e} "
              f"(bitwise {R[f'snp_{scat}_bitwise_vs_ref']}), peak {R[f'snp_{scat}_peak_MiB']:.1f} MiB", flush=True)

# ---------------- shared body, jax.jit
Kj = SB.JaxKernels(nsm); Gj = G.to(jnp); pj = jnp.asarray(p)
def make_jit(**flags):
    st, _ = SB.make_step(jnp, SB.JaxOps(), Gj, Kj, pj, **flags)
    return jax.jit(st)
def run_jit(stepj, n):
    Q, A, t = jnp.asarray(Q0), jnp.asarray(A0), 0.0
    t0 = time.perf_counter(); Q, A = stepj(Q, A, t, dt); Q.block_until_ready(); tc = time.perf_counter() - t0
    t += dt; ts = []
    for _ in range(n - 1):
        t0 = time.perf_counter(); Q, A = stepj(Q, A, t, dt); Q.block_until_ready(); ts.append(time.perf_counter() - t0); t += dt
    return np.asarray(Q), tc, ts
if "sjx" in PARTS:
    stepj = make_jit()
    Qsj, tc, ts = run_jit(stepj, NSTEP)
    np.save(f"f1_{case}{N}_sjx.npy", Qsj)
    R["sjx_compile_s"] = tc; R["sjx_ms_med"] = 1e3 * float(np.median(ts))
    R["sjx_maxdQ_vs_ref"] = float(np.abs(Qsj - Qref).max())
    try: Qsn = np.load(f"f1_{case}{N}_snp.npy"); R["sjx_maxdQ_vs_snp"] = float(np.abs(Qsj - Qsn).max())
    except FileNotFoundError: pass
    def j_mem(n):
        Q, A, t = jnp.asarray(Q0), jnp.asarray(A0), 0.0
        for _ in range(n): Q, A = stepj(Q, A, t, dt); t += dt
        Q.block_until_ready()
    R["sjx_peak_MiB(tracemalloc; XLA buffers invisible)"] = peak(j_mem, MEM)
    print(f"  shared jax.jit: compile+1st {tc:.1f}s, {R['sjx_ms_med']:.2f} ms/step, max|dQ| vs ref {R['sjx_maxdQ_vs_ref']:.2e}, "
          f"vs shared-numpy {R.get('sjx_maxdQ_vs_snp', float('nan')):.2e}", flush=True)

# ---------------- current jax solver (fixed VK on disk) and HEAD-buggy VK
def run_jax_solver(buggy):
    vk_bug.set_buggy(buggy)
    s = JXSolver(time_end=np.inf, compute_dt=timestepping.adaptive(CFL=0.4))
    Q, A = s.setup_simulation(mk(), nsm); t = 0.0; ts = []
    for i in range(NSTEP):
        t0 = time.perf_counter(); Qn = s.step(dt, t, Q, A); Q, A = s.post_step(t + dt, dt, Qn, Q, A); Q.block_until_ready()
        ts.append(time.perf_counter() - t0); t += dt
    vk_bug.set_buggy(False)
    return np.asarray(Q)[:, :R["nc"]], ts
if "jxs" in PARTS:
    for lab, bug in (("jxsolver_fixedVK", False), ("jxsolver_HEADbuggyVK", True)):
        Qx, ts = run_jax_solver(bug); np.save(f"f1_{case}{N}_{lab}.npy", Qx)
        R[f"{lab}_ms_med"] = 1e3 * float(np.median(ts[1:])); R[f"{lab}_first_s"] = ts[0]
        R[f"{lab}_maxdQ_vs_ref"] = float(np.abs(Qx - Qref).max())
        try: R[f"{lab}_maxdQ_vs_sjx"] = float(np.abs(Qx - np.load(f"f1_{case}{N}_sjx.npy")).max())
        except FileNotFoundError: pass
        print(f"  {lab}: {R[f'{lab}_ms_med']:.2f} ms/step (1st {ts[0]:.1f}s), max|dQ| vs ref {R[f'{lab}_maxdQ_vs_ref']:.2e}, vs shared-jax {R.get(f'{lab}_maxdQ_vs_sjx', float('nan')):.2e}", flush=True)
    Qa, Qb = np.load(f"f1_{case}{N}_jxsolver_fixedVK.npy"), np.load(f"f1_{case}{N}_jxsolver_HEADbuggyVK.npy")
    R["jxsolver_bug_effect_maxdQ"] = float(np.abs(Qa - Qb).max())
    print(f"  VK-bug effect inside the jax solver (buggy - fixed): {R['jxsolver_bug_effect_maxdQ']:.2e}", flush=True)

# ---------------- attribution: shared jax body with jax-solver flags
if "flags" in PARTS:
    for lab, fl in (("vkbug", dict(vk_jax_bug=True)), ("lift", dict(bounds_lift=True)),
                    ("stage2aux", dict(stage2_aux=True)),
                    ("lift+stage2aux", dict(bounds_lift=True, stage2_aux=True)),
                    ("vkbug+lift+stage2aux", dict(vk_jax_bug=True, bounds_lift=True, stage2_aux=True))):
        Qf, _, _ = run_jit(make_jit(**fl), NSTEP)
        line = f"  shared-jax[{lab}]: vs shared-jax {np.abs(Qf - np.load(f'f1_{case}{N}_sjx.npy')).max():.2e}"
        R[f"flag_{lab}_vs_sjx"] = float(np.abs(Qf - np.load(f"f1_{case}{N}_sjx.npy")).max())
        for js in ("jxsolver_fixedVK", "jxsolver_HEADbuggyVK"):
            try:
                d = float(np.abs(Qf - np.load(f"f1_{case}{N}_{js}.npy")).max()); R[f"flag_{lab}_vs_{js}"] = d
                line += f"  vs {js} {d:.2e}"
            except FileNotFoundError: pass
        print(line, flush=True)
json.dump(R, open(f"f1_{case}{N}_results.json", "w"), indent=1)
