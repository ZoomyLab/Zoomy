"""Where the JAX symbolic solver's start-up time goes (ANALYSIS 65).

    python jax_compile_time.py CASE KEY [--steps]      # a matrix run, e.g. C1 o2_n40, C8 vam
    python jax_compile_time.py escalante               # the thesis ML-VAM(2) bump (ANALYSIS 59)

Prints the setup split (model / NSM / print / runtime), then the march program's trace, lowering
and XLA compile (jax.monitoring events), its size (jaxpr equations, StableHLO ops, optimised HLO
instructions), its host callbacks and whether the persistent compilation cache was hit.  Set
`JAX_COMPILATION_CACHE_DIR` to test the cache (run twice); `PROFILE=1` cProfiles the setup.
Run from `docs/dev/symbolic_solver/matrix` with its `env.sh` (the case's libraries).
"""
import cProfile
import io
import json
import os
import pstats
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "matrix"), os.path.join(HERE, "..", "gates")]

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
if os.environ.get("JAX_COMPILATION_CACHE_DIR"):
    jax.config.update("jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"])
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_explain_cache_misses", bool(os.environ.get("EXPLAIN")))

EV = {}


def _dur(event, secs, **kw):
    EV.setdefault(event, []).append(secs)


def _ev(event, **kw):
    EV.setdefault(event, []).append(1)


jax.monitoring.register_event_duration_secs_listener(_dur)
jax.monitoring.register_event_listener(_ev)

from zoomy_jax.symsolver import runtime as R  # noqa: E402

try:                                    # main before ANALYSIS 65 has no compile_cache
    from zoomy_jax.symsolver.compile_cache import callback_cache, stats as CB
except ImportError:
    import contextlib
    callback_cache, CB = contextlib.nullcontext, {}

PROGRAMS = []
_hoisted = R.hoisted_jit


def hoisted_jit(fn):
    """`R.hoisted_jit` with the march's trace, lowering and compile timed separately (the
    same jaxpr, the same jit body: the program is the solver's)."""
    cache = {}

    def run(*args):
        flat, tree = jax.tree_util.tree_flatten(args)
        key = (tree, tuple((jax.numpy.shape(x), jax.numpy.result_type(x)) for x in flat))
        if key not in cache:
            rec = {}
            t0 = time.perf_counter()
            closed, out = jax.make_jaxpr(fn, return_shape=True)(*args)
            rec["trace"] = time.perf_counter() - t0
            consts = [jax.numpy.asarray(c) for c in closed.consts]
            body = jax.jit(lambda cs, *xs: jax.core.eval_jaxpr(closed.jaxpr, cs, *xs))
            t0 = time.perf_counter()
            lowered = body.lower(consts, *flat)
            rec["lower"] = time.perf_counter() - t0
            n_cache = len(EV.get("/jax/compilation_cache/cache_hits", [])) + CB.get("hits", 0)
            t0 = time.perf_counter()
            with callback_cache():
                compiled = lowered.compile()
            rec["compile"] = time.perf_counter() - t0
            rec["cache_hit"] = (len(EV.get("/jax/compilation_cache/cache_hits", []))
                                + CB.get("hits", 0)) > n_cache
            rec["jaxpr_eqns"] = count_eqns(closed.jaxpr)
            if os.environ.get("BREAKDOWN"):
                breakdown(closed.jaxpr, os.environ["BREAKDOWN"])
            rec["consts"] = len(consts)
            rec["const_bytes"] = int(sum(c.size * c.dtype.itemsize for c in consts))
            txt = lowered.as_text()
            rec["stablehlo_ops"] = len(re.findall(r"^\s*(%\S+ = )?\"?[a-z_]+\.[a-z_]+", txt,
                                                  re.M))
            rec["stablehlo_bytes"] = len(txt)
            rec["callbacks"] = sorted(set(re.findall(r"xla_ffi\w*python\w*callback", txt)))
            rec["n_callbacks"] = len(re.findall(r"xla_ffi\w*python\w*callback", txt))
            try:
                hlo = compiled.as_text()
                rec["hlo_instr"] = sum(1 for ln in hlo.splitlines() if " = " in ln)
            except Exception as e:          # noqa: BLE001
                rec["hlo_instr"] = f"n/a ({e})"
            PROGRAMS.append(rec)
            cache[key] = (compiled, consts, jax.tree_util.tree_structure(out))
        compiled, consts, out_tree = cache[key]
        return jax.tree_util.tree_unflatten(out_tree, compiled(consts, *flat))
    return run


def count_eqns(jaxpr):
    """Equations of a jaxpr, sub-jaxprs (loops, conds, custom rules) included."""
    n = 0
    for e in jaxpr.eqns:
        n += 1
        for v in e.params.values():
            for j in (v if isinstance(v, (tuple, list)) else (v,)):
                if hasattr(j, "jaxpr") and hasattr(j.jaxpr, "eqns"):
                    n += count_eqns(j.jaxpr)
                elif hasattr(j, "eqns"):
                    n += count_eqns(j)
    return n


R.hoisted_jit = hoisted_jit


def _walk(jaxpr, mult=1):
    """(eqn, multiplicity) over a jaxpr and its sub-jaxprs."""
    for e in jaxpr.eqns:
        yield e
        for v in e.params.values():
            for j in (v if isinstance(v, (tuple, list)) else (v,)):
                if hasattr(j, "jaxpr") and hasattr(j.jaxpr, "eqns"):
                    yield from _walk(j.jaxpr)
                elif hasattr(j, "eqns"):
                    yield from _walk(j)


def breakdown(jaxpr, path):
    """Equations per printed-program line, per runtime / kernel function, per primitive (JSON)."""
    import collections
    by_line, by_fn, by_prim, by_kernel = (collections.Counter() for _ in range(4))
    for e in _walk(jaxpr):
        by_prim[e.primitive.name] += 1
        frames = list(e.source_info.traceback.frames) if e.source_info.traceback else []
        prog = [f for f in frames if f.file_name.startswith("<symsolver")]
        by_line[f"{prog[0].file_name}:{prog[0].line_num}" if prog else "?"] += 1
        rt = [f for f in frames if ("zoomy_jax" in f.file_name or "zoomy_core" in f.file_name)]
        by_fn[" < ".join(f"{os.path.basename(f.file_name)}:{f.function_name}"
                         for f in rt[-3:][::-1]) if rt else "?"] += 1
        lam = [f for f in frames if f.file_name.startswith("<lambdify")]
        by_kernel[lam[0].function_name if lam else "-"] += 1
    with open(path, "w") as f:
        json.dump(dict(line=by_line.most_common(), fn=by_fn.most_common(),
                       prim=by_prim.most_common(), kernel=by_kernel.most_common()), f, indent=0)


def timed(obj, name, label, T):
    """Wrap `obj.name` to add its wall time to `T[label]`."""
    f = getattr(obj, name)

    def g(*a, **k):
        t0 = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            T[label] = T.get(label, 0.0) + time.perf_counter() - t0
    setattr(obj, name, g)


def instrument(T):
    from zoomy_jax.symsolver import solver as S
    from zoomy_jax.transformation.jax_runtime import JaxRuntime
    for cls in (S.SymbolicHyperbolicSolverJax, S.SymbolicChorinSolverJax):
        if "_print" in cls.__dict__:
            timed(cls, "_print", "print template", T)
    timed(S, "default_wave_speed", "default_wave_speed", T)
    timed(S, "build_topology", "topology", T)
    timed(R, "runtime", "R.runtime", T)
    timed(R, "build", "R.build (exec)", T)
    f = JaxRuntime.from_nsm.__func__
    JaxRuntime.from_nsm = classmethod(lambda cls, *a, **k: _t(T, "JaxRuntime.from_nsm", f,
                                                              cls, *a, **k))


def _t(T, label, f, *a, **k):
    t0 = time.perf_counter()
    try:
        return f(*a, **k)
    finally:
        T[label] = T.get(label, 0.0) + time.perf_counter() - t0


def matrix_run(case, key):
    """Setup and march of matrix run `case/key` as backends.Jax (serial / one device)."""
    import numpy as np
    import jax.numpy as jnp
    from cases import CASES, prepare
    from backends import _solver_kw
    from zoomy_core.symsolver.settings import LinearSolver
    from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax, SymbolicHyperbolicSolverJax
    r = next(r for r in CASES[case][1] if r.key == key)
    T = {}
    instrument(T)
    t0 = time.perf_counter()
    mesh, systems, Q0, names = prepare(r)
    T["case build (model, SM, NSM)"] = time.perf_counter() - t0
    cls = SymbolicChorinSolverJax if r.chorin else SymbolicHyperbolicSolverJax
    kw = _solver_kw(r, cls)
    prof = cProfile.Profile() if os.environ.get("PROFILE") else None
    t0 = time.perf_counter()
    if prof:
        prof.enable()
    if r.chorin:
        pls = kw.pop("pressure_linear_solver", None) or LinearSolver(
            os.environ.get("MATRIX_JAX_PRESSURE", "gmres"))
        s = SymbolicChorinSolverJax(pressure_linear_solver=pls, **kw)
        Q, A = s.setup(mesh, *systems)
    else:
        s = SymbolicHyperbolicSolverJax(**kw)
        Q, A = s.setup(mesh, systems)
    if prof:
        prof.disable()
    T["setup (total)"] = time.perf_counter() - t0
    if r.q0 is not None:
        Q = jnp.asarray(r.q0(names, np.asarray(s.mesh_np.cell_centers)[:, :Q.shape[1]]))
        A = s.initial_aux(Q, A)
    else:
        Q = jnp.asarray(Q0)
    t0 = time.perf_counter()
    t, Q, _ = s.run(Q, A)
    T["run (total)"] = time.perf_counter() - t0
    return T, prof, s, np.asarray(Q)


def escalante():
    """ANALYSIS 59's case: thesis escalante_vam_bump/run_mlvam2_jax_jit.py (ML-VAM(2) level 1, 60
    cells, hr, order 2, SSP-RK2, MOOD, dt 1e-3) to T_END (default 0.1)."""
    import numpy as np
    import jax.numpy as jnp
    import sympy as sp
    from zoomy_core.mesh import BaseMesh
    from zoomy_core.model.boundary_conditions import Dirichlet
    from zoomy_core.model.initial_conditions import Constant
    from zoomy_core.model.models.ml_vam import MLVAM
    from zoomy_core.model.models.closures import Newtonian, StressFree
    from zoomy_core.numerics import Limiter, ReconstructionSpec
    from zoomy_core.symsolver.strategies.chorin import predictor_nsm
    from zoomy_core.systemmodel.system_model import SystemModel, _DT_SYMBOL
    from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
    T = {}
    instrument(T)
    t0 = time.perf_counter()
    NL, Q_IN = 2, 0.11197
    bcs = []
    for ell in (1, 2):
        bcs += [Dirichlet("left", on=f"q_{ell}_0", value=Q_IN / NL),
                Dirichlet("left", on=f"q_{ell}_1", value=0.0),
                Dirichlet("left", on=f"r_{ell}_0", value=0.0),
                Dirichlet("left", on=f"r_{ell}_1", value=0.0),
                Dirichlet("right", on=f"P_{ell}_0", value=0.0),
                Dirichlet("right", on=f"P_{ell}_1", value=0.0)]
    model = MLVAM(n_layers=NL, level=1, dimension=2, boundary_conditions=bcs,
                  closures=[Newtonian(), StressFree()], equilibrium_reconstruction="none")
    sm = SystemModel.from_model(model)
    sm.initial_conditions = Constant(constants=lambda n: np.zeros(n))
    sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
    split = model.chorin_split(_DT_SYMBOL, system_model=sm)
    mesh = BaseMesh.create_1d(domain=(-1.5, 1.5), n_inner_cells=60)
    nsm = predictor_nsm(split.SM_pred, split.SM_press, split.SM_corr,
                        ReconstructionSpec(order=2, limiter=Limiter.minmod))
    T["case build (model, SM, split, NSM)"] = time.perf_counter() - t0
    names = [str(v) for v in nsm.sm.state]
    dt, t_end = 1e-3, float(os.environ.get("T_END", "0.1"))
    frames = []
    kw = dict(time_end=t_end - 1e-9, dt=dt, time_order=2, dt_write=100 * dt,
              on_write=lambda t, Q, A: frames.append(float(t)),
              pressure_tol=1e-9, pressure_maxit=200)
    prof = cProfile.Profile() if os.environ.get("PROFILE") else None
    t0 = time.perf_counter()
    if prof:
        prof.enable()
    s = SymbolicChorinSolverJax(**kw)
    Q0, A = s.setup(mesh, nsm, split.SM_press, split.SM_corr)
    if prof:
        prof.disable()
    T["setup (total)"] = time.perf_counter() - t0
    xc = np.asarray(s.mesh_np.cell_centers)[0, :60]
    bump = 0.20 * np.exp(-(xc ** 2) / (2 * 0.20 ** 2))
    Q0 = np.zeros(np.shape(Q0))
    Q0[names.index("b")] = bump
    Q0[names.index("h")] = np.maximum(np.where(xc < 1.0, 0.34 - bump, 0.015), 0.015)
    t0 = time.perf_counter()
    t, Q, _ = s.run(jnp.asarray(Q0), A)
    T["run (total)"] = time.perf_counter() - t0
    T["frames"] = len(frames)
    return T, prof, s, np.asarray(Q)


def main():
    if sys.argv[1] == "escalante":
        T, prof, s, Q = escalante()
    else:
        T, prof, s, Q = matrix_run(sys.argv[1], sys.argv[2])
    print("backend", jax.default_backend(), "jax", jax.__version__)
    for k, v in T.items():
        print(f"  {k:36s} {v:9.2f}" if isinstance(v, float) else f"  {k:36s} {v}")
    for k in ("/jax/core/compile/jaxpr_trace_duration",
              "/jax/core/compile/jaxpr_to_mlir_module_duration",
              "/jax/core/compile/backend_compile_duration"):
        v = EV.get(k, [])
        print(f"  {k.split('/')[-1]:36s} {sum(v):9.2f}  ({len(v)} events, max "
              f"{max(v) if v else 0:.2f})")
    print("  cache hits", len(EV.get("/jax/compilation_cache/cache_hits", [])),
          "requests", len(EV.get("/jax/compilation_cache/compile_requests_use_cache", [])))
    for p in PROGRAMS:
        print("  program", json.dumps(p))
    if os.environ.get("SOURCE"):
        with open(os.environ["SOURCE"], "w") as f:
            f.write(s.source)
    if os.environ.get("SAVE_Q"):
        import numpy as np
        np.save(os.environ["SAVE_Q"], Q)
    if prof:
        out = io.StringIO()
        st = pstats.Stats(prof, stream=out).sort_stats("cumulative")
        st.print_stats(45)
        if os.environ.get("CALLEES"):
            st.print_callees(os.environ["CALLEES"])
        print(out.getvalue())


if __name__ == "__main__":
    main()
