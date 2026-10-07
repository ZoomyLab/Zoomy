"""ITEM 4: capability table — backend caps vs the caps a COMPOSED solver
requires, computed by walking the IR before any lowering.

Then each verdict is checked EMPIRICALLY against the real lowering:
  refused  jax-reverse + stateful (in-trace) coupling predicate
  refused  jax-reverse + ForEachFace with a traced count
  accepted jax-reverse + idempotent Callback output (item 1 measured it)
"""
from __future__ import annotations

import dataclasses

from common import *  # noqa
import numpy as np
import sympy as sp
import jax
import jax.numpy as jnp
from jax.experimental import io_callback

from zoomy_core.solver.ir import (Assign, Call, ForEachFace, Procedure, While,
                                  expression_names)
from ir_ext import Callback, Guard, ProcedureBuilderExt, Root, Select

# ── capability vocabulary ───────────────────────────────────────────────────
CAPS_DOC = {
    "runtime_branch": "Select lowers (lax.cond / if)",
    "runtime_abort": "Guard lowers (error_if / raise / abort())",
    "host_callback_idempotent": "Callback whose replay is harmless (keyed by "
                                "an input counter, e.g. i_snapshot writes)",
    "host_callback_stateful": "Callback with host state advanced per call "
                              "(coupling advance/read, a counter)",
    "stateful_loop_predicate": "a While condition reads a stateful Callback "
                               "result (coupling 'ongoing')",
    "foreach_traced_count": "ForEachFace over a count that is NOT a "
                            "build-time constant",
    "implicit_root": "Root lowers (custom_root / Newton)",
}

BACKEND_CAPS = {
    "numpy": {"runtime_branch", "runtime_abort", "host_callback_idempotent",
              "host_callback_stateful", "stateful_loop_predicate",
              "foreach_traced_count", "implicit_root"},
    # lax.while_loop + jvp: every callback runs once per primal iteration
    "jax-forward": {"runtime_branch", "runtime_abort",
                    "host_callback_idempotent", "host_callback_stateful",
                    "stateful_loop_predicate", "foreach_traced_count",
                    "implicit_root"},
    # eqxi checkpointed While: the backward pass RE-RUNS bodies from
    # checkpoints -> callbacks replay; a traced-count fori_loop becomes a
    # lax.while_loop, which has no transpose.
    "jax-reverse": {"runtime_branch", "runtime_abort",
                    "host_callback_idempotent", "implicit_root"},
    "c-family": {"runtime_branch", "runtime_abort", "host_callback_idempotent",
                 "host_callback_stateful", "stateful_loop_predicate",
                 "foreach_traced_count", "implicit_root"},
}


@dataclasses.dataclass(frozen=True)
class HostCall(Callback):
    """Callback + declared results + replay semantics (scratch extension)."""
    results: tuple = ()
    idempotent: bool = True


def required_caps(proc: Procedure, *, procedures=None, static_names=()):
    """Walk ``proc`` (and every Procedure it Calls, via ``procedures``) and
    return {cap: [where, ...]}."""
    procedures = dict(procedures or {})
    need: dict = {}

    def add(cap, where):
        need.setdefault(cap, []).append(where)

    def walk(block, path, stateful_results):
        for i, s in enumerate(block):
            where = f"{path}[{i}]:{type(s).__name__}"
            if isinstance(s, Select):
                add("runtime_branch", where)
            elif isinstance(s, Guard):
                add("runtime_abort", where)
            elif isinstance(s, Root):
                add("implicit_root", where)
            elif isinstance(s, Callback):
                idem = getattr(s, "idempotent", True)
                add("host_callback_idempotent" if idem
                    else "host_callback_stateful", f"{where}({s.fn})")
                if not idem:
                    stateful_results.update(getattr(s, "results", ()))
            elif isinstance(s, ForEachFace):
                if s.count not in static_names:
                    add("foreach_traced_count", f"{where}(count={s.count})")
            elif isinstance(s, Call) and s.procedure in procedures:
                walk(procedures[s.procedure].stmts,
                     f"{path}[{i}]->{s.procedure}", stateful_results)
            if isinstance(s, While):
                inner: set = set()
                walk(s.body, where + ".body", inner)
                stateful_results |= inner
                used = set(expression_names(s.condition))
                if used & inner:
                    add("stateful_loop_predicate",
                        f"{where}(condition reads {sorted(used & inner)})")
            else:
                for c in s.children():
                    walk(c, where, stateful_results)

    walk(proc.stmts, proc.name, set())
    return need


class CapabilityError(RuntimeError):
    pass


def check(proc, backend, **kw):
    need = required_caps(proc, **kw)
    missing = {c: w for c, w in need.items() if c not in BACKEND_CAPS[backend]}
    if missing:
        lines = [f"  - {c} ({CAPS_DOC[c]}) at {w}" for c, w in missing.items()]
        raise CapabilityError(
            f"backend {backend!r} cannot lower {proc.name!r}:\n"
            + "\n".join(lines))
    return sorted(need)


# ── the three composed solvers ──────────────────────────────────────────────
t, te, x, a = sp.symbols("time t_end x a")
ongoing = sp.Symbol("ongoing")

# (A) coupled march: loop predicate = 'coupling ongoing' from a stateful
#     host call inside the loop (preCICE advance)
coupled = Procedure(name="solver_march", args=("time", "t_end"), stmts=(
    Assign("ongoing", sp.Integer(1)),
    While(sp.And(sp.Lt(t, te), sp.Eq(ongoing, 1)), carry=("x", "time", "ongoing"),
          body=(Assign("x", x * a, declare=False),
                Assign("time", t + 1, declare=False),
                HostCall("solver_coupling_advance", args=("time",),
                         anchor="x", results=("ongoing",),
                         idempotent=False))),))
# (B) a face loop over a TRACED count inside the march
foreach = Procedure(name="solver_march", args=("time", "t_end", "n_faces"),
                    stmts=(While(sp.Lt(t, te), carry=("x", "time"), body=(
                        ForEachFace("f", "n_faces", body=(
                            Assign("x", x * a, declare=False),)),
                        Assign("time", t + 1, declare=False))),))
# (C) output as an idempotent Callback (+ guard + select), item-1 shape
output = Procedure(name="solver_march", args=("time", "t_end"), stmts=(
    While(sp.Lt(t, te), carry=("x", "time"), body=(
        Assign("x", x * a, declare=False),
        Guard(sp.Lt(x, sp.oo), "blow-up", on="x"),
        Select(sp.Gt(x, 100), then=(Assign("x", x / 2, declare=False),),
               otherwise=(), outputs=("x",)),
        Assign("time", t + 1, declare=False),
        HostCall("solver_write_fields", args=("x", "time"), anchor="x",
                 idempotent=True))),))

print("capability table:")
allcaps = sorted(CAPS_DOC)
print(f"  {'cap':28s}" + "".join(f"{b:>13s}" for b in BACKEND_CAPS))
for c in allcaps:
    print(f"  {c:28s}" + "".join(
        f"{'yes' if c in BACKEND_CAPS[b] else '-':>13s}" for b in BACKEND_CAPS))

for name, proc, kw in [("A coupled predicate", coupled, {}),
                       ("B ForEachFace traced count", foreach, {}),
                       ("B' ForEachFace static count", foreach,
                        {"static_names": ("n_faces",)}),
                       ("C Callback output", output, {})]:
    for backend in ("jax-forward", "jax-reverse"):
        try:
            caps = check(proc, backend, **kw)
            print(f"[ACCEPT] {name:30s} on {backend}: needs {caps}")
        except CapabilityError as e:
            print(f"[REFUSE] {name:30s} on {backend}: "
                  + str(e).splitlines()[1].strip())

# ── empirical confirmation with the real lowering ───────────────────────────
print("\nempirical:")
N_WIN = 5
state = {"n": 0, "calls": 0}


def host_advance(time):
    state["calls"] += 1
    state["n"] += 1
    return np.int64(1 if state["n"] < N_WIN else 0)


@jax.custom_jvp
def advance_cb(anchor, time):
    return io_callback(host_advance, jax.ShapeDtypeStruct((), jnp.int64),
                       time, ordered=True)


@advance_cb.defjvp
def _adv_jvp(p, tng):
    return advance_cb(*p), np.zeros((), dtype=jax.dtypes.float0)


class B(ProcedureBuilderExt):
    def _compile_callback(self, stmt):
        if not isinstance(stmt, HostCall) or not stmt.results:
            return super()._compile_callback(stmt)

        def run(env):
            env = dict(env)
            env[stmt.results[0]] = advance_cb(env["x"], env["time"])
            return env
        return run


def run_A(kind, a_):
    fn = B("jax", externals={"solver_coupling_advance": host_advance},
           while_kind=kind, checkpoints=2).build(coupled)
    e = fn(time=jnp.asarray(0.0), t_end=jnp.asarray(100.0),
           x=jnp.asarray(1.0), a=a_)
    return e["x"], e["time"]


state.update(n=0, calls=0)
xf, tf = jax.jit(lambda a_: run_A("lax", a_))(jnp.asarray(1.1))
print(f"A forward: steps {int(tf)} (expected {N_WIN}), x {float(xf):.6f}, "
      f"host calls {state['calls']}")
state.update(n=0, calls=0)
_, dx = jax.jvp(lambda a_: run_A("lax", a_)[0], (jnp.asarray(1.1),),
                (jnp.asarray(1.0),))
print(f"A jvp:     dx/da {float(dx):.6f} (exact {N_WIN * 1.1 ** (N_WIN - 1):.6f}),"
      f" host calls {state['calls']}")
state.update(n=0, calls=0)
try:
    g = jax.grad(lambda a_: run_A("checkpointed", a_)[0])(jnp.asarray(1.1))
    print(f"A grad (checkpointed): dx/da {float(g):.6f} (exact "
          f"{N_WIN * 1.1 ** (N_WIN - 1):.6f}), host calls {state['calls']} "
          f"-> {'WRONG' if abs(float(g) - N_WIN * 1.1 ** (N_WIN - 1)) > 1e-9 else 'ok'}")
except Exception as e:  # noqa
    print(f"A grad (checkpointed): raises {type(e).__name__}: "
          f"{str(e).splitlines()[0][:120]}")


def run_B(kind, a_, nf):
    fn = ProcedureBuilderExt("jax", while_kind=kind, checkpoints=2).build(
        foreach)
    e = fn(time=jnp.asarray(0.0), t_end=jnp.asarray(3.0), x=jnp.asarray(1.0),
           a=a_, n_faces=nf)
    return e["x"]


for nf_label, nf in (("traced", None), ("static", 2)):
    try:
        if nf is None:   # a REAL tracer: count is a jit argument
            g = jax.jit(jax.grad(lambda a_, n: run_B("checkpointed", a_, n)))(
                jnp.asarray(1.1), jnp.asarray(2))
        else:
            g = jax.jit(jax.grad(lambda a_: run_B("checkpointed", a_, nf)))(
                jnp.asarray(1.1))
        print(f"B grad (checkpointed, {nf_label} count): dx/da {float(g):.6f} "
              f"(exact {6 * 1.1 ** 5:.6f})")
    except Exception as e:  # noqa
        print(f"B grad (checkpointed, {nf_label} count): raises "
              f"{type(e).__name__}: {str(e).splitlines()[0][:110]}")
_, dB = jax.jit(lambda a_, n: jax.jvp(lambda b_: run_B("lax", b_, n),
                                       (a_,), (jnp.asarray(1.0),)))(
    jnp.asarray(1.1), jnp.asarray(2))
print(f"B jvp (lax, traced count): dx/da {float(dB):.6f} (exact {6 * 1.1 ** 5:.6f})")

writes = []


def run_C(kind, a_):
    fn = ProcedureBuilderExt(
        "jax", externals={"solver_write_fields":
                          lambda x_, t_: writes.append((float(t_), float(x_)))},
        while_kind=kind, checkpoints=2).build(output)
    return fn(time=jnp.asarray(0.0), t_end=jnp.asarray(8.0),
              x=jnp.asarray(30.0), a=a_)["x"]


g = jax.grad(lambda a_: run_C("checkpointed", a_))(jnp.asarray(1.5))
_, d = jax.jvp(lambda a_: run_C("lax", a_), (jnp.asarray(1.5),),
               (jnp.asarray(1.0),))
uniq = sorted(set(writes))
print(f"C grad {float(g):.10f} jvp {float(d):.10f} equal {float(g) == float(d)}; "
      f"host write calls {len(writes)} for {len(uniq) // 2 if False else len(set(w[0] for w in writes))} "
      f"distinct stamps (replays are harmless: idempotent by stamp)")


# A2: the predicate itself is the stateful host query, evaluated in cond_fun
import equinox.internal as eqxi


def host_ongoing(_t):
    state["calls"] += 1
    state["n"] += 1
    return np.bool_(state["n"] <= N_WIN)


def run_A2(kind, a_):
    def cond(c):
        return io_callback(host_ongoing, jax.ShapeDtypeStruct((), jnp.bool_),
                           c[1], ordered=True)

    def body(c):
        return (c[0] * a_, c[1] + 1.0)
    init = (jnp.asarray(1.0), jnp.asarray(0.0))
    if kind == "lax":
        return jax.lax.while_loop(cond, body, init)
    return eqxi.while_loop(cond, body, init, kind="checkpointed",
                           checkpoints=2)


state.update(n=0, calls=0)
xf, tf = jax.jit(lambda a_: run_A2("lax", a_))(jnp.asarray(1.1))
print(f"A2 (predicate = host query) forward: steps {int(tf)} host calls "
      f"{state['calls']}")
state.update(n=0, calls=0)
try:
    g = jax.jit(jax.grad(lambda a_: run_A2("checkpointed", a_)[0]))(
        jnp.asarray(1.1))
    ex = N_WIN * 1.1 ** (N_WIN - 1)
    print(f"A2 grad (checkpointed): {float(g):.6f} exact {ex:.6f} host calls "
          f"{state['calls']} -> {'WRONG' if abs(float(g) - ex) > 1e-9 else 'ok'}")
except Exception as e:  # noqa
    print(f"A2 grad (checkpointed): raises {type(e).__name__}: "
          f"{str(e).splitlines()[0][:140]}")

# A3: a stateful READ feeds the state (preCICE read of peer data per window)
peer = {"k": 0}


def host_read(_t):
    peer["k"] += 1
    return np.float64(0.01 * peer["k"])     # peer data changes per call


@jax.custom_jvp
def read_cb(anchor, t_):
    return io_callback(host_read, jax.ShapeDtypeStruct((), jnp.float64), t_,
                       ordered=True)


@read_cb.defjvp
def _read_jvp(p, tng):
    return read_cb(*p), jnp.zeros(())


def run_A3(kind, a_):
    def body(c):
        x_, t_ = c
        return (x_ * a_ + read_cb(x_, t_), t_ + 1.0)
    init = (jnp.asarray(1.0), jnp.asarray(0.0))
    cond = lambda c: c[1] < N_WIN  # noqa: E731
    if kind == "lax":
        return jax.lax.while_loop(cond, body, init)[0]
    return eqxi.while_loop(cond, body, init, kind="checkpointed",
                           checkpoints=2)[0]


peer["k"] = 0
_, d3 = jax.jit(lambda a_: jax.jvp(lambda b: run_A3("lax", b), (a_,),
                                   (jnp.asarray(1.0),)))(jnp.asarray(1.1))
calls_jvp = peer["k"]
peer["k"] = 0
g3 = jax.jit(jax.grad(lambda a_: run_A3("checkpointed", a_)))(jnp.asarray(1.1))
print(f"A3 (stateful read feeds the state): jvp {float(d3):.10f} (host reads "
      f"{calls_jvp}) grad(checkpointed) {float(g3):.10f} (host reads "
      f"{peer['k']}) rel {abs(float(g3) - float(d3)) / abs(float(d3)):.2e} -> "
      f"{'WRONG' if abs(float(g3) - float(d3)) > 1e-12 * abs(float(d3)) else 'ok'}")
