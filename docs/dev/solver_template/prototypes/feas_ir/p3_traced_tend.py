"""ITEM 3: time_end as a TRACED argument of the production compiled loop.

Production HyperbolicSolver._run keys the compiled loop on float(time_end)
and closes over it.  The subclass below changes exactly two things:
  * time_end is dropped from the cache key,
  * the jitted loop takes time_end as its last (traced) argument.
Everything else in _run / _compile_time_loop is the production code path
(the loop body is HyperbolicSolver._compile_time_loop itself, called with a
TRACER for time_end from inside an outer jit).

Measures: compiles over 3 chunks with different time_end (prod vs traced),
bitwise equality chunk by chunk, AD (forward + reverse) incl. d/d time_end.
"""
import logging

from common import *  # noqa
import numpy as np
import jax
import jax.numpy as jnp
import zoomy_core.fvm.timestepping as timestepping
from zoomy_core.mesh import LSQMesh
from zoomy_jax.fvm.solver_jax import HyperbolicSolver

COMPILES = []


class _H(logging.Handler):
    def emit(self, rec):
        m = rec.getMessage()
        if m.startswith("Compiling "):
            COMPILES.append(m.split(" ")[1])


jax.config.update("jax_log_compiles", True)
for nm in ("jax._src.interpreters.pxla",):
    lg = logging.getLogger(nm)
    lg.addHandler(_H())
    lg.setLevel(logging.WARNING)
    lg.propagate = False


class TracedEndSolver(HyperbolicSolver):
    """time_end traced: one compiled loop for every time_end."""

    def _run(self, Q, Qaux, write_output, parameters, observer, time0,
             time_end=None):
        assert not write_output, "scratch: the uncoupled no-output path only"
        parameters = (self._rt_parameters if parameters is None
                      else parameters)
        if Qaux is None:
            Qaux = self._qaux_from_q(Q, parameters)

        def save_fields(time, time_stamp, i_snapshot, Q, Qaux):
            return i_snapshot

        n_intervals = max(self.settings.output.snapshots - 1, 1)
        i_snapshot = save_fields(time0, time0, 0.0, Q, Qaux)
        last_written = time0
        time_end = self.time_end if time_end is None else time_end
        self._couple_setup()
        _advance, _couple_proceed = self._advance, self._couple_proceed
        _log_every = int(self.march_log_interval or 0)
        _ad_mode, _ad_ckpt = self.ad_mode, int(self.ad_checkpoints)
        if _ad_mode == "reverse":
            import equinox.internal as eqxi

            def _while_loop(cond, body, init):
                return eqxi.while_loop(cond, body, init, kind="checkpointed",
                                       checkpoints=_ad_ckpt)
        else:
            _while_loop = jax.lax.while_loop

        # NO float(time_end) in the key
        key = ("traced_tend", int(n_intervals), _log_every, False, None,
               _ad_mode, _ad_ckpt, observer)
        loop = self._time_loop_cache.get(key)
        if loop is None:
            compile_ = HyperbolicSolver._compile_time_loop

            @jax.jit
            def loop(time, iteration, i_snapshot, last_written, Q, Qaux,
                     parameters, time_end):
                # the PRODUCTION loop builder, handed a tracer for time_end;
                # its inner jax.jit inlines into this trace.
                inner = compile_(_advance, _couple_proceed, save_fields,
                                 _while_loop, time_end=time_end,
                                 n_intervals=n_intervals,
                                 log_every=_log_every, observer=observer)
                return inner(time, iteration, i_snapshot, last_written, Q,
                             Qaux, parameters)
            self._time_loop_cache[key] = loop
        return loop(time0, 0.0, i_snapshot, last_written, Q, Qaux,
                    parameters, jnp.asarray(time_end, dtype=Q.dtype))

    def run_chunk(self, Q, Qaux=None, parameters=None, time0=0.0,
                  time_end=None):
        Q, Qaux, _ = self._run(Q, Qaux, False, parameters, None, time0,
                               time_end=time_end)
        return Q, Qaux


def make(cls, ad_mode="forward"):
    nsm = nsm_for("stoker", 2)
    mesh = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=200)
    s = cls(time_end=1.0, compute_dt=timestepping.adaptive(CFL=CFL,
                                                            dimension=1),
            march_log_interval=0, ad_mode=ad_mode, ad_checkpoints=4)
    Q, Qaux = s.setup_simulation(mesh, nsm)
    return s, Q, Qaux


CHUNKS = [(0.0, 0.7), (0.7, 1.3), (1.3, 2.0)]

# ── production: time_end is a param -> a new compiled loop per chunk ────────
s, Q, Qaux = make(HyperbolicSolver)
c0 = len(COMPILES)
prod = []
Qc, Qauxc = Q, Qaux
for t0, t1 in CHUNKS:
    s.time_end = t1
    Qc, Qauxc = s.run_simulation(Qc, Qauxc, write_output=False, time0=t0)
    prod.append(np.asarray(Qc))
n_prod = sum(1 for c in COMPILES[c0:] if "time_loop" in c)
print(f"production: compiles of the time loop over 3 chunks = {n_prod} "
      f"(cache entries {len(s._time_loop_cache)})")

# ── traced time_end ─────────────────────────────────────────────────────────
t, Q, Qaux = make(TracedEndSolver)
c0 = len(COMPILES)
trc = []
Qc, Qauxc = Q, Qaux
for t0, t1 in CHUNKS:
    Qc, Qauxc = t.run_chunk(Qc, Qauxc, time0=t0, time_end=t1)
    trc.append(np.asarray(Qc))
n_trc = sum(1 for c in COMPILES[c0:] if c.startswith("jit(loop"))
print(f"traced time_end: compiles of the loop over 3 chunks = {n_trc} "
      f"(cache entries {len(t._time_loop_cache)}; all compiles logged: "
      f"{sorted(set(COMPILES[c0:]))})")
for k, (a, b) in enumerate(zip(prod, trc)):
    print(f"  chunk {k}: bitwise {bool((a == b).all())} "
          f"max|diff| {np.abs(a - b).max():.3e}")

# ── AD through the traced-time_end loop ─────────────────────────────────────
p0 = t._rt_parameters
ig = list(t.nsm.parameter_values.as_dict()).index("g")
eg = jnp.zeros_like(p0).at[ig].set(1.0)


def J(p, T, solver):
    Qf, _ = solver.run_chunk(Q, Qaux, parameters=p, time0=0.0, time_end=T)
    return jnp.sum(Qf[1, :200] ** 2)


tf, _, _ = make(TracedEndSolver, "forward")
tr, _, _ = make(TracedEndSolver, "reverse")
T = 1.0
_, d = jax.jvp(lambda p: J(p, T, tf), (p0,), (eg,))
g = jax.grad(lambda p: J(p, T, tr))(p0)
print(f"AD g: jvp {float(d):.12e} grad {float(g[ig]):.12e} rel "
      f"{abs(float(g[ig]) - float(d)) / abs(float(d)):.2e}")
for eps in (1e-5, 1e-6):
    fd = (J(p0 + eps * eg, T, tf) - J(p0 - eps * eg, T, tf)) / (2 * eps)
    print(f"    FD eps {eps:.0e}: rel {abs(float(fd) - float(d)) / abs(float(d)):.2e}")
# derivative w.r.t. time_end itself (only possible because it is traced)
_, dT = jax.jvp(lambda T_: J(p0, T_, tf), (jnp.asarray(T),), (jnp.asarray(1.0),))
gT = jax.grad(lambda T_: J(p0, T_, tr))(jnp.asarray(T))
for eps in (1e-6, 1e-7):
    fdT = (J(p0, T + eps, tf) - J(p0, T - eps, tf)) / (2 * eps)
    print(f"d/d time_end: jvp {float(dT):.10e} grad {float(gT):.10e} "
          f"FD(eps {eps:.0e}) {float(fdT):.10e} rel "
          f"{abs(float(fdT) - float(dT)) / abs(float(dT)):.2e}")
print(f"forward compiles cache entries after AD: tf {len(tf._time_loop_cache)} "
      f"tr {len(tr._time_loop_cache)}")
