"""Radial dry dam break, O2 + MOOD, three face treatments of the model's
W = (b, eta, u):
  plain   -- the template (no clamp),
  clamp   -- the old path's eta_f >= b_f,
  audusse -- reconstruct (eta, h, u) instead, b = eta - h
             (Audusse et al. 2004, Sec. 3: "work with h and h + z").
Counts MOOD triggers (candidate inadmissible) and whole-step O1 fallbacks;
h_min, mass.  Probe: maps swapped inside the JAX runtime only."""
import jax
import jax.numpy as jnp
import numpy as np
import common as C
import zoomy_jax.symsolver.runtime as RT
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax

orig = RT.runtime
COUNT = {"inadmissible calls": 0, "flagged": 0}
MINS = []


def make_runtime(mode):
    def runtime(nsm, rt, *a, **k):
        m = orig(nsm, rt, *a, **k)
        pw, inad = m["pointwise"], m["inadmissible"]
        names = [str(s) for s in nsm.state]
        b, h = names.index("b"), names.index("h")
        mom = [i for i, n in enumerate(names) if n.startswith("q_")]

        def pointwise(kern, X, Xaux, p):
            if mode == "clamp" and kern is rt.state_from_reconstruction:
                X = X.at[h].set(jnp.maximum(X[h], X[b]))
            if mode == "audusse" and kern is rt.reconstruction_variables:
                W = pw(kern, X, Xaux, p)                  # (b, eta, u)
                return W.at[b].set(W[h]).at[h].set(X[h])  # (eta, h, u)
            if mode == "audusse" and kern is rt.state_from_reconstruction:
                Q = jnp.zeros_like(X).at[b].set(X[b] - X[h]).at[h].set(X[h])
                for i in mom:
                    Q = Q.at[i].set(X[i] * X[h])
                return Q
            return pw(kern, X, Xaux, p)

        def inadmissible(kern, Q, Qaux, p):
            bad = inad(kern, Q, Qaux, p)
            jax.debug.callback(lambda n: COUNT.__setitem__("flagged", COUNT["flagged"] + int(n > 0)),
                               jnp.sum(bad))
            jax.debug.callback(lambda v: MINS.append(float(v)), jnp.min(Q[h]))
            jax.debug.callback(lambda: COUNT.__setitem__("inadmissible calls",
                                                         COUNT["inadmissible calls"] + 1))
            return bad
        m["pointwise"], m["inadmissible"] = pointwise, inadmissible
        return m
    return runtime


for mode in ("plain", "clamp", "audusse"):
    RT.runtime = make_runtime(mode)
    COUNT.update({"inadmissible calls": 0, "flagged": 0})
    MINS.clear()
    mesh, nsm = C.dry_dambreak_2d()
    s = SymbolicHyperbolicSolverJax(time_end=0.02, cfl=0.45)
    Q, A = s.setup(mesh, nsm)
    t, Q, A = s.run(Q, A)
    jax.effects_barrier()
    names = [str(v) for v in nsm.state]
    hh = np.asarray(Q)[names.index("h")]
    # inadmissible runs once per step on the candidate (+ once per repair)
    print(f"{mode:8s} t={float(t):.3g}  checks {COUNT['inadmissible calls']:3d}  "
          f"with flagged cells {COUNT['flagged']:3d}  h_min {hh.min():.3e}  "
          f"finite {np.isfinite(np.asarray(Q)).all()}  mass {hh.sum():.15g}", flush=True)
    m = np.array(MINS)
    print(f"         min h per check: candidates {np.array2string(m[0::2][:8], precision=1)}"
          f"  repairs {np.array2string(m[1::2][:8], precision=1)}  worst {m.min():.1e}")
RT.runtime = orig
