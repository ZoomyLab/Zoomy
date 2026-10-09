"""One preCICE participant on JAX: python participant_jax.py NAME SIDE H0 OUT."""
import sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import sympy as sp
import build as B
import kernels_swe as K
import rt_jax

name, side, h0, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
me, peer = ("L", "R") if name == "Left" else ("R", "L")
ad = rt_jax.PreciceAdapter(name, "precice-config.xml", f"{name}-Mesh",
                           [me + "0", me + "1"], [peer + "0", peer + "1"], 0.5)
src = B.coupled_src("py", side, order=1, fixed_dt=sp.Float(1e-3))
fn = rt_jax.build(src, "hyperbolic_precice",
                  rt_jax.runtime(K.jax_kernels(), couple=ad))
N = 100
Q0 = jnp.stack([jnp.full(N, h0), jnp.zeros(N)])
t, Q, tok = jax.jit(fn)(Q0, 0.0, 9.81, 0.01, 0.0, 0.45, 1.0 / 200, 0.0, jnp.int64(0))
ad.p.finalize()
np.save(out, np.asarray(Q))
print(f"{name}: t={float(t):.15g} effects={int(tok)}")
