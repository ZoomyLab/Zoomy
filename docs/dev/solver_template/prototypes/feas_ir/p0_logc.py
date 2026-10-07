import logging, jax, jax.numpy as jnp
recs=[]
class H(logging.Handler):
    def emit(self, r): recs.append((r.name, r.getMessage()[:100]))
jax.config.update("jax_log_compiles", True)
logging.getLogger().addHandler(H()); logging.getLogger().setLevel(logging.DEBUG)
def run(x, t): return jnp.sin(x)*t
f = jax.jit(run); f(jnp.ones(3), 1.0); f(jnp.ones(3), 2.0)
for r in recs: print(r)
