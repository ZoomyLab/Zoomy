"""Matvecs per pressure solve, old vs symbolic (first 5 steps)."""
import sys, collections
import jax
import zoomy_jax.fvm.solver_chorin_vam_jax as OLD
import zoomy_jax.symsolver.runtime as RT
COUNT = collections.Counter()

def counting(gmres, key):
    def wrapped(A, b, **kw):
        def mv(x):
            jax.debug.callback(lambda: COUNT.update([key]))
            return A(x)
        return gmres(mv, b, **kw)
    return wrapped

OLD.jax_gmres = counting(OLD.jax_gmres, "old")
RT.jax_gmres = counting(RT.jax_gmres, "new")
sys.argv = ["x"]
import profile_chorin as P   # runs 50 timed steps of each after one warm call
n = 52
print("matvecs per step: old", COUNT["old"] / n, "new", COUNT["new"] / n)
