"""Cost of numerical_face_sides vs numerical_face_sides_speed per case: op
counts in the jaxpr (eig, sqrt, total) and us/face."""
import collections, sys, os, time as wall
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cases"))
import common as C
import jax, jax.numpy as jnp, numpy as np
from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax


def malpasset():
    sys.modules.pop("run", None)
    import malpasset as MP
    mesh, b = MP.build()
    return mesh, b["nsm"]


for case, make in (("malpasset", malpasset), ("swe2d", lambda: C.bump_2d(2))):
    mesh, nsm = make()
    print(case, type(nsm.build_numerics()).__name__)
    s = SymbolicHyperbolicSolverJax(time_end=0.1, cfl=0.4)
    Q, A = s.setup(mesh, nsm)
    rt = s.rt
    nf = 20000
    rng = np.random.default_rng(0)
    q = jnp.asarray(np.abs(rng.normal(1, 0.1, (Q.shape[0], nf))))
    a = jnp.asarray(np.abs(rng.normal(1, 0.1, (A.shape[0], nf))))
    n = jnp.zeros((s.mesh_np.dimension, nf)).at[0].set(1.0)
    args = (q, q * 1.01, a, a, jnp.asarray(s.parameters), n)
    for nm in ("numerical_face_sides", "numerical_face_sides_speed"):
        k = getattr(rt, nm)
        cnt = collections.Counter()

        def walk(j):
            for e in j.eqns:
                cnt[e.primitive.name] += 1
                for v in e.params.values():
                    jj = getattr(v, "jaxpr", None)
                    if jj is not None:
                        walk(jj if hasattr(jj, "eqns") else jj.jaxpr)
        walk(jax.make_jaxpr(k)(*args).jaxpr)
        f = jax.jit(k)
        jax.block_until_ready(f(*args))
        t = min((lambda t0: (jax.block_until_ready(f(*args)), wall.perf_counter() - t0)[1])(wall.perf_counter()) for _ in range(10))
        print(f"  {nm:28s} eqns {sum(cnt.values()):5d} eig {cnt.get('eig', 0)} sqrt {cnt['sqrt']} "
              f"max {cnt['max']} {1e6 * t / nf:.3f} us/face")
