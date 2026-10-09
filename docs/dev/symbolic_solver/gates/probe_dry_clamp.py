"""Dry dam break, template reconstruction vs the old eta path: is the old
face clamp eta_f >= b_f the only difference?  A probe applies the clamp
inside the runtime's inverse map (NOT in the code); with it the template
should match the old solver to round-off."""
import numpy as np
import jax.numpy as jnp
import common as C
import gate1_vs_solver_jax as G
import zoomy_jax.symsolver.runtime as RT

orig = RT.runtime


def clamped(nsm, rt, *a, **k):
    m = orig(nsm, rt, *a, **k)
    pw = m["pointwise"]
    names = [str(s) for s in nsm.state]
    b, h = names.index("b"), names.index("h")

    def pointwise(kern, X, Xaux, p):
        if kern is rt.state_from_reconstruction:
            X = X.at[h].set(jnp.maximum(X[h], X[b]))
        return pw(kern, X, Xaux, p)
    m["pointwise"] = pointwise
    return m


def clamped_demoted(nsm, rt, *a, **k):
    """... plus the old dry-cell demotion: phi = 0 where h < wet_dry_eps."""
    m = clamped(nsm, rt, *a, **k)
    ex = m["extrapolate"]
    names = [str(s) for s in nsm.state]
    b, h = names.index("b"), names.index("h")
    eps = float(nsm.parameter_values.wet_dry_eps)

    def extrapolate(W, Wb, G, phi, mesh):
        dry = (W[h] - W[b]) < eps
        phi = jnp.broadcast_to(jnp.asarray(phi, W.dtype), W.shape)
        return ex(W, Wb, G, jnp.where(dry[None, :], 0.0, phi), mesh)
    m["extrapolate"] = extrapolate
    return m


if __name__ == "__main__":
    make = lambda: C.dry_dambreak_2d(ties="old")     # noqa: E731
    Qo, n_old = G.old_march(make, 0.02, 0.45, **G.eta_kw(make))
    for label, rtf in (("template (no clamp)", orig), ("template + probe clamp", clamped)):
        RT.runtime = rtf
        Qn, t, s = G.new_march(make, 0.02, 0.45)
        names = [str(v) for v in s.nsm.state]
        h = Qn[names.index("h")]
        print(f"{label:24s} max|new-old| {np.abs(Qn - Qo).max():.2e}  h_min {h.min():.3e}  "
              f"finite {np.isfinite(Qn).all()}  mass {h.sum():.12g} (old {Qo[names.index('h')].sum():.12g})")
    RT.runtime = orig
