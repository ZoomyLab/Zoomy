"""Gate 1a-1 (jax): every template statement on numpy, jax-forward and
jax-reverse; AD through a march with Select / Guard / Callback / Root /
LinSolve; one compile across time_end / write-interval values; readable
tracebacks; the host/compiled boundary.

Measured (2026-10-07, CPU, x64):
  toy march (all statements)  jax-fwd == jax-rev bitwise; |numpy - jax| 3.7e-13
                              (FD-Newton + scipy GMRES vs AD-Newton + jax GMRES,
                              both at tol 1e-13)
  advection template (MOOD)   jax-fwd == jax-rev bitwise; |numpy - jax| 5e-16
  AD (toy, reverse loop)      |jvp - FD| / |jvp| 3.4e-10 (eps 1e-6),
                              |grad.d - jvp| / |jvp| 1.2e-15
"""
import os
import traceback

import numpy as np
import pytest
import sympy as sp

import jax
import jax.numpy as jnp

from zoomy_core.solver.template import (
    Assign, LinSolve, LinSolveOpts, LoweringError, NumpyBackend, Output,
    Procedure, PythonWalker, Root, RootOpts, Setup, Solver, fill_hooks)
from zoomy_core.solver.template import toy
from zoomy_core.solver.template.toy import advection_bodies, advection_state
from zoomy_jax.template import JaxBackend

pytestmark = pytest.mark.template_gate

BACKENDS = {
    "numpy": (NumpyBackend, np),
    "jax_fwd": (lambda: JaxBackend("forward"), jnp),
    "jax_rev": (lambda: JaxBackend("reverse", checkpoints=4), jnp),
}


def _toy(backend, *, stateful=False, record=None, window_calls=None, flags=None,
         linsolve="krylov"):
    mk, xp = BACKENDS[backend]
    m = fill_hooks(toy.march_proc(stateful=stateful, linsolve=linsolve),
                   toy.toy_fills())
    return PythonWalker(mk()).build(
        m, toy.toy_bodies(xp, record=record, window_calls=window_calls),
        registry=toy.toy_registry(), constants=toy.toy_constants(),
        flags=flags or {"double_acc": True}, solver_name="gate"), xp


def _run(g, xp, p=None, t_end=0.3):
    inp = toy.toy_inputs(xp)
    return g.entry(inp["u"], inp["p"] if p is None else p, inp["s"], 0.0, t_end)


def test_every_statement_kind_numpy_fwd_rev():
    out, recs = {}, {}
    for b in BACKENDS:
        rec = []
        g, xp = _toy(b, record=rec)
        u, t, it, acc = _run(g, xp)
        out[b] = (np.asarray(u), float(t), int(it), float(acc))
        recs[b] = rec
        if b != "numpy":
            assert set(g.lowering.values()) == {"traced"}
    un, tn, itn, an = out["numpy"]
    uf, tf, itf, af = out["jax_fwd"]
    ur, tr, itr, ar = out["jax_rev"]
    assert tn == tf == tr == 0.3 and itn == itf == itr == 26
    assert np.array_equal(uf, ur) and af == ar               # fwd == rev bitwise
    assert np.abs(uf - un).max() < 1e-12 and abs(af - an) < 1e-12 * abs(an)
    # the idempotent callback fired once per step, in order, on every backend
    for b in BACKENDS:
        assert [r[2] for r in recs[b]] == list(range(1, 27)), b
    assert max(np.abs(a[1] - c[1]).max()
               for a, c in zip(recs["numpy"], recs["jax_fwd"])) < 1e-12


def _adv(backend, **kw):
    class Adv(Solver):
        name = "adv"

        def setup_bodies(self, mesh, nsm, B):
            xp = B.xp
            Q, Qaux = advection_state(xp, 40)
            return Setup(bodies=advection_bodies(xp, order=2, limiter="none"),
                         flags={"cell_ncp": False}, geom=None, Q=Q, Qaux=Qaux,
                         parameters=xp.asarray([1.0, 0.0]), dimension=1,
                         degree=1, order=2)
    mk, _ = BACKENDS[backend]
    return Adv(backend=mk(), **kw)


def test_hyperbolic_template_backends_and_single_compile():
    res = {}
    for b in BACKENDS:
        s = _adv(b, time_end=0.25, cfl=0.9, plugins=[Output()])
        Q0, A0 = s.setup_simulation(None, None)
        Q, _ = s.run_simulation(Q0, A0, write_interval=0.05)
        assert float(s.last_outputs["time"]) == 0.25
        res[b] = np.asarray(Q)
        if b == "numpy":
            continue
        g = next(iter(s._compiled.values()))
        assert g.entry._cache_size() == 1
        for te, wi in ((0.2, 0.03), (0.15, 0.1), (0.3, 0.07)):
            s.run_simulation(Q0, A0, time_end=te, write_interval=wi)
            assert float(s.last_outputs["time"]) == te       # traced time_end
        assert g.entry._cache_size() == 1                    # ONE compile
    assert np.array_equal(res["jax_fwd"], res["jax_rev"])
    assert np.abs(res["jax_fwd"] - res["numpy"]).max() < 1e-14


def test_ad_jvp_fd_grad_through_select_guard_callback_root_linsolve():
    gf, _ = _toy("jax_fwd")
    gr, _ = _toy("jax_rev")
    p0 = jnp.asarray(toy.toy_inputs(np)["p"])
    d = jnp.asarray([0.3, -0.5, 1.0, 0.0])

    def J(p, g):
        u, t, it, acc = _run(g, jnp, p=p)
        return jnp.sum(u ** 2) + acc

    _, jv = jax.jvp(lambda p: J(p, gf), (p0,), (d,))
    gd = jax.grad(lambda p: J(p, gr))(p0) @ d
    eps = 1e-6
    fd = (J(p0 + eps * d, gf) - J(p0 - eps * d, gf)) / (2 * eps)
    jv, gd, fd = float(jv), float(gd), float(fd)
    assert abs(fd - jv) / abs(jv) < 1e-8, (jv, fd)
    assert abs(gd - jv) / abs(jv) < 1e-12, (jv, gd)


def test_linsolve_kinds_nullspace_precond_and_root_jacobians():
    x, dt, nu = sp.symbols("x dt nu")
    from zoomy_core.solver.template import BLOCKS, BlockSpec, Call
    reg = BLOCKS.extended([BlockSpec("nlap", "x:cell_state", "y:cell_state",
                                     comm="neighbor")])
    lap = Procedure("lap_neumann", "x:cell_state", "y:cell_state",
                    [Call("nlap", ("x",), ("y",))])
    jac = Procedure("jacobi", "r:cell_state", "z:cell_state",
                    [Assign("z", -sp.Symbol("r") / 2)])

    def nlap(xp):
        def f(v):                                   # pure-Neumann Laplacian
            vp = xp.concatenate([v[..., 1:], v[..., -1:]], axis=-1)
            vm = xp.concatenate([v[..., :1], v[..., :-1]], axis=-1)
            return vp - 2 * v + vm
        return {"nlap": f}

    def build(kind, nullspace=None, precond=None):
        return Procedure("solve_" + kind, "b:cell_state", "x:cell_state", [
            LinSolve(lap, "b", out="x", opts=LinSolveOpts(
                kind=kind, tol=1e-12, restart=40, maxiter=40, bandwidth=1
                if kind == "banded" else None, nullspace=nullspace,
                precond=precond))])

    rng = np.random.default_rng(0)
    b = rng.standard_normal((1, 24))
    b -= b.mean()
    sols = {}
    for name, (mk, xp) in BACKENDS.items():
        for kind, ns, pc in (("krylov", "constant", None), ("krylov", "constant", jac),
                             ("dense", "constant", None)):
            g = PythonWalker(mk()).build(build(kind, ns, pc), nlap(xp), registry=reg,
                                         solver_name=f"ns_{kind}_{pc is not None}")
            xs = np.asarray(g.entry(xp.asarray(b)))
            assert abs(xs.mean()) < 1e-12
            assert np.abs(nlap(np)["nlap"](xs) - b).max() < 1e-9
            sols[(name, kind, pc is not None)] = xs
    ref = sols[("numpy", "dense", False)]
    for k, v in sols.items():
        assert np.abs(v - ref).max() < 1e-9, k
    # banded (tridiagonal Dirichlet Laplacian inside the toy) == Krylov, on jax
    gk, _ = _toy("jax_fwd", linsolve="krylov")
    gb, _ = _toy("jax_fwd", linsolve="banded")
    assert np.abs(np.asarray(_run(gk, jnp)[0]) - np.asarray(_run(gb, jnp)[0])).max() \
        < 1e-11

    # Root: AD / FD / analytic Newton Jacobians, cellwise / dense updates
    res = Procedure("cubic", "x:cell_state u:cell_state", "r:cell_state",
                    [Assign("r", x ** 3 + x - sp.Symbol("u"))])
    jv = Procedure("cubic_jv", "x:cell_state v:cell_state u:cell_state",
                   "jv:cell_state",
                   [Assign("jv", (3 * x ** 2 + 1) * sp.Symbol("v"))])
    roots = []
    for name, (mk, xp) in BACKENDS.items():
        for jk, lk in (("ad", "cellwise"), ("fd", "dense"), ("analytic", "cellwise")):
            proc = Procedure(f"cub_{jk}_{lk}", "u:cell_state", "x:cell_state", [
                Root(res, "u", args=("u",), out="x", opts=RootOpts(
                    tol=1e-13, maxiter=40, jac=jk,
                    jac_proc=jv if jk == "analytic" else None,
                    linsolve=LinSolveOpts(kind=lk)))])
            g = PythonWalker(mk()).build(proc, {}, solver_name="root")
            u = np.linspace(-2, 3, 10)[None, :]
            r = np.asarray(g.entry(xp.asarray(u)))
            assert np.abs(r ** 3 + r - u).max() < 1e-12
            roots.append(r)
    assert max(np.abs(r - roots[0]).max() for r in roots) < 1e-12


def test_guard_in_compiled_code_and_traceback_into_generated_file():
    g, xp = _toy("jax_fwd")
    with pytest.raises(Exception, match="dt not admissible"):
        jax.block_until_ready(_run(g, xp, p=jnp.asarray([jnp.nan, 0.5, 0.01, 1.15])))

    u_, v_ = sp.symbols("u v")
    bad = Procedure("shape_bug", "u:cell_state v:cell_state", "y:cell_state", [
        Assign("y", 2 * u_, note="fine"),
        Assign("y", u_ + v_, note="the deliberate shape error")])
    gb = PythonWalker(JaxBackend("forward")).build(bad, {}, solver_name="bug")
    with pytest.raises(TypeError) as ei:
        gb.entry(jnp.ones((1, 8)), jnp.ones((1, 9)))
    frames = traceback.extract_tb(ei.value.__traceback__)
    user = [f for f in frames if "site-packages" not in f.filename
            and "/jax/" not in f.filename]
    assert user[-1].filename == gb.path and os.path.isfile(gb.path)
    assert "# [shape_bug:body[1]] Assign y -- the deliberate shape error" in \
        user[-1].line


def test_host_compiled_boundary_with_stateful_callback():
    out = {}
    for b in ("numpy", "jax_fwd"):
        wc = []
        g, xp = _toy(b, stateful=True, window_calls=wc, flags={"double_acc": False})
        u, t, it, acc = _run(g, xp)
        out[b] = (np.asarray(u), float(acc))
        assert wc == [0, 1] and float(t) == 0.3
        if b == "jax_fwd":
            assert g.lowering["toy_march"] == "host"
            assert g.lowering["toy_step"] == "traced"
            src = g.source
            # the inner time loop is one compiled unit inside the host window loop
            assert "while (window < c_windows):" in src
            assert "= B.jit(_unit_" in src
    assert np.abs(out["numpy"][0] - out["jax_fwd"][0]).max() < 1e-12
    # the reverse backend has no lowering for a stateful callback
    with pytest.raises(LoweringError, match="stateful"):
        _toy("jax_rev", stateful=True, flags={"double_acc": False})
