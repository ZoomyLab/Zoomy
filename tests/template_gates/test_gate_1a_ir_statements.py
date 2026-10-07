"""Gate 1a-1 (core): the template IR, its checks and the Python walker on numpy.

* every statement kind lowers and runs on numpy in BOTH lowering modes
  (host Python control flow and the traced B.* form) with identical results;
* build-time checks: arity, kind, scope, hook outputs, replicated predicates,
  unresolved / missing build flags, missing bodies;
* generated source is written, readable, and a deliberate shape error's
  traceback ends in a generated line naming the template statement.
"""
import os
import traceback

import numpy as np
import pytest
import sympy as sp

from zoomy_core.solver.template import (
    AnyTrue, Assign, BLOCKS, BlockSpec, Call, Callback, ConstTable, GuardError,
    Hook, IfStatic, IsFinite, LinSolve, LinSolveOpts, MissingBodyError,
    NumpyBackend, Procedure, PythonWalker, ReduceSum, Root, RootOpts, Select,
    TemplateCheckError, While, check_program, fill_hooks, resolve_static,
    show_program)
from zoomy_core.solver.template import toy

pytestmark = pytest.mark.template_gate


def _build(proc, *, lowering="auto", flags=None, record=None, linsolve=None):
    reg, C = toy.toy_registry(), toy.toy_constants()
    w = PythonWalker(NumpyBackend(), lowering=lowering)
    return w.build(proc, toy.toy_bodies(np, record=record), registry=reg,
                   constants=C, flags=flags or {"double_acc": True},
                   solver_name="gate")


def _run(g, **kw):
    inp = toy.toy_inputs(np, **kw)
    return g.entry(inp["u"], inp["p"], inp["s"], 0.0, 0.3)


def test_every_statement_kind_numpy_host_equals_traced():
    m = fill_hooks(toy.march_proc(), toy.toy_fills())
    kinds = set()

    def visit(p):
        from zoomy_core.solver.template.ir import iter_statements
        for s in iter_statements(p.body):
            kinds.add(type(s).__name__)
            for _, q in s.procedures():
                visit(q)
    visit(m)
    assert kinds >= {"Seq", "Assign", "Call", "Hook", "IfStatic", "While",
                     "Repeat", "ForEachFace", "Select", "Root", "LinSolve",
                     "Guard", "Callback"}, kinds

    out = {}
    for low in ("host", "traced"):
        rec = []
        g = _build(m, lowering=low, record=rec)
        assert set(g.lowering.values()) == {low}
        u, t, it, acc = _run(g)
        out[low] = (np.asarray(u), t, it, acc, rec)
    (uh, th, ih, ah, rh), (ut, tt, it_, at, rt) = out["host"], out["traced"]
    assert np.array_equal(uh, ut) and th == tt == 0.3 and ih == it_ and ah == at
    assert isinstance(ih, int) and ih == 26      # int counter stays an int
    assert len(rh) == len(rt) == ih              # one idempotent write per step
    assert [r[2] for r in rh] == list(range(1, ih + 1))
    # Select fired at least once (trouble threshold 1.15 < max u0 = 1.3)
    assert np.isfinite(uh).all()


def test_ifstatic_flags_and_hooks():
    base = toy.march_proc()
    a1 = _run(_build(fill_hooks(base, toy.toy_fills()),
                     flags={"double_acc": True}))[3]
    a0 = _run(_build(fill_hooks(base, toy.toy_fills()),
                     flags={"double_acc": False}))[3]
    w_term = 1e-3 * np.sum(2 * np.linspace(0.0, 1.0, 16))   # ForEachFace part
    assert a1 - w_term == pytest.approx(2 * (a0 - w_term), rel=1e-12)
    # unresolved / missing flags are hard errors
    with pytest.raises(KeyError, match="double_acc"):
        resolve_static(base, {})
    with pytest.raises(TemplateCheckError, match="unresolved IfStatic"):
        check_program(base, toy.toy_registry(), toy.toy_constants())
    # an unfilled hook runs its default body; a fill replaces it
    hp = Procedure("h", "x:real", "y:real", [
        Assign("y", sp.Symbol("x")),
        Hook("adjust", inputs=("y",), outputs=("y",),
             default=[Assign("y", sp.Symbol("y") + 1)])])
    g0 = PythonWalker(NumpyBackend()).build(hp, {})
    g1 = PythonWalker(NumpyBackend()).build(
        fill_hooks(hp, {"adjust": [("plug", [Assign("y", 10 * sp.Symbol("y"))])]}),
        {})
    assert g0.entry(2.0) == 3.0 and g1.entry(2.0) == 20.0


@pytest.mark.parametrize("kind", ["dense", "banded"])
def test_linsolve_kinds_agree_numpy(kind):
    m = fill_hooks(toy.march_proc(linsolve="krylov"), toy.toy_fills())
    mk = fill_hooks(toy.march_proc(linsolve=kind), toy.toy_fills())
    u0 = np.asarray(_run(_build(m))[0])
    u1 = np.asarray(_run(_build(mk))[0])
    assert np.abs(u1 - u0).max() < 1e-11


def test_guard_raises_with_values():
    m = fill_hooks(toy.march_proc(), toy.toy_fills())
    g = _build(m)
    with pytest.raises(GuardError, match=r"dt not admissible \[dt=nan"):
        _run(g, a=float("nan"))


def test_root_and_linsolve_on_fail_guard():
    x = sp.Symbol("x")
    # x^2 + 1 = 0 has no real root -> Newton cannot converge -> D9 Guard
    res = Procedure("noroot", "x:real", "r:real", [Assign("r", x**2 + 1)])
    p = Procedure("f", "x0:real", "x:real", [
        Root(res, "x0", out="x", opts=RootOpts(tol=1e-12, maxiter=5, jac="fd",
                                               linsolve=LinSolveOpts(kind="dense")))])
    g = PythonWalker(NumpyBackend()).build(p, {})
    with pytest.raises(GuardError, match="Newton not converged"):
        g.entry(np.asarray([0.5]))


def _err(proc, match, registry=None, constants=None):
    with pytest.raises(TemplateCheckError, match=match):
        check_program(proc, registry or toy.toy_registry(),
                      constants or toy.toy_constants())


def test_build_time_checks():
    u, p = sp.symbols("u p")
    sig = "u:cell_state p:params"
    _err(Procedure("a", sig, "dt:real", [Call("toy_dt", ("u", "p"), ("dt",))]),
         r"passes 2 argument\(s\)")
    _err(Procedure("a", sig, "dt:real", [
        Call("toy_dt", ("u", "u", "c_cfl"), ("dt",))]), "is cell_state but parameter")
    _err(Procedure("a", sig, "dt:real", [
        Call("toy_dt", ("u", "p", "c_cfl"), ("dt", "x"))]), r"binds 2 result")
    _err(Procedure("a", sig, "y:real", [Assign("y", sp.Symbol("nope"))]),
         "unbound name 'nope'")
    _err(Procedure("a", sig, "u:cell_state", [Assign("u", 1.0)]),
         "rebound with kind")
    _err(Procedure("a", sig, "u:cell_state", [Call("toy_record", (), ())]),
         "use Callback")
    _err(Procedure("a", sig, "u:cell_state", [
        Hook("h", inputs=("u",), outputs=(), body=[Assign("u", 2 * u)])]),
        "not among its declared outputs")
    # replicated-predicate rule: a per-rank value guarding communication
    reg = toy.toy_registry().extended([
        BlockSpec("local_max", "u:cell_state", "m:real", comm="none")])
    _err(Procedure("a", sig, "u:cell_state", [
        Call("local_max", ("u",), ("m",)),
        Select(sp.Symbol("m") > 1, then=[Call("toy_lap", ("u",), ("u",))])]),
        "not replicated", registry=reg)
    # ... and the reduced value is fine
    check_program(Procedure("a", sig, "u:cell_state", [
        Select(AnyTrue(IsFinite(sp.Symbol("u"))),
               then=[Call("toy_lap", ("u",), ("u",))])]), reg, toy.toy_constants())
    # loop carries keep their kinds; names bound inside a loop do not escape
    _err(Procedure("a", "t:real", "y:real", [
        While(sp.Symbol("t") < 1, body=[Assign("y", sp.Symbol("t")),
                                         Assign("t", sp.Symbol("t") + 1)])]),
         "output 'y' is not bound")
    # unknown constant in an option
    x = sp.Symbol("x")
    res = Procedure("r1", "x:real", "r:real", [Assign("r", x - 1)])
    _err(Procedure("a", "x0:real", "x:real", [
        Root(res, "x0", out="x", opts=RootOpts(tol="c_missing"))]),
        "unknown constant")


def test_missing_body_is_reported():
    m = fill_hooks(toy.march_proc(), toy.toy_fills())
    bodies = toy.toy_bodies(np)
    del bodies["toy_lap"]
    with pytest.raises(MissingBodyError, match="toy_lap"):
        PythonWalker(NumpyBackend()).build(
            m, bodies, registry=toy.toy_registry(), constants=toy.toy_constants(),
            flags={"double_acc": True})


def test_generated_source_is_readable_and_tracebacks_point_into_it():
    m = fill_hooks(toy.march_proc(), toy.toy_fills())
    g = _build(m)
    assert os.path.isfile(g.path) and g.path.endswith("gate_toy_march_numpy.py")
    src = open(g.path).read()
    assert src == g.source
    code = [ln for ln in src.splitlines()
            if ln.strip() and not ln.lstrip().startswith(("#", "def ", "return",
                                                          '"""', "(", "xp ="))
            and "=" in ln and "B.jit" not in ln and not ln.startswith(
                ("ENTRY", "PROCEDURES", "c_"))]
    tagged = [ln for ln in code if "# [" in ln]
    assert len(tagged) >= 0.9 * len(code), [ln for ln in code if "# [" not in ln]
    assert "# [toy_step:body[4]] Root toy_reaction_residual" in src

    # deliberate shape error: two states of different cell counts
    u_, v_ = sp.symbols("u v")
    bad = Procedure("shape_bug", "u:cell_state v:cell_state", "y:cell_state", [
        Assign("y", 2 * u_, note="fine"),
        Assign("y", u_ + v_, note="the deliberate shape error")])
    gb = PythonWalker(NumpyBackend()).build(bad, {}, solver_name="bug")
    with pytest.raises(ValueError) as ei:
        gb.entry(np.ones((1, 8)), np.ones((1, 9)))
    last = traceback.extract_tb(ei.value.__traceback__)[-1]
    assert last.filename == gb.path
    assert "# [shape_bug:body[1]] Assign y -- the deliberate shape error" in \
        last.line, last.line


def test_show_prints_the_template():
    m = fill_hooks(toy.march_proc(), toy.toy_fills())
    txt = show_program(m, toy.toy_constants(), toy.toy_registry())
    for frag in ("procedure toy_march", "while time < t_until",
                 "if static double_acc", "hook observe(acc, u) -> (acc)  "
                 "[filled by toy_obs]", "root toy_reaction_residual",
                 "linsolve toy_diffusion_matvec", "callback[idempotent] toy_record",
                 "for face f in range(c_ncell)", "repeat k in range(2)",
                 "toy_dt(u: cell_state, p: params, cfl: real) -> (dt: real)"):
        assert frag in txt, frag
