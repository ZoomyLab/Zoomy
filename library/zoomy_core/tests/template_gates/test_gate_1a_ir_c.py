"""Gate 1a-1 (core): the C walker.

* the emitted SSPRK2 hyperbolic template with D1 MOOD + guards + output
  passes ``g++ -std=c++17 -fsyntax-only -Wall -Wextra``, links against the
  plain-array backend with trivial (1-D advection) bodies, runs, and is
  BITWISE equal to the same template lowered to numpy (state, step count,
  snapshot stamps, number of stage evaluations incl. MOOD re-entries);
* the every-statement toy (Root, LinSolve, Repeat, ForEachFace, Callbacks,
  Select, While, Hook) emits syntactically valid C++;
* the printer re-checks call arity / kinds (F6).
"""
import os
import shutil
import subprocess

import numpy as np
import pytest

from zoomy_core.solver.template import (
    Call, Output, Setup, Solver, fill_hooks)
from zoomy_core.solver.template import toy
from zoomy_core.solver.template.toy import advection_bodies, advection_state

pytestmark = [pytest.mark.template_gate,
              pytest.mark.skipif(shutil.which("g++") is None, reason="no g++")]

HERE = os.path.dirname(os.path.abspath(__file__))
CVEC = os.path.join(HERE, "c_vec")
FLAGS = ["-std=c++17", "-Wall", "-Wextra", "-Wno-unused-parameter",
         "-Wno-unused-but-set-variable"]


def _gxx(args, cwd):
    r = subprocess.run(["g++", *args], capture_output=True, text=True, cwd=cwd)
    assert r.returncode == 0, r.stderr[-4000:]
    return r


class Adv(Solver):
    name = "adv"

    def __init__(self, record, calls, **kw):
        super().__init__(**kw)
        self.record, self.calls = record, calls

    def setup_bodies(self, mesh, nsm, backend):
        Q, Qaux = advection_state(np, 40)
        b = advection_bodies(np, order=2, limiter="none", record=self.record)
        gu = b["gather_update"]

        def counted(*a):
            self.calls.append(1)
            return gu(*a)
        b["gather_update"] = counted
        return Setup(bodies=b, flags={"cell_ncp": False}, geom=None, Q=Q,
                     Qaux=Qaux, parameters=np.asarray([1.0, 0.0]), dimension=1,
                     degree=1, order=2)


def _emit(proc, ctx, flags, out):
    from zoomy_core.solver.template.walker_c import CWalker
    w = CWalker()
    prog = w.emit(proc, registry=ctx.registry, constants=ctx.constants,
                  flags=flags)
    w.write(prog, out)
    shutil.copy(os.path.join(CVEC, "zoomy_template_vec.hpp"), out)
    return prog


def test_emitted_hyperbolic_template_compiles_runs_and_matches_numpy(tmp_path):
    rec, calls = [], []
    s = Adv(rec, calls, time_end=0.25, cfl=0.9, plugins=[Output(snapshots=5)])
    Q0, A0 = s.setup_simulation(None, None)
    Qn, An = s.run_simulation(Q0, A0)
    ref = s.last_outputs

    march, ctx = s.template(order=2, dimension=1, degree=1)
    out = str(tmp_path)
    prog = _emit(march, ctx, {"cell_ncp": False, "aux_from_state": False}, out)
    assert "zoomy_elementwise(" in prog.procs and "struct TemplateWorkspace" in \
        prog.decls and "static constexpr double c_cfl = 0.9;" in prog.decls
    assert "PORTABLE_FN" not in prog.procs                       # F7
    assert "inline void step(Ctx& ctx, CellState& Q, CellAux& Qaux" in prog.procs
    for f in ("advection_bodies.cpp", "main.cpp"):
        shutil.copy(os.path.join(CVEC, f), out)
    _gxx([*FLAGS, "-fsyntax-only", "-I", out, "main.cpp"], out)
    _gxx([*FLAGS, "-O2", "-ffp-contract=off", "-I", out, "main.cpp",
          "advection_bodies.cpp", "-o", "march"], out)
    with open(os.path.join(out, "input.txt"), "w") as f:
        f.write(f"{Q0.shape[1]}\n")
        f.write(" ".join(repr(float(x)) for x in np.ravel(Q0)) + "\n")
        f.write(" ".join(repr(float(x)) for x in np.ravel(A0)) + "\n")
        f.write("1.0 0.0\n0.25\n")
    r = subprocess.run([os.path.join(out, "march"), "input.txt", "result.txt"],
                       capture_output=True, text=True, cwd=out)
    assert r.returncode == 0, r.stderr
    lines = open(os.path.join(out, "result.txt")).read().splitlines()
    snaps = [ln.split() for ln in lines if ln.startswith("snap")]
    fin = next(ln.split() for ln in lines if ln.startswith("final"))
    it, isnap, t, stage_calls = int(fin[1]), int(fin[2]), float(fin[3]), int(fin[4])
    Qc = np.array([float(x) for x in fin[5:]])
    assert (it, isnap, t) == (ref["iteration"], ref["i_snapshot"], ref["time"]) \
        == (34, 5, 0.25)
    assert stage_calls == len(calls) == 106          # MOOD re-entries included
    assert np.array_equal(Qc, np.ravel(Qn))          # bitwise
    assert [(int(a[1]), float(a[2])) for a in snaps] == [(i, tt) for i, tt, _ in rec]
    for a, (_, _, q) in zip(snaps, rec):
        assert np.array_equal(np.array([float(x) for x in a[3:]]), np.ravel(q))


def test_every_statement_toy_emits_valid_cpp(tmp_path):
    m = fill_hooks(toy.march_proc(), toy.toy_fills())
    from zoomy_core.solver.template.stage import BuildContext
    ctx = BuildContext(registry=toy.toy_registry(), constants=toy.toy_constants())
    out = str(tmp_path)
    prog = _emit(m, ctx, {"double_acc": True}, out)
    for frag in ("zoomy_root(ctx, _res", "zoomy_linsolve(ctx, _mv",
                 "for (k = 0; k < 2; ++k)", "for (f = 0; f < c_ncell; ++f)",
                 "w[f] = 2*s[f];", "toy_record(ctx, time, u, iteration);",
                 "zoomy_reduce_sum(ctx", "if (zoomy_reduce_any(ctx",
                 "while (time < t_until)", "zoomy_guard_fail(ctx"):
        assert frag in prog.procs, frag
    with open(os.path.join(out, "syntax_check.cpp"), "w") as f:
        f.write('#include "zoomy_template_vec.hpp"\n#include "template_procs.inc"\n')
    _gxx([*FLAGS, "-fsyntax-only", "-I", out, "syntax_check.cpp"], out)


def test_c_printer_rechecks_call_signatures():
    from zoomy_core.solver.template.check import check_program
    from zoomy_core.solver.template.walker_c import CWalkerError, make_printer_class
    m = fill_hooks(toy.march_proc(), toy.toy_fills())
    from zoomy_core.solver.template.ir import resolve_static
    info = check_program(resolve_static(m, {"double_acc": True}),
                         toy.toy_registry(), toy.toy_constants())
    P = make_printer_class()(info)
    P.pinfo = info.procs["toy_march"]
    plan = {"extra": {}}
    with pytest.raises(CWalkerError, match="arity"):
        P.call_plan(Call("toy_dt", ("u", "p"), ("dt",)), ("body", 0), plan)
    with pytest.raises(CWalkerError, match="is real, parameter"):
        P.call_plan(Call("toy_dt", ("time", "p", "c_cfl"), ("dt",)), ("body", 0),
                    plan)
