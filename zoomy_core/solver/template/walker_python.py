"""Python walker: template IR -> readable Python SOURCE for one backend.

The walker writes a module to the generated-code cache
(``<cache>/<hash>/<solver>_<proc>_<backend>.py``) and imports it with
importlib.  Every emitted line ends in a comment ``[procedure:path] Kind ...``
naming the template statement it lowers, so a traceback ends in a readable
generated line that says where in the template the failure happened.

Lowering goes through a backend primitive object ``B`` (injected into the
module at load time together with the block bodies ``blocks``):

    B.xp            array namespace (numpy / jax.numpy)
    B.jit           compile a unit (identity on numpy)
    B.while_loop    (cond(carry), body(carry), carry) -> carry
    B.repeat        (n, body(i, carry), carry) -> carry       static n
    B.foreach       (n, body(i, carry), carry) -> carry       static n
    B.cond          (pred, then(*ops), otherwise(*ops), ops) -> outputs
    B.root          (residual, x0, **opts) -> (x, ok, rnorm, iters)
    B.linsolve      (matvec, rhs, **opts) -> (x, ok, rel_resid, iters)
    B.guard         (ok, msg, names, values, anchors) -> anchors
    B.warn          (ok, msg, names, values) -> None
    B.callback_idempotent / callback_returns / callback_stateful
    B.scatter_add/min/max, B.reduce_min/max/sum/any/all, B.inner,
    B.set_at, B.safe_div, B.as_int

Two lowering modes, chosen per procedure:

* ``traced``: loops and branches become ``B.while_loop`` / ``B.cond`` /
  ``B.repeat`` over nested functions with an explicit carry -- the form a
  tracing backend (jax) compiles.  The entry procedure is wrapped in
  ``B.jit`` (one compiled unit).
* ``host``: plain Python ``while`` / ``if`` / ``for``.  Used for a whole
  program on a non-compiling backend (numpy), and on a compiling backend for
  every procedure that contains a STATEFUL callback (the host/compiled
  boundary rule): there, maximal runs of statements free of stateful
  callbacks are emitted as separate ``B.jit``-compiled units, so the innermost
  loop free of them is ONE compiled unit.
"""
from __future__ import annotations

import builtins
import dataclasses
import hashlib
import importlib.util
import itertools
import keyword
import os
import sys
import types
from typing import Optional

import sympy as sp
from sympy.printing.pycode import PythonCodePrinter

from zoomy_core.solver.template import expr as E
from zoomy_core.solver.template.check import (
    ProgramInfo, block_has_stateful, check_program, fmt_path,
    free_reads_block, writes_block)
from zoomy_core.solver.template.consts import Const, ConstTable, format_value
from zoomy_core.solver.template.ir import (
    Assign, Call, Callback, ForEachFace, Guard, Hook, LinSolve, Procedure,
    Repeat, Root, Select, Seq, While, resolve_static)
from zoomy_core.solver.template.kinds import is_scalar


class LoweringError(RuntimeError):
    """The backend has no lowering for a statement the template uses."""


class MissingBodyError(LookupError):
    """The body table lacks a block the template calls."""


# ── expressions ─────────────────────────────────────────────────────────────


class PyExprPrinter(PythonCodePrinter):
    """sympy -> Python source over ``xp`` (array namespace) and ``B``."""

    def __init__(self, rename=lambda n: n):
        super().__init__({"fully_qualified_modules": False,
                          "standard": "python3"})
        self.rename = rename

    def _print_Symbol(self, s):
        return self.rename(str(s))

    def _print_Float(self, f):
        return format_value(float(f), "real").replace('float("inf")', "xp.inf")

    def _print_Rational(self, r):
        return repr(float(r))

    _print_Half = _print_Rational

    def _print_Integer(self, i):
        return str(int(i))

    _print_Zero = _print_One = _print_NegativeOne = _print_Integer

    def _print_BooleanTrue(self, e):
        return "True"

    def _print_BooleanFalse(self, e):
        return "False"

    def _print_Infinity(self, e):
        return "xp.inf"

    def _print_NegativeInfinity(self, e):
        return "(-xp.inf)"

    def _print_NaN(self, e):
        return "xp.nan"

    def _print_Pi(self, e):
        return "xp.pi"

    def _print_Exp1(self, e):
        return "xp.e"

    def _nest(self, fn, args):
        a = [self._print(x) for x in args]
        out = a[-1]
        for x in reversed(a[:-1]):
            out = f"{fn}({x}, {out})"
        return out

    def _print_And(self, e):
        return self._nest("xp.logical_and", e.args)

    def _print_Or(self, e):
        return self._nest("xp.logical_or", e.args)

    def _print_Xor(self, e):
        return self._nest("xp.logical_xor", e.args)

    def _print_Not(self, e):
        return f"xp.logical_not({self._print(e.args[0])})"

    def _print_Relational(self, e):
        op = {"==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">",
              ">=": ">="}[e.rel_op]
        return f"({self._print(e.lhs)} {op} {self._print(e.rhs)})"

    _print_Equality = _print_Unequality = _print_StrictLessThan = \
        _print_LessThan = _print_StrictGreaterThan = _print_GreaterThan = \
        _print_Relational

    def _print_Min(self, e):
        return self._nest("xp.minimum", e.args)

    def _print_Max(self, e):
        return self._nest("xp.maximum", e.args)

    def _print_Abs(self, e):
        return f"xp.abs({self._print(e.args[0])})"

    def _print_Piecewise(self, e):
        args = list(e.args)
        if args[-1].cond in (True, sp.true):
            out = self._print(args[-1].expr)
            args = args[:-1]
        else:
            out = "xp.nan"
        for a in reversed(args):
            out = f"xp.where({self._print(a.cond)}, {self._print(a.expr)}, {out})"
        return out

    def _print_Pow(self, e, rational=False):
        b, x = e.args
        if x == sp.Rational(1, 2):
            return f"xp.sqrt({self._print(b)})"
        if x == -sp.Rational(1, 2):
            return f"(1/xp.sqrt({self._print(b)}))"
        if x == -1:
            return f"(1/{self.parenthesize(b, sp.printing.precedence.PRECEDENCE['Pow'])})"
        pb = self.parenthesize(b, sp.printing.precedence.PRECEDENCE["Pow"])
        px = self.parenthesize(x, sp.printing.precedence.PRECEDENCE["Pow"])
        return f"{pb}**{px}"

    def _print_Indexed(self, e):
        idx = ", ".join(self._print(i) for i in e.indices)
        return f"{self.rename(str(e.base.label))}[{idx}]"

    def _print_IsFinite(self, e):
        return f"xp.isfinite({self._print(e.args[0])})"

    def _print_AnyTrue(self, e):
        return f"B.reduce_any({self._print(e.args[0])})"

    def _print_AllTrue(self, e):
        return f"B.reduce_all({self._print(e.args[0])})"

    def _print_ReduceMin(self, e):
        return f"B.reduce_min({self._print(e.args[0])})"

    def _print_ReduceMax(self, e):
        return f"B.reduce_max({self._print(e.args[0])})"

    def _print_ReduceSum(self, e):
        return f"B.reduce_sum({self._print(e.args[0])})"

    def _print_Where(self, e):
        c, a, b = (self._print(x) for x in e.args)
        return f"xp.where({c}, {a}, {b})"

    def _print_FullLike(self, e):
        x, v = e.args
        if isinstance(v, sp.logic.boolalg.BooleanAtom):
            return f"xp.full(xp.shape({self._print(x)}), {self._print(v)})"
        return f"xp.full_like({self._print(x)}, {self._print(v)})"

    def _print_ZerosLike(self, e):
        return f"xp.zeros_like({self._print(e.args[0])})"

    def _print_MaskLike(self, e):
        x, v = e.args
        return f"xp.full(xp.shape({self._print(x)})[-1:], {self._print(v)})"

    def _print_SafeDiv(self, e):
        a, b, f = (self._print(x) for x in e.args)
        return f"B.safe_div({a}, {b}, {f})"

    def _print_Function(self, e):
        raise LoweringError(f"no Python lowering for function {type(e).__name__}")


def _elementary(np_name):
    def f(self, e):
        return f"xp.{np_name}({', '.join(self._print(a) for a in e.args)})"
    return f


for _sym, _np in E.ELEMENTARY.items():
    setattr(PyExprPrinter, f"_print_{_sym}", _elementary(_np))


# ── naming ──────────────────────────────────────────────────────────────────

_RESERVED = (set(keyword.kwlist) | set(keyword.softkwlist) | set(dir(builtins))
             | {"B", "xp", "blocks", "ENTRY", "PROCEDURES"})


def default_cache_dir() -> str:
    d = os.environ.get("ZOOMY_TEMPLATE_CACHE")
    if d:
        return d
    return os.path.join(os.path.expanduser("~"), ".cache", "zoomy",
                        "generated")


# ── emission ────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Generated:
    """A generated, loaded program."""

    entry: object               # the entry callable
    module: types.ModuleType
    path: str
    source: str
    info: ProgramInfo
    lowering: dict              # procedure name -> "traced" | "host"

    @property
    def procedures(self) -> dict:
        return self.module.PROCEDURES


class _Emitter:
    def __init__(self, info: ProgramInfo, backend, lowering: str,
                 solver_name: str):
        self.info = info
        self.B = backend
        self.compiles = bool(getattr(backend, "compiles", False))
        self.lowering = lowering
        self.solver_name = solver_name
        self.counter = itertools.count()
        self.units: list = []           # module-level unit definitions
        self.opt_consts = ConstTable()  # option values emitted as constants
        self.proc_names = {n: self._safe_proc(n) for n in info.procs}
        reserved = set(_RESERVED) | set(self.proc_names.values())
        self._rename = {}
        self._taken = set()
        for c in info.constants.names():
            self._rename[c] = c
            self._taken.add(c)
        self._reserved = reserved
        self.printer = PyExprPrinter(self.rename)
        self.modes = {}
        for n, pi in info.procs.items():
            if lowering == "host":
                self.modes[n] = "host"
            elif lowering == "traced":
                self.modes[n] = "traced"
            else:
                self.modes[n] = ("host" if (not self.compiles or pi.has_stateful)
                                 else "traced")

    # -- names ----------------------------------------------------------------
    def _safe_proc(self, n):
        return n + "_" if (n in _RESERVED or n.startswith("_")) else n

    def rename(self, n: str) -> str:
        if n in self._rename:
            return self._rename[n]
        r = "v" + n if n.startswith("_") else n
        while r in self._reserved or r in self._taken:
            r = r + "_"
        self._rename[n] = r
        self._taken.add(r)
        return r

    def ex(self, e) -> str:
        return self.printer.doprint(e)

    def tup(self, names) -> str:
        names = [self.rename(n) for n in names]
        if len(names) == 1:
            return f"({names[0]},)"
        return "(" + ", ".join(names) + ")"

    def opt(self, v, kind, label) -> str:
        if isinstance(v, str):
            return v
        self.opt_consts.add(Const(label, v, kind, "statement option"))
        return label

    def tag(self, proc, path, s, extra="") -> str:
        t = f"[{proc}:{fmt_path(path)}] {s.kind_name}"
        if extra:
            t += f" {extra}"
        if s.note:
            t += f" -- {s.note}"
        return "  # " + t

    def fresh(self, stem) -> str:
        return f"_{stem}_{next(self.counter)}"

    # -- module -----------------------------------------------------------------
    def module(self) -> str:
        procs_src = []
        for n, pi in self.info.procs.items():
            procs_src.extend(self.procedure(pi))
            procs_src.append("")
        root = self.info.root.name
        entry = self.proc_names[root]
        head = [
            "# GENERATED by zoomy_core.solver.template.walker_python -- do not edit.",
            f"# solver: {self.solver_name}   entry: {root}   backend: "
            f"{getattr(self.B, 'name', type(self.B).__name__)}",
            "# Every line ends in [procedure:statement-path] naming the template "
            "statement it lowers.",
            "# Injected at load time: B (backend primitives), blocks (block "
            "bodies).",
            "# Lowering per procedure: " + ", ".join(
                f"{n}={m}" for n, m in self.modes.items()),
            "",
            "xp = B.xp",
            "",
        ]
        consts = [c for c in self.info.constants if True]
        consts += [c for c in self.opt_consts]
        if consts:
            head.append("# -- emitted constants " + "-" * 50)
            for c in consts:
                if c.value is None:
                    raise LoweringError(f"constant {c.name} has no value")
                prov = f"  # {c.provenance}" if c.provenance else ""
                head.append(f"{c.name} = {format_value(c.value, c.kind)}{prov}")
            head.append("")
        body = []
        if self.units:
            body.append("# -- compiled units (host/compiled boundary) " + "-" * 30)
            for u in self.units:
                body.extend(u)
                body.append("")
        body.extend(procs_src)
        if self.modes[root] == "traced" and self.compiles:
            body.append(f"{entry} = B.jit({entry})  # entry: one compiled unit")
        body.append(f"ENTRY = {entry}")
        body.append("PROCEDURES = {" + ", ".join(
            f"{n!r}: {self.proc_names[n]}" for n in self.info.procs) + "}")
        return "\n".join(head + body) + "\n"

    # -- procedures -------------------------------------------------------------
    def procedure(self, pi) -> list:
        p = pi.proc
        mode = self.modes[p.name]
        fn = self.proc_names[p.name]
        args = ", ".join(self.rename(x.name) for x in p.inputs)
        out = [f"def {fn}({args}):  # procedure {p.name} [{mode}]"]
        if p.doc:
            out.append('    """' + p.doc.strip().replace('"""', "'''") + '"""')
        body = self.block(pi, p.body, ("body",), 1, mode)
        out.extend(body or ["    pass"])
        out.append("    " + self.ret(p.output_names) + f"  # return {p.name}")
        return out

    def ret(self, names) -> str:
        if not names:
            return "return None"
        if len(names) == 1:
            return f"return {self.rename(names[0])}"
        return "return (" + ", ".join(self.rename(n) for n in names) + ")"

    def unpack(self, names) -> str:
        if len(names) == 1:
            return self.rename(names[0])
        return "(" + ", ".join(self.rename(n) for n in names) + ")"

    def block(self, pi, block, prefix, ind, mode) -> list:
        if mode == "host" and self.compiles:
            return self.host_block_with_units(pi, block, prefix, ind)
        out = []
        for i, s in enumerate(block):
            out.extend(self.stmt(pi, s, prefix + (i,), ind, mode))
        return out

    def host_block_with_units(self, pi, block, prefix, ind) -> list:
        out = []
        run = []

        def flush():
            if run:
                out.extend(self.unit(pi, block, prefix, run, ind))
                run.clear()

        for i, s in enumerate(block):
            if block_has_stateful([s], self.info):
                flush()
                out.extend(self.stmt(pi, s, prefix + (i,), ind, "host"))
            else:
                run.append(i)
        flush()
        return out

    def unit(self, pi, block, prefix, idxs, ind) -> list:
        """A maximal run of statements free of stateful callbacks, emitted as
        a module-level function compiled with ``B.jit``."""
        stmts = [block[i] for i in idxs]
        first = pi.info(prefix + (idxs[0],))
        last = pi.info(prefix + (idxs[-1],))
        bound = set(first.env)
        ins = sorted(free_reads_block(stmts) & bound)
        outs = sorted(writes_block(stmts) & set(last.live_after))
        name = self.fresh("unit")
        paths = f"{fmt_path(prefix + (idxs[0],))}..{fmt_path(prefix + (idxs[-1],))}"
        u = [f"def {name}({', '.join(self.rename(n) for n in ins)}):  "
             f"# compiled unit [{pi.proc.name}:{paths}]"]
        for i in idxs:
            u.extend(self.stmt(pi, block[i], prefix + (i,), 1, "traced"))
        u.append("    " + ("return (" + ", ".join(self.rename(n) for n in outs)
                           + ("," if len(outs) == 1 else "") + ")"))
        u.append(f"{name} = B.jit({name})")
        self.units.append(u)
        pad = "    " * ind
        call = f"{name}({', '.join(self.rename(n) for n in ins)})"
        tag = f"  # [{pi.proc.name}:{paths}] compiled unit"
        if outs:
            return [f"{pad}{self.tup(outs)} = {call}{tag}"]
        return [f"{pad}{call}{tag}"]

    # -- statements -------------------------------------------------------------
    def stmt(self, pi, s, path, ind, mode) -> list:
        m = getattr(self, "s_" + type(s).__name__)
        return m(pi, s, path, ind, mode)

    def s_Seq(self, pi, s, path, ind, mode):
        return self.block(pi, s.body, path + ("body",), ind, mode)

    def s_Hook(self, pi, s, path, ind, mode):
        pad = "    " * ind
        label, body = s.blocks()[0]
        who = s.filled_by or "default"
        head = f"{pad}# -- hook {s.name} ({who})" + self.tag(pi.proc.name, path, s)
        lines = self.block(pi, body, path + (label,), ind, mode)
        return [head] + lines

    def s_Assign(self, pi, s, path, ind, mode):
        pad = "    " * ind
        st = pi.info(path)
        value = self.ex(s.expr)
        target_kind = pi.kinds[s.base]
        if s.index is not None:
            line = (f"{self.rename(s.base)} = B.set_at({self.rename(s.base)}, "
                    f"{self.ex(s.index)}, {value})")
        else:
            from zoomy_core.solver.template.check import _Kinds
            ek = _Kinds(lambda n: st.env.get(n) or self.info.constants[n].kind,
                        lambda m: None)(s.expr)
            if target_kind == "int" and ek == "real":
                value = f"B.as_int({value})"
            line = f"{self.rename(s.base)} = {value}"
        return [pad + line + self.tag(pi.proc.name, path, s, s.target)]

    def callee(self, s) -> str:
        if isinstance(s.callee, Procedure):
            return self.proc_names[s.callee.name]
        return f"blocks.{s.callee}"

    def s_Call(self, pi, s, path, ind, mode):
        pad = "    " * ind
        call = f"{self.callee(s)}({', '.join(self.rename(a) for a in s.args)})"
        tag = self.tag(pi.proc.name, path, s, s.callee_name)
        if not s.out:
            return [pad + call + tag]
        return [f"{pad}{self.unpack(s.out)} = {call}{tag}"]

    def s_Callback(self, pi, s, path, ind, mode):
        pad = "    " * ind
        method = f"callback_{s.kind}"
        if getattr(self.B, method, None) is None:
            raise LoweringError(
                f"backend {getattr(self.B, 'name', self.B)!r} has no lowering for "
                f"Callback(kind={s.kind!r}) ({s.fn}, at "
                f"{pi.proc.name}:{fmt_path(path)})")
        ins = "(" + "".join(self.rename(a) + ", " for a in s.inputs) + ")"
        tag = self.tag(pi.proc.name, path, s, f"{s.kind} {s.fn}")
        if s.kind == "idempotent":
            return [f"{pad}B.callback_idempotent(blocks.{s.fn}, {ins}){tag}"]
        st = pi.info(path)
        if s.kind == "returns":
            spec = self.info.registry.get(s.fn)
            like = "(" + "".join(
                (self.rename(o) if o in st.env else repr(p.kind)) + ", "
                for o, p in zip(s.outputs, spec.outputs)) + ")"
            call = f"B.callback_returns(blocks.{s.fn}, {ins}, {like})"
        else:
            call = f"B.callback_stateful(blocks.{s.fn}, {ins}, {len(s.outputs)})"
        if not s.outputs:
            return [pad + call + tag]
        return [f"{pad}{self.tup(s.outputs)} = {call}{tag}"]

    def s_Guard(self, pi, s, path, ind, mode):
        pad = "    " * ind
        st = pi.info(path)
        anchors = st.anchors
        names = "(" + "".join(repr(v) + ", " for v in s.values) + ")"
        vals = "(" + "".join(self.rename(v) + ", " for v in s.values) + ")"
        anc = "(" + "".join(self.rename(a) + ", " for a in anchors) + ")"
        call = f"B.guard({self.ex(s.ok)}, {s.msg!r}, {names}, {vals}, {anc})"
        tag = self.tag(pi.proc.name, path, s)
        if anchors:
            return [f"{pad}{self.tup(anchors)} = {call}{tag}"]
        return [pad + call + tag]

    def s_While(self, pi, s, path, ind, mode):
        pad = "    " * ind
        st = pi.info(path)
        tag = self.tag(pi.proc.name, path, s, str(s.cond))
        if mode == "host":
            lines = [f"{pad}while {self.ex(s.cond)}:{tag}"]
            body = self.block(pi, s.body, path + ("body",), ind + 1, "host")
            return lines + (body or [pad + "    pass"])
        k = next(self.counter)
        carry = st.carry
        ct = self.tup(carry)
        lines = [
            f"{pad}def _while_{k}_cond(_carry):{self.tag(pi.proc.name, path, s, 'cond')}",
            f"{pad}    {ct} = _carry",
            f"{pad}    return {self.ex(s.cond)}",
            f"{pad}def _while_{k}_body(_carry):{self.tag(pi.proc.name, path, s, 'body')}",
            f"{pad}    {ct} = _carry",
        ]
        lines += self.block(pi, s.body, path + ("body",), ind + 1, "traced")
        lines.append(f"{pad}    return {ct}")
        lines.append(f"{pad}{ct} = B.while_loop(_while_{k}_cond, _while_{k}_body, "
                     f"{ct}){tag}")
        return lines

    def _loop(self, pi, s, path, ind, mode, n, index, prim):
        pad = "    " * ind
        st = pi.info(path)
        tag = self.tag(pi.proc.name, path, s, f"n={n}")
        nn = str(n)
        if mode == "host":
            iv = self.rename(index) if index else "_i"
            lines = [f"{pad}for {iv} in range({nn}):{tag}"]
            body = self.block(pi, s.body, path + ("body",), ind + 1, "host")
            return lines + (body or [pad + "    pass"])
        k = next(self.counter)
        ct = self.tup(st.carry)
        lines = [f"{pad}def _{prim}_{k}(_i, _carry):{self.tag(pi.proc.name, path, s, 'body')}",
                 f"{pad}    {ct} = _carry"]
        if index:
            lines.append(f"{pad}    {self.rename(index)} = _i")
        lines += self.block(pi, s.body, path + ("body",), ind + 1, "traced")
        lines.append(f"{pad}    return {ct}")
        lines.append(f"{pad}{ct} = B.{prim}({nn}, _{prim}_{k}, {ct}){tag}")
        return lines

    def s_Repeat(self, pi, s, path, ind, mode):
        return self._loop(pi, s, path, ind, mode, s.n, s.index, "repeat")

    def s_ForEachFace(self, pi, s, path, ind, mode):
        return self._loop(pi, s, path, ind, mode, s.count, s.index, "foreach")

    def s_Select(self, pi, s, path, ind, mode):
        pad = "    " * ind
        st = pi.info(path)
        tag = self.tag(pi.proc.name, path, s, str(s.pred))
        if mode == "host":
            lines = [f"{pad}if {self.ex(s.pred)}:{tag}"]
            lines += (self.block(pi, s.then, path + ("then",), ind + 1, "host")
                      or [pad + "    pass"])
            if s.otherwise:
                lines.append(f"{pad}else:")
                lines += self.block(pi, s.otherwise, path + ("otherwise",), ind + 1,
                                    "host")
            return lines
        k = next(self.counter)
        bound = set(st.env)
        written = writes_block(s.then) | writes_block(s.otherwise)
        operands = [n for n in st.outputs if n in bound]
        operands += sorted((written & bound) - set(operands))
        ops = ", ".join(self.rename(n) for n in operands)
        outs = st.outputs
        lines = []
        for label, blk in (("then", s.then), ("otherwise", s.otherwise)):
            lines.append(f"{pad}def _{label}_{k}({ops}):"
                         f"{self.tag(pi.proc.name, path, s, label)}")
            lines += self.block(pi, blk, path + (label,), ind + 1, "traced")
            lines.append(f"{pad}    return {self.tup(outs) if outs else '()'}")
        call = f"B.cond({self.ex(s.pred)}, _then_{k}, _otherwise_{k}, {self.tup(operands) if operands else '()'})"
        if outs:
            lines.append(f"{pad}{self.tup(outs)} = {call}{tag}")
        else:
            lines.append(pad + call + tag)
        return lines

    def _solver_opts(self, o, k, kind_label, pad, pi, path, s, args):
        """Python source of the linsolve option dict (+ preconditioner def)."""
        lines = []
        pc = "None"
        if o.precond is not None:
            pc = f"_pc_{k}"
            a = "".join(", " + self.rename(x) for x in args)
            lines += [f"{pad}def _pc_{k}(_r):{self.tag(pi.proc.name, path, s, 'precond ' + o.precond.name)}",
                      f"{pad}    return {self.proc_names[o.precond.name]}(_r{a})"]
        tol = self.opt(o.tol, "real", f"c_{kind_label}{k}_tol")
        rst = self.opt(o.restart, "int", f"c_{kind_label}{k}_restart")
        mxi = self.opt(o.maxiter, "int", f"c_{kind_label}{k}_maxiter")
        bw = "None" if o.bandwidth is None else str(int(o.bandwidth))
        d = (f"dict(kind={o.kind!r}, tol={tol}, restart={rst}, maxiter={mxi}, "
             f"precond={pc}, nullspace={o.nullspace!r}, bandwidth={bw}, "
             f"symmetric={o.symmetric!r})")
        return lines, d

    def _on_fail(self, pad, on_fail, ok, stat_names, stat_vals, anchor, msg, tag):
        if on_fail == "ignore":
            return []
        names = "(" + "".join(repr(n) + ", " for n in stat_names) + ")"
        vals = "(" + "".join(v + ", " for v in stat_vals) + ")"
        if on_fail == "warn":
            return [f"{pad}B.warn({ok}, {msg!r}, {names}, {vals}){tag}"]
        return [f"{pad}({anchor},) = B.guard({ok}, {msg!r}, {names}, {vals}, "
                f"({anchor},)){tag}"]

    def s_Root(self, pi, s, path, ind, mode):
        pad = "    " * ind
        k = next(self.counter)
        o = s.opts
        a = "".join(", " + self.rename(x) for x in s.args)
        tag = self.tag(pi.proc.name, path, s, s.residual.name)
        lines = [f"{pad}def _res_{k}(_x):{self.tag(pi.proc.name, path, s, 'residual ' + s.residual.name)}",
                 f"{pad}    return {self.proc_names[s.residual.name]}(_x{a})"]
        jv = "None"
        if o.jac_proc is not None:
            jv = f"_jv_{k}"
            lines += [f"{pad}def _jv_{k}(_x, _v):{self.tag(pi.proc.name, path, s, 'jacobian ' + o.jac_proc.name)}",
                      f"{pad}    return {self.proc_names[o.jac_proc.name]}(_x, _v{a})"]
        pcl, lin = self._solver_opts(o.linsolve, k, "root_lin", pad, pi, path, s,
                                     s.args)
        lines += pcl
        tol = self.opt(o.tol, "real", f"c_root{k}_tol")
        mxi = self.opt(o.maxiter, "int", f"c_root{k}_maxiter")
        stats = s.stats or (f"_ok_{k}", f"_rnorm_{k}", f"_iters_{k}")
        outs = (s.out,) + tuple(stats)
        lines.append(f"{pad}{self.unpack(outs)} = B.root(_res_{k}, "
                     f"{self.rename(s.x0)}, tol={tol}, maxiter={mxi}, "
                     f"jac={o.jac!r}, jvp={jv}, linsolve={lin}){tag}")
        rn = [self.rename(x) for x in stats]
        lines += self._on_fail(
            pad, o.on_fail, rn[0], ("rnorm", "iters"), (rn[1], rn[2]),
            self.rename(s.out),
            f"Root {s.residual.name}: Newton not converged (||R|| > tol)",
            self.tag(pi.proc.name, path, s, "on_fail=" + o.on_fail))
        return lines

    def s_LinSolve(self, pi, s, path, ind, mode):
        pad = "    " * ind
        k = next(self.counter)
        a = "".join(", " + self.rename(x) for x in s.args)
        tag = self.tag(pi.proc.name, path, s, s.matvec.name)
        lines = [f"{pad}def _mv_{k}(_x):{self.tag(pi.proc.name, path, s, 'matvec ' + s.matvec.name)}",
                 f"{pad}    return {self.proc_names[s.matvec.name]}(_x{a})"]
        pcl, lin = self._solver_opts(s.opts, k, "lin", pad, pi, path, s, s.args)
        lines += pcl
        stats = s.stats or (f"_ok_{k}", f"_rres_{k}", f"_iters_{k}")
        outs = (s.out,) + tuple(stats)
        kw = lin[len("dict("):-1]
        lines.append(f"{pad}{self.unpack(outs)} = B.linsolve(_mv_{k}, "
                     f"{self.rename(s.rhs)}, {kw}){tag}")
        rn = [self.rename(x) for x in stats]
        lines += self._on_fail(
            pad, s.opts.on_fail, rn[0], ("rel_resid", "iters"), (rn[1], rn[2]),
            self.rename(s.out),
            f"LinSolve {s.matvec.name}: not converged (rel. residual > tol)",
            self.tag(pi.proc.name, path, s, "on_fail=" + s.opts.on_fail))
        return lines


# ── loading ─────────────────────────────────────────────────────────────────

_LOADED: dict = {}


def load_source(source: str, path: str, backend, bodies) -> types.ModuleType:
    modname = "zoomy_generated_" + hashlib.sha1(
        (path + str(id(backend)) + str(id(bodies))).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    mod.B = backend
    mod.blocks = bodies if isinstance(bodies, types.SimpleNamespace) else \
        types.SimpleNamespace(**dict(bodies or {}))
    spec.loader.exec_module(mod)
    sys.modules[modname] = mod
    return mod


def _write_atomic(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path) as f:
            if f.read() == text:
                return
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


class PythonWalker:
    """Emit, write and load the generated Python for one backend.

    ``lowering``: ``"auto"`` (traced on compiling backends, host otherwise,
    with the host/compiled boundary rule), ``"host"`` or ``"traced"``
    (force one mode everywhere -- used to test both lowerings on numpy)."""

    def __init__(self, backend, *, lowering: str = "auto",
                 cache_dir: Optional[str] = None):
        if lowering not in ("auto", "host", "traced"):
            raise ValueError("lowering must be auto | host | traced")
        self.backend = backend
        self.lowering = lowering
        self.cache_dir = cache_dir or default_cache_dir()

    def emit(self, root: Procedure, *, registry=None, constants=None,
             flags=None, solver_name: str = "template"):
        """Resolve, check and emit; returns ``(source, info, modes)``."""
        if flags is not None:
            root = resolve_static(root, flags)
        info = check_program(root, registry, constants)
        em = _Emitter(info, self.backend, self.lowering, solver_name)
        src = em.module()
        return src, info, dict(em.modes)

    def path_for(self, source, solver_name, proc_name) -> str:
        bname = getattr(self.backend, "name", type(self.backend).__name__)
        h = hashlib.sha1(source.encode()).hexdigest()[:12]
        safe = lambda s: "".join(c if c.isalnum() or c == "_" else "_" for c in s)  # noqa: E731
        return os.path.join(self.cache_dir, h,
                            f"{safe(solver_name)}_{safe(proc_name)}_{safe(bname)}.py")

    def build(self, root: Procedure, bodies, *, registry=None, constants=None,
              flags=None, solver_name: str = "template") -> Generated:
        src, info, modes = self.emit(root, registry=registry,
                                     constants=constants, flags=flags,
                                     solver_name=solver_name)
        table = dict(bodies or {})
        missing = [n for n in info.blocks_used if n not in table]
        if missing:
            raise MissingBodyError(
                f"no body for block(s) {missing} on backend "
                f"{getattr(self.backend, 'name', self.backend)!r}")
        path = self.path_for(src, solver_name, info.root.name)
        _write_atomic(path, src)
        mod = load_source(src, path, self.backend, table)
        return Generated(entry=mod.ENTRY, module=mod, path=path, source=src,
                         info=info, lowering=modes)


__all__ = ["PythonWalker", "Generated", "PyExprPrinter", "LoweringError",
           "MissingBodyError", "load_source", "default_cache_dir"]
