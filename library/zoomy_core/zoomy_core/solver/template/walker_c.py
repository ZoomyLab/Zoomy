"""C walker: template IR -> C++ (host orchestration over backend handles).

Derived from `zoomy_core.transformation.procedure_c.CProcedurePrinter`
(GenericCppBase expression printing: typed Min/Max literals, the math-function
caster, CSE-free scalar printing) with the eight fixes the feasibility
prototype needed to make the real march compile and run
(`docs/dev/solver_template/prototypes/feas_c/feas_printer.py`):

F1  declare-once / typed local table: every local is declared ONCE at the
    head of its procedure with the kind the type checker inferred;
F2  whole-array `Assign` -> `zoomy_elementwise(lambda, target, operands...)`
    (operands must share the target kind; reductions inside are lifted);
F3  written scalars by mutable reference (`double& time`), read scalars by
    value, arrays as typed handles (`const CellState&`);
F4  locals bound to the context workspace (``CellState& Q0 =
    ctx.ws.step__Q0;``), the workspace struct and its allocator are emitted;
F5  `Procedure.results` pass-once: an in/out port is ONE mutable parameter;
    outputs not among the inputs are trailing references; aliasing between a
    read-only and a written actual goes through a temporary;
F6  arity / kind check of every call at print time against the block registry
    / procedure ports;
F7  host-only orchestration (no `PORTABLE_FN`; `orchestration_qualifier`);
F8  an explicit context parameter `Ctx& ctx` on every procedure and block,
    plus emitted `constexpr` constants.

The backend header supplies the handle types, `Ctx` (with `ws` of the
emitted workspace type), `zoomy_elementwise`, `zoomy_copy`,
`zoomy_reduce_{any,all,min,max,sum}`, `zoomy_guard_value` /
`zoomy_guard_fail`, `zoomy_warn`, `zoomy_safe_div`, `zoomy_at`,
`zoomy_root`, `zoomy_linsolve`, `zoomy_alloc_<kind>` and the block
definitions.
"""
from __future__ import annotations

import dataclasses
import itertools
import math

import sympy as sp

from zoomy_core.solver.template import expr as E
from zoomy_core.solver.template.check import (
    ProgramInfo, TemplateCheckError, check_program, fmt_path)
from zoomy_core.solver.template.consts import Const, ConstTable
from zoomy_core.solver.template.ir import (
    Assign, Call, Callback, ForEachFace, Guard, Hook, LinSolve, Procedure,
    Repeat, Root, Select, Seq, While, resolve_static)
from zoomy_core.solver.template.kinds import (
    ARRAY_KINDS, compatible, is_array, is_opaque, is_scalar)


def _base_printer():
    # Lazy: procedure_c pulls generic_c (and the NSM stack) in.
    from zoomy_core.transformation.procedure_c import CProcedurePrinter
    return CProcedurePrinter


HANDLE_TYPES = {
    "real": "double", "int": "int", "bool": "bool",
    "cell_state": "CellState", "cell_aux": "CellAux", "params": "Params",
    "face_state": "FaceState", "face_aux": "FaceAux",
    "face_scalar": "FaceScalar", "face_mask": "FaceMask",
    "cell_scalar": "CellScalar", "cell_mask": "CellMask",
    "cell_grad": "CellGrad", "bc_state": "BCState", "bc_aux": "BCAux",
    "array": "Array", "mask": "Mask",
    "geom": "Geom", "obs": "ObsAcc", "plugin": "PluginState", "any": "AnyHandle",
}

CPP_KEYWORDS = frozenset("""alignas alignof and and_eq asm auto bitand bitor bool
break case catch char char8_t char16_t char32_t class compl concept const
consteval constexpr constinit const_cast continue co_await co_return co_yield
decltype default delete do double dynamic_cast else enum explicit export extern
false float for friend goto if inline int long mutable namespace new noexcept
not not_eq nullptr operator or or_eq private protected public register
reinterpret_cast requires return short signed sizeof static static_assert
static_cast struct switch template this thread_local throw true try typedef
typeid typename union unsigned using virtual void volatile wchar_t while xor
xor_eq ctx std main""".split())


class CWalkerError(TemplateCheckError):
    pass


@dataclasses.dataclass
class CProgram:
    """Emitted C++: `decls` (constants, workspace struct + allocator, block
    and procedure declarations) and `procs` (procedure definitions)."""

    decls: str
    procs: str
    info: ProgramInfo
    workspace: list            # (member, kind)
    entry: str


def _cpp_float(v: float) -> str:
    if math.isinf(v):
        return ("std::numeric_limits<double>::infinity()" if v > 0
                else "(-std::numeric_limits<double>::infinity())")
    if math.isnan(v):
        return "std::numeric_limits<double>::quiet_NaN()"
    r = repr(float(v))
    if "e" not in r and "." not in r and "inf" not in r:
        r += ".0"
    return r


def make_printer_class():
    Base = _base_printer()

    class CTemplatePrinter(Base):
        gpu_enabled = False                  # F7: orchestration is host code
        orchestration_qualifier = ""
        context_type = "Ctx"
        context_name = "ctx"
        handle_types = HANDLE_TYPES

        def __init__(self, info: ProgramInfo, *, workspace_type="TemplateWorkspace"):
            super().__init__()
            self.info = info
            self.workspace_type = workspace_type
            self.opt_consts = ConstTable()
            self.workspace: list = []
            self._ws_seen: set = set()
            self.counter = itertools.count()
            self._elem: dict = {}           # name -> element name (in lambdas)
            self.pinfo = None
            self._rename = {}
            reserved = set(CPP_KEYWORDS) | set(info.procs) | set(
                info.registry.names())
            self._reserved = reserved
            self._taken = set(info.constants.names())
            for c in info.constants.names():
                self._rename[c] = c
            for n in info.procs:
                if n in CPP_KEYWORDS:
                    raise CWalkerError(f"procedure name {n!r} is a C++ keyword")

        # ── names / kinds ──────────────────────────────────────────────────
        def cname(self, n):
            if n in self._rename:
                return self._rename[n]
            r = n
            while r in self._reserved or r in self._taken:
                r += "_"
            self._rename[n] = r
            self._taken.add(r)
            return r

        def handle(self, kind):
            return self.handle_types[kind]

        def kind(self, name):
            k = self.pinfo.kinds.get(name)
            if k is None:
                c = self.info.constants.get(name) or self.opt_consts.get(name)
                if c is None:
                    raise CWalkerError(f"{self.pinfo.proc.name}: no kind for {name!r}")
                return c.kind
            return k

        def is_arr(self, name):
            try:
                return not is_scalar(self.kind(name))
            except CWalkerError:
                return False

        # ── expression printing ───────────────────────────────────────────
        def _print_Symbol(self, s):
            n = str(s)
            if n in self._elem:
                return self._elem[n]
            return self.cname(n)

        def _print_Float(self, f):
            return _cpp_float(float(f))

        def _print_Rational(self, r):
            return _cpp_float(float(r))

        _print_Half = _print_Rational

        def _print_Integer(self, i):
            return str(int(i))

        _print_Zero = _print_One = _print_NegativeOne = _print_Integer

        def _print_Infinity(self, e):
            return _cpp_float(math.inf)

        def _print_NegativeInfinity(self, e):
            return _cpp_float(-math.inf)

        def _print_BooleanTrue(self, e):
            return "true"

        def _print_BooleanFalse(self, e):
            return "false"

        def _print_Pow(self, e):
            b, x = e.args
            pb = self.parenthesize(b, sp.printing.precedence.PRECEDENCE["Mul"])
            if x == sp.Rational(1, 2):
                return f"std::sqrt({self._print(b)})"
            if x == -sp.Rational(1, 2):
                return f"(1.0/std::sqrt({self._print(b)}))"
            if x.is_Integer and 2 <= int(x) <= 4:
                return "(" + "*".join([pb] * int(x)) + ")"
            if x == -1:
                return f"(1.0/{pb})"
            return super()._print_Pow(e)

        def _print_Indexed(self, e):
            base = self.cname(str(e.base.label))
            idx = [self._print(i) for i in e.indices]
            if len(idx) == 1:
                return f"{base}[{idx[0]}]"
            return f"zoomy_at({base}, {', '.join(idx)})"

        def _print_Piecewise(self, e):
            args = list(e.args)
            if args[-1].cond in (True, sp.true):
                out = f"({self._print(args[-1].expr)})"
                args = args[:-1]
            else:
                out = "std::numeric_limits<double>::quiet_NaN()"
            for a in reversed(args):
                out = f"(({self._print(a.cond)}) ? ({self._print(a.expr)}) : {out})"
            return out

        def _print_Where(self, e):
            c, a, b = (self._print(x) for x in e.args)
            return f"(({c}) ? ({a}) : ({b}))"

        def _print_SafeDiv(self, e):
            a, b, f = (self._print(x) for x in e.args)
            return f"zoomy_safe_div({a}, {b}, {f})"

        def _print_IsFinite(self, e):
            return f"std::isfinite({self._print(e.args[0])})"

        def _fill(self, e):
            raise CWalkerError(
                f"{type(e).__name__} is only valid as the whole right-hand side "
                "of an elementwise assignment")

        def _print_FullLike(self, e):
            if self._in_fill:
                return self._print(e.args[1])
            self._fill(e)

        def _print_ZerosLike(self, e):
            if self._in_fill:
                return "0.0"
            self._fill(e)

        def _print_MaskLike(self, e):
            if self._in_fill:
                return self._print(e.args[1])
            self._fill(e)

        _in_fill = False

        def array_operands(self, e) -> list:
            """Array names an elementwise expression reads (shape providers of
            FullLike / ZerosLike / MaskLike and reductions excluded)."""
            found = []

            def walk(x):
                if isinstance(x, (E.REDUCTIONS, sp.Indexed)):
                    return
                if isinstance(x, (E.FullLike, E.MaskLike)):
                    walk(x.args[1])
                    return
                if isinstance(x, E.ZerosLike):
                    return
                if isinstance(x, sp.Symbol):
                    n = str(x)
                    if self.is_arr(n) and n not in found:
                        found.append(n)
                    return
                for a in x.args:
                    walk(a)
            walk(e)
            return found

        def _reduce(self, fn, e):
            arg = e.args[0]
            ops = self.array_operands(arg)
            if not ops:
                return f"({self._print(arg)})"
            saved, saved_fill = dict(self._elem), self._in_fill
            for o in ops:
                self._elem[o] = self.cname(o) + "_e"
            self._in_fill = True
            body = self._print(arg)
            self._elem, self._in_fill = saved, saved_fill
            params = ", ".join(f"const auto& {self.cname(o)}_e" for o in ops)
            operands = ", ".join(self.cname(o) for o in ops)
            return (f"zoomy_reduce_{fn}({self.context_name}, [=]({params}) "
                    f"{{ return {body}; }}, {operands})")

        def _print_AnyTrue(self, e):
            return self._reduce("any", e)

        def _print_AllTrue(self, e):
            return self._reduce("all", e)

        def _print_ReduceMin(self, e):
            return self._reduce("min", e)

        def _print_ReduceMax(self, e):
            return self._reduce("max", e)

        def _print_ReduceSum(self, e):
            return self._reduce("sum", e)

        def ex(self, e) -> str:
            return self.doprint(e)

        # ── program ─────────────────────────────────────────────────────────
        def emit(self) -> CProgram:
            plans = {}
            for n, pi in self.info.procs.items():
                plans[n] = self.plan(pi)
            proc_defs, proc_decls = [], []
            for n, pi in self.info.procs.items():
                d, body = self.procedure(pi, plans[n])
                proc_decls.append(d + ";")
                proc_defs.append(body)
            decls = []
            decls.append("// GENERATED by zoomy_core.solver.template.walker_c -- "
                         "do not edit.")
            decls.append(f"// entry: {self.info.root.name}")
            decls.append("#pragma once")
            decls.append("#include <cmath>")
            decls.append("#include <limits>")
            decls.append("")
            decls.append("// -- emitted constants " + "-" * 50)
            for c in list(self.info.constants) + list(self.opt_consts):
                if c.value is None:
                    raise CWalkerError(f"constant {c.name} has no value")
                v = (_cpp_float(c.value) if c.kind == "real"
                     else ("true" if c.value else "false") if c.kind == "bool"
                     else str(int(c.value)))
                prov = f"  // {c.provenance}" if c.provenance else ""
                decls.append(f"static constexpr {self.handle(c.kind)} {c.name} = "
                             f"{v};{prov}")
            decls.append("")
            decls.append("// -- workspace (F4) " + "-" * 52)
            decls.append(f"struct {self.workspace_type} {{")
            for m, k in self.workspace:
                decls.append(f"    {self.handle(k)} {m};  // {k}")
            decls.append("};")
            decls.append(f"template <class C> inline void allocate_workspace(C& "
                         f"{self.context_name}) {{")
            for m, k in self.workspace:
                decls.append(f"    zoomy_alloc_{k}({self.context_name}, "
                             f"{self.context_name}.ws.{m});")
            if not self.workspace:
                decls.append(f"    (void){self.context_name};")
            decls.append("}")
            decls.append("")
            decls.append(f"struct {self.context_type};")
            decls.append("// -- blocks (bodies supplied by the backend) " + "-" * 27)
            for name in self.info.blocks_used:
                spec = self.info.registry.get(name)
                params = self.sig_params(spec.inputs, spec.outputs)
                decls.append(f"void {name}({self.sig_text(params)});"
                             f"  // {spec.role}, comm={spec.comm}")
            decls.append("")
            decls.append("// -- procedures " + "-" * 56)
            decls.extend(proc_decls)
            procs = ["// GENERATED by zoomy_core.solver.template.walker_c -- "
                     "do not edit.", ""] + proc_defs
            return CProgram(decls="\n".join(decls) + "\n",
                            procs="\n".join(procs) + "\n", info=self.info,
                            workspace=list(self.workspace),
                            entry=self.info.root.name)

        # ── signatures (F3, F5, F8) ────────────────────────────────────────
        @staticmethod
        def sig_params(inputs, outputs, written_inputs=frozenset()):
            outs = [o.name for o in outputs]
            ins = [p.name for p in inputs]
            params = []
            for p in inputs:
                if p.name in outs:
                    params.append((p.name, p.kind, "inout"))
                elif p.name in written_inputs:
                    params.append((p.name, p.kind, "copyin"))
                else:
                    params.append((p.name, p.kind, "in"))
            for o in outputs:
                if o.name not in ins:
                    params.append((o.name, o.kind, "out"))
            return params

        def decl(self, name, kind, mode) -> str:
            t = self.handle(kind)
            n = self.cname(name)
            if mode == "in":
                return f"const {t} {n}" if is_scalar(kind) else f"const {t}& {n}"
            if mode == "copyin":
                return f"{t} {n}" if is_scalar(kind) else f"const {t}& {n}__in"
            return f"{t}& {n}"

        def sig_text(self, params) -> str:
            return ", ".join([f"{self.context_type}& {self.context_name}"]
                             + [self.decl(n, k, m) for n, k, m in params])

        def ws_member(self, proc, name, kind) -> str:
            m = f"{proc}__{name}"
            if m not in self._ws_seen:
                self._ws_seen.add(m)
                self.workspace.append((m, kind))
            return m

        # ── planning (temps, option constants) ───────────────────────────────
        def plan(self, pi):
            """Pre-pass: aliasing temporaries of calls and stat temporaries of
            Root / LinSolve, keyed by statement path."""
            self.pinfo = pi
            plan = {"calls": {}, "solves": {}, "extra": {}}

            def visit(block, prefix):
                for i, s in enumerate(block):
                    path = prefix + (i,)
                    if isinstance(s, (Call, Callback)):
                        plan["calls"][path] = self.call_plan(s, path, plan)
                    if isinstance(s, (Root, LinSolve)):
                        k = next(self.counter)
                        stats = s.stats or tuple(
                            f"_{'root' if isinstance(s, Root) else 'lin'}{k}_{x}"
                            for x in ("ok", "res", "it"))
                        for n, kd in zip(stats, ("bool", "real", "int")):
                            if n not in pi.kinds:
                                plan["extra"][n] = kd
                        plan["solves"][path] = (k, stats)
                    for label, b in s.blocks():
                        visit(b, path + (label,))
            visit(pi.proc.body, ("body",))
            return plan

        def callee_ports(self, s):
            if isinstance(s, Callback):
                spec = self.info.registry.get(s.fn)
                return s.fn, spec.inputs, spec.outputs, s.inputs, s.outputs
            if isinstance(s.callee, Procedure):
                p = s.callee
                return p.name, p.inputs, p.outputs, s.args, s.out
            spec = self.info.registry.get(s.callee)
            return spec.name, spec.inputs, spec.outputs, s.args, s.out

        def call_plan(self, s, path, plan):
            name, ins, outs, args, res = self.callee_ports(s)
            # F6: arity + kind check at print time
            if len(args) != len(ins) or len(res) != len(outs):
                raise CWalkerError(
                    f"{self.pinfo.proc.name}:{fmt_path(path)}: call {name} arity "
                    f"{len(args)}->{len(res)} vs signature {len(ins)}->{len(outs)}")
            for a, p in zip(args, ins):
                if not compatible(self.kind(a), p.kind):
                    raise CWalkerError(
                        f"{self.pinfo.proc.name}:{fmt_path(path)}: call {name}: "
                        f"{a!r} is {self.kind(a)}, parameter {p.name} is {p.kind}")
            params = self.sig_params(ins, outs)
            in_act = dict(zip([p.name for p in ins], args))
            out_act = dict(zip([p.name for p in outs], res))
            const_actuals = {in_act[n] for n, _, m in params if m == "in"}
            pre, post, passed = [], [], []
            for n, k, m in params:
                if m == "in":
                    passed.append(in_act[n])
                    continue
                o = out_act[n]
                target = o
                if o in const_actuals:      # F5 aliasing -> temporary
                    target = f"_alias_{name}_{n}"
                    plan["extra"][target] = k
                    post.append((o, target, k))
                if m == "inout" and in_act[n] != target:
                    pre.append((target, in_act[n], k))
                passed.append(target)
            return pre, passed, post

        # ── procedures ──────────────────────────────────────────────────────
        def procedure(self, pi, plan):
            self.pinfo = pi
            p = pi.proc
            params = self.sig_params(p.inputs, p.outputs, pi.written_inputs)
            sig = (f"{self.orchestration_qualifier}void {p.name}("
                   f"{self.sig_text(params)})")
            pnames = {n for n, _, _ in params}
            lines = []
            if p.doc:
                lines.extend("// " + ln for ln in p.doc.strip().splitlines())
            lines.append(f"inline {sig} {{  // procedure {p.name}")
            # F1/F4: one declaration per local, typed by the checker
            for n, _, m in params:
                if m == "copyin" and not is_scalar(self.kind(n)):
                    k = self.kind(n)
                    mem = self.ws_member(p.name, n, k)
                    lines.append(f"    {self.handle(k)}& {self.cname(n)} = "
                                 f"{self.context_name}.ws.{mem};  // written input")
                    lines.append(f"    zoomy_copy({self.cname(n)}, {self.cname(n)}__in);")
            locals_ = [n for n in pi.locals if n not in pnames]
            locals_ += [n for n in plan["extra"] if n not in pnames
                        and n not in locals_]
            for n in locals_:
                k = pi.kinds.get(n) or plan["extra"][n]
                if is_scalar(k):
                    init = {"real": "0.0", "int": "0", "bool": "false"}[k]
                    lines.append(f"    {self.handle(k)} {self.cname(n)} = {init};")
                else:
                    mem = self.ws_member(p.name, n, k)
                    lines.append(f"    {self.handle(k)}& {self.cname(n)} = "
                                 f"{self.context_name}.ws.{mem};")
            self._extra_kinds = dict(plan["extra"])
            body = self.block(p.body, ("body",), 1, plan)
            lines.extend(body)
            if p.doc is None:
                pass
            lines.append("}")
            lines.append("")
            return sig, "\n".join(lines)

        def kind_any(self, n):
            if n in self._extra_kinds:
                return self._extra_kinds[n]
            return self.kind(n)

        def tag(self, path, s, extra="") -> str:
            t = f"  // [{self.pinfo.proc.name}:{fmt_path(path)}] {s.kind_name}"
            if extra:
                t += f" {extra}"
            if s.note:
                t += f" -- {s.note}"
            return t

        def block(self, block, prefix, ind, plan):
            out = []
            for i, s in enumerate(block):
                out.extend(self.stmt(s, prefix + (i,), ind, plan))
            return out

        def stmt(self, s, path, ind, plan):
            return getattr(self, "s_" + type(s).__name__)(s, path, ind, plan)

        def copy_line(self, dst, src, kind) -> str:
            if is_scalar(kind):
                return f"{self.cname(dst)} = {self.cname(src)};"
            return f"zoomy_copy({self.cname(dst)}, {self.cname(src)});"

        def s_Seq(self, s, path, ind, plan):
            return self.block(s.body, path + ("body",), ind, plan)

        def s_Hook(self, s, path, ind, plan):
            pad = "    " * ind
            label, body = s.blocks()[0]
            head = [f"{pad}// -- hook {s.name} ({s.filled_by or 'default'})"
                    + self.tag(path, s)]
            return head + self.block(body, path + (label,), ind, plan)

        def s_Assign(self, s, path, ind, plan):
            pad = "    " * ind
            tag = self.tag(path, s, s.target)
            if s.index is not None:
                idx = s.index
                tgt = (f"{self.cname(s.base)}[{self.ex(idx)}]"
                       if not isinstance(idx, sp.Tuple) else
                       f"zoomy_at({self.cname(s.base)}, "
                       f"{', '.join(self.ex(i) for i in idx)})")
                return [f"{pad}{tgt} = {self.ex(s.expr)};{tag}"]
            k = self.kind(s.base)
            if is_scalar(k):
                v = self.ex(s.expr)
                if k == "int" and not s.expr.is_Integer and \
                        self._expr_kind(s.expr) == "real":
                    v = f"static_cast<int>({v})"
                return [f"{pad}{self.cname(s.base)} = {v};{tag}"]
            if is_opaque(k):
                if not isinstance(s.expr, sp.Symbol):
                    raise CWalkerError(f"opaque {s.base} assigned an expression")
                return [f"{pad}{self.copy_line(s.base, str(s.expr), k)}{tag}"]
            return self.elementwise(s, k, pad, tag)

        def _expr_kind(self, e):
            from zoomy_core.solver.template.check import _Kinds
            return _Kinds(self.kind_any, lambda m: None)(e)

        def elementwise(self, s, k, pad, tag):
            """F2: whole-array assignment through the backend hook."""
            target = s.base
            e = s.expr
            lifted = []
            reds = [x for x in sp.preorder_traversal(e) if isinstance(x, E.REDUCTIONS)]
            rep = {}
            for j, r in enumerate(dict.fromkeys(reds)):
                nm = f"_red{j}"
                lifted.append(f"const auto {nm} = {self.ex(r)};")
                rep[r] = sp.Symbol(nm)
            if rep:
                e = e.xreplace(rep)
                self._extra_kinds.update({str(v): "real" for v in rep.values()})
            ops = [o for o in self.array_operands(e) if o != target]
            for o in ops:
                if self.kind(o) != k and not compatible(self.kind(o), k):
                    raise CWalkerError(
                        f"elementwise {target} ({k}) = ...: operand {o!r} is "
                        f"{self.kind(o)} (operands must share the target kind)")
            saved, saved_fill = dict(self._elem), self._in_fill
            self._elem[target] = self.cname(target) + "_e"
            for o in ops:
                self._elem[o] = self.cname(o) + "_e"
            self._in_fill = True
            rhs = self.ex(e)
            self._elem, self._in_fill = saved, saved_fill
            params = [f"auto& {self.cname(target)}_e"] + [
                f"const auto& {self.cname(o)}_e" for o in ops]
            operands = ", ".join([self.cname(target)] + [self.cname(o) for o in ops])
            call = (f"zoomy_elementwise([=]({', '.join(params)}) {{ "
                    f"{self.cname(target)}_e = {rhs}; }}, {operands});")
            if lifted:
                return ([f"{pad}{{{tag}"] + [f"{pad}    {ln}" for ln in lifted]
                        + [f"{pad}    {call}", f"{pad}}}"])
            return [f"{pad}{call}{tag}"]

        def s_Call(self, s, path, ind, plan):
            pad = "    " * ind
            pre, passed, post = plan["calls"][path]
            name = s.callee_name
            lines = [pad + self.copy_line(d, a, k) for d, a, k in pre]
            args = ", ".join([self.context_name] + [self.cname(a) for a in passed])
            lines.append(f"{pad}{name}({args});"
                         + self.tag(path, s, s.callee_name))
            lines += [pad + self.copy_line(o, t, k) for o, t, k in post]
            return lines

        def s_Callback(self, s, path, ind, plan):
            pad = "    " * ind
            pre, passed, post = plan["calls"][path]
            lines = [pad + self.copy_line(d, a, k) for d, a, k in pre]
            args = ", ".join([self.context_name] + [self.cname(a) for a in passed])
            lines.append(f"{pad}{s.fn}({args});"
                         + self.tag(path, s, f"{s.kind} (direct call)"))
            lines += [pad + self.copy_line(o, t, k) for o, t, k in post]
            return lines

        def s_Guard(self, s, path, ind, plan):
            pad = "    " * ind
            lines = [f"{pad}if (!({self.ex(s.ok)})) {{" + self.tag(path, s)]
            for v in s.values:
                lines.append(f'{pad}    zoomy_guard_value({self.context_name}, '
                             f'"{v}", {self.cname(v)});')
            lines.append(f'{pad}    zoomy_guard_fail({self.context_name}, '
                         f'{self._cstr(s.msg)});')
            lines.append(f"{pad}}}")
            return lines

        @staticmethod
        def _cstr(msg):
            return '"' + msg.replace("\\", "\\\\").replace('"', '\\"') + '"'

        def s_While(self, s, path, ind, plan):
            pad = "    " * ind
            lines = [f"{pad}while ({self.ex(s.cond)}) {{" + self.tag(path, s)]
            lines += self.block(s.body, path + ("body",), ind + 1, plan)
            return lines + [f"{pad}}}"]

        def _for(self, s, path, ind, plan, n, index):
            pad = "    " * ind
            nn = str(n) if isinstance(n, int) else self.cname(n)
            if index:
                iv = self.cname(index)
                head = f"for ({iv} = 0; {iv} < {nn}; ++{iv}) {{"
            else:
                iv = f"_i{next(self.counter)}"
                head = f"for (int {iv} = 0; {iv} < {nn}; ++{iv}) {{"
            lines = [pad + head + self.tag(path, s)]
            lines += self.block(s.body, path + ("body",), ind + 1, plan)
            return lines + [f"{pad}}}"]

        def s_Repeat(self, s, path, ind, plan):
            return self._for(s, path, ind, plan, s.n, s.index)

        def s_ForEachFace(self, s, path, ind, plan):
            return self._for(s, path, ind, plan, s.count, s.index)

        def s_Select(self, s, path, ind, plan):
            pad = "    " * ind
            lines = [f"{pad}if ({self.ex(s.pred)}) {{" + self.tag(path, s)]
            lines += self.block(s.then, path + ("then",), ind + 1, plan)
            if s.otherwise:
                lines.append(f"{pad}}} else {{")
                lines += self.block(s.otherwise, path + ("otherwise",), ind + 1,
                                    plan)
            return lines + [f"{pad}}}"]

        def _opt(self, v, kind, label):
            if isinstance(v, str):
                return v
            self.opt_consts.add(Const(label, v, kind, "statement option"))
            return label

        def _lambda(self, proc, lead, args, outname="_r"):
            """`[&](const T& _x, T& _r) { proc(ctx, _x, args..., _r); }`"""
            if proc.outputs[0].name in proc.input_names:
                raise CWalkerError(f"{proc.name}: a residual / matvec / "
                                   "preconditioner output must not reuse an "
                                   "input name")
            k_in = proc.inputs[0].kind
            k_out = proc.outputs[0].kind
            lp = [f"const {self.handle(proc.inputs[i].kind)}& {n}"
                  for i, n in enumerate(lead)]
            call_args = [self.context_name] + list(lead) + [
                self.cname(a) for a in args] + [outname]
            return (f"[&]({', '.join(lp)}, {self.handle(k_out)}& {outname}) {{ "
                    f"{proc.name}({', '.join(call_args)}); }}"), k_in

        def _on_fail(self, pad, on_fail, ok, values, msg, path, s):
            if on_fail == "ignore":
                return []
            if on_fail == "warn":
                return [f"{pad}if (!{self.cname(ok)}) zoomy_warn({self.context_name}, "
                        f"{self._cstr(msg)});" + self.tag(path, s, "on_fail=warn")]
            lines = [f"{pad}if (!{self.cname(ok)}) {{" + self.tag(path, s, "on_fail=guard")]
            for label, v in values:
                lines.append(f'{pad}    zoomy_guard_value({self.context_name}, '
                             f'"{label}", {self.cname(v)});')
            lines.append(f"{pad}    zoomy_guard_fail({self.context_name}, "
                         f"{self._cstr(msg)});")
            return lines + [f"{pad}}}"]

        def _lin_args(self, o, k, label, args):
            pc = "nullptr"
            pre = []
            if o.precond is not None:
                lam, _ = self._lambda(o.precond, ["_v"], args, "_z")
                pre.append(f"auto _pc{k} = {lam};")
                pc = f"_pc{k}"
            tol = self._opt(o.tol, "real", f"c_{label}{k}_tol")
            rst = self._opt(o.restart, "int", f"c_{label}{k}_restart")
            mxi = self._opt(o.maxiter, "int", f"c_{label}{k}_maxiter")
            ns = f'"{o.nullspace}"' if o.nullspace else "nullptr"
            bw = str(int(o.bandwidth)) if o.bandwidth is not None else "-1"
            return pre, (f'{pc}, "{o.kind}", {tol}, {rst}, {mxi}, {ns}, {bw}, '
                         f'{"true" if o.symmetric else "false"}')

        def s_Root(self, s, path, ind, plan):
            pad = "    " * ind
            k, stats = plan["solves"][path]
            o = s.opts
            lines = [f"{pad}{{" + self.tag(path, s, s.residual.name)]
            lam, _ = self._lambda(s.residual, ["_x"], s.args)
            lines.append(f"{pad}    auto _res{k} = {lam};")
            jv = "nullptr"
            if o.jac_proc is not None:
                jl, _ = self._lambda(o.jac_proc, ["_x", "_v"], s.args, "_jv")
                lines.append(f"{pad}    auto _jac{k} = {jl};")
                jv = f"_jac{k}"
            pre, lin = self._lin_args(o.linsolve, k, "root_lin", s.args)
            lines += [f"{pad}    {ln}" for ln in pre]
            if s.out != s.x0:
                lines.append(f"{pad}    zoomy_copy({self.cname(s.out)}, "
                             f"{self.cname(s.x0)});")
            tol = self._opt(o.tol, "real", f"c_root{k}_tol")
            mxi = self._opt(o.maxiter, "int", f"c_root{k}_maxiter")
            st = ", ".join(self.cname(x) for x in stats)
            lines.append(f"{pad}    zoomy_root({self.context_name}, _res{k}, {jv}, "
                         f"{self.cname(s.out)}, {tol}, {mxi}, \"{o.jac}\", {lin}, "
                         f"{st});")
            lines.append(f"{pad}}}")
            lines += self._on_fail(pad, o.on_fail, stats[0],
                                   (("rnorm", stats[1]), ("iters", stats[2])),
                                   f"Root {s.residual.name}: Newton not converged",
                                   path, s)
            return lines

        def s_LinSolve(self, s, path, ind, plan):
            pad = "    " * ind
            k, stats = plan["solves"][path]
            lines = [f"{pad}{{" + self.tag(path, s, s.matvec.name)]
            lam, _ = self._lambda(s.matvec, ["_x"], s.args, "_y")
            lines.append(f"{pad}    auto _mv{k} = {lam};")
            pre, lin = self._lin_args(s.opts, k, "lin", s.args)
            lines += [f"{pad}    {ln}" for ln in pre]
            st = ", ".join(self.cname(x) for x in stats)
            lines.append(f"{pad}    zoomy_linsolve({self.context_name}, _mv{k}, "
                         f"{self.cname(s.rhs)}, {self.cname(s.out)}, {lin}, {st});")
            lines.append(f"{pad}}}")
            lines += self._on_fail(pad, s.opts.on_fail, stats[0],
                                   (("rel_resid", stats[1]), ("iters", stats[2])),
                                   f"LinSolve {s.matvec.name}: not converged",
                                   path, s)
            return lines

    return CTemplatePrinter


class CWalker:
    """Emit C++ for a template program.

    `emit(root, registry=..., constants=..., flags=...)` resolves the
    build-time branches, type-checks, and returns a `CProgram`."""

    def __init__(self, *, workspace_type="TemplateWorkspace"):
        self.workspace_type = workspace_type

    def emit(self, root: Procedure, *, registry=None, constants=None,
             flags=None) -> CProgram:
        if flags is not None:
            root = resolve_static(root, flags)
        info = check_program(root, registry, constants)
        P = make_printer_class()
        return P(info, workspace_type=self.workspace_type).emit()

    def write(self, prog: CProgram, directory, stem="template"):
        import os
        os.makedirs(directory, exist_ok=True)
        d = os.path.join(directory, f"{stem}_decls.hpp")
        p = os.path.join(directory, f"{stem}_procs.inc")
        with open(d, "w") as f:
            f.write(prog.decls)
        with open(p, "w") as f:
            f.write(prog.procs)
        return d, p


__all__ = ["CWalker", "CProgram", "CWalkerError", "HANDLE_TYPES",
           "make_printer_class"]
