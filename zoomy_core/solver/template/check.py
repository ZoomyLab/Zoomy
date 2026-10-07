"""Build-time checking of a template program: scope, kinds, arity, liveness.

:func:`check_program` runs two passes per procedure:

1. a FORWARD pass (scope + kinds): every name read is bound, every ``Call`` /
   ``Callback`` matches its signature (arity and kind), loop carries and Select
   operands keep their kinds, hooks only write their declared outputs, and the
   replicated-predicate rule holds (a ``Select`` / ``While`` predicate guarding
   communicating work, and every ``Guard`` predicate, must be the same on all
   ranks: derived from scalars, constants, reductions or global blocks);
2. a BACKWARD pass (liveness): the carry of every loop and the outputs of every
   ``Select`` are inferred as the names written inside, bound before, and live
   afterwards (or read by the next iteration); ``Guard`` anchors are chosen
   among the live values.

The result (:class:`ProgramInfo`) is what both walkers lower from, so the C
and Python code agree on every carry, output and declaration.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import sympy as sp
from sympy.logic.boolalg import BooleanAtom, BooleanFunction

from zoomy_core.solver.template import expr as E
from zoomy_core.solver.template.consts import ConstTable
from zoomy_core.solver.template.ir import (
    Assign, Call, Callback, ForEachFace, Guard, Hook, IfStatic, LinSolve,
    Procedure, Repeat, Root, Select, Seq, Statement, While,
    collect_procedures)
from zoomy_core.solver.template.kinds import (
    ARRAY_KINDS, CELL_AXIS_KINDS, MASK_OF, TRANSIENT_BOOL, compatible,
    is_array, is_opaque, is_scalar)
from zoomy_core.solver.template.registry import BLOCKS, BlockRegistry


class TemplateCheckError(TypeError):
    """A template violates scope, arity, kind or replication rules."""


def where(proc_name, path, stmt=None) -> str:
    loc = f"{proc_name}:" + fmt_path(path)
    if stmt is not None:
        loc += f" ({stmt.kind_name}"
        if stmt.origin:
            loc += f", built at {stmt.origin}"
        loc += ")"
    return loc


def fmt_path(path) -> str:
    """``("body", 3, "then", 1)`` -> ``body[3].then[1]``."""
    out = []
    i = 0
    path = tuple(path)
    while i < len(path):
        label = path[i]
        if i + 1 < len(path) and isinstance(path[i + 1], int):
            out.append(f"{label}[{path[i + 1]}]")
            i += 2
        else:
            out.append(str(label))
            i += 1
    return ".".join(out)


@dataclasses.dataclass
class StmtInfo:
    path: tuple
    stmt: Statement
    env: dict                       # name -> kind bound BEFORE the statement
    carry: tuple = ()               # loops
    outputs: tuple = ()             # Select
    live_after: frozenset = frozenset()
    anchors: tuple = ()             # Guard


@dataclasses.dataclass
class ProcInfo:
    proc: Procedure
    kinds: dict                     # name -> kind (inputs + every local)
    locals: list                    # local names in first-binding order
    stmts: dict                     # path -> StmtInfo
    repl_outputs: frozenset = frozenset()
    has_comm: bool = False
    has_stateful: bool = False
    written_inputs: frozenset = frozenset()

    def info(self, path) -> StmtInfo:
        return self.stmts[tuple(path)]


@dataclasses.dataclass
class ProgramInfo:
    root: Procedure
    procs: dict                     # name -> ProcInfo (callees first)
    registry: BlockRegistry
    constants: ConstTable
    blocks_used: tuple              # registered block / callback names called

    def proc(self, name) -> ProcInfo:
        return self.procs[name]


# ── read / write sets ───────────────────────────────────────────────────────


def expr_names(e) -> set:
    return set(E.names_in(e))


def writes_stmt(s) -> set:
    if isinstance(s, Assign):
        return {s.base}
    if isinstance(s, Call):
        return set(s.out)
    if isinstance(s, Callback):
        return set(s.outputs)
    if isinstance(s, (Root, LinSolve)):
        return {s.out} | set(s.stats)
    out = set()
    for _, b in s.blocks():
        out |= writes_block(b)
    if isinstance(s, (Repeat, ForEachFace)) and s.index:
        out.discard(s.index)
    return out


def writes_block(block) -> set:
    out = set()
    for s in block:
        out |= writes_stmt(s)
    return out


def free_reads_stmt(s) -> set:
    if isinstance(s, Assign):
        r = expr_names(s.expr)
        if s.index is not None:
            r |= expr_names(s.index) | {s.base}
        return r
    if isinstance(s, Call):
        return set(s.args)
    if isinstance(s, Callback):
        return set(s.inputs)
    if isinstance(s, Root):
        return {s.x0} | set(s.args)
    if isinstance(s, LinSolve):
        return {s.rhs} | set(s.args)
    if isinstance(s, Guard):
        return expr_names(s.ok) | set(s.values) | set(s.on or ())
    if isinstance(s, (Seq, Hook)):
        return free_reads_block(s.blocks()[0][1])
    if isinstance(s, Select):
        wt, we = writes_block(s.then), writes_block(s.otherwise)
        return (expr_names(s.pred) | free_reads_block(s.then)
                | free_reads_block(s.otherwise) | (wt ^ we))
    if isinstance(s, While):
        return expr_names(s.cond) | free_reads_block(s.body) | writes_block(s.body)
    if isinstance(s, (Repeat, ForEachFace)):
        r = free_reads_block(s.body) | writes_block(s.body)
        r.discard(s.index)
        return r
    if isinstance(s, IfStatic):
        raise TemplateCheckError("IfStatic must be resolved before checking")
    raise TemplateCheckError(f"unknown statement {type(s).__name__}")


def free_reads_block(block) -> set:
    free, written = set(), set()
    for s in block:
        free |= free_reads_stmt(s) - written
        written |= writes_stmt(s)
    return free


def _const_names_in(s) -> set:
    """Names a statement references as option values (tolerances, counts)."""
    out = set()
    if isinstance(s, Root):
        for v in (s.opts.tol, s.opts.maxiter, s.opts.linsolve.tol,
                  s.opts.linsolve.restart, s.opts.linsolve.maxiter):
            if isinstance(v, str):
                out.add(v)
    if isinstance(s, LinSolve):
        for v in (s.opts.tol, s.opts.restart, s.opts.maxiter):
            if isinstance(v, str):
                out.add(v)
    if isinstance(s, Repeat) and isinstance(s.n, str):
        out.add(s.n)
    if isinstance(s, ForEachFace) and isinstance(s.count, str):
        out.add(s.count)
    return out


# ── expression kinds ────────────────────────────────────────────────────────


class _Kinds:
    """Kind inference of one expression in an environment."""

    def __init__(self, lookup, err):
        self.lookup = lookup        # name -> kind (raises when unbound)
        self.err = err

    def __call__(self, e) -> str:
        return self.kind(e)

    def arith(self, kinds, what) -> str:
        arrays = [k for k in kinds if is_array(k) or k == TRANSIENT_BOOL]
        for k in kinds:
            if is_opaque(k):
                self.err(f"{what}: opaque value of kind {k!r} in an expression")
        if arrays:
            if TRANSIENT_BOOL in arrays:
                self.err(f"{what}: arithmetic on an elementwise predicate")
            specific = [k for k in arrays if k not in ("array", "mask")]
            base = specific[0] if specific else arrays[0]
            for k in arrays:
                if not compatible(k, base):
                    self.err(f"{what}: elementwise operands of different kinds "
                             f"{sorted(set(arrays))} (use a block to combine "
                             "differently shaped arrays)")
            if ARRAY_KINDS[base] == "bool":
                self.err(f"{what}: arithmetic on a boolean mask {base!r}")
            return base
        if "real" in kinds:
            return "real"
        return "int"

    def boolean(self, kinds, what) -> str:
        arrays = [k for k in kinds if not is_scalar(k)]
        if not arrays:
            return "bool"
        if TRANSIENT_BOOL in arrays:
            return TRANSIENT_BOOL
        masks = set()
        for k in arrays:
            if is_opaque(k):
                self.err(f"{what}: opaque value {k!r} in a predicate")
            if ARRAY_KINDS.get(k) == "bool":
                masks.add(k)
            else:
                masks.add(MASK_OF.get(k, TRANSIENT_BOOL))
        if len(masks) > 1:
            if TRANSIENT_BOOL in masks:
                return TRANSIENT_BOOL
            specific = masks - {"mask"}
            if len(specific) > 1:
                self.err(f"{what}: predicates over different shapes {sorted(masks)}")
            return specific.pop() if specific else "mask"
        return masks.pop()

    def kind(self, e) -> str:
        if isinstance(e, BooleanAtom):
            return "bool"
        if isinstance(e, sp.Integer):
            return "int"
        if e.is_Number or isinstance(e, sp.NumberSymbol):
            return "real"
        if isinstance(e, sp.Symbol):
            return self.lookup(str(e))
        if isinstance(e, sp.Indexed):
            base = self.lookup(str(e.base.label))
            if not is_array(base):
                self.err(f"indexed read of non-array {e.base.label} ({base})")
            for i in e.indices:
                if self.kind(sp.sympify(i)) not in ("int", "bool"):
                    self.err(f"index {i} of {e.base.label} is not an int")
            return ARRAY_KINDS[base]
        if isinstance(e, sp.core.relational.Relational):
            k = self.arith([self.kind(a) for a in e.args], str(e))
            return self.boolean([k], str(e))
        if isinstance(e, (sp.And, sp.Or, sp.Not, sp.Xor)):
            ks = [self.kind(a) for a in e.args]
            for k in ks:
                if is_scalar(k) and k != "bool":
                    self.err(f"{e}: logical operator on non-boolean {k}")
            return self.boolean(ks, str(e))
        if isinstance(e, E.IsFinite):
            k = self.kind(e.args[0])
            if is_scalar(k):
                return "bool"
            return self.boolean([k], str(e)) if ARRAY_KINDS.get(k) != "bool" \
                else k
        if isinstance(e, (E.AnyTrue, E.AllTrue)):
            k = self.kind(e.args[0])
            if not (k in ("bool", TRANSIENT_BOOL) or ARRAY_KINDS.get(k) == "bool"):
                self.err(f"{e}: reduction of a non-boolean {k}")
            return "bool"
        if isinstance(e, (E.ReduceMin, E.ReduceMax, E.ReduceSum)):
            k = self.kind(e.args[0])
            if is_opaque(k) or k == TRANSIENT_BOOL or ARRAY_KINDS.get(k) == "bool":
                self.err(f"{e}: reduction of {k}")
            return "real" if (is_array(k) or k == "real") else k
        if isinstance(e, E.Where):
            c = self.kind(e.args[0])
            v = self.arith([self.kind(e.args[1]), self.kind(e.args[2])], str(e))
            if c == TRANSIENT_BOOL or (not is_scalar(c) and ARRAY_KINDS.get(c) != "bool"):
                self.err(f"{e}: condition is not a mask")
            if not is_scalar(c):
                if is_scalar(v):
                    self.err(f"{e}: array condition with scalar values")
                if MASK_OF.get(v) != c and not compatible(v, c) \
                        and c != "mask":
                    self.err(f"{e}: condition {c} does not match values {v}")
            return v
        if isinstance(e, E.FullLike):
            x = self.kind(e.args[0])
            vk = self.kind(e.args[1])
            if not is_array(x):
                self.err(f"{e}: FullLike of non-array {x}")
            if vk == "bool":
                m = x if ARRAY_KINDS[x] == "bool" else MASK_OF.get(x)
                if m is None:
                    self.err(f"{e}: no mask kind of {x}")
                return m
            return x
        if isinstance(e, E.ZerosLike):
            x = self.kind(e.args[0])
            if not is_array(x):
                self.err(f"{e}: ZerosLike of non-array {x}")
            return x
        if isinstance(e, E.MaskLike):
            x = self.kind(e.args[0])
            if x not in CELL_AXIS_KINDS:
                self.err(f"{e}: MaskLike needs a cell-axis array, got {x}")
            if self.kind(e.args[1]) != "bool":
                self.err(f"{e}: MaskLike value must be a bool")
            return "cell_mask"
        if isinstance(e, E.SafeDiv):
            return self.arith([self.kind(a) for a in e.args], str(e))
        if isinstance(e, (sp.Min, sp.Max)):
            return self.arith([self.kind(a) for a in e.args], str(e))
        if isinstance(e, sp.Piecewise):
            vals, conds = [], []
            for a in e.args:
                vals.append(self.kind(a.expr))
                conds.append(self.kind(a.cond))
            v = self.arith(vals, str(e))
            for c in conds:
                if c != "bool" and not (is_array(c) and ARRAY_KINDS[c] == "bool"):
                    self.err(f"{e}: Piecewise condition of kind {c}")
            return v
        if isinstance(e, sp.Abs):
            return self.arith([self.kind(e.args[0])], str(e))
        if isinstance(e, sp.Pow):
            kb = self.kind(e.args[0])
            ke = self.kind(e.args[1])
            k = self.arith([kb, ke], str(e))
            if k == "int" and not (e.args[1].is_Integer and e.args[1] >= 0):
                return "real"
            return k
        if isinstance(e, (sp.Add, sp.Mul)):
            return self.arith([self.kind(a) for a in e.args], str(e))
        if isinstance(e, sp.Function) and type(e).__name__ in E.ELEMENTARY:
            k = self.arith([self.kind(a) for a in e.args], str(e))
            return "real" if k == "int" else k
        if isinstance(e, BooleanFunction):
            self.err(f"unsupported boolean function {type(e).__name__}")
        self.err(f"unsupported expression node {type(e).__name__}: {e}")


def _repl_expr(e, repl) -> bool:
    """Is the (scalar) value of ``e`` identical on every rank?"""
    if isinstance(e, E.REDUCTIONS):
        return True
    if isinstance(e, sp.Symbol):
        return str(e) in repl
    if isinstance(e, sp.Indexed):
        return False
    if not e.args:
        return True
    return all(_repl_expr(a, repl) for a in e.args)


# ── the checker ─────────────────────────────────────────────────────────────


class _ProcChecker:
    def __init__(self, prog, proc: Procedure):
        self.prog = prog
        self.proc = proc
        self.stmts: dict = {}
        self.kinds: dict = {p.name: p.kind for p in proc.inputs}
        self.locals: list = []
        self.has_comm = False
        self.has_stateful = False
        self.written_inputs = set()

    # -- helpers --------------------------------------------------------------
    def err(self, msg, path=None, stmt=None):
        loc = where(self.proc.name, path, stmt) if path is not None else self.proc.name
        raise TemplateCheckError(f"{loc}: {msg}")

    def name_kind(self, name, env, path, stmt):
        if name in env:
            return env[name]
        c = self.prog.constants.get(name)
        if c is not None:
            return c.kind
        self.err(f"reads unbound name {name!r} (bound here: "
                 f"{sorted(env)})", path, stmt)

    def ekind(self, e, env, path, stmt):
        def lookup(n):
            return self.name_kind(n, env, path, stmt)

        def err(m):
            self.err(m, path, stmt)
        return _Kinds(lookup, err)(e)

    def bind(self, name, kind, env, path, stmt):
        if name in self.prog.constants:
            self.err(f"{name!r} is a constant and cannot be assigned", path, stmt)
        old = env.get(name, self.kinds.get(name))
        if old is not None and old != kind and not (
                compatible(kind, old) and (is_array(old) or old == "any")):
            if not (old == "real" and kind == "int"):
                self.err(f"{name!r} rebound with kind {kind!r} (was {old!r}); a "
                         "name has one kind per procedure", path, stmt)
            kind = old
        if name not in self.kinds:
            self.kinds[name] = kind
            self.locals.append(name)
        if name in self.proc.input_names:
            self.written_inputs.add(name)
        env[name] = self.kinds[name]

    def static_count(self, n, path, stmt):
        if isinstance(n, int):
            return
        c = self.prog.constants.get(n)
        if c is None or c.kind != "int":
            self.err(f"static count {n!r} must be an int or an int constant",
                     path, stmt)

    def opt_value(self, v, kind, what, path, stmt):
        if isinstance(v, str):
            c = self.prog.constants.get(v)
            if c is None:
                self.err(f"{what} names unknown constant {v!r}", path, stmt)
            if not compatible(c.kind, kind):
                self.err(f"{what} constant {v!r} is {c.kind}, needs {kind}",
                         path, stmt)

    def check_args(self, what, actual, declared, env, path, stmt):
        if len(actual) != len(declared):
            self.err(f"{what}: passes {len(actual)} argument(s) "
                     f"{list(actual)} but the signature takes "
                     f"{len(declared)} ({', '.join(str(p) for p in declared)})",
                     path, stmt)
        for a, p in zip(actual, declared):
            k = self.name_kind(a, env, path, stmt)
            if not compatible(k, p.kind):
                self.err(f"{what}: argument {a!r} is {k} but parameter "
                         f"{p.name!r} is {p.kind}", path, stmt)

    def bind_outs(self, what, outs, declared, env, path, stmt, repl, out_repl):
        if len(outs) != len(declared):
            self.err(f"{what}: binds {len(outs)} result(s) {list(outs)} but the "
                     f"signature returns {len(declared)} "
                     f"({', '.join(str(p) for p in declared)})", path, stmt)
        if len(set(outs)) != len(outs):
            self.err(f"{what}: duplicate result names {list(outs)}", path, stmt)
        for o, p in zip(outs, declared):
            self.bind(o, p.kind, env, path, stmt)
            if out_repl(p) and is_scalar(p.kind):
                repl.add(o)
            else:
                repl.discard(o)

    # -- blocks ---------------------------------------------------------------
    def block(self, block, env, repl, prefix):
        for i, s in enumerate(block):
            self.stmt(s, env, repl, prefix + (i,))
        return env

    def stmt(self, s, env, repl, path):
        self.stmts[path] = StmtInfo(path=path, stmt=s, env=dict(env))
        for n in _const_names_in(s):
            if n not in self.prog.constants:
                self.err(f"references unknown constant {n!r}", path, s)
        m = getattr(self, "s_" + type(s).__name__, None)
        if m is None:
            self.err(f"no rule for statement {type(s).__name__}", path, s)
        m(s, env, repl, path)

    def s_Seq(self, s, env, repl, path):
        self.block(s.body, env, repl, path + ("body",))

    def s_IfStatic(self, s, env, repl, path):
        self.err(f"unresolved IfStatic({s.flag!r}): call resolve_static() with "
                 "the build flags first", path, s)

    def s_Assign(self, s, env, repl, path):
        k = self.ekind(s.expr, env, path, s)
        if k == TRANSIENT_BOOL:
            self.err("an elementwise predicate over an array without a mask "
                     "kind can only be reduced (AnyTrue/AllTrue), not stored",
                     path, s)
        if s.index is not None:
            base = self.name_kind(s.base, env, path, s)
            if not is_array(base):
                self.err(f"indexed assignment into non-array {s.base}", path, s)
            if self.ekind(s.index, env, path, s) not in ("int", "bool"):
                self.err("index is not an int", path, s)
            if not is_scalar(k):
                self.err("indexed assignment of a non-scalar value", path, s)
            self.bind(s.base, base, env, path, s)
            repl.discard(s.base)
            return
        if s.kind is not None:
            if not (compatible(k, s.kind) or (s.kind == "int" and k == "real")):
                self.err(f"declared kind {s.kind!r} but the expression is {k!r}",
                         path, s)
            k = s.kind
        self.bind(s.base, k, env, path, s)
        if is_scalar(k) and _repl_expr(s.expr, repl | set(self.prog.constants.names())):
            repl.add(s.base)
        else:
            repl.discard(s.base)
        if any(isinstance(a, E.REDUCTIONS) for a in sp.preorder_traversal(s.expr)):
            self.has_comm = True

    def _callee(self, s, path):
        if isinstance(s.callee, Procedure):
            return s.callee, None
        spec = self.prog.registry.get(s.callee, None)
        if spec is None:
            self.err(f"call to unknown block {s.callee!r} (not registered, "
                     "not a Procedure)", path, s)
        if spec.role != "block":
            self.err(f"{s.callee!r} is a {spec.role}; use Callback(...)", path, s)
        return None, spec

    def s_Call(self, s, env, repl, path):
        proc, spec = self._callee(s, path)
        if proc is not None:
            info = self.prog.proc_info(proc)
            self.check_args(f"call {proc.name}", s.args, proc.inputs, env, path, s)
            all_repl = all(a in repl or a in self.prog.constants
                           or not is_scalar(self.name_kind(a, env, path, s))
                           for a in s.args)
            self.bind_outs(f"call {proc.name}", s.out, proc.outputs, env, path, s,
                           repl, lambda p: all_repl and p.name in info.repl_outputs)
            self.has_comm |= info.has_comm
            self.has_stateful |= info.has_stateful
        else:
            self.prog.used.add(spec.name)
            self.check_args(f"call {spec.name}", s.args, spec.inputs, env, path, s)
            scalar_args_repl = all(
                a in repl or a in self.prog.constants
                or not is_scalar(self.name_kind(a, env, path, s)) for a in s.args)
            has_arrays = any(not is_scalar(self.name_kind(a, env, path, s))
                             for a in s.args)
            self.bind_outs(
                f"call {spec.name}", s.out, spec.outputs, env, path, s, repl,
                lambda p: spec.comm == "global"
                or (scalar_args_repl and not has_arrays))
            if spec.comm != "none":
                self.has_comm = True

    def s_Callback(self, s, env, repl, path):
        spec = self.prog.registry.get(s.fn, None)
        if spec is None:
            self.err(f"callback {s.fn!r} is not registered", path, s)
        if spec.role != "callback":
            self.err(f"{s.fn!r} is a {spec.role}, not a callback", path, s)
        self.prog.used.add(spec.name)
        self.check_args(f"callback {s.fn}", s.inputs, spec.inputs, env, path, s)
        self.bind_outs(f"callback {s.fn}", s.outputs, spec.outputs, env, path, s,
                       repl, lambda p: True)
        if s.kind == "stateful":
            self.has_stateful = True

    def s_Hook(self, s, env, repl, path):
        for n in s.inputs:
            self.name_kind(n, env, path, s)
        before = dict(env)
        label, body = s.blocks()[0]
        self.block(body, env, repl, path + (label,))
        for n in s.outputs:
            if n not in env:
                self.err(f"hook {s.name!r} leaves declared output {n!r} unbound",
                         path, s)
        stray = (writes_block(body) & set(before)) - set(s.outputs)
        if stray:
            self.err(f"hook {s.name!r} body ({s.filled_by or 'default'}) writes "
                     f"{sorted(stray)}, which are not among its declared outputs "
                     f"{list(s.outputs)}", path, s)

    def _pred(self, e, env, repl, path, s, what, body_comm):
        k = self.ekind(e, env, path, s)
        if any(isinstance(a, E.REDUCTIONS) for a in sp.preorder_traversal(e)):
            self.has_comm = True
        if k != "bool":
            self.err(f"{what} must be a scalar bool, got {k} (reduce arrays "
                     "with AnyTrue / AllTrue)", path, s)
        if body_comm and not _repl_expr(e, repl | set(self.prog.constants.names())):
            self.err(f"{what} {e} is not replicated across ranks but guards "
                     "communicating work; derive it from a reduction "
                     "(AnyTrue/AllTrue/Reduce*) or a global block", path, s)

    def _loop_body(self, s, env, repl, path, label, body, extra=None):
        inner = dict(env)
        inner_repl = set(repl)
        if extra:
            for n, k in extra.items():
                self.bind(n, k, inner, path, s)
                inner_repl.add(n)
        self.block(body, inner, inner_repl, path + (label,))
        for n in set(env) & set(inner):
            if inner[n] != env[n]:
                self.err(f"loop changes the kind of {n!r}", path, s)
        # replicated status of carried names: replicated only if every write is
        for n in writes_block(body) & set(env):
            if n not in inner_repl:
                repl.discard(n)
        if s.carry is not None:
            for n in s.carry:
                if n not in env:
                    self.err(f"explicit carry {n!r} is not bound before the loop",
                             path, s)
        return inner

    def s_While(self, s, env, repl, path):
        comm = self.prog.block_has_comm(s.body)
        self._loop_body(s, env, repl, path, "body", s.body)
        self._pred(s.cond, env, repl, path, s, "While condition", comm)

    def s_Repeat(self, s, env, repl, path):
        self.static_count(s.n, path, s)
        extra = {s.index: "int"} if s.index else None
        self._loop_body(s, env, repl, path, "body", s.body, extra)

    def s_ForEachFace(self, s, env, repl, path):
        self.static_count(s.count, path, s)
        self._loop_body(s, env, repl, path, "body", s.body, {s.index: "int"})

    def s_Select(self, s, env, repl, path):
        comm = (self.prog.block_has_comm(s.then)
                or self.prog.block_has_comm(s.otherwise))
        self._pred(s.pred, env, repl, path, s, "Select predicate", comm)
        et, rt = dict(env), set(repl)
        self.block(s.then, et, rt, path + ("then",))
        ee, re_ = dict(env), set(repl)
        self.block(s.otherwise, ee, re_, path + ("otherwise",))
        for n in (set(et) & set(ee)) - set(env):
            if et[n] != ee[n]:
                self.err(f"branches bind {n!r} with kinds {et[n]} / {ee[n]}", path, s)
            env[n] = et[n]
            if n in rt and n in re_:
                repl.add(n)
        for n in (writes_block(s.then) | writes_block(s.otherwise)) & set(env):
            if not (n in rt and n in re_):
                repl.discard(n)
        if s.outputs is not None:
            for n in s.outputs:
                if n not in env:
                    self.err(f"explicit Select output {n!r} is not bound after "
                             "both branches", path, s)

    def s_Root(self, s, env, repl, path):
        r = s.residual
        self.prog.proc_info(r)
        if len(r.inputs) != 1 + len(s.args) or len(r.outputs) != 1:
            self.err(f"residual {r.name} must take (x, *args) with "
                     f"{len(s.args)} args and return one value", path, s)
        xk = r.inputs[0].kind
        k0 = self.name_kind(s.x0, env, path, s)
        if not compatible(k0, xk):
            self.err(f"x0 {s.x0!r} is {k0} but residual x is {xk}", path, s)
        if not compatible(r.outputs[0].kind, xk):
            self.err(f"residual returns {r.outputs[0].kind}, x is {xk}", path, s)
        self.check_args(f"Root residual {r.name}", s.args, r.inputs[1:], env,
                        path, s)
        if s.opts.jac_proc is not None:
            j = s.opts.jac_proc
            self.prog.proc_info(j)
            if len(j.inputs) != 2 + len(s.args) or len(j.outputs) != 1:
                self.err(f"jac_proc {j.name} must take (x, v, *args) and return Jv",
                         path, s)
            self.check_args(f"Root jac {j.name}", (s.x0, s.x0) + s.args,
                            j.inputs, env, path, s)
        self._check_linsolve_opts(s.opts.linsolve, s.x0, s.args, env, path, s)
        self.opt_value(s.opts.tol, "real", "Root tol", path, s)
        self.opt_value(s.opts.maxiter, "int", "Root maxiter", path, s)
        self.bind(s.out, xk if xk not in ("array",) else k0, env, path, s)
        repl.discard(s.out)
        self._bind_stats(s, env, repl, path)
        self.has_comm = True

    def _check_linsolve_opts(self, o, x_name, args, env, path, s):
        self.opt_value(o.tol, "real", "LinSolve tol", path, s)
        self.opt_value(o.restart, "int", "LinSolve restart", path, s)
        self.opt_value(o.maxiter, "int", "LinSolve maxiter", path, s)
        if o.precond is not None:
            pc = o.precond
            self.prog.proc_info(pc)
            if len(pc.inputs) != 1 + len(args) or len(pc.outputs) != 1:
                self.err(f"precond {pc.name} must take (r, *args) and return z",
                         path, s)
            self.check_args(f"precond {pc.name}", (x_name,) + tuple(args),
                            pc.inputs, env, path, s)

    def _bind_stats(self, s, env, repl, path):
        if s.stats:
            for n, k in zip(s.stats, ("bool", "real", "int")):
                self.bind(n, k, env, path, s)
                repl.add(n)

    def s_LinSolve(self, s, env, repl, path):
        mv = s.matvec
        self.prog.proc_info(mv)
        if len(mv.inputs) != 1 + len(s.args) or len(mv.outputs) != 1:
            self.err(f"matvec {mv.name} must take (x, *args) with {len(s.args)} "
                     "args and return one value", path, s)
        rk = self.name_kind(s.rhs, env, path, s)
        if not compatible(rk, mv.inputs[0].kind) or not compatible(
                mv.outputs[0].kind, rk):
            self.err(f"rhs {s.rhs!r} ({rk}) does not match matvec "
                     f"{mv.inputs[0].kind} -> {mv.outputs[0].kind}", path, s)
        self.check_args(f"LinSolve matvec {mv.name}", s.args, mv.inputs[1:], env,
                        path, s)
        self._check_linsolve_opts(s.opts, s.rhs, s.args, env, path, s)
        self.bind(s.out, rk, env, path, s)
        repl.discard(s.out)
        self._bind_stats(s, env, repl, path)
        self.has_comm = True

    def s_Guard(self, s, env, repl, path):
        self._pred(s.ok, env, repl, path, s, "Guard predicate", True)
        for n in s.values + tuple(s.on or ()):
            self.name_kind(n, env, path, s)


class _ProgramChecker:
    def __init__(self, root, registry, constants):
        self.root = root
        self.registry = registry
        self.constants = constants
        self.procs = collect_procedures(root)
        self.infos: dict = {}
        self.used: set = set()
        self._active: set = set()

    def proc_info(self, proc: Procedure) -> ProcInfo:
        if proc.name in self.infos:
            return self.infos[proc.name]
        if proc.name in self._active:
            raise TemplateCheckError(f"recursive procedure {proc.name}")
        self._active.add(proc.name)
        for p in proc.inputs:
            if p.name in self.constants:
                raise TemplateCheckError(
                    f"{proc.name}: input {p.name!r} shadows a constant")
        c = _ProcChecker(self, proc)
        env = {p.name: p.kind for p in proc.inputs}
        repl = {p.name for p in proc.inputs
                if is_scalar(p.kind) or p.kind == "params"}
        c.block(proc.body, env, repl, ("body",))
        for p in proc.outputs:
            if p.name not in env:
                raise TemplateCheckError(
                    f"{proc.name}: output {p.name!r} is not bound at the end of "
                    "the body")
            if not compatible(env[p.name], p.kind):
                raise TemplateCheckError(
                    f"{proc.name}: output {p.name!r} is {env[p.name]} but is "
                    f"declared {p.kind}")
        info = ProcInfo(proc=proc, kinds=dict(c.kinds), locals=list(c.locals),
                        stmts=c.stmts,
                        repl_outputs=frozenset(p.name for p in proc.outputs
                                               if p.name in repl),
                        has_comm=c.has_comm, has_stateful=c.has_stateful,
                        written_inputs=frozenset(c.written_inputs))
        _liveness(info, self)
        self._active.discard(proc.name)
        self.infos[proc.name] = info
        return info

    def block_has_comm(self, block) -> bool:
        for s in block:
            if isinstance(s, (Root, LinSolve)):
                return True
            if isinstance(s, Call):
                if isinstance(s.callee, Procedure):
                    if self.proc_info(s.callee).has_comm:
                        return True
                else:
                    spec = self.registry.get(s.callee, None)
                    if spec is not None and spec.comm != "none":
                        return True
            for ex in _exprs_of(s):
                if any(isinstance(a, E.REDUCTIONS) for a in sp.preorder_traversal(ex)):
                    return True
            for _, b in s.blocks():
                if self.block_has_comm(b):
                    return True
        return False


def _exprs_of(s):
    if isinstance(s, Assign):
        return (s.expr,)
    if isinstance(s, (While,)):
        return (s.cond,)
    if isinstance(s, Select):
        return (s.pred,)
    if isinstance(s, Guard):
        return (s.ok,)
    return ()


def block_has_stateful(block, prog_info: "ProgramInfo") -> bool:
    for s in block:
        if isinstance(s, Callback) and s.kind == "stateful":
            return True
        if isinstance(s, Call) and isinstance(s.callee, Procedure):
            if prog_info.procs[s.callee.name].has_stateful:
                return True
        for _, p in s.procedures():
            if not isinstance(s, Call) and prog_info.procs[p.name].has_stateful:
                return True
        for _, b in s.blocks():
            if block_has_stateful(b, prog_info):
                return True
    return False


# ── liveness ────────────────────────────────────────────────────────────────


def _liveness(info: ProcInfo, prog):
    _live_block(info.proc.body, ("body",), set(info.proc.output_names), info, prog)


def _live_block(block, prefix, live_out, info, prog) -> set:
    live = set(live_out)
    for i in reversed(range(len(block))):
        p = prefix + (i,)
        st = info.stmts[p]
        st.live_after = frozenset(live)
        live = _live_stmt(block[i], p, live, st, info, prog)
    return live


def _live_stmt(s, p, live, st, info, prog) -> set:
    bound = set(st.env)
    if isinstance(s, Assign):
        out = set(live)
        if s.index is None:
            out.discard(s.base)
        return out | free_reads_stmt(s)
    if isinstance(s, Call):
        return (live - set(s.out)) | set(s.args)
    if isinstance(s, Callback):
        return (live - set(s.outputs)) | set(s.inputs)
    if isinstance(s, (Root, LinSolve)):
        return (live - {s.out} - set(s.stats)) | free_reads_stmt(s)
    if isinstance(s, Guard):
        if s.on is not None:
            anchors = tuple(s.on)
        else:
            cands = [n for n in tuple(s.values) + tuple(sorted(expr_names(s.ok)))
                     if n in live and n in bound]
            scal = [n for n in cands if is_scalar(st.env[n])]
            pick = scal or cands
            if not pick:
                rest = sorted(n for n in live if n in bound)
                scal = [n for n in rest if is_scalar(st.env[n])]
                pick = scal or rest
            anchors = tuple(dict.fromkeys(pick[:1]))
        st.anchors = anchors
        return live | free_reads_stmt(s) | set(anchors)
    if isinstance(s, (Seq, Hook)):
        label, body = s.blocks()[0]
        return _live_block(body, p + (label,), live, info, prog)
    if isinstance(s, Select):
        wt, we = writes_block(s.then), writes_block(s.otherwise)
        if s.outputs is not None:
            outs = tuple(s.outputs)
            need = (wt | we) & live & bound
            missing = need - set(outs)
            if missing:
                raise TemplateCheckError(
                    f"{where(info.proc.name, p, s)}: explicit outputs "
                    f"{list(outs)} miss written live names {sorted(missing)}")
        else:
            new_both = (_defs_block(s.then) & _defs_block(s.otherwise)) - bound
            outs = tuple(n for n in _ordered_writes(s.then, s.otherwise)
                         if n in live and (n in bound or n in new_both))
        st.outputs = outs
        lt = _live_block(s.then, p + ("then",), set(outs), info, prog)
        le = _live_block(s.otherwise, p + ("otherwise",), set(outs), info, prog)
        return ((live - set(outs)) | lt | le | expr_names(s.pred)
                | (set(outs) & bound))
    if isinstance(s, While):
        cond = expr_names(s.cond)
        carry = _carry(s, bound, live | free_reads_block(s.body) | cond, info, p)
        st.carry = carry
        lb = _live_block(s.body, p + ("body",), set(carry) | cond, info, prog)
        return live | set(carry) | cond | lb
    if isinstance(s, (Repeat, ForEachFace)):
        carry = _carry(s, bound, live | free_reads_block(s.body), info, p)
        st.carry = carry
        lb = _live_block(s.body, p + ("body",), set(carry), info, prog)
        lb.discard(s.index)
        return live | set(carry) | lb
    raise TemplateCheckError(f"no liveness rule for {type(s).__name__}")


def _carry(s, bound, needed, info, p) -> tuple:
    w = _ordered_writes(s.body)
    idx = getattr(s, "index", None)
    inferred = tuple(n for n in w if n in bound and n in needed and n != idx)
    if s.carry is not None:
        missing = set(inferred) - set(s.carry)
        if missing:
            raise TemplateCheckError(
                f"{where(info.proc.name, p, s)}: explicit carry {list(s.carry)} "
                f"misses written live names {sorted(missing)}")
        return tuple(s.carry)
    if not inferred and not w:
        raise TemplateCheckError(
            f"{where(info.proc.name, p, s)}: loop body writes nothing")
    return inferred


def _defs_block(block) -> set:
    """Names definitely bound after a block (scope rules)."""
    out = set()
    for s in block:
        if isinstance(s, (Assign, Call, Callback, Root, LinSolve)):
            out |= writes_stmt(s)
        elif isinstance(s, (Seq, Hook)):
            out |= _defs_block(s.blocks()[0][1])
        elif isinstance(s, Select):
            out |= _defs_block(s.then) & _defs_block(s.otherwise)
    return out


def _ordered_writes(*blocks) -> list:
    out = []

    def add(n):
        if n not in out:
            out.append(n)

    def visit(block):
        for s in block:
            if isinstance(s, Assign):
                add(s.base)
            elif isinstance(s, Call):
                for n in s.out:
                    add(n)
            elif isinstance(s, Callback):
                for n in s.outputs:
                    add(n)
            elif isinstance(s, (Root, LinSolve)):
                add(s.out)
                for n in s.stats:
                    add(n)
            for _, b in s.blocks():
                visit(b)

    for b in blocks:
        visit(b)
    return out


# ── entry point ─────────────────────────────────────────────────────────────


def check_program(root: Procedure, registry: Optional[BlockRegistry] = None,
                  constants: Optional[ConstTable] = None) -> ProgramInfo:
    """Check ``root`` and every procedure it references; return the analysis
    the walkers lower from.  Raises :class:`TemplateCheckError`."""
    registry = BLOCKS if registry is None else registry
    constants = ConstTable() if constants is None else constants
    pc = _ProgramChecker(root, registry, constants)
    for p in pc.procs.values():
        pc.proc_info(p)
    procs = {n: pc.infos[n] for n in pc.procs}
    return ProgramInfo(root=root, procs=procs, registry=registry,
                       constants=constants, blocks_used=tuple(sorted(pc.used)))


__all__ = ["check_program", "ProgramInfo", "ProcInfo", "StmtInfo",
           "TemplateCheckError", "fmt_path", "free_reads_block",
           "writes_block", "block_has_stateful"]
