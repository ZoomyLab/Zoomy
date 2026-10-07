"""Scratch prototype: the minimal fixes to core's CProcedurePrinter that make
the EMITTED march compile and run as host C++.  Subclass only; core untouched.

Fixes (numbered as in the report):
  F1  declare-once scoping: a second ``declare=True`` Assign of a name already
      bound in an enclosing scope prints as an assignment (type must agree).
  F2  whole-array Assign -> elementwise hook (generic lambda handed to a
      backend ``zoomy_elementwise``); operands must share the target's kind.
  F3  written scalar params by mutable reference (``double& time``), read
      scalars by value; arrays as typed handles (``const CellState&``).
  F4  declaration table for every name a body uses but no param/Assign binds:
      workspace arrays -> ``CellState& Q0 = ctx.ws.Q0;``, mesh slots ->
      ``ctx.mesh.<n>``, scalar call results -> ``double dt = 0;``.  An
      undeclared name RAISES at print time (catches the skipped-constant
      class, e.g. ``c_dt_max`` when nsm.dt_max is None).
  F5  call convention: args, then results not already among the args (in/out
      passed ONCE by reference); Procedure callees get out-params from
      PROC_RESULTS (stand-in for a ``Procedure.results`` field).
  F6  arity + kind + mutability check of every Call against the callee's
      signature (internal Procedure or EXTERNAL_SIGS) at print time.
  F7  host-only qualifier for orchestration (no PORTABLE_FN).
  F8  an explicit context parameter ``Ctx& ctx`` on every procedure/external
      (the bodies need MeshRT + workspace; core's IR has no handle slot).
"""
from __future__ import annotations

import sympy as sp

from zoomy_core.solver.arg_slots import SOLVER_ARG_MAPPING
from zoomy_core.solver.ir import (Assign, Call, ForEachFace, Procedure, While,
                                  expression_names, require_resolved,
                                  split_target, written_names)
from zoomy_core.transformation.procedure_c import CProcedurePrinter

from feas_tables import (ARRAY_KINDS, EXTERNAL_SIGS, HANDLE_TYPES,
                         PROC_RESULTS, binding_of, kind_of)


class CallSignatureError(TypeError):
    """A Call does not match its callee's declared signature."""


class FeasCProcedurePrinter(CProcedurePrinter):
    gpu_enabled = False                 # F7: orchestration is host code
    orchestration_qualifier = ""        # amrex: "" ; never PORTABLE_FN
    context_type = "Ctx"                # F8
    context_name = "ctx"

    def __init__(self, procedures=(), externals=None, proc_results=None,
                 **kw):
        super().__init__(**kw)
        self.procs = {p.name: p for p in procedures}
        self.externals = dict(EXTERNAL_SIGS if externals is None else externals)
        self.proc_results = dict(PROC_RESULTS if proc_results is None
                                 else proc_results)
        self.workspace_used: dict = {}  # name -> kind, for the ws struct

    # ── names / kinds ──────────────────────────────────────────────────────
    def cname(self, key):
        return SOLVER_ARG_MAPPING.get(key, key)

    def handle(self, kind):
        return HANDLE_TYPES[kind]

    def decl_param(self, name, kind, mutable):
        t = self.handle(kind)
        if kind in ARRAY_KINDS:
            return f"{t}& {name}" if mutable else f"const {t}& {name}"
        return f"{t}& {name}" if mutable else f"const {t} {name}"

    # ── signatures ─────────────────────────────────────────────────────────
    def proc_params(self, proc: Procedure):
        """(name, kind, mutable) of an internal Procedure: its args, then its
        out-results not among the args (F5)."""
        written = set(written_names(proc.stmts))
        names = [self.cname(a) for a in proc.args]
        names += [self.cname(r) for r in self.proc_results.get(proc.name, ())
                  if self.cname(r) not in names]
        outs = {self.cname(r) for r in self.proc_results.get(proc.name, ())}
        return tuple((n, kind_of(n), n in written or n in outs) for n in names)

    def callee_params(self, name):
        if name in self.procs:
            return self.proc_params(self.procs[name])
        if name in self.externals:
            return self.externals[name]
        raise CallSignatureError(f"call to undeclared procedure {name!r}")

    # ── F6: the call check ───────────────────────────────────────────────
    def check_call(self, call: Call):
        actual = [self.cname(a) for a in call.args]
        results = [self.cname(r) for r in call.results]
        actual += [r for r in results if r not in actual]
        params = self.callee_params(call.procedure)
        if len(actual) != len(params):
            raise CallSignatureError(
                f"{call.procedure}: call passes {len(actual)} "
                f"({', '.join(actual)}) but the declaration takes "
                f"{len(params)} ({', '.join(p[0] for p in params)})")
        for a, (pn, pk, pm) in zip(actual, params):
            ak = kind_of(a)
            if ak != pk and not (ak == "int" and pk == "real"):
                raise CallSignatureError(
                    f"{call.procedure}: argument {a!r} is {ak} but parameter "
                    f"{pn!r} is {pk}")
            if a in results and not pm:
                raise CallSignatureError(
                    f"{call.procedure}: call expects {a!r} written back but "
                    f"parameter {pn!r} is read-only")
            if pm and a not in results:
                raise CallSignatureError(
                    f"{call.procedure}: callee writes parameter {pn!r} but the "
                    f"call does not list {a!r} among its results")
        return actual

    # ── procedure ──────────────────────────────────────────────────────────
    def print_procedure(self, proc: Procedure) -> str:
        require_resolved(proc)
        params = self.proc_params(proc)
        self._scopes = [set(n for n, _, _ in params) | {self.context_name}]
        self._params = {n: k for n, k, _ in params}
        self._decl_types = {}
        self._locals = set(self._params)
        decls = self._hoisted_decls(proc, params)
        body = self._block(proc.stmts, indent=2)
        if proc.returns:
            body.append(f"        return {self.cname(proc.returns)};")
        sig = ",\n        ".join(
            [f"{self.context_type}& {self.context_name}"]
            + [self.decl_param(n, k, m) for n, k, m in params])
        doc = "".join(f"    // {ln}\n" for ln in proc.doc.strip().splitlines()) \
            if proc.doc else ""
        q = self.orchestration_qualifier
        ret = self._return_type(proc)
        return (f"{doc}    {q}inline {ret} {proc.name}(\n        {sig})\n"
                f"    {{\n" + "\n".join(decls + body) + "\n    }\n")

    def _used_names(self, stmts):
        out = []

        def add(n):
            if n not in out:
                out.append(n)

        def walk(block):
            for s in block:
                if isinstance(s, Assign):
                    add(split_target(s.target)[0])
                    exprs = (s.expr.args if isinstance(s.expr, sp.Piecewise)
                             else [s.expr])
                    for e in exprs:
                        for n in expression_names(e):
                            add(n)
                elif isinstance(s, Call):
                    for a in (*s.args, *s.results):
                        add(self.cname(a))
                elif isinstance(s, While):
                    for n in expression_names(s.condition):
                        add(n)
                    walk(s.body)
                elif isinstance(s, ForEachFace):
                    add(self.cname(s.count))
                    walk(s.body)
        walk(stmts)
        return out

    def _assign_declared(self, stmts):
        out = set()
        for s in stmts:
            if isinstance(s, Assign) and s.declare:
                out.add(s.target)
            for ch in s.children():
                out |= self._assign_declared(ch)
        return out

    def _hoisted_decls(self, proc, params):
        """F4: bind every used-but-unbound name from the declaration table."""
        bound = {n for n, _, _ in params}
        assigned = self._assign_declared(proc.stmts)
        lines = []
        for n in self._used_names(proc.stmts):
            if n in bound or n in assigned:
                continue
            kind = kind_of(n)            # RAISES on an undeclared name
            b = binding_of(n)
            t = self.handle(kind)
            if b == "ws":
                self.workspace_used[n] = kind
                lines.append(f"        {t}& {n} = {self.context_name}.ws.{n};")
            elif b == "mesh":
                if kind in ARRAY_KINDS:
                    lines.append(f"        const {t}& {n} = "
                                 f"{self.context_name}.mesh.{n};")
                else:
                    lines.append(f"        const {t} {n} = "
                                 f"{self.context_name}.mesh.{n};")
            elif b == "local":
                lines.append(f"        {t} {n} = 0;")
            elif b == "arg":
                raise CallSignatureError(
                    f"{proc.name}: march-state name {n!r} used but not a "
                    "parameter")
            else:  # "const": an emitted constant whose Assign was skipped
                raise CallSignatureError(
                    f"{proc.name}: constant {n!r} is referenced but never "
                    "assigned (its MarchConstants value is None, so "
                    "consts.assigns() skipped it)")
            self._scopes[0].add(n)
            self._locals.add(n)
            self._decl_types[n] = t
        return lines

    # ── statements ─────────────────────────────────────────────────────────
    def _statement(self, stmt, indent):
        pad = " " * (4 * indent)
        if isinstance(stmt, While):
            cond = self.doprint(sp.sympify(stmt.condition))
            self._scopes.append(set())
            body = self._block(stmt.body, indent + 1)
            self._scopes.pop()
            return [f"{pad}while ({cond}) {{", *body, pad + "}"]
        if isinstance(stmt, ForEachFace):
            self._scopes.append({stmt.index})
            out = super()._statement(stmt, indent)
            self._scopes.pop()
            return out
        return super()._statement(stmt, indent)

    def _bound(self, name):
        return any(name in s for s in self._scopes)

    def _assign(self, stmt: Assign):
        base, idx = split_target(stmt.target)
        if stmt.declare:
            want = stmt.ctype or self.real_type
            if base in self._params or self._bound(base):
                # F1: declare-once — already bound (param, hoisted local or an
                # earlier stage's Assign): print as an assignment, same type.
                have = (self.handle(self._params[base]) if base in self._params
                        else self._decl_types.get(base, want))
                if have != want:
                    raise CallSignatureError(
                        f"{base!r} redeclared as {want} (bound as {have})")
                stmt = Assign(stmt.target, stmt.expr, stmt.shape,
                              declare=False, ctype=stmt.ctype)
            else:
                self._scopes[-1].add(base)
                self._decl_types[base] = want
        # F2: whole-array assign
        if idx is None and not stmt.shape and self._is_array(base):
            return self._assign_elementwise(stmt)
        return super()._assign(stmt)

    def _is_array(self, name):
        try:
            return kind_of(name) in ARRAY_KINDS
        except KeyError:
            return False

    def _assign_elementwise(self, stmt: Assign):
        target = split_target(stmt.target)[0]
        tkind = kind_of(target)
        expr = sp.sympify(stmt.expr)
        arrays = [n for n in expression_names(expr)
                  if n != target and kind_of(n) in ARRAY_KINDS]
        for a in arrays:
            if kind_of(a) != tkind:
                raise CallSignatureError(
                    f"elementwise {target} = ...: operand {a!r} is "
                    f"{kind_of(a)}, target is {tkind}")
        reads_target = target in expression_names(expr)
        ops = ([target + "_in"] if reads_target else []) + arrays
        rep = {sp.Symbol(a): sp.Symbol(f"{a}_e") for a in arrays}
        if reads_target:
            rep[sp.Symbol(target)] = sp.Symbol(f"{target}_e")
        rhs = expr.xreplace(rep)
        saved = set(self._locals)
        self._locals |= {str(s) for s in rep.values()}
        rhs_s = self.doprint(rhs)
        self._locals = saved
        params = [f"auto& {target}_e"] + [f"const auto& {a}_e" for a in arrays]
        operands = ", ".join([target] + arrays)
        return [f"zoomy_elementwise([=]({', '.join(params)}) "
                f"{{ {target}_e = {rhs_s}; }}, {operands});"]

    def _call(self, stmt: Call) -> str:
        actual = self.check_call(stmt)
        return f"{stmt.procedure}({', '.join([self.context_name] + actual)});"

    # ── emitted support: workspace struct + external declarations ───────
    def workspace_struct(self, name="MarchWorkspace"):
        lines = [f"struct {name} {{"]
        for n, k in self.workspace_used.items():
            lines.append(f"    {self.handle(k)} {n};   // {k}")
        lines.append("};")
        lines.append(f"template <class C> inline void allocate_march_workspace(C& {self.context_name}) {{")
        for n, k in self.workspace_used.items():
            lines.append(f"    zoomy_alloc_{k}({self.context_name}, "
                         f"{self.context_name}.ws.{n});")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def external_declarations(self):
        out = []
        for name, params in self.externals.items():
            sig = ", ".join([f"{self.context_type}& {self.context_name}"]
                            + [self.decl_param(n, k, m) for n, k, m in params])
            ret = "double" if name == "solver_reduce_min" else "void"
            out.append(f"{ret} {name}({sig});")
        return "\n".join(out) + "\n"
