"""Walker (a): the C-family Procedure printer.

Emits a `Procedure` as C++ statements for the
dmplex / amrex / openfoam backends.  It REUSES the existing machinery rather
than reinventing it — `GenericCppBase`
supplies expression printing (including the `c_functions` print-map, the
`_NON_RESOLVABLE` shield, the REQ-66 literal caster and the Min/Max typing),
`convert_expression_body`'s CSE emission shape, the Piecewise if/else
lowering, `get_array_declaration` / `format_assignment` and
`_sm_signature` / `_sm_arg_decl` for the parameter list.

What this printer adds is only the STATEMENT layer: the five IR node kinds
lowered to C++ text.  `IfStatic` never reaches it — the entry guard
`require_resolved` rejects an unresolved procedure,
so a build-time decision cannot become a runtime branch.
"""
from __future__ import annotations

import textwrap

import sympy as sp

from zoomy_core.solver.arg_slots import SOLVER_ARG_KINDS, SOLVER_ARG_MAPPING
from zoomy_core.solver.ir import (
    Assign,
    Call,
    ForEachFace,
    IfStatic,
    Procedure,
    While,
    require_resolved,
    written_names,
)
from zoomy_core.transformation.generic_c import GenericCppBase


class CProcedurePrinter(GenericCppBase):
    """Print solver Procedures as C++ free functions.

    Constructed WITHOUT a SystemModel: solver blocks speak the march-level
    argument vocabulary, not a model's operator signature.  A backend that
    wants its own spellings subclasses this exactly as `FoamPrinter`
    subclasses `GenericCppBase` (override `ARG_MAPPING` / `real_type` /
    `_sm_arg_decl`).
    """

    #: Emitted return type of a procedure.  Solver blocks communicate through
    #: their (written) array/scalar arguments; they never return a value.
    procedure_return_type = "void"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Locals bound by Assign statements print as bare identifiers.  Kept
        # as a set (not a symbol_map) so an Assign target shadows nothing.
        self._locals: set = set()

    # ── entry point ────────────────────────────────────────────────────────

    def print_procedure(self, proc: Procedure) -> str:
        """Full C++ text of one procedure (signature + body)."""
        require_resolved(proc)
        written = set(written_names(proc.stmts))
        args_str = ",\n        ".join(
            self._proc_arg_decl(a, written) for a in proc.args)
        self._locals = set()
        lines = self._block(proc.stmts, indent=2)
        if proc.returns:
            lines.append(f"        return {self.ARG_MAPPING[proc.returns]};")
        body = "\n".join(lines)
        doc = ""
        if proc.doc:
            doc = "".join(f"    // {line}\n"
                          for line in proc.doc.strip().splitlines())
        qualifier = "PORTABLE_FN " if self.gpu_enabled else ""
        return (
            f"{doc}    {qualifier}static inline {self._return_type(proc)} "
            f"{proc.name}(\n        {args_str})\n    {{\n{body}\n    }}\n"
        )

    def _proc_arg_decl(self, slot: str, written: set) -> str:
        """One parameter declaration, const-correct FOR THIS PROCEDURE.

        The slot's declared kind states the array's nature; whether *this*
        block writes it decides the `const`.  So `lam_lo_f` is `T*` in
        `solver_dt_pass` (which fills it) and `const T*` in
        `solver_reduce_dt` (which only reduces over it) — from ONE
        declaration table, with no per-block hand-spelling.
        """
        kind = SOLVER_ARG_KINDS[slot]
        name = self.ARG_MAPPING[slot]
        mutable = {"const_real_ptr": "real_ptr", "const_int_ptr": "int_ptr"}
        constant = {v: k for k, v in mutable.items()}
        if name in written:
            kind = mutable.get(kind, kind)
        else:
            kind = constant.get(kind, kind)
        return self._decl_for_kind(kind, name)

    def _return_type(self, proc: Procedure) -> str:
        """Return type from the returned slot's storage kind."""
        if not proc.returns:
            return self.procedure_return_type
        kind = SOLVER_ARG_KINDS[proc.returns]
        return self.real_type if kind == "scalar_real" else "int"

    def print_procedures(self, procs) -> str:
        """Several procedures, emission order preserved."""
        return "\n".join(self.print_procedure(p) for p in procs)

    # ── statement dispatch ─────────────────────────────────────────────────

    def _block(self, stmts, indent) -> list:
        lines = []
        for s in stmts:
            lines.extend(self._statement(s, indent))
        return lines

    def _statement(self, stmt, indent) -> list:
        pad = " " * (4 * indent)
        if isinstance(stmt, Assign):
            return [pad + ln if ln else ln
                    for ln in self._assign(stmt)]
        if isinstance(stmt, Call):
            return [pad + self._call(stmt)]
        if isinstance(stmt, ForEachFace):
            head = (f"{pad}for (int {stmt.index} = 0; "
                    f"{stmt.index} < {self.ARG_MAPPING[stmt.count]}; "
                    f"++{stmt.index}) {{")
            self._locals.add(stmt.index)
            body = self._block(stmt.body, indent + 1)
            return [head, *body, pad + "}"]
        if isinstance(stmt, While):
            cond = self.doprint(sp.sympify(stmt.condition))
            body = self._block(stmt.body, indent + 1)
            return [f"{pad}while ({cond}) {{", *body, pad + "}"]
        if isinstance(stmt, IfStatic):
            raise ValueError(
                "IfStatic reached the C printer — resolve() it at build time")
        raise TypeError(f"unknown statement kind {type(stmt).__name__}")

    # ── Assign ─────────────────────────────────────────────────────────────

    def _assign(self, stmt: Assign) -> list:
        """One assignment, CSE'd through the same shape
        `convert_expression_body` uses (`T t0 = …;` temporaries first,
        then the target writes)."""
        if isinstance(stmt.expr, sp.Piecewise):
            return self._assign_piecewise(stmt)
        if stmt.shape:
            return self._assign_array(stmt)
        return self._assign_scalar(stmt)

    def _assign_scalar(self, stmt: Assign) -> list:
        expr = sp.sympify(stmt.expr)
        temps, (simplified,) = sp.cse([expr], symbols=sp.numbered_symbols("t"))
        lines = [f"{self.real_type} {self.doprint(l)} = {self.doprint(r)};"
                 for l, r in temps]
        if stmt.declare:
            self._locals.add(stmt.target)
            lines.append(f"{stmt.ctype or self.real_type} {stmt.target} = "
                         f"{self.doprint(simplified)};")
        else:
            lines.append(f"{stmt.target} = {self.doprint(simplified)};")
        return lines

    def _assign_array(self, stmt: Assign) -> list:
        flat = list(sp.flatten(
            list(stmt.expr) if isinstance(stmt.expr, sp.MatrixBase)
            else stmt.expr))
        temps, simplified = sp.cse(flat, symbols=sp.numbered_symbols("t"))
        lines = [f"{self.real_type} {self.doprint(l)} = {self.doprint(r)};"
                 for l, r in temps]
        if stmt.declare:
            self._locals.add(stmt.target)
            lines.append(self.get_array_declaration(
                stmt.target, stmt.shape, init_zero=False))
        for i, val in enumerate(simplified):
            lines.append(f"{stmt.target}[{i}] = {self.doprint(val)};")
        return lines

    def _assign_piecewise(self, stmt: Assign) -> list:
        """Piecewise lowered to the same if/else-if/else ladder the kernel
        emitter builds (`_print_piecewise_structure`), but assigning instead
        of returning — the statement-position form."""
        lines = []
        if stmt.declare:
            self._locals.add(stmt.target)
            if stmt.shape:
                lines.append(self.get_array_declaration(
                    stmt.target, stmt.shape, init_zero=True))
            else:
                lines.append(
                    f"{stmt.ctype or self.real_type} {stmt.target} = 0;")
        branches = list(stmt.expr.args)
        # A trailing "otherwise: keep the current value" branch is a no-op on
        # an existing lvalue — drop it rather than emitting ``x = x;``.
        if (not stmt.declare and len(branches) > 1
                and branches[-1].cond in (True, sp.true)
                and branches[-1].expr == sp.Symbol(stmt.target)):
            branches = branches[:-1]
        for i, arg in enumerate(branches):
            val, cond = arg.expr, arg.cond
            if i == 0:
                lines.append(f"if ({self.doprint(cond)}) {{")
            elif cond in (True, sp.true):
                lines.append("} else {")
            else:
                lines.append(f"}} else if ({self.doprint(cond)}) {{")
            inner = Assign(target=stmt.target, expr=val,
                           shape=stmt.shape, declare=False)
            lines.extend(textwrap.indent(ln, "    ")
                         for ln in self._assign(inner))
        lines.append("}")
        return lines

    # ── Call ───────────────────────────────────────────────────────────────

    def _call(self, stmt: Call) -> str:
        """A statement-position call.  Results are passed as trailing
        (written) arguments — the C family has no multiple return, and the
        stored-face-array contract wants the callee to write in place."""
        names = [self._arg_name(a) for a in (*stmt.args, *stmt.results)]
        return f"{stmt.procedure}({', '.join(names)});"

    def _arg_name(self, key: str) -> str:
        return self.ARG_MAPPING.get(key, key)

    # ── symbol printing ────────────────────────────────────────────────────

    def _print_Symbol(self, s):
        """Bare identifier for anything the procedure's own argument list or
        an `Assign` bound; otherwise fall back to the base printer (which
        consults the pushed `symbol_maps`)."""
        name = str(s)
        if name in self._locals or name in self.ARG_MAPPING.values():
            return name
        return super()._print_Symbol(s)
