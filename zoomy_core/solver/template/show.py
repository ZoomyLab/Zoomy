"""Human-readable rendering of a template (`Solver.show()`).

Pseudo-code, one statement per line, nested blocks indented.  Unresolved
`IfStatic` nodes are shown with both branches (teaching view); hooks show
which plugin filled them.
"""
from __future__ import annotations

from sympy.printing.str import StrPrinter

from zoomy_core.solver.template.ir import (
    Assign, Call, Callback, ForEachFace, Guard, Hook, IfStatic, LinSolve,
    Procedure, Repeat, Root, Select, Seq, While, collect_procedures)


class _ShowPrinter(StrPrinter):
    def _print_Float(self, f):
        return repr(float(f))


_P = _ShowPrinter()


def _e(x) -> str:
    return _P.doprint(x)


def _tuple(names) -> str:
    return ", ".join(names)


def _lines(block, ind, out):
    pad = "  " * ind
    for s in block:
        note = f"    # {s.note}" if s.note else ""
        if isinstance(s, Assign):
            out.append(f"{pad}{s.target} = {_e(s.expr)}"
                       + (f"  : {s.kind}" if s.kind else "") + note)
        elif isinstance(s, Call):
            lhs = f"{_tuple(s.out)} = " if s.out else ""
            out.append(f"{pad}{lhs}{s.callee_name}({_tuple(s.args)}){note}")
        elif isinstance(s, Callback):
            lhs = f"{_tuple(s.outputs)} = " if s.outputs else ""
            out.append(f"{pad}{lhs}callback[{s.kind}] {s.fn}({_tuple(s.inputs)})"
                       f"{note}")
        elif isinstance(s, Hook):
            state = f"filled by {s.filled_by}" if s.body is not None else "default"
            out.append(f"{pad}hook {s.name}({_tuple(s.inputs)}) -> "
                       f"({_tuple(s.outputs)})  [{state}]{note}")
            body = s.effective
            if body:
                _lines(body, ind + 1, out)
            else:
                out.append(f"{pad}  pass")
        elif isinstance(s, IfStatic):
            out.append(f"{pad}if static {s.flag}:{note}")
            _lines(s.then, ind + 1, out) if s.then else out.append(f"{pad}  pass")
            if s.otherwise:
                out.append(f"{pad}else static:")
                _lines(s.otherwise, ind + 1, out)
        elif isinstance(s, While):
            carry = f"  carry({_tuple(s.carry)})" if s.carry is not None else ""
            out.append(f"{pad}while {_e(s.cond)}:{carry}{note}")
            _lines(s.body, ind + 1, out)
        elif isinstance(s, Repeat):
            idx = f"{s.index} in " if s.index else ""
            out.append(f"{pad}repeat {idx}range({s.n}):{note}")
            _lines(s.body, ind + 1, out)
        elif isinstance(s, ForEachFace):
            out.append(f"{pad}for face {s.index} in range({s.count}):{note}")
            _lines(s.body, ind + 1, out)
        elif isinstance(s, Select):
            out.append(f"{pad}select {_e(s.pred)}:{note}")
            _lines(s.then, ind + 1, out) if s.then else out.append(f"{pad}  pass")
            if s.otherwise:
                out.append(f"{pad}else:")
                _lines(s.otherwise, ind + 1, out)
        elif isinstance(s, Root):
            st = f", stats=({_tuple(s.stats)})" if s.stats else ""
            o = s.opts
            out.append(f"{pad}{s.out} = root {s.residual.name}(x; {_tuple(s.args)}) "
                       f"from {s.x0}  [tol={o.tol}, maxiter={o.maxiter}, "
                       f"jac={o.jac}, linsolve={o.linsolve.kind}, "
                       f"aux={o.aux_policy}, on_fail={o.on_fail}{st}]{note}")
        elif isinstance(s, LinSolve):
            st = f", stats=({_tuple(s.stats)})" if s.stats else ""
            o = s.opts
            pc = f", precond={o.precond.name}" if o.precond is not None else ""
            ns = f", nullspace={o.nullspace}" if o.nullspace else ""
            out.append(f"{pad}{s.out} = linsolve {s.matvec.name}(x; "
                       f"{_tuple(s.args)}) = {s.rhs}  [{o.kind}, tol={o.tol}, "
                       f"restart={o.restart}, maxiter={o.maxiter}{pc}{ns}, "
                       f"on_fail={o.on_fail}{st}]{note}")
        elif isinstance(s, Guard):
            v = f"  report({_tuple(s.values)})" if s.values else ""
            out.append(f"{pad}guard {_e(s.ok)}  \"{s.msg}\"{v}{note}")
        elif isinstance(s, Seq):
            _lines(s.body, ind, out)
        else:
            out.append(f"{pad}<{type(s).__name__}>")


def show_procedure(proc: Procedure) -> str:
    ins = ", ".join(str(p) for p in proc.inputs)
    outs = ", ".join(str(p) for p in proc.outputs)
    out = [f"procedure {proc.name}({ins}) -> ({outs})"]
    if proc.doc:
        out.extend("  # " + ln for ln in proc.doc.strip().splitlines())
    _lines(proc.body, 1, out)
    return "\n".join(out)


def show_program(root: Procedure, constants=None, registry=None) -> str:
    procs = collect_procedures(root)
    parts = [show_procedure(p) for p in reversed(list(procs.values()))]
    if constants is not None and constants.names():
        parts.append("constants:\n" + "\n".join(
            "  " + ln for ln in constants.report().splitlines()))
    if registry is not None:
        used = set()
        for p in procs.values():
            from zoomy_core.solver.template.ir import iter_statements
            for s in iter_statements(p.body):
                if isinstance(s, Call) and isinstance(s.callee, str):
                    used.add(s.callee)
                if isinstance(s, Callback):
                    used.add(s.fn)
        if used:
            parts.append("blocks:\n" + "\n".join(
                "  " + registry.get(n).signature() for n in sorted(used)
                if n in registry))
    return "\n\n".join(parts)


__all__ = ["show_procedure", "show_program"]
