"""Every Call site in the emitted march vs the THREE signature sources core
has: ExternalProcedure.args/results, the doc-string "(...) -> (...)", and
arg_slots.SOLVER_ARG_SLOTS.  Counts under the raw printer convention
(args + results, duplicates passed twice) and the dedupe convention
(in/out once)."""
import re
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from emit_raw import build_nsm
from zoomy_core.solver.arg_slots import SOLVER_ARG_SLOTS
from zoomy_core.solver.external import REQUIRED_PROCEDURES
from zoomy_core.solver.ir import Call
from zoomy_core.solver.march import emit_march

DOC = re.compile(r"^\(([^)]*)\)\s*->\s*(?:\(([^)]*)\)|(\w+))")


def doc_sig(doc):
    m = DOC.match(doc.strip())
    if not m:
        return None
    ins = [s.strip() for s in m.group(1).split(",") if s.strip()]
    outs = m.group(2) if m.group(2) is not None else (m.group(3) or "")
    outs = [s.strip() for s in outs.split(",") if s.strip()]
    return ins, outs


def calls(stmts):
    for s in stmts:
        if isinstance(s, Call):
            yield s
        for ch in s.children():
            yield from calls(ch)


def main():
    seen = {}
    for order, kw in ((1, dict(dt_floor=1e-12)), (2, dict(mood=True, coupled=True,
                       flags=dict(implicit_source=True, implicit_diffusion=True))),
                      (2, dict(mood=True))):
        nsm = build_nsm(order)
        prog = emit_march(nsm, cfl=0.9, dimension=1, **kw)
        for proc in (prog.march, prog.step, prog.should_write):
            for c in calls(proc.stmts):
                key = (c.procedure, c.args, c.results)
                seen.setdefault(key, proc.name)
    procs = {"solver_should_write", "solver_hyperbolic_step"}
    rows = []
    for (name, args, results), caller in seen.items():
        raw = len(args) + len(results)
        dedup = len(args) + len([r for r in results if r not in args])
        if name in procs:
            rows.append((name, caller, raw, dedup, "Procedure", "-", "-"))
            continue
        ep = REQUIRED_PROCEDURES[name]
        if ep.args or ep.results:
            decl = f"{len(ep.args)}+{len(ep.results)}"
            decl_n = len(ep.args) + len([r for r in ep.results if r not in ep.args])
            src = "args"
        else:
            ds = doc_sig(ep.doc)
            if ds is None:
                decl, decl_n, src = "?", None, "doc(unparsed)"
            else:
                ins, outs = ds
                decl = f"{len(ins)}->{len(outs)}"
                decl_n = len(ins) + len([o for o in outs if o not in ins])
                src = "doc"
        slots = SOLVER_ARG_SLOTS.get(name)
        verdict = "OK" if decl_n == dedup else f"MISMATCH ({decl_n} vs {dedup})"
        rows.append((name, caller, raw, dedup, f"{src}:{decl}",
                     len(slots) if slots else "-", verdict))
    w = max(len(r[0]) for r in rows)
    print(f"{'callee':{w}}  caller                  raw dedup  core-decl    ARG_SLOTS verdict")
    for r in sorted(rows):
        print(f"{r[0]:{w}}  {r[1]:22}  {r[2]:>3} {r[3]:>5}  {r[4]:12} {str(r[5]):>9} {r[6]}")


if __name__ == "__main__":
    main()
