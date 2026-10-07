"""F6 demo: the print-time call check, run against core's OWN declarations
(ExternalProcedure.args/results; doc "(..)->(..)" where args are empty).
Unknown doc names take the kind of the actual argument (arity+mutability
only).  Also: the F4 check on a None-valued constant (c_dt_max)."""
import sys, types, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from emit_raw import build_nsm
from arity_report import doc_sig, calls
from zoomy_core.solver.arg_slots import SOLVER_ARG_KINDS, SOLVER_ARG_MAPPING
from zoomy_core.solver.external import REQUIRED_PROCEDURES
from zoomy_core.solver.march import emit_march, build_march
from zoomy_core.solver.constants import march_constants
from feas_printer import FeasCProcedurePrinter, CallSignatureError
from feas_tables import kind_of, TYPED

def core_sigs():
    out = {}
    for n, ep in REQUIRED_PROCEDURES.items():
        if ep.args or ep.results:
            ins = [SOLVER_ARG_MAPPING[a] for a in ep.args]
            outs = [SOLVER_ARG_MAPPING[r] for r in ep.results]
            written = {SOLVER_ARG_MAPPING[a] for a in ep.args
                       if SOLVER_ARG_KINDS[a] in ("real_ptr", "int_ptr")}
        else:
            ins, outs = doc_sig(ep.doc) or ([], [])
            written = set()
        names = ins + [o for o in outs if o not in ins]
        out[n] = tuple((x, None, x in outs or x in written) for x in names)
    return out

class CorePrinter(FeasCProcedurePrinter):
    def check_call(self, call):
        params = self.callee_params(call.procedure)
        # unknown kinds: adopt the actual's (arity + mutability only)
        actual = [self.cname(a) for a in call.args]
        actual += [r for r in map(self.cname, call.results) if r not in actual]
        if call.procedure in self.externals and len(actual) == len(params):
            self.externals[call.procedure] = tuple(
                (pn, kind_of(a), pm) for a, (pn, _, pm) in zip(actual, params))
        return super().check_call(call)

errs = []
nsm2 = build_nsm(2)
prog = emit_march(nsm2, cfl=0.9, dimension=1, dt_max=1.0, mood=True, coupled=True,
                  flags=dict(implicit_source=True, implicit_diffusion=True))
procs = [prog.should_write, prog.step, prog.march]
pr = CorePrinter(procedures=procs, externals=core_sigs())
for proc in procs:
    for c in calls(proc.stmts):
        try:
            pr.check_call(c)
        except CallSignatureError as e:
            if str(e) not in errs:
                errs.append(str(e))
print(f"F6 vs CORE declarations: {len(errs)} call-site errors")
for e in errs:
    print("  -", e)

# F4 on a None-valued constant
proxy = types.SimpleNamespace(**{k: getattr(nsm2, k) for k in
        ("state", "eigenvalues", "parameters", "parameter_values")}, dt_max=None)
m = build_march(nsm2, march_constants(proxy, cfl=0.9, dimension=1)).resolve(
    dict(coupled=False, dt_floor=False, write_output=True))
try:
    FeasCProcedurePrinter(procedures=[prog.should_write, prog.step, m]).print_procedure(m)
    print("F4: no error (unexpected)")
except CallSignatureError as e:
    print("F4 with nsm.dt_max=None:", e)
