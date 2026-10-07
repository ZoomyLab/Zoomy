import sys
sys.path.insert(0, ".")
from emit_raw import build_nsm
from zoomy_core.solver.march import emit_march
from zoomy_core.transformation.procedure_c import CProcedurePrinter
nsm = build_nsm(1)
print("nsm.dt_max =", getattr(nsm, "dt_max", "<missing>"))
print("nsm.eigenvalues type:", type(getattr(nsm, "eigenvalues", None)))
ev = getattr(nsm, "eigenvalues", None)
print("eigenvalues:", ev if ev is None else list(ev)[:3])
prog = emit_march(nsm, cfl=0.9, dimension=1)
txt = CProcedurePrinter().print_procedure(prog.march)
print([l for l in txt.splitlines() if "c_dt_max" in l or "c_eps_h" in l])
