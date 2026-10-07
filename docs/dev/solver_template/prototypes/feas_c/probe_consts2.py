import sys, types
sys.path.insert(0, ".")
from emit_raw import build_nsm
from zoomy_core.solver.march import build_march
from zoomy_core.solver.constants import march_constants
from zoomy_core.transformation.procedure_c import CProcedurePrinter
nsm = build_nsm(1)
proxy = types.SimpleNamespace(**{k: getattr(nsm, k) for k in ("state", "eigenvalues", "parameters", "parameter_values")})
proxy.dt_max = None
consts = march_constants(proxy, cfl=0.9, dimension=1)
m = build_march(nsm, consts).resolve(dict(coupled=False, dt_floor=False, write_output=True))
txt = CProcedurePrinter().print_procedure(m)
print([l.strip() for l in txt.splitlines() if "c_dt_max" in l])
