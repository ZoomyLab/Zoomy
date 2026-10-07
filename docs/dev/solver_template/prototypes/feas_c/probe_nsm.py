import sys
sys.path.insert(0, ".")
from emit_raw import build_nsm
nsm = build_nsm(2)
for a in ("state", "aux_state", "aux_variables", "parameters", "parameter_values", "dt_max", "reconstruction", "nonconservative_matrix", "flux", "source"):
    v = getattr(nsm, a, "<missing>")
    try:
        s = str(list(v) if hasattr(v, "__iter__") and not isinstance(v, (str, dict)) else v)
    except Exception as e:
        s = repr(v)
    print(f"{a}: {s[:400]}")
