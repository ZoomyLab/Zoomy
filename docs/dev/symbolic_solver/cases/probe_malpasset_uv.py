import sys
sys.argv = ["x", "2"]
import malpasset as MP
import sympy as sp
mesh, b = MP.build()
nsm = b["nsm"]
print("state", [str(s) for s in nsm.state], "e2s", list(nsm.equation_to_state_index))
uv = nsm.update_variables
print("shape", getattr(uv, "shape", None))
for i, e in enumerate(sp.flatten(uv.tolist() if hasattr(uv, "tolist") else uv)):
    print(i, str(e)[:300])
print("aux", [str(s) for s in nsm.aux_state])
f = nsm.operator("update_variables")
print("args", {k: [str(x) for x in (v.values() if hasattr(v, "values") else [v])] for k, v in f.args.items()})
