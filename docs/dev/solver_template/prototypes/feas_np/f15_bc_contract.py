"""F15: BC kernel contract — FACE value vs GHOST value on a linear field."""
import numpy as np, sympy as sp
from common import *
import jax.numpy as jnp

TAGS = {1: ("left", "right"), 2: ("left", "right", "bottom", "top")}
G = 9.81

def run(dim, nmesh):
    if dim == 1:
        mesh = LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=nmesh)
    else:
        mesh = LSQMesh.create_2d(domain=(0.0, 1.0, 0.0, 1.0), nx=nmesh, ny=nmesh - 1)
    geo = bf_geometry(mesh, dim)
    xL, xR = 0.0, 1.0
    # linear field u_k = a_k + b_k x (x only, so a per-tag constant can match it on left/right)
    a = np.array([0.1, 1.0, 5.0, 0.3][:dim + 2])     # b, h, hu, (hv)   -> supercritical u~5
    bcoef = np.array([0.05, 0.4, -0.7, 0.2][:dim + 2])
    uf = lambda x: a + bcoef * x
    ii = {"b": 0, "h": 1, "hu": 2, "hv": 3}
    hu = 2
    def lin_presc(x):   # prescribed values at a left/right face (all slots h, hu)
        u = uf(x); return {1: float(u[1]), 2: float(u[2])}
    lam_fields = {1: lambda t, X, dX, Q, Qa, p, n: a[1] + bcoef[1] * X[0],
                  2: lambda t, X, dX, Q, Qa, p, n: a[2] + bcoef[2] * X[0]}
    tags = TAGS[dim]
    other = [BC.Extrapolation(tag=t) for t in tags[2:]]
    cases = {
      "Extrapolation":   ([BC.Extrapolation(tag=t) for t in tags], "lin", [1, 2]),
      "Periodic":        ([BC.Periodic(tag="left", periodic_to_physical_tag="right"),
                           BC.Periodic(tag="right", periodic_to_physical_tag="left")] + other, "lin", [1, 2]),
      "Coupled":         ([BC.Coupled(tag=t, mesh_name="m") for t in tags[:2]] + other, "lin", [1, 2]),
      "Dirichlet(on=h)": ([BC.Dirichlet("left", on="h", value=lin_presc(xL)[1]),
                           BC.Dirichlet("right", on="h", value=lin_presc(xR)[1])] + other, "lin", [1]),
      "InflowOutflow":   ([BC.InflowOutflow(tag="left", prescribe_fields=lin_presc(xL)),
                           BC.InflowOutflow(tag="right", prescribe_fields=lin_presc(xR))] + other, "lin", [1, 2]),
      "Lambda(a+b*X0)":  ([BC.Lambda(tag=t, prescribe_fields=lam_fields) for t in tags[:2]] + other, "lin", [1, 2]),
      "FromData":        ([BC.FromData(tag="left", timeline=np.array([0.0, 1.0]),
                                       prescribe_fields={k: np.array([v, v]) for k, v in lin_presc(xL).items()}),
                           BC.FromData(tag="right", timeline=np.array([0.0, 1.0]),
                                       prescribe_fields={k: np.array([v, v]) for k, v in lin_presc(xR).items()})] + other, "lin", [1, 2]),
      "Characteristic(target=lin, supercrit in at left)":
                          ([BC.Characteristic(tag=t, prescribe_fields={k: (lambda kk: (lambda tt, X, dX, Q, Qa, p, n: a[kk] + bcoef[kk] * X[0]))(k) for k in (1, 2)})
                            for t in tags[:2]] + other, "lin", [1, 2]),
      "Wall":            ([BC.Wall(tag=t) for t in tags], "wall", [2]),
      "RoughWall":       ([BC.RoughWall(tag=t) for t in tags], "wall", [2]),
      "FromModel(wall)": ([BC.FromModel(tag=t, definition="wall") for t in tags], "wall", [2]),
      "CharacteristicWall": ([BC.CharacteristicWall(tag=t) for t in tags], "wall", [1, 2]),
    }
    out_rows = []
    for name, (bcl, kind, slots) in cases.items():
        try:
            model = SWE(dimension=dim, boundary_conditions=BC.BoundaryConditions(bcl))
            nsm = build_nsm(model)
            rt_np, rt_jx, p = runtimes(nsm)
            bc_tags = list(nsm.sm._bc_source.list_sorted_function_names)
        except Exception as e:
            out_rows.append((name, f"BUILD FAIL {type(e).__name__}: {str(e)[:120]}")); continue
        res = {}
        for (backend, side), fn in [((b_, sd), f_) for b_, f_ in (("numpy", rt_np.boundary_conditions), ("jax", rt_jx.boundary_conditions)) for sd in ("left", "right")]:
            ef, eg, ec = [], [], []
            for i in range(geo["n_bf"]):
                tag = geo["tag"][i]
                if tag != side:
                    continue
                br = bc_tags.index(tag)                       # route by NAME (contract test only)
                xc, xf = geo["cc"][i, 0], geo["fc"][i, 0]
                if kind == "lin":
                    qc = uf(xc); qf = uf(xf)
                else:   # momentum field vanishing at THIS face: q = s (x - x_f); h,b linear
                    qc = uf(xc).copy(); qf = uf(xf).copy()
                    qc[hu] = 2.0 * (xc - xf); qf[hu] = 0.0
                    qc[1] = 1.0 + 0.4 * xc; qf[1] = 1.0 + 0.4 * xf   # h ~ 1, slow flow (subcritical)
                    if dim == 2: qc[3] = 0.0; qf[3] = 0.0
                qg = 2 * qf - qc
                aux = np.zeros(nsm.sm.n_aux_variables)
                if nsm.sm.n_aux_variables: aux[-1] = 1.0 / qc[1]
                o = np.asarray(fn(br, 0.0, geo["fc"][i], geo["d"][i], qc, aux, p, geo["nrm"][:, i]),
                               dtype=float).ravel()[:dim + 2]
                ef.append(np.abs(o[slots] - qf[slots]).max())
                eg.append(np.abs(o[slots] - qg[slots]).max())
                ec.append(np.abs(o[slots] - qc[slots]).max())
            ef, eg, ec = max(ef), max(eg), max(ec)
            tol = 1e-12
            verdict = ("FACE" if ef < tol else "GHOST" if eg < tol else
                       "IDENTITY(=cell)" if ec < tol else "MIXED")
            res[backend + "/" + side] = (ef, eg, ec, verdict)
        out_rows.append((name, res))
    return out_rows

for dim, n in ((1, 10), (2, 6)):
    print(f"\n=== create_{dim}d  (slots tested; |out-u_face|, |out-u_ghost|, |out-u_cell| maxed over left/right faces) ===")
    for name, res in run(dim, n):
        if isinstance(res, str):
            print(f"{name:52s} {res}"); continue
        same = all(res["numpy/" + sd][3] == res["jax/" + sd][3] and abs(res["numpy/" + sd][0] - res["jax/" + sd][0]) < 1e-14 for sd in ("left", "right"))
        s = "  ".join(f"{sd}: f={res['numpy/'+sd][0]:.2e} g={res['numpy/'+sd][1]:.2e} c={res['numpy/'+sd][2]:.2e} -> {res['numpy/'+sd][3]}" for sd in ("left", "right")) + f"   numpy==jax: {same}"
        print(f"{name:52s} {s}")
