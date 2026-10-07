"""F5: PositiveNonconservativeRusanov batched vs per-face numerical_flux /
numerical_fluctuations on random wet/dry states incl. h=0 and h=1e-12."""
import warnings, numpy as np
from common import *
from zoomy_core.fvm.solver_numpy import _build_free_surface_numerics
from zoomy_core.fvm.riemann_solvers import NonconservativeRusanov
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import Newtonian, NavierSlip, StressFree
warnings.simplefilter("ignore")
rng = np.random.default_rng(1)

def states(nv, nf, dim):
    h = rng.uniform(0, 2, nf)
    special = rng.choice(nf, size=nf // 2, replace=False)
    h[special[: nf // 6]] = 0.0
    h[special[nf // 6: nf // 3]] = 1e-12
    h[special[nf // 3:]] = rng.choice([1e-14, 1e-8, 1e-3, 0.0099, 0.0101], size=len(special) - nf // 3)
    b = rng.uniform(-0.5, 0.5, nf)
    q = [rng.uniform(-2, 2, nf) * np.where(h > 0, 1.0, rng.choice([0.0, 1e-9], nf)) for _ in range(nv - 2)]
    return np.vstack([b, h] + q)

for label, model in (("SWE 1-D", SWE(dimension=1)), ("SWE 2-D", SWE(dimension=2)),
                     ("SME(level=1) 1-D", SME(level=1, dimension=2, closures=[Newtonian(), NavierSlip(), StressFree()],
                                              parameters={"nu": 1e-6, "lambda_s": 0.0}))):
    nsm = build_nsm(model, order=1); sm = nsm.sm
    rt_model = NumpyRuntimeModel.from_system_model(sm)
    p = np.array(list(sm.parameter_values.values()), float)
    dim = sm.dimension; nv = len(sm.state); na = sm.n_aux_variables
    for cls_name, numerics in (("PositiveNonconservativeRusanov", _build_free_surface_numerics(sm)),
                               ("NonconservativeRusanov(batched-validated)", NonconservativeRusanov(sm))):
        rn = numerics.to_runtime_numpy()
        nf = 3000
        qA, qB = states(nv, nf, dim), states(nv, nf, dim)
        # half of the pairs: same bed (lake-at-rest-ish), rest random bed step
        qB[0, : nf // 2] = qA[0, : nf // 2]
        def aux_of(Q):
            if na == 0: return np.zeros((0, Q.shape[1]))
            A0 = np.zeros((na, Q.shape[1]))
            out = getattr(rt_model, "update_aux_variables", None)
            if out is None: return A0
            return np.asarray(out(Q, A0, p, 0.0, np.zeros(3)), float).reshape(na, -1)
        aA, aB = aux_of(qA), aux_of(qB)
        th = rng.uniform(0, 2 * np.pi, nf)
        n = np.vstack([np.cos(th), np.sin(th)])[:dim] if dim == 2 else rng.choice([-1.0, 1.0], (1, nf))
        res = {}
        for kname in ("numerical_flux", "numerical_fluctuations"):
            fn = getattr(rn, kname)
            nrow = nv if kname == "numerical_flux" else 2 * nv
            with np.errstate(all="ignore"):
                loop = np.array([np.asarray(fn(qA[:, i], qB[:, i], aA[:, i], aB[:, i], p, n[:, i]), float).reshape(-1) for i in range(nf)]).T
                loop1 = np.array([np.asarray(fn(qA[:, [i]], qB[:, [i]], aA[:, [i]], aB[:, [i]], p, n[:, [i]]), float).reshape(-1) for i in range(nf)]).T
                try:
                    bat = np.asarray(fn(qA, qB, aA, aB, p, n), float).reshape(nrow, -1)
                except Exception as e:
                    res[kname] = f"FAILED {type(e).__name__}: {str(e)[:90]}"; continue
            nanl, nanb = ~np.isfinite(loop), ~np.isfinite(bat)
            same_nan = np.array_equal(nanl, nanb)
            fin = ~nanl & ~nanb
            eq = np.array_equal(np.where(fin, bat, 0), np.where(fin, loop, 0)) and same_nan
            nd = int((bat[fin] != loop[fin]).sum())
            md = np.abs(bat[fin] - loop[fin]).max() if fin.any() else 0.0
            dry = (qA[1] <= 1e-12) | (qB[1] <= 1e-12)
            rel = (np.abs(bat[fin] - loop[fin]) / np.maximum(np.abs(loop[fin]), 1e-300)).max() if fin.any() else 0.0
            eq1 = np.array_equal(bat, loop1)
            mag = np.abs(loop[fin]).max()
            res[kname] = (f"bitwise vs scalar loop={eq} nonfinite={int(nanl.sum())}/{int(nanb.sum())} "
                          f"differing={nd}/{fin.sum()} max|d|={md:.1e} max rel={rel:.1e} (max|val|={mag:.1e}); "
                          f"bitwise vs loop of (nv,1)-column calls={eq1}")
        print(f"{label:16s} {cls_name:42s} supports_batched_faces={getattr(numerics, 'supports_batched_faces', None)}")
        for k, v in res.items():
            print(f"      {k:24s} {v}")
