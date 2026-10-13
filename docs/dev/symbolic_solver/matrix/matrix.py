"""Cross-backend acceptance matrix of the symbolic solvers (README.md).

    python matrix.py run BACKEND CASE [CONFIG] [--force]   # runs -> store/
    python matrix.py table                                 # store -> table.md
    python matrix.py run --tier large CASE [ARGS]          # large.py (never by default)

BACKEND: jax foam amrex.  CASE: C1 ... C34 (cases.py).  CONFIG: serial
(default) mpi2 mpi4 gpu1 gpu2 (= gpu-mpi).  A run in the store is skipped
(--force reruns).  A run that raises NotImplementedError is stored as
"unsupported", any other exception as a failure.
"""
import functools
import json
import os
import subprocess
import sys
import time as clock
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gates"))
import common  # noqa: E402,F401  (x64)
from cases import (CASES, CHANNEL, C10_N, C10_VARIANTS, C31_K, C31_SWE, C31_VAM, C4_N,  # noqa: E402
                   DIFF_ALL, DIFF_KINDS, DIFF_PATHS, NA, PERTURB, prepare)
from backends import BACKENDS  # noqa: E402

STORE = Path(os.environ.get("MATRIX_STORE", HERE / "store"))
WORK = Path(os.environ.get("MATRIX_WORK", HERE / "work"))
CONFIGS = ("serial", "mpi2", "mpi4", "gpu1", "gpu2")
HEAD = {"gpu2": "gpu-mpi"}


def path(backend, config, case, key):
    return STORE / backend / config / case / f"{key}.npz"


def load(backend, config, case, key):
    p = path(backend, config, case, key)
    return dict(np.load(p, allow_pickle=True)) if p.exists() else None


def provenance():
    """The library commits a run used (PYTHONPATH entries; `MATRIX_PROVENANCE`: a JSON file
    with them, for a library copy without git, e.g. mounted into a container)."""
    if os.environ.get("MATRIX_PROVENANCE"):
        return Path(os.environ["MATRIX_PROVENANCE"]).read_text()
    out = {}
    for d in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if d:
            r = subprocess.run(["git", "-C", d, "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True)
            out[Path(d).name] = r.stdout.strip()
    return json.dumps(out)


def _rank0():
    if "OMPI_COMM_WORLD_RANK" in os.environ or "PMI_RANK" in os.environ:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank() == 0
    return True


def run(backend, case, config="serial", force=False):
    b = BACKENDS[backend]()
    runs = [r for r in CASES[case][1] if backend in r.backends and config in r.configs]
    if config not in b.configs or not runs:
        print(f"{backend} {config} {case}: n/a")
        return
    child = os.environ.get("MATRIX_CHILD")
    cmd = None if child else b.launch(config)
    if cmd and any(r.launch for r in runs):     # re-run under mpiexec / another env
        env = dict(os.environ, MATRIX_CHILD="1")
        argv = cmd + [__file__, "run", backend, case, config] + (["--force"] if force else [])
        subprocess.run(argv, env=env)
    t_start, longest = clock.perf_counter(), 0.0
    budget = float(os.environ.get("MATRIX_BUDGET", "inf"))   # s per command (env.sh: 285)
    for r in runs:
        if (cmd and r.launch) or (child and not r.launch):
            continue
        if clock.perf_counter() - t_start + longest > budget:
            print(f"{backend} {config} {case}: budget reached, rerun to continue", flush=True)
            break
        p = path(backend, config, case, r.key)
        if p.exists() and not force:
            continue
        work = WORK / backend / config / case / r.key
        t0 = clock.perf_counter()
        err, uns = p.with_suffix(".err"), p.with_suffix(".unsupported")
        try:
            if r.custom is not None:
                out = r.custom(backend, config, work)
            else:
                mesh, systems, Q0, names = prepare(r)
                out = b.run(r, mesh, systems, Q0, config, work)
        except Exception as e:                # a backend failure is a result
            msg = f"{type(e).__name__}: {e}"
            first = [ln for ln in msg.splitlines() if "error:" in ln][:3]
            f = uns if isinstance(e, NotImplementedError) else err
            if _rank0():
                p.parent.mkdir(parents=True, exist_ok=True)
                f.write_text("\n".join(first + ["..."]) + "\n" + msg[-3000:])
            print(f"{backend} {config} {case} {r.key}: {f.suffix[1:].upper()} ({msg[:150]})",
                  flush=True)
            continue
        wall = clock.perf_counter() - t0
        longest = max(longest, wall)
        if out is None or not _rank0():      # non-root MPI rank
            continue
        err.unlink(missing_ok=True)
        uns.unlink(missing_ok=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        if r.custom is not None:
            np.savez(p, metrics=json.dumps(out), wall=wall, lib=provenance())
            print(f"{backend} {config} {case} {r.key}: {out} [{wall:.1f}s]", flush=True)
            continue
        nc = Q0.shape[1]
        np.savez(p, t=out[0], Q=out[1], Q0=Q0, names=np.array(names), wall=wall, lib=provenance(),
                 xc=np.asarray(mesh.cell_centers)[:3, :nc],
                 vol=np.asarray(mesh.cell_volumes)[:nc])
        print(f"{backend} {config} {case} {r.key}: t={out[0]:.6g} [{wall:.1f}s]", flush=True)


# ── measures ────────────────────────────────────────────────────────────────
def names(d):
    return [str(n) for n in d["names"]]


def h(d):
    """The depth (SME) or the density (Euler)."""
    n = names(d)
    return d["Q"][n.index("h") if "h" in n else n.index("rho")]


def relmax(a, b):
    return float(np.abs(a["Q"] - b["Q"]).max() / max(np.abs(b["Q"]).max(), 1e-300))


def l1rel(a, b):
    i = [k for k, n in enumerate(names(b)) if n != "b"]
    return float(np.abs(a["Q"][i] - b["Q"][i]).sum() / np.abs(b["Q"][i]).sum())


def mass(d, row=None):
    n = names(d)
    i = row if row is not None else (n.index("h") if "h" in n else n.index("rho"))
    m0 = float((d["Q0"][i] * d["vol"]).sum())
    return abs(float((d["Q"][i] * d["vol"]).sum()) - m0) / abs(m0)


def _grid(xc_grid, xc):
    """Cell index of points ``xc`` on the Cartesian grid of centres ``xc_grid``."""
    idx, stride = np.zeros(xc.shape[1], dtype=int), 1
    for ax in range(min(xc_grid.shape[0], xc.shape[0])):
        u = np.unique(np.round(xc_grid[ax], 9))
        d = u[1] - u[0] if len(u) > 1 else 1.0
        idx += stride * np.clip(np.floor((xc[ax] - u[0] + d / 2) / d), 0, len(u) - 1).astype(int)
        stride *= len(u)
    return idx, stride


def cartesian(d):
    return len(np.unique(_grid(d["xc"], d["xc"])[0])) == d["xc"].shape[1]


def err_ref(d, ref):
    """L1(h) vs a fine Cartesian reference: averaged onto d's grid, or (d on
    triangles) sampled at d's cell centres."""
    if cartesian(d):
        idx, n = _grid(d["xc"], ref["xc"])
        avg = np.bincount(idx, h(ref), n) / np.bincount(idx, None, n)
        return float(np.abs(h(d) - avg[_grid(d["xc"], d["xc"])[0]]).mean())
    return float(np.abs(h(d) - h(ref)[_grid(ref["xc"], d["xc"])[0]]).mean())


def err_exact(d, variant):
    from cases import bernoulli
    return float(np.abs(h(d) - bernoulli(variant, d["xc"][0])).mean())


def order(e_coarse, e_fine, ratio=2.0):
    return float(np.log(e_coarse / e_fine) / np.log(ratio)) if e_fine > 0 else float("inf")


def metrics(d):
    return json.loads(str(d["metrics"]))


# ── checks ──────────────────────────────────────────────────────────────────
# A threshold is (op, value, source); source: theory | tolerance | spec |
# observed (about 2x the worst value seen, README).  None: information.
RO = ("<=", 1e-12, "tolerance (round-off)")
RO_L1 = ("<=", 1e-11, "tolerance (round-off, L1 rel): foam = JAX")
PRESS = ("<=", 1e-9, "tolerance (10 x pressure rtol 1e-10, over the steps)")
TH = {
    "C1 O1": (">=", 0.6, "observed 0.71 (pre-asymptotic, theory 1)"),
    "C1 O2": (">=", 1.8, "theory 2"),
    "C11 O1": (">=", 0.9, "theory 1"),
    "C11 O2": (">=", 1.5, "observed 1.67-1.70 on JAX and foam (1-D minmod; theory 2)"),
    "C12 O1": (">=", 0.5, "observed 0.65 (coarse triangles, theory 1)"),
    "C12 O2": (">=", 1.4, "observed 1.69 (coarse triangles, theory 2)"),
    "C2": (">=", 0.6, "spec"),
    "C2 foam": ("<=", 1.5e-3, "observed (foam 7.1e-4)"),
    "C3": ("<=", 1e-12, "spec (well-balanced: round-off)"),
    "C4 order": {("sub", 1): (">=", 0.45, "observed 0.58, pre-asymptotic (theory 1)"),
                 ("sub", 2): (">=", 1.8, "theory 2"),
                 ("trans", 1): (">=", 0.55, "observed 0.71, pre-asymptotic (theory 1)"),
                 ("trans", 2): (">=", 1.8, "theory 2")},
    "C5 foam": ("<=", 3.5e-2, "observed (foam 1.7e-2)"),
    "C5 amrex": ("<=", 1e-9, "observed 6e-10 (wet/dry eigenvalue cluster)"),
    "C8 foam": ("<=", 3e-2, "observed (foam 1.6e-2)"),
    "C10": {(1, 1): (">=", 0.8, "theory 1"), (2, 1): (">=", 0.8, "theory 1 (time)"),
            (2, 2): (">=", 1.5, "observed (theory 2, minmod)")},
    "C14 O1": (">=", 0.5, "observed 0.63 (pre-asymptotic, theory 1)"),
    "C14 O2": (">=", 0.8, "theory 1 (Lie splitting)"),
    "shrink": (">=", 0.5, "observed: foam-JAX difference shrinks under refinement"),
    "shrink tri": (">=", 0.25, "observed 0.33-0.39 (O1 triangles: both converge slowly)"),
    "newton": ("<=", 1e-8, "tolerance (Newton 1e-10 vs GMRES 1e-10, over the steps)"),
    "fd": ("<=", 1e-5, "tolerance (central FD, eps 1e-3..1e-5)"),
    "rev": ("<=", 1e-10, "tolerance (checkpointed reverse vs forward)"),
    "couple": ("<=", 1e-14, "tolerance (gate 26: coupled = single domain at O1)"),
}
CLOSED = ("C1", "C2", "C3", "C5", "C11", "C12", "C13", "C14", "C20", "C23", "C32", "C33", "C36")
TH.update({
    "C18 L1": {1: ("<=", 4.2e-2, "observed 2.09e-2 (3-D worker)"), 2: ("<=", 1.0e-2, "observed 5.13e-3")},
    "C18 order": {1: (">=", 0.45, "observed 0.54 (discontinuous: theory <= 1)"),
                  2: (">=", 0.8, "observed 0.90 (discontinuous: theory <= 1)")},
    "C19": {"o1": (">=", 0.75, "observed 0.85, pre-asymptotic"), "o2none": (">=", 1.8, "theory 2"),
            "o2minmod": (">=", 1.8, "theory 2 (observed 2.53)")},
    "C24 wave": {"o1": (">=", 0.85, "observed 0.96 (theory 1)"), "o2none": (">=", 1.8, "theory 2"),
                 "o2minmod": (">=", 1.7, "observed 1.83")},
    "fd3": ("<=", 1e-6, "observed 4.4e-8 (best eps of 1e-3..1e-6)"),
    "C23": (">=", 1.5, "observed 2.11 (different stencils, converging)"),
    "C31 swe": {"lie1": (">=", 0.8, "theory 1"),
                "lie2": ("<=", 1.4, "theory 1: Lie splitting caps SSP-RK2 (observed 1.22, the "
                                    "dt/32 reference lifts the last pair)"),
                "ssp2": (">=", 1.8, "theory 2")},
    "C31 vam": {"t1": (">=", 0.8, "theory 1"),
                "t2": (">=", 0.8, "theory 1 (the Lie source per stage; observed 1.12, increment "
                                  "projection, ANALYSIS 45e)"),
                "ssp2": (">=", 1.8, "theory 2 (increment projection, ANALYSIS 45e)"),
                "pred": (">=", 1.8, "theory 2")},
    "C34 order": (">=", 1.8, "theory 2"),
    "C34 err": ("<=", 6e-3, "observed 3.9e-3 (q_y), 2.3e-3 (q_x) on JAX vs the exact div(h tau')"),
    "C34 xb": ("<=", 2e-7, "observed 7e-8 foam (its own O2 stencils in the explicit stage: O(dt) in L), "
                           "6e-10 AMReX (round-off / dt)"),
    "C32 order": (">=", 1.8, "theory 2"),
    "C32 err": ("<=", 7e-4, "observed 3.3e-4 on all three (the O(dy^2) error of the viscous flux)"),
    "C32 control": (">=", 0.1, "observed 0.42 (a wall that acts: the no-slip profile is not a free-slip solution)"),
    "C32 inviscid": ("<=", 4e-4, "observed 1.7e-4 (a regression bound; 2.5e-3 at g = 9.81)"),
    "C32 inviscid split": ("<=", 8e-4, "observed 4.1e-4 (a regression bound: O2 Rusanov dissipation, no wall exit)"),
    "C32 foam": ("<=", 0.5, "observed 0.21 (foam's own stencils at second order)"),
    "C33 err": {"imex": ("<=", 1e-3, "observed 5.1e-4 (implicit Euler, theory O(dt))"),
                "rk2": ("<=", 7e-6, "observed 3.3e-6 (SSP-RK2, theory O(dt^2))")},
    "C33 scheme": {"imex": ("<=", 1e-10, "tolerance (Newton 1e-12; observed 3e-12 C++, 4e-15 JAX)"),
                   "rk2": RO},
    "C33 xb": {"imex": ("<=", 1e-10, "tolerance (Newton 1e-12; observed 1.3e-12)"), "rk2": RO},
})


def euler_checks(add, b, g, J):
    """C18-C24: compressible Euler (exact Riemann / smooth wave, symmetry,
    conservation, cross-backend)."""
    import euler_common as E
    ok = lambda *ds: all(x is not None for x in ds)  # noqa: E731

    def sod_err(d, axis):
        rho_ex = E.exact_riemann(E.SOD[0], E.SOD[1], (d["xc"][axis] - 0.5) / float(d["t"]))[0]
        return float(np.abs(d["Q"][0] - rho_ex).mean())

    def profile(d, axis):
        key = np.lexsort(tuple(np.round(d["xc"][k], 9) for k in range(3) if k != axis)
                         + (np.round(d["xc"][axis], 9),))
        q = d["Q"][:, key].copy()
        q[[1, 1 + axis]] = q[[1 + axis, 1]]          # the axis momentum first
        return q
    for case, dim, ns, thr_L1 in (("C18", 3, (50, 100), True), ("C24", 2, (50, 100), False)):
        pre = "sod_" if case == "C24" else ""
        for o in (1, 2):
            e = [g(case, f"{pre}o{o}_n{n}_ax0") for n in ns]
            if ok(*e):
                es = [sod_err(x, 0) for x in e]
                add(case, f"Sod O{o} order {ns[0]}->{ns[1]}", b, "serial", order(*es),
                    TH["C18 order"][o], "L1(rho) " + " / ".join(f"{x:.2e}" for x in es))
                add(case, f"Sod O{o} L1(rho) n{ns[1]}", b, "serial", es[1],
                    TH["C18 L1"][o] if thr_L1 else None, "")
            runs = [g(case, f"{pre}o{o}_n{ns[1]}_ax{a}") for a in range(dim)]
            if ok(*runs):
                ps = [profile(x, a) for a, x in enumerate(runs)]
                add(case, f"Sod O{o} x / y / z profiles (max)", b, "serial",
                    max(float(np.abs(p - ps[0]).max()) for p in ps[1:]), RO, "")
                add(case, f"Sod O{o} max|u_perp|", b, "serial",
                    max(float(np.abs(np.delete(x["Q"][1:1 + dim], a, 0) / x["Q"][0]).max())
                        for a, x in enumerate(runs)), RO, "")
        ns_w = (8, 16, 24) if dim == 3 else (16, 32, 64)
        wcase, thw = ("C19", TH["C19"]) if dim == 3 else ("C24", TH["C24 wave"])
        for v in ("o1", "o2none", "o2minmod"):
            e = [g(wcase, f"{v}_n{n}") for n in ns_w]
            if ok(*e):
                vel = (1.0, 0.5, 0.25)[:dim]
                es = [float(np.abs(x["Q"][0] - E.wave_exact(x["xc"], float(x["t"]), vel)).mean())
                      for x in e]
                add(wcase, f"wave {v} order {ns_w[1]}->{ns_w[2]}", b, "serial",
                    order(es[1], es[2], ns_w[2] / ns_w[1]), thw[v],
                    "L1(rho) " + " / ".join(f"{x:.2e}" for x in es))
                rho, u, pr = E.primitive(e[-1]["Q"])
                add(wcase, f"wave {v} max|u - u0|, |p - 1|", b, "serial",
                    max(float(np.abs(u - np.asarray(vel)[:, None]).max()), float(np.abs(pr - 1).max())),
                    RO, "contact: velocity and pressure constant")
    for o in (1, 2):                            # C20: conservation and symmetry
        d = g("C20", f"o{o}")
        if d is None:
            continue
        add("C20", f"O{o} energy change (rel)", b, "serial", mass(d, -1), RO, "")
        n = int(round(d["xc"].shape[1] ** (1 / 3)))
        ijk = np.rint(d["xc"] * n - 0.5).astype(int)
        lin = lambda a: (a[0] * n + a[1]) * n + a[2]   # noqa: E731
        pos = np.empty(n ** 3, dtype=int)
        pos[lin(ijk)] = np.arange(d["Q"].shape[1])
        imgs = ((n - 1 - ijk[0], ijk[1], ijk[2]), (ijk[0], n - 1 - ijk[1], ijk[2]),
                (ijk[0], ijk[1], n - 1 - ijk[2]), (ijk[1], ijk[0], ijk[2]), (ijk[2], ijk[1], ijk[0]))
        sym = max(float(np.abs(d["Q"][r] - d["Q"][r][pos[lin(np.asarray(im))]]).max())
                  for im in imgs for r in (0, -1)) / float(np.abs(d["Q"]).max())
        add("C20", f"O{o} mirror / permutation symmetry (max rel)", b, "serial", sym, RO, "")
    d, j = g("C26", "visc"), J("C26", "visc")
    if b not in ("jax", "numpy") and ok(d, j):  # C26: implicit diffusion vs JAX
        if b == "amrex":
            add("C26", "max rel vs JAX", b, "serial", relmax(d, j),
                ("<=", 1e-8, "tolerance (Newton 1e-12, GMRES 1e-13)"), "")
        else:
            add("C26", "L1 rel vs JAX", b, "serial", l1rel(d, j), TH.get("C26 foam"), "")
    if b not in ("jax", "numpy"):               # C23: cross-backend
        for n in (8, 16, 24):
            if ok(g("C23", f"o1_n{n}"), J("C23", f"o1_n{n}")):
                add("C23", f"O1 n{n} max rel vs JAX", b, "serial",
                    relmax(g("C23", f"o1_n{n}"), J("C23", f"o1_n{n}")), RO, "same stencil")
        d = [(g("C23", f"o2_n{n}"), J("C23", f"o2_n{n}")) for n in (16, 24)]
        if b == "amrex" and all(ok(*x) for x in d):
            add("C23", "O2 max rel vs JAX", b, "serial", max(relmax(*x) for x in d), RO, "")
        elif all(ok(*x) for x in d):
            dl = [float(np.abs(a["Q"] - j["Q"]).mean()) for a, j in d]
            add("C23", "O2 L1|backend - JAX| order 16->24", b, "serial", order(*dl, 1.5),
                TH["C23"], " / ".join(f"{x:.2e}" for x in dl))
    for c in CONFIGS:                           # C22: AD
        for k in ("o1_dt", "o1_adaptive", "o2_dt"):
            f, r = g("C22", f"{k}_fwd", c), g("C22", f"{k}_rev", c)
            if f is not None:
                add("C22", f"{k} |FD - jvp| (rel)", b, c, metrics(f)["fd_rel"], TH["fd3"], "")
            if ok(f, r):
                fw = metrics(f)["fwd"]
                add("C22", f"{k} |rev - jvp| (rel)", b, c, abs(metrics(r)["rev"] - fw) / abs(fw), RO, "")


def gate_checks(add, b, g, J):
    """C27-C30, C34: the wrapped gates 40 (SSP-RK order), 41 / 43 (diffusion blocks
    at the boundaries), 44 (LSQ boundary closure), 39 (LSQ polynomials), 45
    (in-plane viscous operator)."""
    SLACK = 0.2
    for part, lab, pair in (("ode", "ODE", "32->64 steps"), ("wave", "wave", "dt/2->dt/4")):   # C27
        for to in (1, 2, 3):
            d = g("C27", f"{part}_t{to}")
            if d is not None:
                m = metrics(d)
                add("C27", f"{lab} time_order {to} order {pair}", b, "serial", order(*m["err"][-2:]),
                    (">=", to - SLACK, f"theory {to}, slack {SLACK}"),
                    "err " + " / ".join(f"{x:.2e}" for x in m["err"]))
    d = g("C27", "ode_control")
    if d is not None:
        m = metrics(d)
        add("C27", "ODE control (RK3, wrong stage time) order", b, "serial", order(*m["err"][-2:]),
            ("<=", 2.5, "gate 40 control: a wrong stage time loses the order"),
            "err " + " / ".join(f"{x:.2e}" for x in m["err"]))
    for kind in DIFF_KINDS:                         # C28
        for path in DIFF_PATHS:
            d = g("C28", f"orders_{kind}_{path}")   # gate 41 (JAX): orders
            if d is not None:
                m = metrics(d)
                for case in ("dir0", "neu", "dirL", "per"):
                    add("C28", f"{kind} {path} {case} L_inf order 32->64", b, "serial",
                        order(*m[case][-2:]), (">=", 1.8, "theory 2, gate 41"),
                        "L_inf " + " / ".join(f"{x:.2e}" for x in m[case]))
                add("C28", f"{kind} {path} steady max|u - L|", b, "serial", m["steady"], RO,
                    "linear Dirichlet field kept to round-off")
            for case in DIFF_ALL:                   # gate 43: foam / AMReX / numpy vs JAX
                if b == "amrex" and kind == "tri":
                    if (path, case) == (DIFF_PATHS[0], DIFF_ALL[0]):
                        add("C28", "tri max rel vs JAX (n/a: AMReX takes Cartesian grids only)", b,
                            "serial", "n/a", None, "")
                    continue
                d, j = g("C28", f"{kind}_{path}_{case}"), J("C28", f"{kind}_{path}_{case}")
                if d is None or b == "jax":
                    continue
                m = metrics(d)
                bound = ("<=", 1e-12, "tolerance, gate 43: the same two-point fluxes and ghosts") \
                    if path == "expl" else ("<=", 1e-9, "tolerance, gate 43: Newton 1e-10, GMRES 1e-13")
                if j is not None:
                    u, uj = np.asarray(m["u"]), np.asarray(metrics(j)["u"])
                    add("C28", f"{kind} {path} {case} max rel vs JAX", b, "serial",
                        float(np.abs(u - uj).max() / np.abs(uj).max()), bound,
                        f"L_inf vs exact {m['err']:.2e}")
                if case == "steady":
                    add("C28", f"{kind} {path} steady max|u - L|", b, "serial", m["err"], RO,
                        "on the backend itself")
    for p in (2, 3):                                # C29
        for bc in ("dir", "neu"):
            d = g("C29", f"p{p}_{bc}")
            if d is None:
                continue
            m = metrics(d)
            for where in ("boundary", "interior"):
                for k in range(1, p + 1):
                    e = m[f"closure_{where}_d{k}"]
                    add("C29", f"p{p} {bc} d{k} {where} order 24->48", b, "serial", order(*e[-2:]),
                        (">=", p + 1 - k - SLACK, f"theory p+1-k = {p + 1 - k}, slack {SLACK}, gate 44"),
                        "max err " + " / ".join(f"{x:.2e}" for x in e)
                        + "; ghost as sample: " + " / ".join(f"{x:.2e}" for x in m[f"ghost_{where}_d{k}"]))
    ds = [g("C34", f"n{n}") for n in (16, 32, 64)]   # C34 (gate 45)
    if all(x is not None for x in ds):
        for k in ("q_x_0", "q_y_0"):
            e = [metrics(x)[f"{k} exact"] for x in ds]
            add("C34", f"{k}: vs the exact div(h tau') (mixed derivatives), order 32->64", b,
                "serial", order(*e[-2:]), TH["C34 order"],
                "max rel " + " / ".join(f"{x:.2e}" for x in e) + "; vs the SystemModel operator "
                + " / ".join(f"{metrics(x)[f'{k} full']:.2e}" for x in ds)
                + "; vs its diagonal-index part only "
                + " / ".join(f"{metrics(x)[f'{k} diag']:.2e}" for x in ds))
            add("C34", f"{k}: vs the exact div(h tau') at nx 64 (max rel)", b, "serial", e[-1],
                TH["C34 err"], "")
    d = g("C34", "n16")
    if b != "jax" and d is not None:
        m = metrics(d)
        add("C34", "nx 16 max rel |L - L_JAX|", b, "serial", m["rel_vs_jax"], TH["C34 xb"],
            "vs the exact div(h tau') " + " / ".join(f"{m[f'{k} exact']:.2e}" for k in ("q_x_0", "q_y_0")))
    d = g("C30", "polynomials")                     # C30
    if d is not None:
        m = metrics(d)
        add("C30", "gate 39 checks failing", b, "serial", m["fails"],
            ("<=", 0, f"spec, gate 39: exact to {m['tol']:g}"),
            f"PASS={m['passes']} FAIL={m['fails']} (tol {m['tol']:g}); the ghost as a plain sample "
            f"is inexact in {m['sample_inexact']} cases (sensitivity)")


def restart_checks(add, b, g):
    """C35: a run restarted at T/2 from the backend's own restart state = the run 0 -> T with a
    checkpoint at T/2, bitwise; a misspelt restart_from fails at once, naming the path."""
    for s in ("c1", "c8"):
        for c in CONFIGS:
            d = g("C35", f"{s}_restart", c)
            if d is not None:
                add("C35", f"{s}: restarted at T/2 vs 0 -> T (max abs)", b, c,
                    metrics(d)["max_abs"], ("<=", 0.0, "spec: bitwise"), "")
            d = g("C35", f"{s}_typo", c)
            if d is not None:
                m = metrics(d)
                add("C35", f"{s}: misspelt restart_from fails at once", b, c, m["clear"],
                    (">=", 1, "spec: FileNotFoundError naming restart_from"),
                    f"after {m['seconds']:.1f} s: {m['message']}")


def imex_checks(add, b, g, J):
    """C31: time order at a constant dt on a fixed mesh, L1(h) (SWE: vs dt/32
    of the same scheme; VAM: Richardson), and the C++ backends vs JAX."""
    ok = lambda *ds: all(x is not None for x in ds)  # noqa: E731
    for s in C31_SWE:
        e, ref = [g("C31", f"swe_{s}_k{k}") for k in C31_K], g("C31", f"swe_{s}_ref")
        if ok(ref, *e):
            es = [err_ref(x, ref) for x in e]
            add("C31", f"SWE {s} order dt/4->dt/8", b, "serial", order(*es[-2:]), TH["C31 swe"][s],
                "L1(h) vs dt/32 " + " / ".join(f"{x:.2e}" for x in es))
    for s in (*C31_VAM, "pred"):
        e = [g("C31", f"{'pred_ssp2' if s == 'pred' else 'vam_' + s}_k{k}") for k in C31_K]
        if ok(*e):
            d = [err_ref(e[k], e[k + 1]) for k in range(len(e) - 1)]
            add("C31", f"VAM {'predictor alone ssp2' if s == 'pred' else s} order (Richardson)", b,
                "serial", order(*d[-2:]), TH["C31 vam"][s],
                "L1(h) of dt/2^k - dt/2^(k+1) " + " / ".join(f"{x:.2e}" for x in d))
    for key, th in (("swe_ssp2_k0", RO), ("vam_ssp2_k0", PRESS)):
        d, j = g("C31", key), J("C31", key)
        if b not in ("jax", "numpy") and ok(d, j):
            add("C31", f"{key} max rel vs JAX", b, "serial", relmax(d, j), th, "")
@functools.lru_cache(maxsize=None)
def _moment_jacobian():
    import cases
    return cases.moment_jacobian()


def visc_checks(add, b, g, J):
    """C32 (viscous shear decay between walls, exact u_t = nu u_yy) and C33
    (SME(2) vertical viscosity of a uniform state, exact q(t) = expm(t J) q0)."""
    import cases as K
    from scipy.linalg import expm
    ok = lambda *ds: all(x is not None for x in ds)  # noqa: E731
    S, ny = K.SHEAR, K.SHEAR["ny"]

    def shear_err(d, v):
        return float(np.abs(d["Q"][names(d).index("q_x_0")]
                            - K.shear_exact(v, d["xc"][1], float(d["t"]))).max())
    for v in ("noslip", "free", "noslip_split"):              # C32
        runs = [g("C32", f"{v}_n{n}") for n in ny]
        if ok(*runs):                                          # JAX: the order
            es = [shear_err(x, v) for x in runs]
            add("C32", f"{v} L_inf order {ny[-2]}->{ny[-1]}", b, "serial", order(es[-2], es[-1]),
                TH["C32 order"], "L_inf " + " / ".join(f"{x:.2e}" for x in es)
                + f"; order {ny[0]}->{ny[1]} {order(es[0], es[1]):.2f}")
        d, j = g("C32", f"{v}_n{ny[0]}"), J("C32", f"{v}_n{ny[0]}")
        if d is None:
            continue
        n_ = names(d)
        add("C32", f"{v} n{ny[0]} L_inf vs exact", b, "serial", shear_err(d, v),
            TH["C32 err"], "")
        add("C32", f"{v} n{ny[0]} max|h - 1|", b, "serial",
            float(np.abs(d["Q"][n_.index("h")] - 1.0).max()), RO, "theory: h stays 1")
        add("C32", f"{v} n{ny[0]} max|q_y|", b, "serial",
            float(np.abs(d["Q"][n_.index("q_y_0")]).max()), RO, "theory: v stays 0")
        u = d["Q"][n_.index("q_x_0")]                          # x-independent: every x-derivative is 0
        _, grp = np.unique(np.round(d["xc"][1], 9), return_inverse=True)
        lo, hi = np.full(grp.max() + 1, np.inf), np.full(grp.max() + 1, -np.inf)
        np.minimum.at(lo, grp, u)
        np.maximum.at(hi, grp, u)
        add("C32", f"{v} n{ny[0]} x-variation of q_x (max rel)", b, "serial",
            float((hi - lo).max() / np.abs(u).max()), RO, "theory: u = u(y)")
        if b == "amrex" and j is not None:
            add("C32", f"{v} n{ny[0]} max rel vs JAX", b, "serial", relmax(d, j),
                ("<=", 1e-8, "tolerance (Newton 1e-12, GMRES 1e-13)"), "")
        elif b == "foam" and j is not None:
            add("C32", f"{v} n{ny[0]} L_inf|foam - JAX| / L_inf(JAX - exact)", b, "serial",
                float(np.abs(d["Q"] - j["Q"]).max() / shear_err(j, v)), TH["C32 foam"],
                "foam's own stencils at second order")
    c = g("C32", f"ctrl_n{ny[0]}")                             # C32 negative control (JAX)
    if c is not None:
        add("C32", "control: no-slip profile on free-slip walls, L_inf vs the no-slip exact", b,
            "serial", shear_err(c, "ctrl"), TH["C32 control"], "the walls act")
    c = g("C32", f"inviscid_n{ny[0]}")                         # C32: the Dirichlet wall's own friction (JAX)
    if c is not None:
        add("C32", "nu 0 on no-slip walls: max|u - u0| (the Riemann state's wall friction)", b,
            "serial", shear_err(c, "inviscid"), TH["C32 inviscid"],
            "Dirichlet(momentum) also sets the Riemann outer state: Rusanov damps the wall cells")
        add("C32", "nu 0 on no-slip walls: wall friction, rel. change of sum(q_x vol)", b, "serial",
            mass(c, names(c).index("q_x_0")), None, "the interior fluxes cancel: the walls' share")
    c = g("C32", f"inviscid_split_n{ny[0]}")                   # C32: the split wall, nu 0 (JAX)
    if c is not None:
        add("C32", "nu 0 on split walls: wall friction, rel. change of sum(q_x vol)", b, "serial",
            mass(c, names(c).index("q_x_0")), RO,
            "Wall face: the reflected Riemann state carries no x-momentum flux")
        add("C32", "nu 0 on split walls: max|u - u0|", b, "serial", shear_err(c, "inviscid_split"),
            TH["C32 inviscid split"],
            "the interior Rusanov dissipation of the O2 face jumps, piled up in the wall cells")
    M = K.MOMENT                                               # C33
    for integ, scheme in (("imex", "implicit Euler (I - dt J)^-N"),
                          ("rk2", "SSP-RK2 (1 + dt J + (dt J)^2 / 2)^N")):
        ds = [g("C33", f"{integ}_k{k}") for k in M["ks"]]
        if not ok(*ds):
            continue
        J_, q0, I = _moment_jacobian(), np.array(M["q0"]), np.eye(len(M["q0"]))
        rows = [names(ds[0]).index(f"q_{i}") for i in range(len(q0))]
        ex = expm(M["t_end"] * J_) @ q0
        es, sch = [], []
        for k, d in zip(M["ks"], ds):
            dt = M["dt0"] / 2 ** k
            n_step = int(round(M["t_end"] / dt))
            step = np.linalg.inv(I - dt * J_) if integ == "imex" else I + dt * J_ + (dt * J_) @ (dt * J_) / 2
            q = d["Q"][rows, 0]
            es.append(float(np.abs(q - ex).max() / np.abs(ex).max()))
            sch.append(float(np.abs(q - np.linalg.matrix_power(step, n_step) @ q0).max() / np.abs(ex).max()))
        p = 1 if integ == "imex" else 2
        add("C33", f"{integ} q(T) vs expm, order dt/2 -> dt/4", b, "serial", order(es[-2], es[-1]),
            (">=", p - 0.2, f"theory {p}, slack 0.2"),
            "rel err " + " / ".join(f"{x:.2e}" for x in es) + f" (dt {M['dt0']} / 2^k); {scheme}")
        add("C33", f"{integ} q(T) vs expm at dt {M['dt0'] / 4:g} (rel)", b, "serial", es[-1],
            TH["C33 err"][integ], "")
        add("C33", f"{integ} q(T) vs its own scheme, {scheme} (max rel)", b, "serial", max(sch),
            TH["C33 scheme"][integ], "the integrator in use, to round-off / the implicit tolerance")
        add("C33", f"{integ} uniform state: max|Q - Q[:, 0]|, |h - 1|", b, "serial",
            float(max(np.abs(ds[-1]["Q"] - ds[-1]["Q"][:, :1])[[r for r in range(len(names(ds[-1])))
                                                                 if names(ds[-1])[r] != "b"]].max(),
                      np.abs(ds[-1]["Q"][names(ds[-1]).index("h")] - M["h"]).max())), RO,
            "theory: no flux divergence, h stays")
        jd = [J("C33", f"{integ}_k{k}") for k in M["ks"]]
        if b not in ("jax", "numpy") and ok(*jd):
            add("C33", f"{integ} max rel vs JAX", b, "serial", max(relmax(d, j) for d, j in zip(ds, jd)),
                TH["C33 xb"][integ], "")


#: numpy = JAX: the tolerance of a case's implicit solves (Newton / GMRES), else round-off
NUMPY_TOL = {"C8": PRESS, "C10": PRESS,
             "C14": ("<=", 1e-8, "tolerance (implicit Newton 1e-8)"),
             "C26": ("<=", 1e-8, "tolerance (Newton 1e-12, GMRES 1e-13)"),
             "C32": ("<=", 1e-8, "tolerance (Newton 1e-12, GMRES 1e-13)"),
             "C36": ("<=", 1e-8, "tolerance (Newton 1e-12, GMRES 1e-13)")}


def numpy_checks(add, g, J):
    """numpy = JAX per case: the max rel difference of the final states over the case's runs
    (round-off, or the implicit-solve tolerance); custom runs with a `rel_vs_jax` metric."""
    for case, (_, runs) in CASES.items():
        worst, tol = None, NUMPY_TOL.get(case, RO)
        for r in runs:
            d = g(case, r.key)
            if d is None or "numpy" not in r.backends:
                continue
            if r.custom is not None:            # C27, C34: the run compares with JAX itself
                v = metrics(d).get("rel_vs_jax") if case in ("C27", "C34") else None
                th = TH["C34 xb"] if case == "C34" else RO
            else:
                j = J(case, r.key)
                v = None if j is None else relmax(d, j)
                th = tol
                if case == "C31":
                    th = PRESS if r.key.startswith("vam") else NUMPY_TOL["C14"]
                elif case == "C33":
                    th = TH["C33 xb"]["imex" if r.key.startswith("imex") else "rk2"]
                elif J(case, r.key + "_pert") is not None:      # round-off amplified by the run
                    sens = relmax(J(case, r.key + "_pert"), j)
                    if 10 * sens > th[1]:
                        th = ("<=", 10 * sens, f"10 x JAX round-off sensitivity ({sens:.1e})")
            if v is not None and (worst is None or v / th[1] > worst[0] / worst[1][1]):
                worst = (v, th, r.key)
        if worst is not None:
            add(case, "numpy = JAX (max rel, worst run)", "numpy", "serial", worst[0], worst[1],
                f"worst run {worst[2]}")


def measure():
    """[(case, check, backend, config, value, threshold, note)]."""
    out = []
    add = lambda *a: out.append(a)  # noqa: E731
    for f in sorted(STORE.glob("*/*/*/*.err")) + sorted(STORE.glob("*/*/*/*.unsupported")):
        b, c, case = f.parts[-4:-1]
        first = f.read_text().splitlines()[0][:160]
        if f.suffix == ".err":
            add(case, f"{f.stem} runs", b, c, float("nan"), ("<=", 0, "the run completes"),
                f"{f.relative_to(STORE)}: {first}")
        else:
            add(case, f"{f.stem} runs", b, c, "unsupported", None,
                f"{f.relative_to(STORE)}: {first}")
    for case, (_, runs) in CASES.items():      # whole backends without the case
        for b in BACKENDS:
            if runs and not any(b in r.backends for r in runs):
                why = NA.get((case, b))
                add(case, "runs" if why is None else f"runs (n/a: {why})", b, "serial", "n/a", None, "")
    J = lambda case, k, c="serial": load("jax", c, case, k)  # noqa: E731
    for b in BACKENDS:
        g = lambda case, k, c="serial", b=b: load(b, c, case, k)  # noqa: E731
        ok = lambda *ds: all(x is not None for x in ds)  # noqa: E731

        def conv(case, label, keys, err, th, ns):
            e = [g(case, k) for k in keys]
            if ok(*e):
                es = [err(x) for x in e]
                add(case, f"{label} order {ns[-2]}->{ns[-1]}", b, "serial", order(es[-2], es[-1]),
                    th, "L1(h) " + " / ".join(f"{x:.2e}" for x in es))
            return e

        def cross(case, label, keys, ns, tri=False, tol=RO):
            """AMReX = JAX (round-off); foam O1 on quads = JAX (the same
            stencil); otherwise the foam - JAX difference shrinks."""
            d = [(g(case, k), J(case, k)) for k in keys]
            if b in ("jax", "numpy") or not all(ok(*x) for x in d):
                return
            if b == "amrex" or ("O1" in label and not tri):
                # round-off amplified by the run (limiter switches, a
                # transcritical flow): 10x JAX's own sensitivity, measured
                # with Q0 perturbed at PERTURB (the ``*_pert`` runs)
                sens = [relmax(J(case, k + "_pert"), J(case, k)) for k in keys
                        if J(case, k + "_pert") is not None and J(case, k) is not None]
                if sens and 10 * max(sens) > tol[1]:
                    tol = ("<=", 10 * max(sens), f"10 x JAX round-off sensitivity "
                           f"(Q0 perturbed {PERTURB:.0e}: {max(sens):.1e})")
                add(case, f"{label} max rel vs JAX".strip(), b, "serial",
                    max(relmax(*x) for x in d), tol, "same stencil" if b == "foam" else "")
            elif len(d) > 1:
                dl = [l1rel(*x) for x in d]
                if dl[-1] <= RO_L1[1]:           # foam = JAX at the finest mesh (the
                    # limiters differ, OpenFOAM's cellLimited acts on coarse
                    # meshes only): an order of round-off is noise
                    add(case, f"{label} foam-JAX L1 diff n{ns[-1]}".strip(), b, "serial", dl[-1],
                        RO_L1, " / ".join(f"{x:.1e}" for x in dl))
                    return
                add(case, f"{label} foam-JAX L1 diff order {ns[-2]}->{ns[-1]}", b, "serial",
                    order(dl[-2], dl[-1]), TH["shrink tri" if tri else "shrink"],
                    " / ".join(f"{x:.1e}" for x in dl))
        ref, r11, r14 = J("C1", "ref_n320", "mpi8"), J("C11", "smooth_ref_n1280"), J("C14", "ref_n160")
        for o in (1, 2):                        # convergence vs fine JAX references
            k3 = lambda pre, ns: [f"{pre}o{o}_n{n}" for n in ns]  # noqa: E731
            if ref is not None:
                conv("C1", f"O{o}", k3("", (20, 40, 80)), lambda x: err_ref(x, ref),
                     TH[f"C1 O{o}"], (20, 40, 80))
                conv("C12", f"bump O{o}", k3("bump_", (10, 20, 40)), lambda x: err_ref(x, ref),
                     TH[f"C12 O{o}"], (10, 20, 40))
            if r11 is not None:
                conv("C11", f"smooth O{o}", k3("smooth_", (40, 80, 160)),
                     lambda x: err_ref(x, r11), TH[f"C11 O{o}"], (40, 80, 160))
            if r14 is not None:
                conv("C14", f"O{o}", k3("", (20, 40, 80)), lambda x: err_ref(x, r14),
                     TH[f"C14 O{o}"], (20, 40, 80))
            cross("C1", f"O{o}", k3("", (20, 40, 80)), (20, 40, 80))
            cross("C11", f"smooth O{o}", k3("smooth_", (40, 80, 160)), (40, 80, 160))
            cross("C12", f"bump O{o}", k3("bump_", (10, 20, 40)), (10, 20, 40), tri=True)
            cross("C14", f"O{o}", k3("", (20, 40, 80)), (20, 40, 80),
                  tol=("<=", 1e-8, "tolerance (implicit Newton 1e-8; foam: FD Jacobian)"))
        ref2 = J("C2", "ref_n320", "mpi8")      # C2
        if ref2 is not None:
            conv("C2", "L1", ["n40", "n80"], lambda x: err_ref(x, ref2), TH["C2"], (40, 80))
        if b == "foam" and ok(g("C2", "n80"), J("C2", "n80")):
            add("C2", "L1 vs JAX n80 (rel)", b, "serial", l1rel(g("C2", "n80"), J("C2", "n80")),
                TH["C2 foam"], "")
        if b == "amrex":
            cross("C2", "", ["n40", "n80"], (40, 80))
            cross("C3", "", ["o1", "o2"], (24,))
            cross("C13", "", ["o1_x2.5", "o2_x2.5"], (40,))
        for c in CONFIGS:                       # C3
            for o in (1, 2):
                d = g("C3", f"o{o}", c)
                if d is not None:
                    q = d["Q"][[i for i, n in enumerate(names(d)) if n.startswith("q")]]
                    add("C3", f"O{o} max|u| after 100 steps", b, c,
                        float(np.abs(q / h(d)).max()), TH["C3"], "")
        for case, vs in (("C4", ("sub", "trans")), ("C11", ("sub",)), ("C12", ("sub",))):
            for v in vs:                        # C4 / C11 / C12: Bernoulli
                for o in (1, 2):
                    keys = [f"{v}_o{o}_n{n}" for n in C4_N]
                    e = conv(case, f"{v} O{o}", keys, lambda x: err_exact(x, v),
                             TH["C4 order"][(v, o)], C4_N)
                    if ok(*e):
                        q = e[-1]["Q"][2]          # [b, h, q_x, ...]
                        add(case, f"{v} O{o} max|q/q_in - 1| n{C4_N[-1]}", b, "serial",
                            float(np.abs(q / CHANNEL[v]["q"] - 1).max()), None,
                            "steady discharge = inflow (open-domain mass balance)")
                        if "q_y_0" in names(e[-1]):
                            add(case, f"{v} O{o} max|q_y| n{C4_N[-1]}", b, "serial",
                                float(np.abs(e[-1]["Q"][names(e[-1]).index("q_y_0")]).max()),
                                ("<=", 1e-12, "theory (1-D flow, symmetric stencils)")
                                if case == "C4" else None, "")
                    cross(case, f"{v} O{o}", keys, C4_N, tri=case == "C12")
        for c in CONFIGS:                       # C5
            d = g("C5", "n48", c)
            if d is not None:
                add("C5", "h_min", b, c, float(h(d).min()), (">=", 0.0, "theory (positivity)"), "")
        if b not in ("jax", "numpy") and ok(g("C5", "n48"), J("C5", "n48")):
            fn, th = (l1rel, TH["C5 foam"]) if b == "foam" else (relmax, TH["C5 amrex"])
            add("C5", f"{'L1' if b == 'foam' else 'max'} rel vs JAX", b, "serial",
                fn(g("C5", "n48"), J("C5", "n48")), th, "")
        for k in ("vam", "vam_newton", "vam_t2", "vam_tri"):    # C8
            d, j = g("C8", k), J("C8", k)
            if b not in ("jax", "numpy") and ok(d, j):
                fn, th = (l1rel, TH["C8 foam"]) if b == "foam" else (relmax, PRESS)
                add("C8", f"{k} {'L1' if b == 'foam' else 'max'} rel vs JAX", b, "serial",
                    fn(d, j), th, "")
        if ok(g("C8", "vam_newton"), g("C8", "vam")):
            add("C8", "Newton vs linear pressure (max rel)", b, "serial",
                relmax(g("C8", "vam_newton"), g("C8", "vam")), TH["newton"], "")
        for o, to in C10_VARIANTS:              # C10: Richardson, 3 meshes
            e = [g("C10", f"s{o}_t{to}_n{n}") for n in C10_N]
            if ok(*e):
                d1, d2 = err_ref(e[0], e[1]), err_ref(e[1], e[2])
                add("C10", f"space O{o} time O{to} order (Richardson)", b, "serial",
                    order(d1, d2), TH["C10"][(o, to)],
                    f"|h{C10_N[0]}-h{C10_N[1]}| {d1:.2e}, |h{C10_N[1]}-h{C10_N[2]}| {d2:.2e}")
        for c in CONFIGS:                       # C13: periodic shift invariance
            for o in (1, 2):
                a, s = g("C13", f"o{o}_x2.5", c), g("C13", f"o{o}_x7.5", c)
                if ok(a, s):
                    from cases import PERIODIC
                    L = PERIODIC["L"]
                    ka = np.lexsort((a["xc"][0], a["xc"][1]))
                    ks = np.lexsort(((s["xc"][0] - L / 2) % L, s["xc"][1]))
                    add("C13", f"O{o} shift by L/2 (max rel)", b, c,
                        float(np.abs(a["Q"][:, ka] - s["Q"][:, ks]).max() / np.abs(a["Q"]).max()),
                        RO, "")
        if b == "foam" and ok(g("C13", "o1_x2.5"), J("C13", "o1_x2.5")):
            add("C13", "O1 max rel vs JAX", b, "serial",
                relmax(g("C13", "o1_x2.5"), J("C13", "o1_x2.5")), RO, "O1: the same stencil")
        for c in CONFIGS:                       # C15: AD
            for k in ("hyperbolic", "imex", "chorin"):
                f, r = g("C15", f"{k}_fwd", c), g("C15", f"{k}_rev", c)
                if f is not None:
                    add("C15", f"{k} |FD - jvp| (rel)", b, c, metrics(f)["fd_rel"], TH["fd"], "")
                if ok(f, r):
                    fw, rv = metrics(f)["fwd"], metrics(r)["rev"]
                    add("C15", f"{k} |rev - jvp| (rel)", b, c, abs(rv - fw) / abs(fw), TH["rev"], "")
                s = g("C15", f"{k}_fwd")
                if c != "serial" and ok(f, s):
                    add("C15", f"{k} jvp vs serial (rel)", b, c,
                        abs(metrics(f)["fwd"] - metrics(s)["fwd"]) / abs(metrics(s)["fwd"]),
                        PRESS if k == "chorin" else RO, "")
        for c in CONFIGS:                       # C16: preCICE
            for pair in ("jax_jax", "jax_foam", "foam_foam", "amrex_amrex", "amrex_jax"):
                d = g("C16", pair, c)
                if d is not None:
                    m = metrics(d)
                    add("C16", f"{pair} coupled vs single (max)", b, c, m["coupled_vs_single"],
                        TH["couple"], "")
                    add("C16", f"{pair} mass drift", b, c, m["mass"], TH["couple"], "")
        if b == "numpy":
            numpy_checks(add, g, J)
        euler_checks(add, b, g, J)
        gate_checks(add, b, g, J)
        imex_checks(add, b, g, J)
        restart_checks(add, b, g)
        visc_checks(add, b, g, J)
        for c in CONFIGS:                       # C17: AMR (gate 37)
            for k in ("cover_o2", "cover_dry", "wave16", "wave32", "box", "boxnr", "lake", "dry"):
                d = g("C17", k, c)
                if d is None:
                    continue
                m = metrics(d)
                for key, lab, th in (
                        ("vs_cpu1", "vs CPU 1 rank (max rel)", RO),
                        ("fine_vs_single", "fine level vs fine single level (max rel)", RO),
                        ("L1", "L1(rho - exact)", None),
                        ("mass", "mass drift", RO if k != "boxnr" else None),
                        ("energy", "energy drift", RO if k != "boxnr" else None),
                        ("u", "max|u|", ("<=", 1e-12, "theory (well-balanced)")),
                        ("h_min", "h_min", (">=", 0.0, "theory (positivity)"))):
                    if key in m:
                        add("C17", f"{k} {lab}", b, c, m[key], th,
                            "no reflux: not conservative" if k == "boxnr" else "")
            w = [g("C17", f"wave{n}") for n in (16, 32)]
            if c == "serial" and ok(*w):
                add("C17", "wave order 16->32 (composite)", b, c,
                    order(metrics(w[0])["L1"], metrics(w[1])["L1"]),
                    (">=", 1.5, "observed (O2, static patch; theory 2)"), "")
        for case in CLOSED:                     # C6: mass of every closed-domain run
            for c in CONFIGS:
                ds = [g(case, r.key, c) for r in CASES[case][1]
                      if r.custom is None and not r.key.startswith("sub")]
                ds = [x for x in ds if x is not None]
                if ds:
                    add("C6", f"{case} mass change (rel)", b, c, max(map(mass, ds)), RO,
                        f"max over {len(ds)} runs")
        for case, key in (("C1", "o2_n40"), ("C5", "n48"), ("C8", "vam"), ("C13", "o2_x2.5"),
                          ("C14", "o2_n40"), ("C21", "pulse_o1"), ("C21", "pulse_o2"),
                          ("C21", "sod_y"), ("C26", "visc")):   # C7 / C9: parallel and GPU = serial CPU
            for c in CONFIGS[1:]:
                d, ser = g(case, key, c), g(case, key)
                if ok(ser, d):
                    gpu = c.startswith("gpu")
                    add("C9" if gpu else "C7",
                        f"{case} {key} {'GPU' if gpu else 'parallel'} vs serial CPU (max rel)",
                        b, c, relmax(d, ser), PRESS if case == "C8" else RO, "")
    return out


def table():
    rows = measure()
    lines = ["| case | check | backend | " + " | ".join(HEAD.get(c, c) for c in CONFIGS)
             + " | threshold (source) |", "|---|---|---|" + "---|" * len(CONFIGS) + "---|"]
    fails, seen, ths = [], {}, {}
    for case, check, b, c, v, th, note in rows:            # each cell its own threshold
        seen.setdefault((case, check, b), {})[c] = (v, note, th)
        if th is not None or (case, check, b) not in ths:
            ths[(case, check, b)] = th
    for k in sorted(seen, key=lambda k: (int(k[0][1:]), k[2], k[1])):
        case, check, b = k
        th, cells, txt = ths[k], seen[k], []
        for c in CONFIGS:
            if c not in cells:
                txt.append("-" if c in BACKENDS[b].configs else "n/a")
                continue
            v, note, tc = cells[c]
            if isinstance(v, str):
                txt.append(v)
            elif tc is None:
                txt.append(f"{v:.2e}")
            else:
                ok = bool(v >= tc[1] if tc[0] == ">=" else v <= tc[1])
                txt.append(f"{v:.2e} {'PASS' if ok else '**FAIL**'}")
                if not ok:
                    fails.append(f"- {case} {check}, {b} {c}: {v:.3e} (need {tc[0]} {tc[1]:g}) {note}")
        thr = "information" if th is None else f"{th[0]} {th[1]:g} ({th[2]})"
        lines.append(f"| {case} | {check} | {b} | " + " | ".join(txt) + f" | {thr} |")
    notes = [f"- {case} {check}, {b} {c}: {note}" for case, check, b, c, v, th, note in rows if note]
    walls = {}
    for f in STORE.glob("*/*/*/*.npz"):
        walls[f.parts[-4]] = walls.get(f.parts[-4], 0.0) + float(np.load(f)["wall"])
    import large
    lines += large.table()
    text = "\n".join(lines + ["", "Failures:"] + (fails or ["- none"]) + ["", "Notes:"] + notes
                     + ["", "Summed run wall time (s, without process start-up): "
                        + ", ".join(f"{k} {v:.0f}" for k, v in sorted(walls.items()))])
    (HERE / "table.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if x != "--force"]
    if a[:3] == ["run", "--tier", "large"]:
        import large
        large.main(a[3:])
    elif a[0] == "run":
        run(*a[1:4], force="--force" in sys.argv)
    else:
        table()
