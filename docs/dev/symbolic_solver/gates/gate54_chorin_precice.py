"""Gate 54: the Chorin march as two preCICE participants coupled by overlapping Schwarz
(`ChorinTemplate.precice_march`) vs the single domain.

VAM(1) in 1-D (one horizontal dimension), C10's wave over the bump on [-1.5, 1.5] (60 cells,
left q = r = 0, right P = 0), O1, time_order 1, constant dt = the window = 2^-7 to t = 0.25
(32 windows).  Split at x = 0 into Left (30 cells) | Right (30 cells), each mesh reaching W cells
across the interface (the overlap); an implicit coupling scheme repeats each window until the
exchanged data (all state rows of the mirror cells) converge.  Measured: max |coupled - single|
over the own cells, |mass coupled - mass single| (both drift alike through the open right end),
the coupling iterations per window.

    python gate54_chorin_precice.py [W] [SCHEME] [BACKEND] [PROJECTION] [ACCEL]   # the gate
    python gate54_chorin_precice.py side NAME W WORK BACKEND PROJECTION     # one participant
    python gate54_chorin_precice.py single WORK BACKEND PROJECTION          # the single domain

SCHEME: serial-implicit (default) | parallel-implicit; BACKEND: jax (default) | numpy | foam |
amrex;
PROJECTION: standard (default) | increment; ACCEL: none (default) | iqn (IQN-ILS on P).
GATE54_DIM=2: C10's own quasi-1-D mesh (quads, ny = 2 on [0, 0.1], VAM(1) with two horizontal
dimensions, Extrapolation at the sides): the overlap rings and the vertex sets in 2-D.
GATE54_TIME_ORDER=2 / 3: SSP-RK2 / RK3 projection stages, each stage's pressure exchanged too.
"""
import json
import os
import signal
import subprocess
import sys
import time as wall
from pathlib import Path

import numpy as np

import common  # noqa: F401  (x64)

HERE = Path(__file__).resolve().parent
DOMAIN, NX, DT, T = (-1.5, 1.5), 60, 2.0 ** -7, 0.25
DX = (DOMAIN[1] - DOMAIN[0]) / NX
WAVE = dict(H=0.34, a=0.01, x0=-0.5, s=0.3)
PRESSURE = dict(pressure_tol=float(os.environ.get("GATE54_PRESSURE_TOL", "1e-13")),
                pressure_restart=120, pressure_maxit=20)
COUPLING_TOL = dict(rel=float(os.environ.get("GATE54_COUPLING_TOL", "1e-12")),   # preCICE:
                    abs=1e-13)                                              # abs > 1e-14
TIMEOUT = 600
DIM = int(os.environ.get("GATE54_DIM", "1"))     # horizontal dimensions (2: C10's quads, ny = 2)
NAMES = (["b", "h", "q_0", "q_1", "r_0", "r_1", "P_0", "P_1"] if DIM == 1 else
         ["b", "h", "q_x_0", "q_x_1", "q_y_0", "q_y_1", "r_0", "r_1", "P_0", "P_1"])
TIME_ORDER = int(os.environ.get("GATE54_TIME_ORDER", "1"))
STAGES = {1: 1, 2: 2, 3: 3}[TIME_ORDER]
N_STATE = len(NAMES)
# the exchanged rows: the state, then (several stages) each stage's pressure (P_0, P_1); the
# iterations read the pressure rows (IQN-ILS acts on them)
N_ROWS = N_STATE + (0 if STAGES == 1 else 2 * STAGES)
P_ROWS = ((NAMES.index("P_0"), NAMES.index("P_1")) if STAGES == 1
          else tuple(range(N_STATE, N_ROWS)))
AREA = DX * (1.0 if DIM == 1 else 0.05)          # cell size (2-D: [0, 0.1] in 2 rows)


def split():
    """VAM(1), `DIM` horizontal dimensions: left q = r = 0, right P = 0 (C10's boundaries; 2-D:
    Extrapolation at the sides)."""
    import zoomy_core.model.boundary_conditions as BC
    from zoomy_core.model.initial_conditions import Constant
    from zoomy_core.model.models.closures import Newtonian, StressFree
    from zoomy_core.model.models.vam import VAM
    from zoomy_core.systemmodel.system_model import SystemModel, _DT_SYMBOL
    q = ("q_0", "q_1") if DIM == 1 else ("q_x_0", "q_x_1")
    bcs = ([BC.Dirichlet("left", on=f, value=0.0) for f in (*q, "r_0", "r_1")]
           + [BC.Dirichlet("right", on=f"P_{i}", value=0.0) for i in range(2)]
           + ([] if DIM == 1 else [BC.Extrapolation(tag=t) for t in ("bottom", "top")]))
    model = VAM(level=1, dimension=DIM + 1, boundary_conditions=bcs,
                closures=[Newtonian(), StressFree()])
    sm = SystemModel.from_model(model)
    sm.initial_conditions = Constant(constants=lambda n: np.zeros(n))
    sm.aux_initial_conditions = Constant(constants=lambda n: np.zeros(n))
    r = model.chorin_split(_DT_SYMBOL, system_model=sm)
    return r.SM_pred, r.SM_press, r.SM_corr


def wave(names, x):
    """C10's wave: eta = H + a exp(-(x - x0)^2 / 2 s^2) over the bump b, at rest."""
    q = np.zeros((len(names), len(x)))
    b = 0.20 * np.exp(-x ** 2 / (2 * 0.20 ** 2))
    eta = WAVE["H"] + WAVE["a"] * np.exp(-(x - WAVE["x0"]) ** 2 / (2 * WAVE["s"] ** 2))
    q[names.index("b")], q[names.index("h")] = b, eta - b
    return q


CXX = ("foam", "amrex")                         # C++ backends: the solver runs the case


def mesh_of(domain, n):
    """`n` cells along x on `domain` (2-D: quads, 2 rows on [0, 0.1])."""
    from zoomy_core.mesh import LSQMesh
    if DIM == 1:
        return LSQMesh.create_1d(domain=domain, n_inner_cells=n)
    return LSQMesh.create_2d(domain=(*domain, 0.0, 0.1), nx=n, ny=2)


def solver(backend, projection, **kw):
    """The Chorin solver of `backend`, constant dt, the gate's pressure tolerance (unpreconditioned
    GMRES on every backend)."""
    from zoomy_core.symsolver.settings import LinearSolver, Projection
    kw = dict(dt=DT, cfl=0.9, projection=Projection(projection), time_order=TIME_ORDER,
              **PRESSURE, **kw)
    if backend in CXX:
        kw["linear_solver"] = LinearSolver.gmres
    else:
        kw["pressure_linear_solver"] = LinearSolver.gmres
    if backend == "jax":
        from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
        return SymbolicChorinSolverJax(**kw)
    if backend == "foam":
        from zoomy_foam.symsolver import SymbolicChorinSolverFoam
        return SymbolicChorinSolverFoam(**kw)
    if backend == "amrex":
        from zoomy_amrex.symsolver import SymbolicChorinSolverAmrex
        return SymbolicChorinSolverAmrex(**kw)
    from zoomy_core.symsolver.numpy_backend.solver import SymbolicChorinSolverNumpy
    return SymbolicChorinSolverNumpy(**kw)


def march(s, mesh, backend, case=None):
    """March the wave on `mesh` with `s` (a C++ backend: its case in `case`); returns
    `(names, x, Q, t)` of all its cells, `x` their centres `(2, cells)`."""
    from zoomy_core.mesh import ensure_lsq_mesh
    from zoomy_core.symsolver.strategies.chorin import predictor_nsm
    pred, press, corr = split()
    nsm = predictor_nsm(pred, press, corr)
    if backend in CXX:
        mesh = ensure_lsq_mesh(mesh, nsm)
        names = [str(v) for v in nsm.state]
        x = np.asarray(mesh.cell_centers)[:2, :int(mesh.n_inner_cells)]
        t, Q = s.run(mesh, (nsm, press, corr), wave(names, x[0]), str(case))
        return names, x, np.asarray(Q), float(t)
    Q, A = s.setup(mesh, nsm, press, corr)
    names = [str(v) for v in s.nsm.state]
    x = np.asarray(s.mesh_np.cell_centers)[:2, :Q.shape[1]]
    Q = wave(names, x[0])
    if backend == "jax":
        import jax.numpy as jnp
        Q = jnp.asarray(Q)
    A = s.initial_aux(Q, A)
    t, Q, A = s.run(Q, A)
    return names, x, np.asarray(Q), float(t)


def iterations_log(work, name):
    """The coupling iterations per window from preCICE's `precice-<name>-iterations.log`
    (written by the participant that checks convergence) in `work` or below."""
    logs = sorted(Path(work).rglob(f"precice-{name}-iterations.log"))
    if not logs:
        return []
    rows = [ln.split() for ln in logs[0].read_text().splitlines()[1:] if ln.strip()]
    return [int(r[2]) for r in rows]


def single(work, backend, projection):
    s = solver(backend, projection, time_end=T)
    names, x, Q, t = march(s, mesh_of(DOMAIN, NX), backend, Path(work) / "single")
    np.save(Path(work) / "single.npy", Q)
    np.save(Path(work) / "single_x.npy", x)
    print(f"single: t = {t:.12g}", flush=True)


def config(work, n_state, W, scheme, accel="none"):
    """The preCICE configuration: each side writes its n_state rows on its mesh (mirror and
    overlap cell centres), reads the peer's (nearest neighbour: the cell centres coincide);
    `accel` iqn: IQN-ILS on the pressure rows (the data the iterations read: of the second
    participant in a serial scheme, of both in a parallel one)."""
    data = "".join(f'<data:scalar name="{s}{i}" />' for s in "LR" for i in range(n_state))
    use = "".join(f'<use-data name="{s}{i}" />' for s in "LR" for i in range(n_state))
    parts, exch, conv = [], [], []
    for me, peer, name, other in (("L", "R", "Left", "Right"), ("R", "L", "Right", "Left")):
        parts.append(
            f'<participant name="{name}"><provide-mesh name="{name}-Mesh" />'
            f'<receive-mesh name="{other}-Mesh" from="{other}" geometric-filter="no-filter" />'
            + "".join(f'<write-data name="{me}{i}" mesh="{name}-Mesh" />' for i in range(n_state))
            + "".join(f'<read-data name="{peer}{i}" mesh="{name}-Mesh" />' for i in range(n_state))
            + f'<mapping:nearest-neighbor direction="read" from="{other}-Mesh" to="{name}-Mesh" '
              'constraint="consistent" /></participant>')
        for i in range(n_state):
            exch.append(f'<exchange data="{me}{i}" mesh="{name}-Mesh" from="{name}" to="{other}" '
                        'initialize="true" />')
            conv.append(f'<absolute-or-relative-convergence-measure abs-limit="{COUPLING_TOL["abs"]}" '
                        f'rel-limit="{COUPLING_TOL["rel"]}" data="{me}{i}" mesh="{name}-Mesh" />')
    acc = ""
    if accel == "iqn":
        sides = (("R", "Right"),) if scheme.startswith("serial") else (("L", "Left"), ("R", "Right"))
        acc = ("<acceleration:IQN-ILS>"
               + "".join(f'<data name="{me}{i}" mesh="{name}-Mesh" />'
                         for me, name in sides for i in P_ROWS)
               + '<preconditioner type="residual-sum" /><filter type="QR2" limit="1e-3" />'
               '<initial-relaxation value="1.0" /><max-used-iterations value="50" />'
               '<time-windows-reused value="0" /></acceleration:IQN-ILS>')
    cfg = (f'<?xml version="1.0" encoding="UTF-8" ?><precice-configuration>{data}'
           f'<mesh name="Left-Mesh" dimensions="2">{use}</mesh>'
           f'<mesh name="Right-Mesh" dimensions="2">{use}</mesh>{"".join(parts)}'
           f'<m2n:sockets acceptor="Left" connector="Right" exchange-directory="{work}" />'
           f'<coupling-scheme:{scheme}><participants first="Left" second="Right" />'
           f'<max-time value="{T}" /><time-window-size value="{DT}" />'
           f'<max-iterations value="200" />{"".join(exch)}{"".join(conv)}{acc}'
           f'</coupling-scheme:{scheme}></precice-configuration>')
    path = Path(work) / "precice-config.xml"
    path.write_text(cfg)
    return path


def side(name, W, work, backend, projection):
    """Participant `name`: its 30 cells and W overlap cells; writes `work/name.npz` (own cells,
    their x, the coupling iterations of each window)."""
    from zoomy_core.mesh.overlap import overlap_layers
    left = name == "Left"
    me, peer = ("L", "R") if left else ("R", "L")
    n_state = N_ROWS
    tag = "right" if left else "left"
    domain = (DOMAIN[0], W * DX) if left else (-W * DX, DOMAIN[1])
    mesh = mesh_of(domain, NX // 2 + W)
    config = str(Path(work) / "precice-config.xml")
    write, read = [f"{me}{i}" for i in range(n_state)], [f"{peer}{i}" for i in range(n_state)]
    t0 = wall.perf_counter()
    if backend in CXX:
        s = solver(backend, projection, couple=dict(
            participant=name, config=config, mesh=f"{name}-Mesh", write=write, read=read,
            tag=tag, overlap=W))
        names, x, Q, t = march(s, mesh, backend, Path(work) / name)
        over, _ = overlap_layers(mesh, tag, W)
        own = np.setdiff1d(np.arange(x.shape[1]), over)
        its = iterations_log(Path(work) / name, name)
        np.savez(Path(work) / f"{name}.npz", Q=Q[:, own], x=x[:, own], iterations=np.asarray(its))
        print(f"{name} ({backend}): t = {t:.12g} [{wall.perf_counter() - t0:.0f}s]", flush=True)
        return
    if backend == "jax":
        from zoomy_jax.symsolver.runtime import PreciceAdapter
    else:
        from zoomy_core.symsolver.precice_adapter import PreciceAdapter
    adapter = PreciceAdapter(name, config, f"{name}-Mesh", write, read)
    if os.environ.get("GATE54_TRACE"):          # the change of the read data per iteration
        read, last = adapter.read_iterate, [None]

        def traced():
            r = read()
            if last[0] is not None:
                print(f"{name} it {adapter._iteration}: |d read| per row "
                      + " ".join(f"{v:.1e}" for v in np.abs(r - last[0]).max(axis=1)), flush=True)
            last[0] = r
            return r
        adapter.read_iterate = traced
    s = solver(backend, projection, couple=dict(adapter=adapter, tag=tag, overlap=W))
    names, x, Q, t = march(s, mesh, backend)
    adapter.finalize()
    over, _ = overlap_layers(s.mesh_np, tag, W)
    own = np.setdiff1d(np.arange(x.shape[1]), over)
    np.savez(Path(work) / f"{name}.npz", Q=Q[:, own], x=x[:, own],
             iterations=np.asarray(adapter.iterations))
    print(f"{name}: t = {t:.12g}, {len(adapter.iterations)} windows, iterations "
          f"{min(adapter.iterations)}..{max(adapter.iterations)} "
          f"[{wall.perf_counter() - t0:.0f}s]", flush=True)


def launch(args, log):
    """A subprocess in its own session (killed as a group)."""
    return subprocess.Popen(["timeout", "-k", "5", str(TIMEOUT), sys.executable, __file__, *args],
                            start_new_session=True, stdout=open(log, "w"),
                            stderr=subprocess.STDOUT)


def gate(W, scheme, backend, projection, accel="none"):
    work = HERE / (f"run_gate54_{backend}_{projection}_W{W}_{scheme}_{accel}"
                   + ("" if DIM == 1 else f"_{DIM}d") + ("" if STAGES == 1 else f"_t{TIME_ORDER}"))
    work.mkdir(exist_ok=True)
    config(work, N_ROWS, W, scheme, accel)
    procs = [launch(["single", str(work), backend, projection], work / "single.log")]
    procs += [launch(["side", n, str(W), str(work), backend, projection], work / f"{n}.log")
              for n in ("Left", "Right")]
    t0 = wall.time()
    try:
        while any(p.poll() is None for p in procs) and wall.time() - t0 < TIMEOUT + 10:
            if any(p.poll() not in (None, 0) for p in procs):
                break
            wall.sleep(0.5)
    finally:
        for p in procs:                     # never leave a participant waiting
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
    rc = [p.returncode for p in procs]
    if any(rc):
        for n in ("single", "Left", "Right"):
            print(f"--- {n}.log\n" + "".join(open(work / f"{n}.log").readlines()[-15:]))
        print(f"FAILED rc={rc}")
        sys.exit(1)
    Qs, xs = np.load(work / "single.npy"), np.load(work / "single_x.npy")
    L, R = np.load(work / "Left.npz"), np.load(work / "Right.npz")
    its = [int(k) for k in (L["iterations"] if len(L["iterations"]) else R["iterations"])]
    Qc, xc = np.concatenate([L["Q"], R["Q"]], 1), np.concatenate([L["x"], R["x"]], 1)
    oc, os_ = np.lexsort(np.round(xc[::-1], 9)), np.lexsort(np.round(xs[::-1], 9))
    Qc, xc, Qs, xs = Qc[:, oc], xc[:, oc], Qs[:, os_], xs[:, os_]   # both by (x, y)
    assert np.abs(xc - xs).max() < 1e-12, "the own cells are not the single domain's"
    names = NAMES
    Q0 = wave(names, xs[0])
    err = np.abs(Qc - Qs).max(axis=1)
    out = dict(W=W, scheme=scheme, accel=accel, backend=backend, projection=projection,
               coupled_vs_single=float(err.max()),
               per_row={n: float(e) for n, e in zip(names, err)},
               mass=float(abs(Qc[1].sum() - Qs[1].sum()) * AREA),
               mass_drift_single=float(abs(Qs[1].sum() - Q0[1].sum()) * AREA),
               iterations=its, dim=DIM, time_order=TIME_ORDER,
               wall=round(wall.time() - t0, 1))
    print("GATE54 " + json.dumps(out), flush=True)
    it = (f"{min(its)}..{max(its)} (mean {np.mean(its):.1f})" if its else "n/a")
    print(f"{DIM}-D t{TIME_ORDER} W {W} {scheme} {accel} {backend} {projection}: "
          "max|coupled - single| "
          f"{out['coupled_vs_single']:.3e}, |mass coupled - single| {out['mass']:.3e} (single's "
          f"drift through the boundaries {out['mass_drift_single']:.3e}), iterations per window "
          f"{it}", flush=True)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "side":
        side(a[1], int(a[2]), a[3], a[4], a[5])
    elif a and a[0] == "single":
        single(a[1], a[2], a[3])
    else:
        gate(int(a[0]) if a else 3, a[1] if len(a) > 1 else "serial-implicit",
             a[2] if len(a) > 2 else "jax", a[3] if len(a) > 3 else "standard",
             a[4] if len(a) > 4 else "none")
