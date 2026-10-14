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

SCHEME: serial-implicit (default) | parallel-implicit; BACKEND: jax (default) | numpy;
PROJECTION: standard (default) | increment; ACCEL: none (default) | iqn (IQN-ILS on P).
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
PRESSURE = dict(pressure_tol=1e-13, pressure_restart=120, pressure_maxit=20)
COUPLING_TOL = dict(rel=1e-12, abs=1e-13)        # preCICE: abs-limit must exceed 1e-14
TIMEOUT = 600
P_ROWS = (6, 7)                                  # P_0, P_1 of [b, h, q_0, q_1, r_0, r_1, P_0, P_1]


def split():
    """VAM(1), one horizontal dimension: left q = r = 0, right P = 0 (C10's boundaries)."""
    import zoomy_core.model.boundary_conditions as BC
    from zoomy_core.model.initial_conditions import Constant
    from zoomy_core.model.models.closures import Newtonian, StressFree
    from zoomy_core.model.models.vam import VAM
    from zoomy_core.systemmodel.system_model import SystemModel, _DT_SYMBOL
    bcs = ([BC.Dirichlet("left", on=f, value=0.0) for f in ("q_0", "q_1", "r_0", "r_1")]
           + [BC.Dirichlet("right", on=f"P_{i}", value=0.0) for i in range(2)])
    model = VAM(level=1, dimension=2, boundary_conditions=bcs,
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


def solver(backend, projection, **kw):
    """The Chorin solver of `backend`, constant dt, the gate's pressure tolerance."""
    from zoomy_core.symsolver.settings import LinearSolver, Projection
    kw = dict(dt=DT, cfl=0.9, projection=Projection(projection), **PRESSURE, **kw)
    kw["pressure_linear_solver"] = LinearSolver.gmres
    if backend == "jax":
        from zoomy_jax.symsolver.solver import SymbolicChorinSolverJax
        return SymbolicChorinSolverJax(**kw)
    from zoomy_core.symsolver.numpy_backend.solver import SymbolicChorinSolverNumpy
    return SymbolicChorinSolverNumpy(**kw)


def march(s, mesh, backend):
    """Set up `s` on `mesh` with the wave, march; returns `(names, x, Q)` of all its cells."""
    from zoomy_core.symsolver.strategies.chorin import predictor_nsm
    pred, press, corr = split()
    Q, A = s.setup(mesh, predictor_nsm(pred, press, corr), press, corr)
    names = [str(v) for v in s.nsm.state]
    x = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    Q = wave(names, x)
    if backend == "jax":
        import jax.numpy as jnp
        Q = jnp.asarray(Q)
    A = s.initial_aux(Q, A)
    t, Q, A = s.run(Q, A)
    return names, x, np.asarray(Q), float(t)


def single(work, backend, projection):
    from zoomy_core.mesh import LSQMesh
    s = solver(backend, projection, time_end=T)
    names, x, Q, t = march(s, LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=NX), backend)
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
    from zoomy_core.mesh import LSQMesh
    from zoomy_core.mesh.overlap import overlap_layers
    left = name == "Left"
    me, peer = ("L", "R") if left else ("R", "L")
    n_state = 8
    if backend == "jax":
        from zoomy_jax.symsolver.runtime import PreciceAdapter
    else:
        from zoomy_core.symsolver.precice_adapter import PreciceAdapter
    adapter = PreciceAdapter(name, str(Path(work) / "precice-config.xml"), f"{name}-Mesh",
                             [f"{me}{i}" for i in range(n_state)],
                             [f"{peer}{i}" for i in range(n_state)])
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
    tag = "right" if left else "left"
    domain = (DOMAIN[0], W * DX) if left else (-W * DX, DOMAIN[1])
    mesh = LSQMesh.create_1d(domain=domain, n_inner_cells=NX // 2 + W)
    s = solver(backend, projection, couple=dict(adapter=adapter, tag=tag, overlap=W))
    t0 = wall.perf_counter()
    names, x, Q, t = march(s, mesh, backend)
    adapter.finalize()
    over, _ = overlap_layers(s.mesh_np, tag, W)
    own = np.setdiff1d(np.arange(len(x)), over)
    np.savez(Path(work) / f"{name}.npz", Q=Q[:, own], x=x[own],
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
    work = HERE / f"run_gate54_{backend}_{projection}_W{W}_{scheme}_{accel}"
    work.mkdir(exist_ok=True)
    config(work, 8, W, scheme, accel)
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
    Qc, xc = np.concatenate([L["Q"], R["Q"]], 1), np.concatenate([L["x"], R["x"]])
    assert np.abs(xc - xs).max() < 1e-12, "the own cells are not the single domain's"
    from zoomy_core.mesh import LSQMesh  # noqa: F401  (x of the initial state)
    names = ["b", "h", "q_0", "q_1", "r_0", "r_1", "P_0", "P_1"]
    Q0 = wave(names, xs)
    err = np.abs(Qc - Qs).max(axis=1)
    out = dict(W=W, scheme=scheme, accel=accel, backend=backend, projection=projection,
               coupled_vs_single=float(err.max()),
               per_row={n: float(e) for n, e in zip(names, err)},
               mass=float(abs(Qc[1].sum() - Qs[1].sum()) * DX),
               mass_drift_single=float(abs(Qs[1].sum() - Q0[1].sum()) * DX),
               iterations=[int(k) for k in L["iterations"]],
               wall=round(wall.time() - t0, 1))
    print("GATE54 " + json.dumps(out), flush=True)
    print(f"W {W} {scheme} {accel} {backend} {projection}: max|coupled - single| "
          f"{out['coupled_vs_single']:.3e}, |mass coupled - single| {out['mass']:.3e} (single's "
          f"drift through the boundaries {out['mass_drift_single']:.3e}), iterations per window "
          f"{min(out['iterations'])}..{max(out['iterations'])} "
          f"(mean {np.mean(out['iterations']):.1f})", flush=True)
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
