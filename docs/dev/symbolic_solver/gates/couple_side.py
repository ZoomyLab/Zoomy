"""Gate 3's 1-D dam break (SME(0), O1, 200 cells, dam at x = 0.5) split at x = 0.5 into two
preCICE participants, parallel explicit, window 1e-3 = the step: the setup of gates 10, 26, 38,
matrix C16 and the cross-container C16 of tests/backend_e2e.  Coupled = the single-domain march
(gate3_precice.single, same fixed step) to round-off, mass exact.

The setup of a side is backend-neutral (zoomy_core, no JAX), so a C++ participant runs in its
backend's image alone; the JAX participant is participant.py.

    python couple_side.py BACKEND NAME WORK [RANKS]   # foam | amrex, Left | Right: reads
                                                      # WORK/precice-config.xml, writes WORK/NAME.npy
    python couple_side.py config WORK [EXCHANGE]      # write_config
    python couple_side.py check WORK                  # vs_single of WORK/{Left,Right}.npy (JSON)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

import common as C
from zoomy_core.mesh import ensure_lsq_mesh

HERE = Path(__file__).resolve().parent
T, WINDOW, h, dx = 0.05, 1e-3, 1, 1 / 200


def write_config(work, exchange=None):
    """`work/precice-config.xml`: gates/precice-config.xml with T, WINDOW, the exchange directory
    `exchange` (default `work`; two containers: the path both see) and no geometric filter (a
    point interface in 1-D has zero-size bounding boxes, the filter would drop it)."""
    cfg = (HERE / "precice-config.xml").read_text()
    cfg = re.sub(r'time-window-size value="[^"]*"', f'time-window-size value="{WINDOW}"', cfg)
    cfg = re.sub(r'max-time value="[^"]*"', f'max-time value="{T}"', cfg)
    cfg = cfg.replace('exchange-directory="."', f'exchange-directory="{exchange or work}"')
    cfg = re.sub(r'(<receive-mesh [^/]*)/>', r'\1geometric-filter="no-filter" />', cfg)
    path = Path(work) / "precice-config.xml"
    path.write_text(cfg)
    return path


def half(name):
    """`(mesh, nsm, Q0)` of the half `name`: Left x < 0.5 (h 1), Right x > 0.5 (h 0.5)."""
    left = name == "Left"
    mesh, nsm = C.dambreak_1d(0, 1, domain=(0.0, 0.5) if left else (0.5, 1.0),
                              nx=100, x_dam=1.0 if left else 0.0)
    mesh = ensure_lsq_mesh(mesh, nsm)
    nc = int(mesh.n_inner_cells)
    Q0 = nsm.sm.initial_conditions.apply(np.asarray(mesh.cell_centers)[:, :nc],
                                         np.zeros((len(nsm.sm.state), nc)))
    return mesh, nsm, np.asarray(Q0, dtype=float)


def cxx_side(backend, name, config, work, ranks=1):
    """Participant `name` on `backend` (foam | amrex) with `ranks` MPI ranks, built now (before
    the peer waits for it); returns its run, `() -> (t, Q)` of the half."""
    left = name == "Left"
    me, peer = ("L", "R") if left else ("R", "L")
    couple = dict(participant=name, config=str(config), mesh=f"{name}-Mesh",
                  write=[f"{me}{i}" for i in range(3)], read=[f"{peer}{i}" for i in range(3)],
                  tag="right" if left else "left", side=0 if left else 1)
    mesh, nsm, Q0 = half(name)
    if backend == "foam":
        from zoomy_foam.symsolver import SymbolicHyperbolicSolverFoam
        s = SymbolicHyperbolicSolverFoam(cfl=0.9, nprocs=ranks, couple=couple)
    elif backend == "amrex":
        from zoomy_amrex.symsolver import SymbolicHyperbolicSolverAmrex
        s = SymbolicHyperbolicSolverAmrex(cfl=0.9, ranks=ranks, couple=couple)
    else:
        raise ValueError(f"no C++ participant for {backend!r} (JAX: participant.py)")
    s.build(nsm, 1)
    return lambda: s.run(mesh, nsm, Q0, Path(work) / name)


def vs_single(Qc):
    """C16's measures of the coupled state `Qc` (Left | Right): max|Qc - single domain| and the
    mass drift."""
    from gate3_precice import single
    Q0, Qs = single(1, fixed_dt=WINDOW)
    return dict(coupled_vs_single=float(np.abs(Qc - Qs).max()),
                mass=float(abs(Qc[h].sum() - Q0[h].sum()) * dx))


if __name__ == "__main__":
    if sys.argv[1] == "config":
        print(write_config(*sys.argv[2:4]))
    elif sys.argv[1] == "check":
        w = Path(sys.argv[2])
        Qc = np.concatenate([np.load(w / "Left.npy"), np.load(w / "Right.npy")], 1)
        print("COUPLE_CHECK " + json.dumps(vs_single(Qc)), flush=True)
    else:
        backend, name, work = sys.argv[1:4]
        ranks = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        t, Q = cxx_side(backend, name, Path(work) / "precice-config.xml", work, ranks)()
        np.save(Path(work) / f"{name}.npy", np.asarray(Q))
        print(f"{name} ({backend}, {ranks} rank(s)): t = {float(t):.12g}", flush=True)
