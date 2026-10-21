"""The Chorin march coupled by overlapping Schwarz (`ChorinTemplate.precice_march`), in process:
two NumPy participants in two threads exchange through an in-memory implicit coupling (the
preCICE calls of `PreciceAdapter`, parallel-implicit, no acceleration) and must reproduce the
single-domain march; the overlap rings of `zoomy_core.mesh.overlap`.  Gate 54 runs the same over
preCICE (docs/dev/symbolic_solver/gates/gate54_chorin_precice.py).
"""
import copy
import functools
import threading

import numpy as np
import pytest

pytestmark = [pytest.mark.solver, pytest.mark.small]

DOMAIN, NX, DT, STEPS, W = (-1.5, 1.5), 24, 2.0 ** -6, 4, 3
DX = (DOMAIN[1] - DOMAIN[0]) / NX


def test_overlap_layers_1d_and_quads():
    from zoomy_core.mesh import LSQMesh
    from zoomy_core.mesh.overlap import overlap_layers
    over, mirror = overlap_layers(LSQMesh.create_1d(domain=(0.0, 1.0), n_inner_cells=10),
                                  "right", 3)
    assert list(over) == [9, 8, 7] and list(mirror) == [6, 5, 4]
    m = LSQMesh.create_2d(domain=(0.0, 1.0, 0.0, 1.0), nx=6, ny=3)
    over, mirror = overlap_layers(m, "left", 2)
    x = np.asarray(m.cell_centers)[0]
    assert sorted(np.round(x[over] * 12).astype(int)) == [1] * 3 + [3] * 3
    assert sorted(np.round(x[mirror] * 12).astype(int)) == [5] * 3 + [7] * 3
    with pytest.raises(ValueError):
        overlap_layers(m, "left", 4)


class Hub:
    """Two participants' implicit coupling in memory, parallel-implicit: at each `advance` both
    meet; the window repeats until the written data change by at most `tol` (max norm)."""

    def __init__(self, tol=1e-13, max_iterations=200):
        self.tol, self.max_iterations = tol, max_iterations
        self.sides, self.time, self.iteration, self.repeat = {}, 0.0, 0, False
        self.barrier = threading.Barrier(2, action=self._exchange)
        self.start = threading.Barrier(2)

    def _exchange(self):
        """Both sides advanced: accept or repeat the window."""
        change = max(np.abs(s["pending"] - s["end"]).max() for s in self.sides.values())
        for s in self.sides.values():
            s["end"] = s["pending"]
        self.iteration += 1
        self.repeat = change > self.tol and self.iteration < self.max_iterations
        if not self.repeat:
            for s in self.sides.values():
                s["start"] = s["end"]
            self.time += DT
            self.iteration = 0


class FakeAdapter:
    """The `PreciceAdapter` calls of an overlap participant, against a `Hub`."""

    def __init__(self, name, peer, hub, n_state):
        self.name, self.peer, self.hub = name, peer, hub
        self.wn = self.rn = [f"q{i}" for i in range(n_state)]
        self.iterations, self._it = [], 0

    def set_overlap_vertices(self, write, read):
        self.hub.sides[self.name] = dict(write=np.asarray(write)[:, 0],
                                         read=np.asarray(read)[:, 0])

    def _map(self, data):
        """The peer's values at my read vertices (the coincident cell centres)."""
        me, peer = self.hub.sides[self.name], self.hub.sides[self.peer]
        at = [int(np.argmin(np.abs(peer["write"] - x))) for x in me["read"]]
        return data[:, at]

    def init(self, q):
        s = self.hub.sides[self.name]
        s["start"] = s["end"] = s["pending"] = np.asarray(q, dtype=float)
        self.hub.start.wait()
        return np.bool_(True)

    def max_dt(self):
        return np.float64(DT)

    def read(self):
        return self._map(self.hub.sides[self.peer]["start"])

    def read_iterate(self):
        return self._map(self.hub.sides[self.peer]["end"])

    def write(self, q):
        self.hub.sides[self.name]["pending"] = np.asarray(q, dtype=float)
        return ()

    def advance(self, dt):
        self.hub.barrier.wait()
        self._repeat = self.hub.repeat
        self._it += 1
        if not self._repeat:
            self.iterations.append(self._it)
            self._it = 0
        return np.bool_(self.hub.time < STEPS * DT - 1e-12)

    def repeat(self):
        return np.bool_(self._repeat)


def _split():
    """A fresh copy of the VAM(1) split (an NSM's setup marks the SMs it is built from)."""
    return copy.deepcopy(_split_once())


@functools.lru_cache(maxsize=None)
def _split_once():
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


def _march(mesh, couple=None):
    """VAM(1) 1-D, a wave over the bump, `STEPS` steps of `DT`; returns `(x, Q)`."""
    from zoomy_core.symsolver.numpy_backend.solver import SymbolicChorinSolverNumpy
    from zoomy_core.symsolver.settings import LinearSolver
    from zoomy_core.symsolver.strategies.chorin import predictor_nsm
    pred, press, corr = _split()
    s = SymbolicChorinSolverNumpy(dt=DT, time_end=STEPS * DT, cfl=0.9, pressure_tol=1e-13,
                                  pressure_restart=2 * NX, pressure_linear_solver=LinearSolver.gmres,
                                  couple=couple)
    Q, A = s.setup(mesh, predictor_nsm(pred, press, corr), press, corr)
    names = [str(v) for v in s.nsm.state]
    x = np.asarray(s.mesh_np.cell_centers)[0, :Q.shape[1]]
    b = 0.2 * np.exp(-x ** 2 / 0.08)
    Q = np.zeros(Q.shape)
    Q[names.index("b")] = b
    Q[names.index("h")] = 0.34 + 0.01 * np.exp(-(x + 0.5) ** 2 / 0.18) - b
    t, Q, _ = s.run(Q, s.initial_aux(Q, A))
    assert abs(t - STEPS * DT) < 1e-12
    return x, np.asarray(Q), s


def test_schwarz_coupled_pair_is_the_single_domain():
    from zoomy_core.mesh import LSQMesh
    from zoomy_core.mesh.overlap import overlap_layers
    xs, Qs, _ = _march(LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=NX))
    hub, out, errors = Hub(), {}, []

    def side(name):
        try:
            left = name == "Left"
            a = FakeAdapter(name, "Right" if left else "Left", hub, Qs.shape[0])
            tag = "right" if left else "left"
            domain = (DOMAIN[0], W * DX) if left else (-W * DX, DOMAIN[1])
            x, Q, s = _march(LSQMesh.create_1d(domain=domain, n_inner_cells=NX // 2 + W),
                             dict(adapter=a, tag=tag, overlap=W))
            own = np.setdiff1d(np.arange(len(x)), overlap_layers(s.mesh_np, tag, W)[0])
            out[name] = (x[own], Q[:, own], a.iterations)
        except Exception as e:                    # noqa: BLE001  (reported below)
            errors.append(e)
            hub.barrier.abort()
            hub.start.abort()

    threads = [threading.Thread(target=side, args=(n,)) for n in ("Left", "Right")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(600)
    assert not errors, errors
    x = np.concatenate([out["Left"][0], out["Right"][0]])
    Q = np.concatenate([out["Left"][1], out["Right"][1]], axis=1)
    assert np.abs(x - xs).max() < 1e-12
    assert len(out["Left"][2]) == STEPS and min(out["Left"][2]) > 1   # coupling iterations
    assert np.abs(Q - Qs).max() < 1e-12
    assert abs(Q[1].sum() - Qs[1].sum()) * DX < 1e-14                 # mass, the interface
