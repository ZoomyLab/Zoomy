"""The lifted shared face (`precice_march(interface=)`): a coupled face between two levels of
the SME hierarchy posed as ONE Riemann problem in the richer model (Dahm, Giesselmann & Helzel
2024, Theorem 4.4), both participants on the column-lifted pair, each keeping its own rows.

* offline (no preCICE): the two sides' rows of the face telescope bit-exactly on the depth and
  (flat bed) discharge rows, the low side's rows are the richer kernel's, and the template
  refuses an interface that lacks the own rows;
* preCICE (two processes): SME(1) | SME(2) over a dam break conserves mass to round-off.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

NZ, FIELDS = 8, ("b", "h", "u", "v", "w", "p")


def nsm_of(level, tags=("left", "right"), coupled=None):
    """SME(level), Newtonian + Navier slip + stress free, walls (or `coupled` on one tag), a
    dam break at x = -0.05."""
    from zoomy_core.model.boundary_conditions import BoundaryConditions, Coupled, FromModel
    from zoomy_core.model.initial_conditions import RP
    from zoomy_core.model.models import SME
    from zoomy_core.model.models import closures as C
    from zoomy_core.numerics import NumericalSystemModel, ReconstructionSpec
    from zoomy_core.systemmodel.system_model import SystemModel
    bcs = [Coupled(tag=t, mesh_name="M") if t == coupled else FromModel(tag=t, definition="wall")
           for t in tags]
    sm = SystemModel.from_model(SME(
        level=level, project_nz=NZ, closures=[C.Newtonian(), C.NavierSlip(), C.StressFree()],
        boundary_conditions=BoundaryConditions(bcs),
        initial_conditions=RP(high=lambda n: np.array([0.0, 0.18] + [0.0] * (n - 2)),
                              low=lambda n: np.array([0.0, 0.10] + [0.0] * (n - 2)),
                              jump_position_x=-0.05)))
    sm.parameter_values["nu"], sm.parameter_values["lambda_s"] = 1e-3, 0.5
    return NumericalSystemModel.from_system_model(sm, reconstruction=ReconstructionSpec(order=1))


def offline():
    """A `ColumnPreciceAdapter` (its column maps) that never opens a preCICE participant."""
    from zoomy_jax.symsolver.runtime import ColumnPreciceAdapter

    class Offline(ColumnPreciceAdapter):
        def __init__(self):
            self.nz, self.y, self.columns, self.rt = NZ, 0.5, None, None
            self.wn = self.rn = list(FIELDS)

        def set_vertices(self, coords):
            self.n_faces = len(coords)
    return Offline()


def offline_side(level, side):
    """The blocks of one participant (SME(level), the coupled face on its interface end) with
    an adapter that never opens preCICE; returns `(solver, adapter)`."""
    from zoomy_core.mesh import LSQMesh
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
    tag = "right" if side == 0 else "left"
    a = offline()
    s = SymbolicHyperbolicSolverJax(cfl=0.9, dt=1e-3, couple=dict(
        adapter=a, tag=tag, side=side, interface=nsm_of(2)))
    s.setup(LSQMesh.create_1d(domain=(-0.1, 0.0) if side == 0 else (0.0, 0.1),
                              n_inner_cells=10), nsm_of(level, coupled=tag))
    return s, a


@pytest.mark.small
def test_lifted_face_telescopes_and_truncates():
    low, a_low = offline_side(1, 0)
    high, a_high = offline_side(2, 1)
    assert "interface.numerical_face_sides" in low.source
    assert "lifted_interface_flux" in high.source
    qL = np.array([[0.0], [0.15], [0.03], [0.004]])                  # SME(1)
    qR = np.array([[0.0], [0.11], [-0.01], [0.002], [0.0007]])       # SME(2)
    # the exchanged columns, lifted into SME(2) by either side alike
    pL, pR = (a.lift.project(a._column(q, None)) for a, q in ((a_low, qL), (a_high, qR)))
    assert np.abs(pL[:4] - qL).max() < 1e-15 and abs(pL[4, 0]) < 1e-15   # zero-padded
    assert np.abs(pR - qR).max() < 1e-15
    p = np.asarray(low.rt.parameters)
    W = []
    for s, own, peer, side in ((low, pL, pR, 0), (high, pR, pL, 1)):
        f = s.module["lifted_interface_flux"](
            s.module["interface"].numerical_face_sides, s.module["interface"].update_aux_variables,
            len(s.interface.sm.aux_state), side, own, peer,
            tuple(range(len(s.nsm.sm.state))), p, s.mesh)
        W.append(np.asarray(f)[:, np.flatnonzero(np.any(np.asarray(f) != 0, axis=0))][:, 0])
    assert W[0].shape == (4,) and W[1].shape == (5,)
    assert W[0][1] + W[1][1] == 0.0                      # depth: bit-exact
    assert W[0][2] + W[1][2] == 0.0                      # discharge, flat bed: bit-exact
    assert abs(W[1][4]) > 0                              # q_2 crosses into SME(2) only


@pytest.mark.small
def test_interface_must_carry_the_own_rows():
    from zoomy_core.mesh import LSQMesh
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
    s = SymbolicHyperbolicSolverJax(cfl=0.9, dt=1e-3, couple=dict(
        adapter=offline(), tag="left", side=1, interface=nsm_of(1)))
    with pytest.raises(ValueError, match="lacks the own rows"):
        s.setup(LSQMesh.create_1d(domain=(0.0, 0.1), n_inner_cells=10),
                nsm_of(2, coupled="left"))


# ── preCICE: two participants ───────────────────────────────────────────────
T, DT, N = 0.1, 1e-3, 20


def write_config(work):
    """Parallel explicit, window = DT, initial data both ways, the six column fields."""
    L, R = [f"L_{f}" for f in FIELDS], [f"R_{f}" for f in FIELDS]
    use = "".join(f'<use-data name="{n}" />' for n in L + R)

    def part(me, peer, w, r):
        return (f'<participant name="{me}"><provide-mesh name="{me}-Mesh" />'
                f'<receive-mesh name="{peer}-Mesh" from="{peer}" geometric-filter="no-filter" />'
                + "".join(f'<write-data name="{n}" mesh="{me}-Mesh" />' for n in w)
                + "".join(f'<read-data name="{n}" mesh="{me}-Mesh" />' for n in r)
                + f'<mapping:nearest-neighbor direction="read" from="{peer}-Mesh" '
                f'to="{me}-Mesh" constraint="consistent" /></participant>\n')
    ex = "".join(f'<exchange data="{n}" mesh="{a}-Mesh" from="{a}" to="{b}" initialize="true" />'
                 for names, a, b in ((L, "Left", "Right"), (R, "Right", "Left")) for n in names)
    Path(work, "precice-config.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8" ?>\n<precice-configuration>\n'
        + "".join(f'<data:scalar name="{n}" />' for n in L + R)
        + f'<mesh name="Left-Mesh" dimensions="3">{use}</mesh>'
        f'<mesh name="Right-Mesh" dimensions="3">{use}</mesh>\n'
        + part("Left", "Right", L, R) + part("Right", "Left", R, L)
        + f'<m2n:sockets acceptor="Left" connector="Right" exchange-directory="{work}" />'
        '<coupling-scheme:parallel-explicit><participants first="Left" second="Right" />'
        f'<max-time value="{T}" /><time-window-size value="{DT}" />{ex}'
        '</coupling-scheme:parallel-explicit>\n</precice-configuration>\n')


def participant(name, work):
    """One side: SME(1) on [-0.1, 0] (Left) or SME(2) on [0, 0.1] (Right); writes NAME.npz."""
    import jax
    from zoomy_core.mesh import LSQMesh
    from zoomy_jax.symsolver.runtime import ColumnPreciceAdapter
    from zoomy_jax.symsolver.solver import SymbolicHyperbolicSolverJax
    left = name == "Left"
    tag, me, peer = ("right", "L", "R") if left else ("left", "R", "L")
    a = ColumnPreciceAdapter(name, str(Path(work, "precice-config.xml")), f"{name}-Mesh",
                             [f"{me}_{f}" for f in FIELDS], [f"{peer}_{f}" for f in FIELDS], NZ)
    frames = []
    s = SymbolicHyperbolicSolverJax(
        cfl=0.9, dt=DT, dt_write=DT, on_write=lambda t, Q, A: frames.append(np.array(Q)),
        couple=dict(adapter=a, tag=tag, side=0 if left else 1, interface=nsm_of(2)))
    Q, A = s.setup(LSQMesh.create_1d(domain=(-0.1, 0.0) if left else (0.0, 0.1),
                                     n_inner_cells=N), nsm_of(1 if left else 2, coupled=tag))
    s.run(Q, A)
    a.finalize()
    jax.effects_barrier()
    np.save(Path(work, f"{name}.npy"), np.stack(frames))


@pytest.mark.medium
def test_sme1_sme2_pair_conserves_mass(tmp_path):
    pytest.importorskip("precice")
    write_config(tmp_path)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    procs = [subprocess.Popen([sys.executable, __file__, n, str(tmp_path)], env=env,
                              cwd=tmp_path, stdout=subprocess.DEVNULL,
                              stderr=open(tmp_path / f"{n}.err", "w"))
             for n in ("Left", "Right")]
    try:
        for p in procs:
            p.wait(timeout=600)
    finally:
        for p in procs:
            p.kill()
    assert all(p.returncode == 0 for p in procs), [
        (tmp_path / f"{n}.err").read_text()[-2000:] for n in ("Left", "Right")]
    QL, QR = (np.load(tmp_path / f"{n}.npy") for n in ("Left", "Right"))
    assert len(QL) == len(QR) == round(T / DT) + 1
    mass = (QL[:, 1].sum(1) + QR[:, 1].sum(1)) * (0.1 / N)
    assert np.abs(mass - mass[0]).max() < 1e-15
    assert np.abs(QR[:, 4]).max() > 1e-3                 # q_2 is live in the SME(2) arm


if __name__ == "__main__":                               # a participant of the preCICE test
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    participant(sys.argv[1], sys.argv[2])
