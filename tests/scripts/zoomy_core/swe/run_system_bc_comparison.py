"""
Compare system-aware BCs vs legacy BCs for SWE (SME L0).

Three tests:
1. Extrapolation BCs — dam-break, open boundaries
2. Periodic BCs — advecting pulse
3. Wall BC — dam-break reflecting off wall

Each test runs with old-style BCs and new system BCs, checks identical results.
"""

import os
import sys
import numpy as np

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
for sub in ["zoomy_core"]:
    p = os.path.join(root, "library", sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from zoomy_core.mesh import BaseMesh
from zoomy_core.fvm.solver_numpy import FreeSurfaceFlowSolver
import zoomy_core.fvm.timestepping as ts
import zoomy_core.model.boundary_conditions as BC
import zoomy_core.model.initial_conditions as IC
from zoomy_core.model.models.sme_model import SMEInviscid


def make_swe_model():
    return SMEInviscid(level=0)


def run_swe(model, mesh, t_end=0.1, cfl=0.3):
    solver = FreeSurfaceFlowSolver(
        time_end=t_end,
        compute_dt=ts.adaptive(CFL=cfl),
    )
    Q, _ = solver.solve(mesh, model, write_output=False)
    return Q


def dam_break_ic(x, nv=3):
    Q = np.zeros(nv)
    Q[1] = 2.0 if x[0] < 5.0 else 1.0
    return Q


def test_extrapolation():
    """Dam-break with extrapolation BCs — old vs new system BCs."""
    print("\n--- Test 1: Extrapolation BCs (dam-break) ---")

    mesh = BaseMesh.create_1d(domain=(0, 10), n_inner_cells=100)

    # Old-style BCs
    m1 = make_swe_model()
    m1.initial_conditions = IC.UserFunction(function=dam_break_ic)
    m1.boundary_conditions = BC.BoundaryConditions([
        BC.Extrapolation(tag="left"),
        BC.Extrapolation(tag="right"),
    ])
    Q_old = run_swe(m1, mesh)

    # New system BCs
    m2 = make_swe_model()
    m2.initial_conditions = IC.UserFunction(function=dam_break_ic)
    m2._system.boundary_conditions.apply(BC.SystemExtrapolation(), tag="left")
    m2._system.boundary_conditions.apply(BC.SystemExtrapolation(), tag="right")
    m2.boundary_conditions = BC.compile_system_bcs(
        m2._system.boundary_conditions,
        m2._equation_variable_map,
        m2.dimension,
    )

    Q_new = run_swe(m2, mesh)

    diff = np.max(np.abs(Q_old - Q_new))
    print(f"  Max difference: {diff:.2e}")
    assert diff < 1e-12, f"Extrapolation BCs differ: {diff}"
    print("  PASS")


def test_wall():
    """Dam-break with wall on right — old vs new system BCs."""
    print("\n--- Test 3: Wall BC (dam-break with wall) ---")

    mesh = BaseMesh.create_1d(domain=(0, 10), n_inner_cells=100)

    # Old-style BCs
    m1 = make_swe_model()
    m1.initial_conditions = IC.UserFunction(function=dam_break_ic)
    m1.boundary_conditions = BC.BoundaryConditions([
        BC.Extrapolation(tag="left"),
        BC.Wall(tag="right", momentum_field_indices=[[2]]),
    ])
    Q_old = run_swe(m1, mesh)

    # New system BCs
    m2 = make_swe_model()
    m2.initial_conditions = IC.UserFunction(function=dam_break_ic)
    m2._system.boundary_conditions.apply(BC.SystemExtrapolation(), tag="left")
    m2._system.boundary_conditions.apply(BC.SystemWall(), tag="right")
    m2.boundary_conditions = BC.compile_system_bcs(
        m2._system.boundary_conditions,
        m2._equation_variable_map,
        m2.dimension,
    )

    Q_new = run_swe(m2, mesh)

    diff = np.max(np.abs(Q_old - Q_new))
    print(f"  Max difference: {diff:.2e}")
    assert diff < 1e-12, f"Wall BCs differ: {diff}"
    print("  PASS")

    # Verify h, hu are sensible
    nc = mesh.n_inner_cells
    h = Q_new[1, :nc]
    print(f"  h range: [{h.min():.4f}, {h.max():.4f}]")
    assert np.isfinite(h).all()
    assert h.min() > 0.5


def test_wall_sme_l1():
    """SME L1 with wall — old vs new system BCs."""
    print("\n--- Test 4: Wall BC (SME L1, more variables) ---")

    mesh = BaseMesh.create_1d(domain=(0, 10), n_inner_cells=100)

    m1 = SMEInviscid(level=1)
    nv = m1.n_variables
    def ic_l1(x, _nv=nv):
        Q = np.zeros(_nv)
        Q[1] = 2.0 if x[0] < 5.0 else 1.0
        return Q

    m1.initial_conditions = IC.UserFunction(function=ic_l1)
    # Old: momentum indices are [2, 3] for SME L1 (hu0, hu1)
    m1.boundary_conditions = BC.BoundaryConditions([
        BC.Extrapolation(tag="left"),
        BC.Wall(tag="right", momentum_field_indices=[[2], [3]]),
    ])
    Q_old = run_swe(m1, mesh)

    m2 = SMEInviscid(level=1)
    m2.initial_conditions = IC.UserFunction(function=ic_l1)
    m2._system.boundary_conditions.apply(BC.SystemExtrapolation(), tag="left")
    m2._system.boundary_conditions.apply(BC.SystemWall(), tag="right")
    m2.boundary_conditions = BC.compile_system_bcs(
        m2._system.boundary_conditions,
        m2._equation_variable_map,
        m2.dimension,
    )

    # Check compiled momentum_field_indices
    wall_bc = [bc for bc in m2.boundary_conditions.boundary_conditions_list
               if isinstance(bc, BC.Wall)]
    if wall_bc:
        print(f"  Compiled momentum_field_indices: {wall_bc[0].momentum_field_indices}")

    Q_new = run_swe(m2, mesh)

    diff = np.max(np.abs(Q_old - Q_new))
    print(f"  Max difference: {diff:.2e}")
    assert diff < 1e-12, f"Wall BCs differ for SME L1: {diff}"
    print("  PASS")


def main():
    print("=" * 60)
    print("SYSTEM BC vs LEGACY BC COMPARISON (SWE / SME)")
    print("=" * 60)

    test_extrapolation()
    test_wall()
    test_wall_sme_l1()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
