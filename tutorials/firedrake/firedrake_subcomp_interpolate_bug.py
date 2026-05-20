"""Standalone reproducer: ``Q.sub(i).interpolate``-into-scalar
silently corrupts ``Q.sub(0)`` for ``i >= 2`` under MPI.

Symptom
-------
Given a vector-valued DG(1) Function ``Q`` on a triangular mesh, doing

    qi = Function(V_scalar)
    qi.interpolate(Q.sub(i))     # i in {2, 3}

silently zeroes scattered values in ``Q.dat.data[:, 0]`` (the first
component, ``b``) on partitions touched by an MPI rank boundary.
``i = 0`` and ``i = 1`` do not show the symptom.  The corruption is
small per call (a few cells per edge rank) but compounds across
iterations of a component loop, and the resulting "limiter touches
the bathymetry" finger-print is what brought this script into
existence.

Reproduction
------------
::

    # Serial — should not show the symptom, since there are no halos.
    python tutorials/firedrake/firedrake_subcomp_interpolate_bug.py

    # MPI N=4 — expected to FAIL on at least one of the i=2 / i=3
    # iterations on partitions sharing a rank boundary with another.
    mpirun -n 4 python tutorials/firedrake/firedrake_subcomp_interpolate_bug.py

What the script does
--------------------

1. Builds a 30x30 ``UnitSquareMesh`` (small, fast).
2. Creates ``V = VectorFunctionSpace(mesh, "DG", 1, dim=4)`` and a
   scalar ``V0 = FunctionSpace(mesh, "DG", 1)``.
3. Fills ``Q.dat.data`` with **per-component constants**
   (``Q[:, i] = 10 * (i+1)``) so any change to ``Q[:, 0]`` is
   immediately visible.
4. Iterates ``i = 0..3``:
   - Snapshots ``Q.dat.data[:, 0]``.
   - Performs ``qi.interpolate(Q.sub(i))`` (the suspected-buggy call).
   - Snapshots ``Q.dat.data[:, 0]`` again.
   - Reports per-rank ``Δb_max``.
5. Asserts that ``Δb_max == 0`` for every ``i``.

Workaround
----------
Replace the call with a direct numpy slice on the underlying ``dat``::

    qi.dat.data[:] = Q.dat.data_ro[:, i]

— see ``library/zoomy_firedrake/zoomy_firedrake/firedrake_compat.py``
for the wrapper used in production.

When this is filed upstream and fixed, the assertion at the end of
this script should pass without the workaround.

Tested on Firedrake 2025.10.x (the version that ships in
``zoomy-fd-test:local``).
"""

import sys

import firedrake as fd
from firedrake.petsc import PETSc
from mpi4py import MPI


def main() -> int:
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # Small mesh — keep the test cheap.
    mesh = fd.UnitSquareMesh(30, 30)
    V = fd.VectorFunctionSpace(mesh, "DG", 1, dim=4)
    V0 = fd.FunctionSpace(mesh, "DG", 1)

    Q = fd.Function(V)
    # Spatially varying per-component data (mirror Malpasset where
    # ``b`` ranges -20..+100, ``h`` 0..50, etc.).  Constant-per-
    # component data was not enough to trigger the bug observed in
    # production — try a non-trivial spatial gradient on every
    # component.
    x, y = fd.SpatialCoordinate(mesh)
    Q.interpolate(fd.as_vector([
        50.0 + 50.0 * x,                      # component 0 ("b"): 50..100
        20.0 + 20.0 * y,                      # component 1 ("h"): 20..40
        5.0  * x * y,                         # component 2 ("hu"): 0..5
        -3.0 * (x - 0.5) * (y - 0.5),         # component 3 ("hv"): -0.75..0.75
    ]))
    # Apply a vertex-based slope limiter once to put the data into a
    # state closer to what the Malpasset run produces between steps
    # (Kuzmin output, possibly with halo state we haven't seen yet).
    V0_scalar = fd.FunctionSpace(mesh, "DG", 1)
    pre_limiter = fd.VertexBasedLimiter(V0_scalar)
    for j in range(4):
        scratch = fd.Function(V0_scalar)
        scratch.dat.data[:] = Q.dat.data_ro[:, j]
        pre_limiter.apply(scratch)
        Q.dat.data[:, j] = scratch.dat.data_ro[:]

    PETSc.Sys.Print(
        f"[reproducer] mesh: 30x30 UnitSquareMesh  size={size}  "
        f"V=VectorFunctionSpace(dim=4)  V0=FunctionSpace(DG, 1)",
        comm=MPI.COMM_WORLD,
    )

    failures = []
    # The bug we observed in the Malpasset run requires the
    # **previous iteration's direct write** to ``Q.dat.data[:, j]``
    # before the next ``interpolate(Q.sub(k))`` for ``k != j``: the
    # direct numpy write bypasses PyOP2's halo dirty-tracking, and
    # the next interpolation reads stale halo state to "correct"
    # what it thinks are inconsistent DOFs — silently zeroing
    # scattered values in ``Q.sub(0)``.
    #
    # Vanilla calls (no interleaved writes) do NOT trigger the bug.
    # The fix at the call site is to gather all reads first and
    # complete all writes after — never interleave a read and write
    # within the same loop.
    for i in range(4):
        b_pre = Q.dat.data_ro[:, 0].copy()
        qi = fd.Function(V0)
        # The suspected-buggy call:
        qi.interpolate(Q.sub(i))
        b_post = Q.dat.data_ro[:, 0]
        delta = float(abs(b_post - b_pre).max())
        # Mimic the production write-back to trigger the dat-state
        # that breaks the *next* interpolate.
        if i > 0:  # don't touch b's column
            Q.dat.data[:, i] = qi.dat.data_ro[:]
        # All ranks contribute their local delta to the global max.
        delta_global = MPI.COMM_WORLD.allreduce(delta, op=MPI.MAX)
        PETSc.Sys.Print(
            f"[reproducer] qi.interpolate(Q.sub({i}))  →  "
            f"global Δb_max = {delta_global:.3e}  "
            f"(rank {rank} local Δb_max = {delta:.3e})",
            comm=MPI.COMM_WORLD,
        )
        # Each call's local delta is also printed per-rank for diagnosis.
        PETSc.Sys.syncPrint(
            f"  rank={rank}  i={i}  local Δb_max = {delta:.3e}"
        )
        PETSc.Sys.syncFlush()
        if delta_global > 0.0:
            failures.append((i, delta_global))

    PETSc.Sys.Print("", comm=MPI.COMM_WORLD)
    if failures:
        PETSc.Sys.Print(
            "[reproducer] FAILED: ``Q.sub(i).interpolate`` corrupted "
            "``Q.sub(0)`` for the following components:",
            comm=MPI.COMM_WORLD,
        )
        for i, d in failures:
            PETSc.Sys.Print(
                f"   i = {i}:  global Δb_max = {d:.3e}",
                comm=MPI.COMM_WORLD,
            )
        return 1

    PETSc.Sys.Print(
        "[reproducer] OK: no spurious modification of ``Q.sub(0)`` "
        "observed on any component.",
        comm=MPI.COMM_WORLD,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
