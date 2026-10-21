"""SPMD through the REAL solver: the halo hook in ``HyperbolicSolver``
(``_explicit_hyperbolic_step`` → ``_halo_wrap``) lets the actual solver ``step``
run inside ``jax.shard_map`` over a partitioned mesh — one code path, sharded or
not.  We assert the run is **transparent to the device count** (bit-identical
across {3,4} devices): the correctness guarantee for parallelisation.

Covered: explicit hyperbolic, order 1 & 2, 1D (``partition_1d_contiguous``) and
2D x-strips (``partition_xaxis_structured``).  Four host devices are simulated on
CPU via ``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
from __future__ import annotations

import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
from loguru import logger
logger.remove()

import zoomy_core.model.initial_conditions as IC
import zoomy_core.model.boundary_conditions as BC
from zoomy_core.mesh import LSQMesh
from zoomy_core.model.models import SME
from zoomy_core.model.models.closures import (
    ManningFriction, Newtonian, StressFree)
from zoomy_core.numerics import Limiter, NumericalSystemModel
from zoomy_core.numerics.numerical_system_model import ReconstructionSpec
from zoomy_core.systemmodel import SystemModel
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.mesh import partition_1d_contiguous, partition_xaxis_structured
from zoomy_jax.fvm.spmd_jax import (shard_global_state, gather_owned,
                                    run_solver_sharded, distributed_gmres,
                                    collective_inner, spmd_device_mesh,
                                    halo_exchange_periodic)

from functools import partial
from jax.sharding import PartitionSpec as P
try:
    from jax.experimental.shard_map import shard_map
except ImportError:                                   # pragma: no cover
    from jax import shard_map

N_STEPS = 6


# ── 2D SWE model (flat bed) ──────────────────────────────────────────────────
# DERIVED, not hand-built: ``SME(level=0, dimension=3)`` IS the 2-horizontal
# shallow-water system — state ``[b, h, q_x_0, q_y_0]``.  A hand-written
# ``StructuredDerivativeModel`` used to stand here, and because it registered no
# ``interpolate_to_3d`` rows the Wall BC could not derive its momentum vector
# and the test had to PIN ``momentum_field_indices=[[1, 2]]``.  The derived
# model carries the rows, so ``Wall`` resolves the group ``[[2, 3]]`` itself and
# no pin is needed anywhere in this file.
def _swe_closures():
    """The closure set of the suite's shallow-water model (``models.swe``): a
    DECLARED closure that reaches the SystemModel unbound raises
    ``UnclosedClosureError``, so ``SME(level=0)`` is never built bare."""
    return [Newtonian(), ManningFriction(), StressFree()]


def _swe2d_model(bcs, ic):
    return SME(level=0, dimension=3, closures=_swe_closures(),
               boundary_conditions=bcs, initial_conditions=ic)


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
@pytest.mark.parametrize("order", [1, 2])
def test_spmd_real_solver_1d_transparent(order):
    """SME(0) 1D: real solver.step over {3,4} devices is bit-identical."""
    if jax.device_count() < 4:
        pytest.skip("need 4 devices")
    N_TOTAL = 48; DOMAIN = (0.0, 1.0); DX = 1.0 / N_TOTAL; DT = 0.15 * DX
    halo = 1 if order == 1 else 2
    smooth = lambda x: 1.0 + 0.3 * np.sin(2 * np.pi * x)

    def run(n_devs):
        bcs = BC.BoundaryConditions([
            BC.Periodic(tag="left", periodic_to_physical_tag="right"),
            BC.Periodic(tag="right", periodic_to_physical_tag="left")])
        sm = SystemModel.from_model(
            SME(level=0, dimension=2, closures=_swe_closures()))
        sm.attach_boundary_conditions(bcs)
        names = [str(s) for s in sm.state]; ih = names.index("h")
        sm.initial_conditions = IC.UserFunction(
            function=lambda x: np.array([smooth(float(x[0])) if i == ih else 0.0
                                         for i in range(len(names))]))
        sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
        nsm = NumericalSystemModel.from_system_model(
            sm, reconstruction=ReconstructionSpec(order=order, limiter=Limiter.venkatakrishnan))
        gmesh = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=N_TOTAL)
        parts = partition_1d_contiguous(gmesh, n_parts=n_devs, halo=halo)
        solver = HyperbolicSolver()
        _, Qaux0 = solver.setup_simulation(parts[1], nsm)
        naux = int(np.asarray(Qaux0).shape[0])
        xc = DOMAIN[0] + (np.arange(N_TOTAL) + 0.5) * DX
        u0 = np.zeros((len(names), N_TOTAL)); u0[ih] = smooth(xc)
        Q_pad, n_local = shard_global_state(u0, n_devs, halo)
        Qaux_pad = jnp.zeros((naux, Q_pad.shape[1]), dtype=Q_pad.dtype)
        Q_out, _ = run_solver_sharded(solver, n_devs, halo, N_STEPS, DT)(Q_pad, Qaux_pad)
        return gather_owned(np.asarray(Q_out), n_devs, n_local, halo)

    a, b = run(3), run(4)
    assert np.isfinite(b).all()
    assert np.max(np.abs(a - b)) < 1e-10, "1D real-solver SPMD not device-count transparent"


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
@pytest.mark.parametrize("order", [1, 2])
def test_spmd_real_solver_2d_transparent(order):
    """SWE2D x-strips: real solver.step over {3,4} devices is bit-identical."""
    if jax.device_count() < 4:
        pytest.skip("need 4 devices")
    NX, NY = 12, 4; LX, LY = 12.0, 4.0; DOMAIN = (0.0, LX, 0.0, LY)
    DX = LX / NX; DT = 0.15 * DX
    halo_x = 1 if order == 1 else 2; halo = halo_x * NY
    smooth = lambda x: 0.5 + 0.1 * np.sin(2 * np.pi * x / LX)

    def run(n_devs):
        bcs = BC.BoundaryConditions([
            BC.Periodic(tag="left", periodic_to_physical_tag="right"),
            BC.Periodic(tag="right", periodic_to_physical_tag="left"),
            # NO ``momentum_field_indices`` pin: the derived model registers the
            # ``interpolate_to_3d`` rows Wall derives the momentum vector from.
            BC.Wall(tag="bottom"),
            BC.Wall(tag="top")])
        # state [b, h, q_x_0, q_y_0]: flat bed, smooth depth, momentum at rest
        model = _swe2d_model(bcs, IC.UserFunction(
            function=lambda x: np.array([0.0, smooth(float(x[0])), 0.0, 0.0])))
        nsm = NumericalSystemModel.from_system_model(
            model, reconstruction=ReconstructionSpec(order=order, limiter=Limiter.minmod))
        gmesh = LSQMesh.create_2d(domain=DOMAIN, nx=NX, ny=NY)
        parts = partition_xaxis_structured(gmesh, n_parts=n_devs, halo=halo_x,
                                           domain=DOMAIN, shape=(NX, NY))
        solver = HyperbolicSolver()
        _, Qaux0 = solver.setup_simulation(parts[1], nsm)
        naux = int(np.asarray(Qaux0).shape[0])
        u0 = np.zeros((4, NX * NY))
        for ix in range(NX):
            for iy in range(NY):
                u0[1, ix * NY + iy] = smooth((ix + 0.5) * DX)
        Q_pad, n_local = shard_global_state(u0, n_devs, halo)
        Qaux_pad = jnp.zeros((naux, Q_pad.shape[1]), dtype=Q_pad.dtype)
        Q_out, _ = run_solver_sharded(solver, n_devs, halo, N_STEPS, DT)(Q_pad, Qaux_pad)
        return gather_owned(np.asarray(Q_out), n_devs, n_local, halo)

    a, b = run(3), run(4)
    assert np.isfinite(b).all()
    assert np.max(np.abs(a - b)) < 1e-10, "2D real-solver SPMD not device-count transparent"


# The spmd tier's slowest test, MEASURED 195 s when the tier runs on its own
# (next slowest 64 s; whole tier 386 s) — under the 300 s size-rule threshold,
# hence ``small``.  The cost is XLA COMPILE, not arithmetic: the N-step Arnoldi
# is UNROLLED inside ``shard_map`` with a collective per inner product.
#
# REPORTED, not accommodated: in a SINGLE process that has already run the rest
# of the zoomy_jax tier, this test (and others in this tier) blow past 10 min
# apiece — the shard_map compile time grows with what the process has already
# traced.  Marking one test ``large`` does not fix that; it is a property of
# running the whole tier in one interpreter, and it needs a real fix (smaller
# N, or a per-file process) rather than a mark that hides it.
@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.jax
def test_distributed_gmres_solves_global_elliptic_system():
    """Stage 2 core: distributed_gmres (halo-in-matvec + psum inner products)
    solves the GLOBAL 1D periodic diffusion system (I - alpha*L) x = b across 4
    devices — matching the single-device dense solve.  This is the reusable
    primitive the global-implicit source / diffusion / Chorin-pressure solves
    need (their local jax_gmres would otherwise converge each device's slab)."""
    if jax.device_count() < 4:
        pytest.skip("need 4 devices")
    # N=16, not 64: `maxiter=N` unrolls an N-step Arnoldi with O(N^2) H updates
    # + collectives under shard_map, and XLA compile blows past 20 min at N=64.
    # 16 still gives a full-dimension Krylov space (=> exact solve) on 4 devices.
    N, N_DEVS, HALO, ALPHA = 16, 4, 1, 0.35
    rng = np.random.default_rng(0)
    b_np = rng.standard_normal(N)
    L = -2*np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)
    L[0, -1] = 1.0; L[-1, 0] = 1.0
    A = np.eye(N) - ALPHA*L
    x_ref = np.linalg.solve(A, b_np)

    def matvec(v):
        vp = jnp.pad(v, ((0, 0), (HALO, HALO)))
        vp = halo_exchange_periodic(vp, HALO, "cells", N_DEVS)
        lap = vp[:, :-2] - 2*vp[:, 1:-1] + vp[:, 2:]
        return v - ALPHA*lap

    dmesh = spmd_device_mesh(N_DEVS, "cells")
    inner = collective_inner("cells")

    def make_solve(m):
        # m is the STATIC Krylov dimension (the Arnoldi loop is a python
        # `range(m)`), so it must be closed over, not passed through shard_map.
        @partial(shard_map, mesh=dmesh, in_specs=P(None, "cells"),
                 out_specs=(P(None, "cells"), P(None)), check_rep=False)
        def solve(b):
            x, rel = distributed_gmres(matvec, b, inner=inner, maxiter=m)
            return x, jnp.asarray([rel])
        return solve

    solve = make_solve(N)
    x_spmd, rel = solve(jnp.asarray(b_np[None, :]))
    x_spmd = np.asarray(x_spmd)[0]
    assert np.max(np.abs(x_spmd - x_ref)) < 1e-8
    assert np.max(np.abs(A @ x_spmd - b_np)) < 1e-8
    # the reported residual must match the residual the caller can measure —
    # a diagnostic nobody can trust is what this return value replaced.
    assert float(np.asarray(rel).ravel()[0]) < 1e-8
    assert np.isclose(float(np.asarray(rel).ravel()[0]),
                      np.linalg.norm(A @ x_spmd - b_np) / np.linalg.norm(b_np),
                      atol=1e-8)

    # ⚠ maxiter=N gives a FULL-dimension Krylov space — an exact solve in
    # disguise, which is why the original version of this test could not have
    # caught a convergence failure.  Starve the subspace and the primitive must
    # SAY so rather than return a confident-looking wrong answer.
    x_bad, rel_bad = make_solve(3)(jnp.asarray(b_np[None, :]))
    rel_bad = float(np.asarray(rel_bad).ravel()[0])
    assert rel_bad > 1e-6, "starved Krylov space must report a large residual"
    assert np.isclose(rel_bad,
                      np.linalg.norm(A @ np.asarray(x_bad)[0] - b_np)
                      / np.linalg.norm(b_np), rtol=1e-5)
