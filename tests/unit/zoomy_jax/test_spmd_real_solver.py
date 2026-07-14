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
from zoomy_core.misc.misc import ZArray
from zoomy_core.model.derivative_workflow import StructuredDerivativeModel
from zoomy_core.model.models import SME
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.numerics.numerical_system_model import ReconstructionSpec
from zoomy_core.systemmodel import SystemModel
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.mesh import partition_1d_contiguous, partition_xaxis_structured
from zoomy_jax.fvm.spmd_jax import (shard_global_state, gather_owned,
                                    run_solver_sharded)

from sympy import Matrix

N_STEPS = 6


# ── 2D SWE model (flat bed) ──────────────────────────────────────────────────
class _SWE2D(StructuredDerivativeModel):
    dimension = 2
    variables = ["h", "hu", "hv"]
    parameters = {"g": (9.81, "positive")}

    def flux(self):
        h, hu, hv = self.Q.h, self.Q.hu, self.Q.hv
        g = self.params.g
        u, v = hu / h, hv / h
        F = Matrix.zeros(self.n_variables, self.dimension)
        F[0, 0] = hu; F[0, 1] = hv
        F[1, 0] = hu * u + 0.5 * g * h * h; F[1, 1] = hu * v
        F[2, 0] = hv * u; F[2, 1] = hv * v + 0.5 * g * h * h
        return ZArray(F)

    def source(self):
        return ZArray.zeros(self.n_variables)


class _SWE2DSolver(HyperbolicSolver):
    def _build_reconstruction(self, mesh, symbolic_model):
        from zoomy_jax.fvm.reconstruction_jax import (
            ConstantReconstruction, FreeSurfaceLSQMUSCLJAX)
        if self.nsm.reconstruction.order >= 2:
            return FreeSurfaceLSQMUSCLJAX(mesh, symbolic_model.dimension, h_index=0,
                                          eps_wet=1e-6, limiter=self.nsm.reconstruction.limiter)
        return ConstantReconstruction(mesh, symbolic_model.dimension)


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
        sm = SystemModel.from_model(SME(level=0, dimension=2))
        sm.attach_boundary_conditions(bcs)
        names = [str(s) for s in sm.state]; ih = names.index("h")
        sm.initial_conditions = IC.UserFunction(
            function=lambda x: np.array([smooth(float(x[0])) if i == ih else 0.0
                                         for i in range(len(names))]))
        sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
        nsm = NumericalSystemModel.from_system_model(
            sm, reconstruction=ReconstructionSpec(order=order, limiter="venkatakrishnan"))
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
            BC.Wall(tag="bottom"), BC.Wall(tag="top")])
        model = _SWE2D(boundary_conditions=bcs,
                       initial_conditions=IC.UserFunction(
                           function=lambda x: np.array([smooth(float(x[0])), 0.0, 0.0])))
        nsm = NumericalSystemModel.from_system_model(
            model, reconstruction=ReconstructionSpec(order=order, limiter="minmod"))
        gmesh = LSQMesh.create_2d(domain=DOMAIN, nx=NX, ny=NY)
        parts = partition_xaxis_structured(gmesh, n_parts=n_devs, halo=halo_x,
                                           domain=DOMAIN, shape=(NX, NY))
        solver = _SWE2DSolver()
        _, Qaux0 = solver.setup_simulation(parts[1], nsm)
        naux = int(np.asarray(Qaux0).shape[0])
        u0 = np.zeros((3, NX * NY))
        for ix in range(NX):
            for iy in range(NY):
                u0[0, ix * NY + iy] = smooth((ix + 0.5) * DX)
        Q_pad, n_local = shard_global_state(u0, n_devs, halo)
        Qaux_pad = jnp.zeros((naux, Q_pad.shape[1]), dtype=Q_pad.dtype)
        Q_out, _ = run_solver_sharded(solver, n_devs, halo, N_STEPS, DT)(Q_pad, Qaux_pad)
        return gather_owned(np.asarray(Q_out), n_devs, n_local, halo)

    a, b = run(3), run(4)
    assert np.isfinite(b).all()
    assert np.max(np.abs(a - b)) < 1e-10, "2D real-solver SPMD not device-count transparent"
