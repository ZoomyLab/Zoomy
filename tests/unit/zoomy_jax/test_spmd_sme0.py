"""SPMD (multi-device) execution of the jax ``HyperbolicSolver`` on the live
``SME(level=0)`` shallow-water model.

Restored off the deleted ``advection`` fixture onto ``SME(level=0,
dimension=2)`` (1-D SWE, state ``[b, h, q_0]``).  Confirms the solver's flux
operator composes with ``jax.shard_map`` + ``ppermute`` halo exchange and
produces a result **bit-identical** to a replicated single-device run — the
property that lets the solver scale across devices (no MPI; just shard_map),
the parallel-computing path documented in ``zoomy_jax/README.md`` and
``fvm/halo_exchange_jax.py``.

Four host devices are simulated on CPU via
``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from functools import partial

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map  # stable signature in this jax

import zoomy_core.model.initial_conditions as IC
from zoomy_core.mesh import LSQMesh
from zoomy_core.systemmodel import SystemModel
from zoomy_core.model.models import SME
from zoomy_core.model.boundary_conditions import BoundaryConditions, Extrapolation
from zoomy_core.numerics import NumericalSystemModel
from zoomy_core.numerics.numerical_system_model import ReconstructionSpec
from zoomy_jax.fvm.solver_jax import HyperbolicSolver
from zoomy_jax.mesh.partition_jax import partition_1d_contiguous

N_TOTAL = 32
N_DEVS = 4
N_LOCAL = N_TOTAL // N_DEVS
DOMAIN = (0.0, 1.0)
DX = (DOMAIN[1] - DOMAIN[0]) / N_TOTAL
DT = 0.2 * DX
N_STEPS = 3


def _smooth(x):
    """Smooth periodic depth profile (positive everywhere)."""
    return 1.0 + 0.3 * np.sin(2 * np.pi * x)


def _periodic_halo(Q_pad, halo, axis_name, n):
    left_owned = Q_pad[:, halo:2 * halo]
    right_owned = Q_pad[:, -2 * halo:-halo]
    perm_r = [(i, (i + 1) % n) for i in range(n)]
    perm_l = [(i, (i - 1) % n) for i in range(n)]
    fill_l = lax.ppermute(right_owned, perm=perm_r, axis_name=axis_name)
    fill_r = lax.ppermute(left_owned, perm=perm_l, axis_name=axis_name)
    Q_pad = Q_pad.at[:, :halo].set(fill_l)
    Q_pad = Q_pad.at[:, -halo:].set(fill_r)
    return Q_pad


def _periodic_halo_np(padded_per_dev, halo):
    out = [np.array(a) for a in padded_per_dev]
    n = len(out)
    for d in range(n):
        out[d][:, :halo] = out[(d - 1) % n][:, N_LOCAL:N_LOCAL + halo]
        out[d][:, -halo:] = out[(d + 1) % n][:, halo:halo + halo]
    return out


def _setup(order):
    """Build the SME(0) solver + interior partition flux op at ``order``."""
    bcs = BoundaryConditions([Extrapolation(tag="left"),
                              Extrapolation(tag="right")])
    model = SME(level=0, dimension=2)
    sm = SystemModel.from_model(model)
    sm.attach_boundary_conditions(bcs)
    ns = len(sm.state)
    ih = [str(s) for s in sm.state].index("h")

    def ic(x):
        out = np.zeros(ns)
        out[ih] = _smooth(float(x[0]))          # depth; b=0, q_0=0
        return out

    sm.initial_conditions = IC.UserFunction(function=ic)
    sm.aux_initial_conditions = IC.Constant(constants=lambda n: np.zeros(n))
    nsm = NumericalSystemModel.from_system_model(
        sm, reconstruction=ReconstructionSpec(order=order,
                                              limiter="venkatakrishnan"))
    mesh_np = LSQMesh.create_1d(domain=DOMAIN, n_inner_cells=N_TOTAL)
    solver = HyperbolicSolver()
    Q_setup, Qaux_setup = solver.setup_simulation(mesh_np, nsm)
    return solver, Q_setup, Qaux_setup, ns, ih


def _run_case(order, halo):
    """Run N_STEPS SPMD (shard_map) vs replicated single-device; return the
    max owned-cell discrepancy (should be ~0 — bit-identical)."""
    if jax.device_count() < N_DEVS:
        pytest.skip(f"Need {N_DEVS} devices")
    spmd_mesh = Mesh(np.array(jax.devices()[:N_DEVS]), axis_names=("cells",))

    solver, Q_setup, Qaux_setup, ns, ih = _setup(order)
    runtime = solver._rt_model
    gmesh = solver._rt_mesh
    parameters = solver._rt_parameters

    parts = partition_1d_contiguous(gmesh, n_parts=N_DEVS, halo=halo)
    part_mesh = parts[1]                          # interior — no global BC
    flux_op = solver.get_flux_operator(part_mesh, runtime)

    # global IC: depth smooth in x, b=q=0
    xc = DOMAIN[0] + (np.arange(N_TOTAL) + 0.5) * DX
    u0 = np.zeros((ns, N_TOTAL), dtype=np.float64)
    u0[ih] = _smooth(xc)

    pad = lambda c: np.concatenate(
        [np.zeros((ns, halo)), c, np.zeros((ns, halo))], axis=1)
    chunks = [u0[:, d * N_LOCAL:(d + 1) * N_LOCAL] for d in range(N_DEVS)]
    Q_pad = jnp.asarray(np.concatenate([pad(c) for c in chunks], axis=1),
                        dtype=Q_setup.dtype)
    n_pad = N_LOCAL + 2 * halo
    Qaux_pad = jnp.zeros((Qaux_setup.shape[0], Q_pad.shape[1]),
                         dtype=Q_setup.dtype)
    dt_j = jnp.asarray(DT, dtype=Q_setup.dtype)
    t_j = jnp.asarray(0.0, dtype=Q_setup.dtype)

    # reference: replicated, per-device, host halo exchange
    ref = [np.asarray(Q_pad[:, d * n_pad:(d + 1) * n_pad]).copy()
           for d in range(N_DEVS)]
    for _ in range(N_STEPS):
        ref = _periodic_halo_np(ref, halo)
        for d in range(N_DEVS):
            Q_d = jnp.asarray(ref[d])
            dQ = flux_op(dt_j, t_j, Q_d, Qaux_pad[:, :n_pad], parameters,
                         jnp.zeros_like(Q_d))
            ref[d] = np.asarray(Q_d + DT * dQ)

    # SPMD path
    def spmd_step(Qp, Qap):
        Qp = _periodic_halo(Qp, halo, "cells", N_DEVS)
        dQ = flux_op(dt_j, t_j, Qp, Qap, parameters, jnp.zeros_like(Qp))
        return Qp + DT * dQ

    @partial(shard_map, mesh=spmd_mesh,
             in_specs=(P(None, "cells"), P(None, "cells")),
             out_specs=P(None, "cells"), check_rep=False)
    def run(Qp, Qap):
        Qf, _ = lax.scan(lambda c, _: (spmd_step(c, Qap), None),
                         Qp, jnp.arange(N_STEPS))
        return Qf

    Q_spmd = np.asarray(run(Q_pad, Qaux_pad))
    err = 0.0
    for d in range(N_DEVS):
        owned_spmd = Q_spmd[:, d * n_pad + halo:d * n_pad + halo + N_LOCAL]
        owned_ref = ref[d][:, halo:halo + N_LOCAL]
        err = max(err, float(np.max(np.abs(owned_spmd - owned_ref))))
    return err


@pytest.mark.jax
def test_spmd_sme0_bit_identity_order1():
    """SME(0) RK1: 4-device shard_map == replicated single-device."""
    err = _run_case(order=1, halo=1)
    assert err < 1e-10, f"SPMD vs single-device drift {err:.3e}"


@pytest.mark.jax
def test_spmd_sme0_bit_identity_order2():
    """SME(0) order-2 MUSCL (RK2-equivalent flux): the full 2nd-order LSQ
    pipeline composes with shard_map, halo=2."""
    err = _run_case(order=2, halo=2)
    assert err < 1e-10, f"SPMD vs single-device drift {err:.3e}"


@pytest.mark.jax
def test_spmd_sme0_parameters_passed_through_shard_map():
    """Parameters reach the sharded flux op as an ARGUMENT (not closure-baked):
    perturbing g changes the sharded result, proving live parameter passing."""
    if jax.device_count() < N_DEVS:
        pytest.skip(f"Need {N_DEVS} devices")
    spmd_mesh = Mesh(np.array(jax.devices()[:N_DEVS]), axis_names=("cells",))
    solver, Q_setup, Qaux_setup, ns, ih = _setup(order=1)
    runtime, gmesh = solver._rt_model, solver._rt_mesh
    halo = 1
    parts = partition_1d_contiguous(gmesh, n_parts=N_DEVS, halo=halo)
    flux_op = solver.get_flux_operator(parts[1], runtime)
    xc = DOMAIN[0] + (np.arange(N_TOTAL) + 0.5) * DX
    u0 = np.zeros((ns, N_TOTAL)); u0[ih] = _smooth(xc)
    pad = lambda c: np.concatenate(
        [np.zeros((ns, halo)), c, np.zeros((ns, halo))], axis=1)
    chunks = [u0[:, d * N_LOCAL:(d + 1) * N_LOCAL] for d in range(N_DEVS)]
    Q_pad = jnp.asarray(np.concatenate([pad(c) for c in chunks], axis=1),
                        dtype=Q_setup.dtype)
    Qaux_pad = jnp.zeros((Qaux_setup.shape[0], Q_pad.shape[1]),
                         dtype=Q_setup.dtype)
    dt_j = jnp.asarray(DT, dtype=Q_setup.dtype)
    t_j = jnp.asarray(0.0, dtype=Q_setup.dtype)

    def make_run(params):
        @partial(shard_map, mesh=spmd_mesh,
                 in_specs=(P(None, "cells"), P(None, "cells")),
                 out_specs=P(None, "cells"), check_rep=False)
        def run(Qp, Qap):
            Qp = _periodic_halo(Qp, halo, "cells", N_DEVS)
            dQ = flux_op(dt_j, t_j, Qp, Qap, params, jnp.zeros_like(Qp))
            return Qp + DT * dQ
        return run

    base = np.asarray(solver._rt_parameters).copy()
    p2 = base.copy()
    ig = [str(k) for k in solver._rt_model.parameter_names].index("g") \
        if hasattr(solver._rt_model, "parameter_names") else 0
    p2[ig] = base[ig] * 2.0
    r1 = np.asarray(make_run(jnp.asarray(base))(Q_pad, Qaux_pad))
    r2 = np.asarray(make_run(jnp.asarray(p2))(Q_pad, Qaux_pad))
    assert float(np.max(np.abs(r1 - r2))) > 1e-8, "g not live through shard_map"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
