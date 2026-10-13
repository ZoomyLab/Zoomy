"""Accessible entry point for SPMD (multi-device) FV runs.

The SPMD machinery — mesh partitioning (`zoomy_jax.mesh.partition_jax`) +
`ppermute` halo exchange (`zoomy_jax.fvm.halo_exchange_jax`) inside
`jax.shard_map` — is proven **bit-identical to single-device** on {2,4}
devices by `tests/unit/zoomy_jax/test_spmd_*.py` (halo, partition, advection,
LSQ-MUSCL, solver-integration, SME(0)).  Historically the actual sharded *run*
had to be hand-assembled from those pieces (each test rolls its own
`shard_map` body).  This module factors that verified pattern into a small
importable API so callers don't reverse-engineer the tests:

    >>> from zoomy_jax.fvm.spmd_jax import shard_global_state, build_sharded_flux_run
    >>> solver = HyperbolicSolver(); solver.setup_simulation(mesh, nsm)
    >>> run = build_sharded_flux_run(solver, n_devices=4, halo=1, n_steps=100, dt=dt)
    >>> Q_pad, n_local = shard_global_state(Q_global, n_parts=4, halo=1)
    >>> Qaux_pad = jnp.zeros((Qaux.shape[0], Q_pad.shape[1]), Q_pad.dtype)
    >>> Q_pad = run(Q_pad, Qaux_pad)                       # advances on N devices
    >>> Q_global_owned = gather_owned(Q_pad, n_parts=4, n_local=n_local, halo=1)

Scope (what is verified): the explicit flux operator with a **ring/periodic**
halo (`halo_exchange_inplace`) over a
1-D contiguous partition — the configuration the SME(0)/advection tests certify.
Per-rank global-BC handling and 2-D strip decomposition
(`partition_xaxis_structured`) are exported building blocks but not wrapped
here; pass your own `halo_exchange`/flux composition for those.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P

try:                                    # jax<0.8 path; >=0.8 promotes it
    from jax.experimental.shard_map import shard_map
except ImportError:                     # pragma: no cover
    from jax import shard_map

from zoomy_jax.mesh.partition_jax import partition_1d_contiguous
from zoomy_jax.fvm.halo_exchange_jax import halo_exchange_inplace

__all__ = [
    "spmd_device_mesh",
    "shard_global_state",
    "gather_owned",
    "build_sharded_flux_run",
    "halo_exchange_periodic",
    "run_solver_sharded",
    "collective_inner",
    "distributed_gmres",
]


def collective_inner(axis_name="cells"):
    """Return an inner-product `inner(a, b)` that sums over the FULL domain.
    Inside a `shard_map` over `axis_name` it is `lax.psum` across devices
    (so a Krylov solve converges the *global* system, not each device's local
    slab); outside shard_map the `psum` over a trivial axis is a no-op, so the
    same callable works single-device."""
    def inner(a, b):
        return lax.psum(jnp.vdot(a, b), axis_name)
    return inner


def distributed_gmres(matvec, b, *, inner=None, maxiter=30, x0=None):
    """Matrix-free GMRES whose inner products go through `inner` — pass
    `collective_inner` and a halo-exchanging `matvec` and the SAME code
    solves the GLOBAL system under `shard_map`.  Non-restarted Arnoldi (Krylov
    dim = `maxiter`) with modified Gram-Schmidt; the small least-squares is
    solved on the (replicated, tiny) Hessenberg.  Only the Arnoldi inner
    products and the residual norms are collective — the basis combination is
    local — so cross-device traffic is O(maxiter) scalars per iteration.

    `matvec(v) -> A v` MUST include the halo exchange when sharded (so `A` is
    the global operator).

    Returns `(x, rel_resid)` where `rel_resid = ||b - A x|| / ||b||` comes
    free from the Arnoldi least-squares.  **The caller MUST inspect it.**  The
    Krylov dimension is the static `maxiter`, so a solve that has not
    converged returns the best-in-subspace `x` and is otherwise
    indistinguishable from a converged one.

    This used to take a `tol` argument and return only `x`.  `tol` was
    never read — the loop always ran `maxiter` steps — so callers passing
    `tol=1e-12` got bit-identical results to `tol=1e-1` while the docstring
    promised `x` "solving A x = b".  jax's own `gmres` has the same shape of
    hole from the other side: its `info` is a hard-coded 0 placeholder (its
    docstring says so), so `_, info = jax_gmres(...)` can never report
    non-convergence.  Reporting the residual is the only honest option here."""
    inner = inner or (lambda a, c: jnp.vdot(a, c))
    x = jnp.zeros_like(b) if x0 is None else x0
    r = b - matvec(x)
    beta = jnp.sqrt(jnp.real(inner(r, r)))
    m = int(maxiter)
    eps = jnp.asarray(1e-300, dtype=beta.dtype)

    V = [r / jnp.where(beta > 0, beta, 1.0)]
    H = jnp.zeros((m + 1, m), dtype=b.dtype)
    for j in range(m):
        w = matvec(V[j])
        for i in range(j + 1):
            hij = inner(V[i], w)
            H = H.at[i, j].set(hij)
            w = w - hij * V[i]
        hjp = jnp.sqrt(jnp.real(inner(w, w)))
        H = H.at[j + 1, j].set(hjp)
        V.append(w / jnp.where(hjp > 0, hjp, 1.0))

    # least squares min || beta e1 - H y ||  (H is (m+1, m), tiny + replicated)
    e1 = jnp.zeros((m + 1,), dtype=b.dtype).at[0].set(beta)
    y, *_ = jnp.linalg.lstsq(H, e1, rcond=None)
    for j in range(m):
        x = x + y[j] * V[j]
    # ||beta e1 - H y|| IS the true residual norm of the Arnoldi solution, so
    # this costs no extra matvec (H, e1, y are replicated -> no collective).
    # Normalise by ||b||, NOT by beta: they coincide only when x0 = 0, and a
    # residual quietly measured against ||b - A x0|| is the kind of almost-right
    # number this function exists to stop shipping.  One scalar collective.
    bnorm = jnp.sqrt(jnp.real(inner(b, b)))
    rel_resid = jnp.linalg.norm(e1 - H @ y) / jnp.where(bnorm > 0, bnorm, 1.0)
    return x, rel_resid


def halo_exchange_periodic(Q_pad, halo, axis_name, n_devices):
    """Periodic ring halo: rank `i` sends its edge slabs to `(i±1) mod n`.
    Unlike `halo_exchange_inplace` (which
    zeros the outer halo at the two global ends for the inline BC to overwrite),
    this wraps around — so a periodic domain sharded across `n` devices equals
    the same periodic domain on 1 device.  Layout `[halo | owned | halo]`."""
    left_owned = Q_pad[:, halo:2 * halo]
    right_owned = Q_pad[:, -2 * halo:-halo]
    perm_r = [(i, (i + 1) % n_devices) for i in range(n_devices)]
    perm_l = [(i, (i - 1) % n_devices) for i in range(n_devices)]
    fill_left = lax.ppermute(right_owned, perm=perm_r, axis_name=axis_name)
    fill_right = lax.ppermute(left_owned, perm=perm_l, axis_name=axis_name)
    Q_pad = Q_pad.at[:, :halo].set(fill_left)
    Q_pad = Q_pad.at[:, -halo:].set(fill_right)
    return Q_pad


def run_solver_sharded(solver, n_devices, halo, n_steps, dt, *,
                       t=0.0, axis_name="cells",
                       halo_exchange=halo_exchange_periodic):
    """Run the **real** solver (its own `step` + `post_step` — full physics:
    flux, order-1/2 reconstruction, explicit/local-implicit source) across
    `n_devices` via `jax.shard_map`.

    `solver` must already be set up (`setup_simulation`) on the **per-
    partition padded mesh** (e.g. `partition_1d_contiguous(global_mesh, ...)[1]`
    or `partition_xaxis_structured(...)`).  This sets `solver._halo_exchange`
    so the flux operator refreshes `Q`/`Qaux` halos every stage (see
    `HyperbolicSolver._halo_wrap`); the same solver code then runs on every
    device.  With the periodic halo (default) the result is transparent to the
    device count (sharded-1 == sharded-N).  `dt` is fixed (a global adaptive dt
    would need a `lax.pmin` collective — a small refinement).  Returns a
    callable `run(Q_pad, Qaux_pad) -> (Q_pad, Qaux_pad)`."""
    solver._halo_exchange = lambda x: halo_exchange(x, halo, axis_name, n_devices)
    dmesh = spmd_device_mesh(n_devices, axis_name)
    dt_j = jnp.asarray(dt)
    t0 = jnp.asarray(t)

    @partial(shard_map, mesh=dmesh,
             in_specs=(P(None, axis_name), P(None, axis_name)),
             out_specs=(P(None, axis_name), P(None, axis_name)), check_rep=False)
    def run(Q_pad, Qaux_pad):
        def body(carry, _):
            Q, Qaux, tc = carry
            Qn = solver.step(dt_j, tc, Q, Qaux)
            Qn, Qauxn = solver.post_step(tc + dt_j, dt_j, Qn, Q, Qaux)
            return (Qn, Qauxn, tc + dt_j), None

        (Q, Qaux, _), _ = lax.scan(body, (Q_pad, Qaux_pad, t0), None,
                                   length=n_steps)
        return Q, Qaux

    return run


def spmd_device_mesh(n_devices=None, axis_name="cells"):
    """A 1-D `jax.sharding.Mesh` over `n_devices` (default: all) along
    `axis_name` — the cell-partition axis used by every SPMD helper here."""
    devs = jax.devices() if n_devices is None else jax.devices()[:n_devices]
    return Mesh(np.array(devs), axis_names=(axis_name,))


def shard_global_state(Q, n_parts, halo):
    """Chop a global `(n_var, n_cells)` state into `n_parts` contiguous
    per-device slabs, each padded with `halo` zero-cells on both sides, and
    concatenate them along the cell axis (the sharded layout `shard_map`
    consumes).  Returns `(Q_pad, n_local)` with ``n_local = n_cells //
    n_parts``.  Requires `n_cells % n_parts == 0`."""
    Q = np.asarray(Q)
    ns, nt = Q.shape
    if nt % n_parts:
        raise ValueError(f"n_cells={nt} not divisible by n_parts={n_parts}")
    n_local = nt // n_parts
    pad = lambda c: np.concatenate(
        [np.zeros((ns, halo)), c, np.zeros((ns, halo))], axis=1)
    chunks = [Q[:, d * n_local:(d + 1) * n_local] for d in range(n_parts)]
    Q_pad = np.concatenate([pad(c) for c in chunks], axis=1)
    return jnp.asarray(Q_pad), n_local


def gather_owned(Q_pad, n_parts, n_local, halo):
    """Inverse of `shard_global_state`: strip the halos and concatenate
    the owned cells back into a global `(n_var, n_parts*n_local)` array."""
    Q_pad = np.asarray(Q_pad)
    n_pad = n_local + 2 * halo
    owned = [Q_pad[:, d * n_pad + halo:d * n_pad + halo + n_local]
             for d in range(n_parts)]
    return np.concatenate(owned, axis=1)


def build_sharded_flux_run(solver, n_devices, halo, n_steps, dt, *,
                           t=0.0, axis_name="cells",
                           halo_exchange=halo_exchange_inplace):
    """Return a callable `run(Q_pad, Qaux_pad) -> Q_pad` that advances
    `n_steps` explicit forward-Euler flux steps across `n_devices` via
    `shard_map`, exchanging `halo` cells with ppermute each step.

    `solver` must already be set up (`setup_simulation` called), so its
    `_rt_mesh` / `_rt_model` / `_rt_parameters` are live.  Bit-identical
    to a replicated single-device run (see `test_spmd_sme0`).  This is the
    verified ring-halo / interior-partition path; for global BCs or 2-D strips
    compose the exported partition/halo primitives yourself."""
    gmesh = solver._rt_mesh
    runtime = solver._rt_model
    parameters = solver._rt_parameters
    parts = partition_1d_contiguous(gmesh, n_parts=n_devices, halo=halo)
    flux_op = solver.get_flux_operator(parts[1], runtime)   # interior partition
    dmesh = spmd_device_mesh(n_devices, axis_name)
    dt_j = jnp.asarray(dt)
    t_j = jnp.asarray(t)

    def _step(Qp, Qap):
        Qp = halo_exchange(Qp, halo, axis_name, n_devices)
        dQ = flux_op(dt_j, t_j, Qp, Qap, parameters, jnp.zeros_like(Qp))
        return Qp + dt_j * dQ

    @partial(shard_map, mesh=dmesh,
             in_specs=(P(None, axis_name), P(None, axis_name)),
             out_specs=P(None, axis_name), check_rep=False)
    def run(Q_pad, Qaux_pad):
        Qf, _ = lax.scan(lambda c, _: (_step(c, Qaux_pad), None),
                         Q_pad, jnp.arange(n_steps))
        return Qf

    return run
