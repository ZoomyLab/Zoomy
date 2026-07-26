"""JAX **UserFunctions** — the opaque kernels a backend must supply (REQ-168).

The ``module`` dict handed to ``sympy.lambdify`` *is* the UserFunctions table:
it resolves every opaque symbol the printer emits.  Historically it was an
anonymous dict literal duplicated across ``transformation/to_jax.py`` (legacy
``JaxRuntimeModel`` / ``JaxRuntimeSymbolic``) and ``transformation/jax_runtime.py``
(the LIVE ``JaxRuntime``), so a missing kernel was invisible instead of a build
error.  This module is the single source; the runtimes build their ``module``
from :func:`jax_userfunctions`.

Kernels
-------
``compute_derivative`` — NON-LOCAL spatial-derivative aux.  Stays ``None`` here;
the solver injects the mesh-bound impl (``lsq_gradient_per_field``) before the
``update_aux_variables`` slot is lambdified (mirrors numpy).

``eigensystem(idx, *A_flat)`` — idx-th of the flat stack ``[λ(n), R(n·n),
L=R⁻¹(n·n)]`` of the row-major matrix ``A``.  Consumed by ``NonconservativeRoe``
and (once core lands REQ-168 GAP 1) by ``local_max_abs_eigenvalue``.

``solve(idx, *args)`` — idx-th component of the per-cell linear solve ``A⁻¹b``
(``args`` = row-major ``A`` (n·n) then ``b`` (n)); REQ-68.

⚠ PERFORMANCE / AD CAVEAT (REQ-168, jax): ``jnp.linalg.eig`` is CPU-only, so
``eigensystem`` here routes through ``jax.pure_callback`` — a HOST round-trip.
Both consumers sit inside the fused JIT hot loop (the wave speed is evaluated
every step inside ``lax.while_loop``; Roe evaluates per face inside the flux
operator), so making this the DEFAULT ``dt`` path would break the fusion that
gives the jax backend its throughput, and ``pure_callback`` has no JVP rule so
forward-AD through the solver would break too.  The table is complete (Roe and
opt-in paths work), but the default wave speed should use the lighter λ-only
``eigenvalues`` kernel (on-device) once core exposes it.  See the REQ-168 thread.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

__all__ = ["eigensystem", "eigenvalues", "solve", "newton_solve",
           "jax_userfunctions"]


# ── newton_solve (higher-order root-find; Bernoulli moving-equilibrium WB) ────
def newton_solve(residual, x0, n_iter=60):
    """Batched scalar Newton root-find of ``residual(x) = 0`` from guess ``x0``.

    The jax twin of ``zoomy_core.fvm.userfunctions.newton_solve`` — same
    contract, on-device and jit/AD-safe.  The reconstruction that lives in core
    builds the ``residual`` closure (elementwise over the grid) and the guess;
    this backend runs the loop (``reconstruction lives in core, backend is a
    loop``).  Consumed by the moving-equilibrium (Bernoulli) WB reconstruction:
    at SME level 0 the per-face root is the SWE specific-energy surface ``η*``.

    Fixed ``n_iter`` via ``lax.scan`` (NO data-dependent ``while`` — the trace
    is static, so it fuses into the flux hot loop and stays reverse/forward-AD
    safe).  The residual is elementwise (each lane independent → diagonal
    Jacobian), so ``jax.jvp`` with a unit tangent returns the EXACT per-lane
    derivative in one pass — no eigen/callback host round-trip."""
    def body(x, _):
        f, fp = jax.jvp(residual, (x,), (jnp.ones_like(x),))
        return x - f / fp, None
    x, _ = jax.lax.scan(body, x0, None, length=int(n_iter))
    return x


# ── eigenvalues (λ-only; REQ-168 GAP 1) ──────────────────────────────────────
def eigenvalues(idx, *a_flat):
    """idx-th eigenvalue (real part) of the row-major ``n×n`` ``A_flat``.

    The λ-only companion of :func:`eigensystem`, for the wave-speed / CFL bound
    ``max|λ_i(A_n)|`` when the model has no closed-form spectrum (SME / VAM).

    **This one does NOT go through ``pure_callback``** — unlike ``eigensystem``
    it lowers to ``jnp.linalg.eigvals`` directly, so it stays inside the jit
    (no host round-trip, no broken JVP) and skips the eigenvector solve + the
    ``R⁻¹`` inverse entirely.  That matters because the wave speed is evaluated
    EVERY step inside the fused ``lax.while_loop``: a per-step host sync there
    would collapse the fusion the jax backend's throughput depends on.

    ⚠ Still CPU-only: jaxlib implements ``eig``/``eigvals`` on CPU only, so this
    kernel does not run on GPU.  A genuinely device-portable spectral radius
    (e.g. power iteration for ``max|λ|`` on a hyperbolic — hence real,
    diagonalisable — system) is the next step if GPU + ``eigenvalues=None`` is
    needed; it would also be AD-safe.  Batched: each arg may be a scalar or a
    ``(n_faces,)`` array."""
    n = int(round(len(a_flat) ** 0.5))
    cols = jnp.broadcast_arrays(*[jnp.asarray(a) for a in a_flat])
    A = jnp.stack(cols, axis=-1).reshape(cols[0].shape + (n, n))
    return jnp.real(jnp.linalg.eigvals(A))[..., idx]


# ── eigensystem ──────────────────────────────────────────────────────────────
def _eig_host(A):
    """Host (numpy) eigendecomposition — returns the flat [λ, R, L] stack."""
    w, V = np.linalg.eig(A)
    w = np.real(w)
    V = np.real(V)
    try:
        L = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        L = np.linalg.pinv(V)
    n = A.shape[-1]
    return np.concatenate(
        [w, V.reshape(*A.shape[:-2], n * n), L.reshape(*A.shape[:-2], n * n)],
        axis=-1,
    )


def eigensystem(idx, *a_flat):
    """idx-th component of ``[λ(n), R(n·n), L(n·n)]`` for row-major ``A``.

    Batched: each arg may be a scalar or a ``(n_faces,)`` array (broadcast).
    Routes through ``pure_callback`` because ``jnp.linalg.eig`` is CPU-only —
    see the module-level performance caveat."""
    n = int(round(len(a_flat) ** 0.5))
    cols = jnp.broadcast_arrays(*[jnp.asarray(a) for a in a_flat])
    A = jnp.stack(cols, axis=-1).reshape(cols[0].shape + (n, n))
    out = jax.ShapeDtypeStruct(A.shape[:-2] + (n + 2 * n * n,), A.dtype)
    stack = jax.pure_callback(_eig_host, out, A, vmap_method="broadcast_all")
    return stack[..., idx]


# ── solve (REQ-68) ───────────────────────────────────────────────────────────
def solve(idx, *args):
    """idx-th component of the per-cell linear solve ``A⁻¹b``.

    ``args`` = row-major ``A`` (n·n) followed by ``b`` (n); ``n`` inferred from
    ``n·n + n``.  Batched over the grid (each arg scalar or ``(n_cells,)``), so
    the NSM point-implicit source lowers to ONE batched solve.  On-device
    (``jnp.linalg.solve``) — no callback, jit/vmap/AD-safe."""
    m = len(args)
    n = int(round((-1.0 + (1.0 + 4.0 * m) ** 0.5) / 2.0))
    cols = jnp.broadcast_arrays(*[jnp.asarray(a) for a in args])
    batch = cols[0].shape
    A = jnp.stack(cols[: n * n], axis=-1).reshape(batch + (n, n))
    b = jnp.stack(cols[n * n:], axis=-1).reshape(batch + (n,))
    return jnp.linalg.solve(A, b[..., None])[..., 0][..., idx]


# ── the table ────────────────────────────────────────────────────────────────
def jax_userfunctions() -> dict:
    """The complete jax UserFunctions table (a fresh dict per runtime, since
    runtimes inject per-instance entries such as ``compute_derivative``)."""
    return {
        # arithmetic / structural
        "ones_like": jnp.ones_like,
        "zeros_like": jnp.zeros_like,
        "array": jnp.array,
        "squeeze": jnp.squeeze,
        "conditional": lambda c, t, f: jnp.where(c, t, f),
        # solver-injected (mesh-bound) — see module docstring
        "compute_derivative": None,
        # opaque numerical kernels
        "eigensystem": eigensystem,
        "eigenvalues": eigenvalues,
        "solve": solve,
        # higher-order root-find (reconstruction supplies the residual callable
        # — NOT a lambdify-lowered REQUIRED_KERNEL, like the numpy table's)
        "newton_solve": newton_solve,
    }
