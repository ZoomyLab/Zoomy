"""JAX **UserFunctions** — the opaque kernels a backend must supply (REQ-168).

The `module` dict handed to `sympy.lambdify` *is* the UserFunctions table:
it resolves every opaque symbol the printer emits.  Historically it was an
anonymous dict literal duplicated across `transformation/to_jax.py` (legacy
`JaxRuntimeModel` / `JaxRuntimeSymbolic`) and `transformation/jax_runtime.py`
(the LIVE `JaxRuntime`), so a missing kernel was invisible instead of a build
error.  This module is the single source; the runtimes build their `module`
from `jax_userfunctions`.

Kernels
-------
`compute_derivative` — NON-LOCAL spatial-derivative aux.  Stays `None` here;
the solver injects the mesh-bound impl (`lsq_gradient_per_field`) before the
`update_aux_variables` slot is lambdified (mirrors numpy).

`eigensystem(*A_flat)` — the FULL packed stack `[λ(n), R(n·n), L=R⁻¹(n·n)]`
of the row-major matrix `A`.  Component access is ``pick(eigensystem(*a),
idx)`` — the canonical form the producers emit, cse-hoisted to ONE call per
face.  Consumed by `NonconservativeRoe` and the characteristic BCs.

`solve(*args)` — the per-cell linear solve `A⁻¹b` (whole vector; `args` =
row-major `A` (n·n) then `b` (n)); component access ``pick(solve(*a),
idx)``.  REQ-68.

`pick(packed, idx)` — trailing-axis index read of a packed kernel result.

⚠ PERFORMANCE / AD CAVEAT (REQ-168, jax): `jnp.linalg.eig` is CPU-only, so
`eigensystem` here routes through `jax.pure_callback` — a HOST round-trip.
Both consumers sit inside the fused JIT hot loop (the wave speed is evaluated
every step inside `lax.while_loop`; Roe evaluates per face inside the flux
operator), so making this the DEFAULT `dt` path would break the fusion that
gives the jax backend its throughput, and `pure_callback` has no JVP rule so
forward-AD through the solver would break too.  The table is complete (Roe and
opt-in paths work), but the default wave speed should use the lighter λ-only
`eigenvalues` kernel (on-device) once core exposes it.  See the REQ-168 thread.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

__all__ = ["eigensystem", "eigenvalues", "solve", "pick", "newton_solve",
           "solve_steady_ode", "jax_userfunctions"]


# ── newton_solve (higher-order root-find; Bernoulli moving-equilibrium WB) ────
def newton_solve(residual, x0, n_iter=60):
    """Batched scalar Newton root-find of `residual(x) = 0` from guess `x0`.

    The jax twin of `zoomy_core.fvm.userfunctions.newton_solve` — same
    contract, on-device and jit/AD-safe.  The reconstruction that lives in core
    builds the `residual` closure (elementwise over the grid) and the guess;
    this backend runs the loop (``reconstruction lives in core, backend is a
    loop``).  Consumed by the moving-equilibrium (Bernoulli) WB reconstruction:
    at SME level 0 the per-face root is the SWE specific-energy surface `η*`.

    Fixed `n_iter` via `lax.scan` (NO data-dependent `while` — the trace
    is static, so it fuses into the flux hot loop and stays reverse/forward-AD
    safe).  The residual is elementwise (each lane independent → diagonal
    Jacobian), so `jax.jvp` with a unit tangent returns the EXACT per-lane
    derivative in one pass — no eigen/callback host round-trip."""
    def body(x, _):
        f, fp = jax.jvp(residual, (x,), (jnp.ones_like(x),))
        return x - f / fp, None
    x, _ = jax.lax.scan(body, x0, None, length=int(n_iter))
    return x


# ── solve_steady_ode (moving-equilibrium WB with friction) ────────────────────
def solve_steady_ode(slope, U0, ds, n_iter=8):
    """Integrate one collocation step of the local stationary ODE `U_x = G(U)`.

    The jax twin of `zoomy_core.fvm.userfunctions.solve_steady_ode` — same
    contract, on-device and jit-safe.  The moving-equilibrium WB reconstruction
    (core) builds the `slope` closure `G` from the emitted operators and the
    `solve` linear-solve kernel; this backend runs the RK loop.

    Reversible / implicit collocation — **implicit midpoint** (1-stage Gauss),
    the reversible RK the fully-WB property needs.  Solves the endpoint
    `U1 = U0 + ds·G((U0+U1)/2)` by `n_iter` fixed-point sweeps via
    `lax.scan` (NO data-dependent `while` → fuses into the flux hot loop,
    stays forward/reverse-AD safe).  `U0` is `(n_state, nf)`, `ds` a
    scalar or `(nf,)` signed step; returns the endpoint `(n_state, nf)`."""
    def body(U1, _):
        Umid = 0.5 * (U0 + U1)
        return U0 + ds * slope(Umid), None
    U1, _ = jax.lax.scan(body, U0, None, length=int(n_iter))
    return U1


# ── eigenvalues (λ-only; REQ-168 GAP 1) ──────────────────────────────────────
def eigenvalues(*a_flat):
    """The λ-only spectrum vector `(…, n)` (real part) of the row-major
    `n×n` `A_flat`.  Component access is `pick(eigenvalues(*a), idx)`.

    The λ-only companion of `eigensystem`, for the wave-speed / CFL bound
    `max|λ_i(A_n)|` when the model has no closed-form spectrum (SME / VAM).

    **This one does NOT go through `pure_callback`** — unlike `eigensystem`
    it lowers to `jnp.linalg.eigvals` directly, so it stays inside the jit
    (no host round-trip, no broken JVP) and skips the eigenvector solve + the
    `R⁻¹` inverse entirely.  That matters because the wave speed is evaluated
    EVERY step inside the fused `lax.while_loop`: a per-step host sync there
    would collapse the fusion the jax backend's throughput depends on.

    ⚠ Still CPU-only: jaxlib implements `eig`/`eigvals` on CPU only, so this
    kernel does not run on GPU.  A genuinely device-portable spectral radius
    (e.g. power iteration for `max|λ|` on a hyperbolic — hence real,
    diagonalisable — system) is the next step if GPU + `eigenvalues=None` is
    needed; it would also be AD-safe.  Batched: each arg may be a scalar or a
    `(n_faces,)` array."""
    n = int(round(len(a_flat) ** 0.5))
    cols = jnp.broadcast_arrays(*[jnp.asarray(a) for a in a_flat])
    A = jnp.stack(cols, axis=-1).reshape(cols[0].shape + (n, n))
    return jnp.real(jnp.linalg.eigvals(A))


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


def eigensystem(*a_flat):
    """The FULL packed stack `[λ(n), R(n·n), L(n·n)]` (length `n+2n²`) for
    row-major `A`.  Component access is `pick(eigensystem(*a), idx)`.

    Batched: each arg may be a scalar or a `(n_faces,)` array (broadcast).
    Routes through `pure_callback` because `jnp.linalg.eig` is CPU-only —
    see the module-level performance caveat."""
    n = int(round(len(a_flat) ** 0.5))
    cols = jnp.broadcast_arrays(*[jnp.asarray(a) for a in a_flat])
    A = jnp.stack(cols, axis=-1).reshape(cols[0].shape + (n, n))
    out = jax.ShapeDtypeStruct(A.shape[:-2] + (n + 2 * n * n,), A.dtype)
    return jax.pure_callback(_eig_host, out, A, vmap_method="broadcast_all")


# ── solve (REQ-68) ───────────────────────────────────────────────────────────
def solve(*args):
    """The per-cell linear solve `A⁻¹b` (whole vector `(…, n)`).

    `args` = row-major `A` (n·n) followed by `b` (n); `n` inferred from
    `n·n + n`.  Batched over the grid (each arg scalar or `(n_cells,)`), so
    the NSM point-implicit source lowers to ONE batched solve; component access
    is `pick(solve(*a), idx)`.  On-device (`jnp.linalg.solve`) — no
    callback, jit/vmap/AD-safe."""
    m = len(args)
    n = int(round((-1.0 + (1.0 + 4.0 * m) ** 0.5) / 2.0))
    cols = jnp.broadcast_arrays(*[jnp.asarray(a) for a in args])
    batch = cols[0].shape
    A = jnp.stack(cols[: n * n], axis=-1).reshape(batch + (n, n))
    b = jnp.stack(cols[n * n:], axis=-1).reshape(batch + (n,))
    return jnp.linalg.solve(A, b[..., None])[..., 0]


# ── pick (component read of a packed kernel result) ──────────────────────────
def pick(packed, idx):
    """The `idx`-th component of a packed opaque-kernel result
    (`eigensystem` / `eigenvalues` / `solve`).  Trailing-axis
    index read — the producers emit `pick(K(*a), idx)` against ONE cse-hoisted
    `K(*a)` node so each face runs ONE decomposition / solve."""
    return packed[..., int(idx)]


# ── the table ────────────────────────────────────────────────────────────────
def spectral_radius(k, safety, *a_flat):
    """`safety * sqrt(rho(A^2))`, `rho(A^2)` by `k` power steps on
    `A^2` from `x = 1` (see `kernel_functions.spectral_radius`).
    Matvecs only: runs on any device; batched over the leading axes."""
    n = int(round(len(a_flat) ** 0.5))
    cols = jnp.broadcast_arrays(*[jnp.asarray(a) for a in a_flat])
    A = jnp.stack(cols, axis=-1).reshape(cols[0].shape + (n, n))
    mv = lambda x: jnp.einsum("...ij,...j->...i", A, x)       # noqa: E731

    def step(_, carry):
        x, _lam = carry
        y = mv(mv(x))
        ny = jnp.linalg.norm(y, axis=-1)
        return y / jnp.where(ny > 0, ny, 1.0)[..., None], ny   # |x| = 1
    x0 = jnp.ones(A.shape[:-1], dtype=A.dtype) / jnp.sqrt(n)
    _, lam = jax.lax.fori_loop(0, int(k), step, (x0, jnp.zeros(A.shape[:-2], A.dtype)))
    return safety * jnp.sqrt(lam)


def jax_userfunctions() -> dict:
    """The complete jax UserFunctions table (a fresh dict per runtime, since
    runtimes inject per-instance entries such as `compute_derivative`)."""
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
        "spectral_radius": spectral_radius,
        "solve": solve,
        # component read of a packed kernel result (canonical pick(K(*a), idx))
        "pick": pick,
        # higher-order root-find (reconstruction supplies the residual callable
        # — NOT a lambdify-lowered REQUIRED_KERNEL, like the numpy table's)
        "newton_solve": newton_solve,
        # higher-order steady-ODE integrator (moving-equilibrium WB w/ friction)
        "solve_steady_ode": solve_steady_ode,
    }
