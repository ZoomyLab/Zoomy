"""NumPy backend **UserFunctions** — the concrete implementations of the opaque
`zoomy_core.model.kernel_functions` kernels for the numpy runtime.

This is numpy's `UserFunctions` table, the python mirror of the C++ backends'
`UserFunctions.H` (REQ-168).  Every backend-supplied kernel named in
`zoomy_core.model.kernel_functions.REQUIRED_KERNELS` has an entry in
`KERNELS` here, so a missing kernel is a caught omission (see the contract
test) rather than a silent `NameError` at lambdify time.  `to_numpy.module`
is built from these tables via `numpy_module`.

Kept a leaf module (imports only numpy) so `to_numpy` can import it without a
`fvm` ↔ `transformation` cycle.

Each kernel returns the WHOLE packed result of ONE decomposition / solve
(`eigensystem` → `[λ, R, L]`, `eigenvalues` → `λ`, `solve` → `A⁻¹b`);
component access is `pick(K(*a), idx)`.  The producers emit every component
read against the SAME `K(*a)` node, so lambdify's cse hoists it to a single
call per face — R, |Λ| and L therefore come from ONE eigenbasis by construction
(`|A| = R|Λ|L` consistent), with no per-call cache to keep them in sync.
"""

import numpy as np


def _stack_square(a_flat, n):
    """Broadcast the flat row-major args into an `(..., n, n)` array.

    Each arg is a scalar or a `(n_cells,)` grid row; `np.broadcast_arrays`
    lifts scalars to the common grid shape so the eig / solve is batched over
    every cell at once.  Scalar-only inputs give a `(n, n)` matrix (0-d
    broadcast), so the scalar paths still work."""
    bcast = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in a_flat])
    return np.stack(bcast, axis=-1).reshape(bcast[0].shape + (n, n))


def _scalarize(x):
    """Return a python float for a 0-d result (scalar-input path), else the
    array — so a scalar evaluation matches the old `float(...)` return."""
    return float(x) if np.ndim(x) == 0 else x


def eigensystem(*a_flat):
    """Opaque `eigensystem` kernel: the FULL packed stack
    `[λ(n), R(n·n), L=R⁻¹(n·n)]` (row-major, length `n+2n²`) of the row-major
    `n×n` matrix `a_flat` (`n = round(√len)`), batched over the grid.
    Component access is `pick(eigensystem(*a), idx)` — the canonical form the
    Roe scheme emits; the producers share ONE `eigensystem(*a)` node, so
    lambdify's cse hoists it to a single `eig` per face (R, |Λ|, L from ONE
    eigenbasis by construction).

    REQ-168 inf-guard: LAPACK raises on non-finite input, so non-finite batch
    members get λ = +inf with an identity eigenbasis (R = L = I) — the +inf wave
    speed flags the garbage state; the identity basis keeps `|A| = R|Λ|L`
    well-defined without inventing an inverse of a garbage matrix."""
    n = int(round(len(a_flat) ** 0.5))
    A = _stack_square(a_flat, n)
    flat = A.reshape(-1, n, n)
    m = flat.shape[0]
    ok = np.isfinite(flat).all(axis=(1, 2))
    w = np.full((m, n), np.inf)
    V = np.broadcast_to(np.eye(n), (m, n, n)).copy()
    L = V.copy()
    if ok.any():
        wk, Vk = np.linalg.eig(flat[ok])
        V[ok] = np.real(Vk)
        w[ok] = np.real(wk)
        try:
            L[ok] = np.linalg.inv(V[ok])
        except np.linalg.LinAlgError:
            L[ok] = np.linalg.pinv(V[ok])
    return np.concatenate(
        [w, V.reshape(m, n * n), L.reshape(m, n * n)],
        axis=-1).reshape(A.shape[:-2] + (n + 2 * n * n,))


def gelfand_bound(j, delta, *a_flat):
    """`kernel_functions.gelfand_bound`, batched: `j` squarings of `A`, each after dividing
    by its max row sum `s_i`, then `w = max_i (|C| x)_i / x_i` with `x` the row sums of `|C|`
    plus `delta` their max; returns `s_0 sqrt(s_1 sqrt(... sqrt(s_(j-1) sqrt(w))))`, `+inf` for
    a non-finite entry."""
    n = int(round(len(a_flat) ** 0.5))
    C = _stack_square(a_flat, n)
    finite = np.isfinite(C).all(axis=(-2, -1))
    C = np.where(finite[..., None, None], C, 0.0)
    scales = []
    for _ in range(int(j)):
        s = np.abs(C).sum(-1).max(-1)
        s = np.where(s > 0, s, 1.0)
        C = C / s[..., None, None]
        scales.append(s)
        C = C @ C
    M = np.abs(C)
    r = M.sum(-1)
    x = r + (np.asarray(delta, float) * r.max(-1))[..., None]   # > 0: w continuous in A
    y = np.einsum("...ik,...k->...i", M, x)
    w = np.where(x > 0, y / np.where(x > 0, x, 1.0), 0.0).max(-1)
    t = w
    for s in reversed(scales):
        t = s * np.sqrt(t)
    return np.where(finite, t, np.inf)


def eigenvalues(*a_flat):
    """Opaque `eigenvalues` kernel (λ-only): the spectrum vector `(n,)` of
    the row-major `n×n` matrix `a_flat` (`n = round(√len)`), batched over
    the grid.  The light companion of `eigensystem` (no eigenvectors) for
    the wave-speed / CFL bound; component access is ``pick(eigenvalues(*a),
    idx)`` (cse-hoisted to ONE `eigvals` per face).

    REQ-168 inf-guard: eig the finite batch members, +inf for the rest (order-2
    MOOD / transient BC-ghost feed non-finite candidate face states BY DESIGN —
    an infinite wave speed is exactly what a garbage face should report; dt
    clamps, MOOD flags the candidate)."""
    n = int(round(len(a_flat) ** 0.5))
    A = _stack_square(a_flat, n)
    flat = A.reshape(-1, n, n)
    ok = np.isfinite(flat).all(axis=(1, 2))
    out = np.full((flat.shape[0], n), np.inf)
    if ok.any():
        out[ok] = np.real(np.linalg.eigvals(flat[ok]))
    return out.reshape(A.shape[:-2] + (n,))


def solve(*args):
    """Opaque `solve` kernel: the per-cell linear solve `A⁻¹ b` ``(ncells,
    n)``.  `args` = row-major `A` (n·n) followed by `b` (n); `n`
    inferred from the count `n·n + n`.  Batched over the grid — the NSM
    point-implicit `source` lowers to ONE batched `np.linalg.solve`;
    component access is `pick(solve(*a), idx)`."""
    m = len(args)
    n = int(round((-1.0 + (1.0 + 4.0 * m) ** 0.5) / 2.0))
    arrs = [np.asarray(a, dtype=float) for a in args]
    ncells = max((a.shape[0] for a in arrs if a.ndim > 0), default=1)

    def _col(a):
        if a.ndim > 0 and a.shape[0] == ncells:
            return a
        return np.full(ncells, float(a))

    cols = [_col(a) for a in arrs]
    A = np.stack(cols[:n * n], axis=-1).reshape(ncells, n, n)
    # Column RHS ``(ncells, n, 1)`` — NumPy 2.0 reads a 2-D ``b`` as a matrix,
    # so the vector RHS must be an explicit column.
    b = np.stack(cols[n * n:], axis=-1).reshape(ncells, n, 1)
    return np.linalg.solve(A, b)[..., 0]   # (ncells, n)


def pick(packed, idx):
    """idx-th component of a packed opaque-kernel result (`eigensystem` /
    `eigenvalues` / `solve`).  Cheap trailing-axis index read — the
    producers emit `pick(K(*a), idx)` against ONE cse-hoisted `K(*a)` node
    so each face runs ONE decomposition / solve.  `_scalarize` keeps the
    scalar-input path returning a float, matching a scalar `K` evaluation."""
    return _scalarize(packed[..., int(idx)])


def newton_solve(residual, x0, n_iter=60, tol=1e-14):
    """Batched scalar Newton root-find of `residual(x) = 0` from guess `x0`.

    A backend-supplied opaque kernel in the `eigensystem` / `solve` family,
    but a HIGHER-ORDER one: the reconstruction that lives in core builds the
    `residual` closure and the initial guess, and the backend runs the
    iteration loop (`reconstruction lives in core, backend is a loop`).  Used
    by the moving-equilibrium (Bernoulli) WB reconstruction — at SME level 0 the
    per-face root is the SWE specific-energy relation for the reconstructed
    surface `η*`.

    Elementwise over the grid (each lane is an independent scalar Newton), so
    the derivative is the per-lane slope taken by forward finite difference: the
    residual is a black-box callable with no symbolic derivative.  Fixed max
    `n_iter` with an early ‖residual‖∞ exit.  The jax twin
    (`zoomy_jax.fvm.userfunctions.newton_solve`) is bit-for-bit the same
    contract, using `jax.jvp` for the exact diagonal derivative and
    `lax.scan` for the loop (no data-dependent `while`, so it stays
    jit/AD-safe)."""
    x = np.array(x0, dtype=float)
    for _ in range(int(n_iter)):
        f = np.asarray(residual(x), dtype=float)
        if np.max(np.abs(f)) < tol:
            break
        dx = 1e-8 * (np.abs(x) + 1.0)
        fp = (np.asarray(residual(x + dx), dtype=float) - f) / dx
        x = x - f / fp
    return x


def solve_steady_ode(slope, U0, ds, n_iter=8):
    """Integrate one collocation step of the local stationary ODE `U_x = G(U)`.

    The higher-order opaque kernel of the FULLY well-balanced moving-equilibrium
    reconstruction (Pimentel-García, *Fully WB Methods for the SW Linearized
    Moment Model with Friction*, HYP2022): with friction `R` there is NO
    closed form for the steady state, so the local stationary solution is
    obtained by numerically integrating

        U_x = G(U) = -(J_F(U)+B(U))_VV^{-1} (S(U)·H_x + R(U))

    (the b/topography row dropped, its column carried into `S·H_x`).  Same
    contract as `newton_solve`: the reconstruction that lives in core
    builds the `slope` closure `G` from the EMITTED operators and the
    existing `solve` linear-solve kernel; the backend runs the RK loop.

    Reversible / implicit collocation — the 2-stage-equivalent **implicit
    midpoint** (1-stage Gauss, Butcher `[1/2|1/2 ; 1]`), the reversible RK the
    paper requires for the well-balanced property.  Solves the endpoint

        U1 = U0 + ds · G((U0+U1)/2)

    by `n_iter` fixed-point sweeps (jit-safe fixed length; the jax twin uses
    `lax.scan`).  `U0` is `(n_state, nf)`, `ds` a scalar or `(nf,)`
    signed step; returns the endpoint `(n_state, nf)`.

    WB-exactness: implicit midpoint is reversible, so the sequence of cell
    averages it produces is a *discrete* stationary solution the scheme
    preserves to the fixed-point residual — "fully well-balanced" in the paper's
    sense (near machine precision), not "exactly WB" (which would need the
    stationary solution in closed form).  The cell's OWN node→intercell step is
    the exact-midpoint special case (midpoint = the cell node) and needs no
    iteration; this kernel is the neighbour-node extension where the midpoint is
    implicit."""
    dsb = np.asarray(ds, dtype=float)
    U1 = np.array(U0, dtype=float)
    for _ in range(int(n_iter)):
        Umid = 0.5 * (U0 + U1)
        U1 = U0 + dsb * slope(Umid)
    return U1


# Backend-supplied opaque kernels (kernel_functions.REQUIRED_KERNELS).
# ``compute_derivative`` is None here: the SOLVER injects the mesh-bound impl
# (``mesh.compute_derivatives``) before the ``update_aux_variables`` slot is
# compiled — same seam as the other backends.
KERNELS = {
    "compute_derivative": None,
    "eigensystem": eigensystem,
    "eigenvalues": eigenvalues,
    "gelfand_bound": gelfand_bound,
    "solve": solve,
    # Component read of a packed kernel result (numpy-internal, not a
    # cross-backend REQUIRED_KERNEL — the per-component C/UFL backends restore
    # ``K(idx, *a)`` from ``pick(K(*a), idx)`` instead of resolving ``pick``).
    "pick": pick,
    # Higher-order root-find (numpy-internal — the reconstruction supplies the
    # residual callable, so it is NOT a lambdify-lowered REQUIRED_KERNEL).
    "newton_solve": newton_solve,
    # Higher-order steady-ODE integrator (moving-equilibrium WB with friction);
    # the reconstruction supplies the slope closure, so — like newton_solve —
    # it is NOT a lambdify-lowered REQUIRED_KERNEL.
    "solve_steady_ode": solve_steady_ode,
}

# Arithmetic / printer-lowered helpers the numpy printer emits — NOT part of the
# backend-supplied kernel contract (``conditional`` lowers to ``np.where``).
ARITHMETIC = {
    "ones_like": np.ones_like,
    "zeros_like": np.zeros_like,
    "array": np.array,
    "squeeze": np.squeeze,
    "conditional": lambda c, t, f: np.where(c, t, f),
}


def numpy_module():
    """The full numpy runtime module dict = arithmetic helpers + the
    UserFunctions kernel table.  `to_numpy.NumpyRuntimeModel.module` is this."""
    return dict(ARITHMETIC, **KERNELS)
