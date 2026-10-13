"""Minimal numpy backend primitive object `B` (host lowering).

Never imports jax (pyodide-safe).  Python `while` / `if`, `raise` for
guards, direct calls for callbacks, Newton with an FD / analytic Jacobian
action, scipy GMRES and dense / cell-block / banded direct solves.  The full
numpy solver backend (bodies, presets) is stage 2; this object is what the
walker needs to lower every statement kind.
"""
from __future__ import annotations

import warnings

import numpy as np

from zoomy_core.solver.template.errors import GuardError, format_guard

#: central-difference step scale (cube root of the float64 epsilon)
FD_EPS = float(np.finfo(float).eps) ** (1.0 / 3.0)


def as_outputs(value, n):
    """Normalise a body's return to a tuple of `n` outputs."""
    if n == 0:
        return ()
    if n == 1:
        return (value,)
    value = tuple(value)
    if len(value) != n:
        raise ValueError(f"expected {n} outputs, got {len(value)}")
    return value


def fd_jvp(f, x, v, fx=None):
    """Central finite-difference directional derivative of `f` at `x`."""
    vmax = float(np.max(np.abs(v))) if np.size(v) else 0.0
    if vmax == 0.0:
        return np.zeros_like(np.asarray(f(x) if fx is None else fx))
    h = FD_EPS * (1.0 + float(np.max(np.abs(x)))) / vmax
    return (np.asarray(f(x + h * v)) - np.asarray(f(x - h * v))) / (2.0 * h)


def _cell_view(b):
    """(rows, cells) view of an array whose trailing axis is the cell axis."""
    b = np.asarray(b)
    return b.reshape(-1, b.shape[-1]) if b.ndim > 1 else b.reshape(1, -1)


def project_nullspace(v, mode):
    if mode is None:
        return v
    v = np.asarray(v)
    if mode == "constant":
        return v - np.mean(v)
    if mode == "constant_per_row":
        return v - np.mean(v, axis=-1, keepdims=True)
    raise ValueError(mode)


class NumpyLinearSolvers:
    """The linear-solve kinds of `LinSolveOpts` on numpy.  Each
    returns `(x, ok, rel_resid, iters)`."""

    @staticmethod
    def solve(matvec, b, *, kind, tol, restart=30, maxiter=50, precond=None,
              nullspace=None, bandwidth=None, symmetric=False):
        b = np.asarray(b, dtype=float)
        shape = b.shape
        rhs = project_nullspace(b, nullspace)

        def A(x):
            return np.asarray(matvec(x.reshape(shape)), dtype=float).reshape(shape)

        if kind == "krylov":
            x, it = _gmres_scipy(A, rhs, tol=tol, restart=restart,
                                 maxiter=maxiter, precond=precond)
        elif kind == "dense":
            n = rhs.size
            cols = [A(np.eye(1, n, j).reshape(shape)).ravel() for j in range(n)]
            M = np.stack(cols, axis=1)
            x = _dense_solve(M, rhs.ravel(), nullspace).reshape(shape)
            it = 1
        elif kind == "cellwise":
            x = cellwise_solve(A, rhs)
            it = 1
        elif kind == "banded":
            x = banded_solve(A, rhs, int(bandwidth))
            it = 1
        else:
            raise ValueError(f"unknown linear solve kind {kind!r}")
        x = project_nullspace(x, nullspace)
        bn = float(np.linalg.norm(rhs))
        rr = float(np.linalg.norm(rhs - A(x))) / (bn if bn > 0 else 1.0)
        return x, bool(rr <= tol), rr, int(it)


def _dense_solve(M, b, nullspace):
    if nullspace is None:
        return np.linalg.solve(M, b)
    return np.linalg.lstsq(M, b, rcond=None)[0]


def _gmres_scipy(A, b, *, tol, restart, maxiter, precond):
    from scipy.sparse.linalg import LinearOperator, gmres
    shape = b.shape
    n = b.size
    op = LinearOperator((n, n), matvec=lambda v: A(np.asarray(v)).ravel(),
                        dtype=float)
    M = None
    if precond is not None:
        M = LinearOperator((n, n), dtype=float, matvec=lambda v: np.asarray(
            precond(np.asarray(v).reshape(shape)), dtype=float).ravel())
    count = [0]

    def cb(_):
        count[0] += 1

    x, info = gmres(op, b.ravel(), rtol=float(tol), atol=0.0,
                    restart=int(min(restart, n)), maxiter=int(maxiter), M=M,
                    callback=cb, callback_type="pr_norm")
    return np.asarray(x).reshape(shape), count[0]


def cellwise_solve(A, b):
    """Solve with an operator that is block-diagonal over the trailing (cell)
    axis: the per-cell blocks are read off with one probe per row."""
    b = np.asarray(b, dtype=float)
    B2 = _cell_view(b)
    nv, nc = B2.shape
    cols = []
    for k in range(nv):
        e = np.zeros_like(B2)
        e[k] = 1.0
        cols.append(_cell_view(A(e.reshape(b.shape))))
    J = np.stack(cols, axis=-1)                      # (nv_i, nc, nv_k)
    J = np.transpose(J, (1, 0, 2))                   # (nc, i, k)
    x = np.linalg.solve(J, B2.T[:, :, None])[:, :, 0].T
    return x.reshape(b.shape)


def band_probe(A, b, bw):
    """Columns of an operator block-banded over the cell axis (half-bandwidth
    `bw` cells), recovered with `nv*(2bw+1)` probes.  Returns the dense
    matrix in cell-major ordering (index `c*nv + v`)."""
    b = np.asarray(b, dtype=float)
    B2 = _cell_view(b)
    nv, nc = B2.shape
    stride = 2 * bw + 1
    n = nv * nc
    M = np.zeros((n, n))
    for v in range(nv):
        for color in range(stride):
            e = np.zeros_like(B2)
            e[v, color::stride] = 1.0
            col = _cell_view(A(e.reshape(b.shape)))          # (nv, nc)
            for c in range(color, nc, stride):
                lo, hi = max(0, c - bw), min(nc, c + bw + 1)
                for r in range(lo, hi):
                    M[r * nv:(r + 1) * nv, c * nv + v] = col[:, r]
    return M


def banded_solve(A, b, bw):
    from scipy.linalg import solve_banded
    b = np.asarray(b, dtype=float)
    B2 = _cell_view(b)
    nv, nc = B2.shape
    M = band_probe(A, b, bw)
    l = u = nv * (bw + 1) - 1
    n = nv * nc
    ab = np.zeros((l + u + 1, n))
    for j in range(n):
        for i in range(max(0, j - u), min(n, j + l + 1)):
            ab[u + i - j, j] = M[i, j]
    rhs = B2.T.reshape(-1)                               # cell-major
    x = solve_banded((l, u), ab, rhs)
    return x.reshape(nc, nv).T.reshape(b.shape)


class NumpyBackend:
    """Backend primitive object for numpy (host lowering everywhere)."""

    name = "numpy"
    compiles = False
    xp = np

    # -- control flow ---------------------------------------------------------
    @staticmethod
    def jit(f):
        return f

    @staticmethod
    def while_loop(cond, body, carry):
        while bool(cond(carry)):
            carry = body(carry)
        return carry

    @staticmethod
    def repeat(n, body, carry):
        for i in range(int(n)):
            carry = body(i, carry)
        return carry

    foreach = repeat

    @staticmethod
    def cond(pred, then, otherwise, operands):
        return then(*operands) if bool(pred) else otherwise(*operands)

    # -- checks / effects -----------------------------------------------------
    @staticmethod
    def guard(ok, msg, names, values, anchors):
        if not bool(np.all(ok)):
            raise GuardError(format_guard(msg, names, values))
        return tuple(anchors)

    @staticmethod
    def warn(ok, msg, names, values):
        if not bool(np.all(ok)):
            warnings.warn(format_guard(msg, names, values), RuntimeWarning,
                          stacklevel=2)

    @staticmethod
    def callback_idempotent(fn, args):
        fn(*args)

    @staticmethod
    def callback_returns(fn, args, like):
        return as_outputs(fn(*args), len(like))

    @staticmethod
    def callback_stateful(fn, args, n_out):
        return as_outputs(fn(*args), n_out)

    # -- solves ---------------------------------------------------------------
    @staticmethod
    def linsolve(matvec, rhs, **opts):
        return NumpyLinearSolvers.solve(matvec, rhs, **opts)

    @staticmethod
    def root(residual, x0, *, tol, maxiter, jac="ad", jvp=None, linsolve=None):
        """Newton.  numpy has no AD: `jac='ad'` uses the central FD Jacobian
        action (the root is the same to `tol`); `analytic` uses `jvp`."""
        lin = dict(linsolve or {"kind": "cellwise"})
        x = np.array(x0, dtype=float, copy=True)
        r = np.asarray(residual(x), dtype=float)
        rn = float(np.linalg.norm(r))
        it = 0
        while it < int(maxiter) and rn > tol:
            if jac == "analytic":
                def mv(v, _x=x):
                    return np.asarray(jvp(_x, v), dtype=float)
            else:
                def mv(v, _x=x, _r=r):
                    return fd_jvp(residual, _x, v, _r)
            dx, _, _, _ = NumpyLinearSolvers.solve(mv, -r, **lin)
            x = x + dx
            r = np.asarray(residual(x), dtype=float)
            rn = float(np.linalg.norm(r))
            it += 1
        return x, bool(rn <= tol), rn, it

    # -- array primitives -----------------------------------------------------
    @staticmethod
    def _at(target, index):
        return (slice(None),) * (np.ndim(target) - 1) + (index,)

    @classmethod
    def scatter_add(cls, target, index, values):
        out = np.array(target, copy=True)
        np.add.at(out, cls._at(out, index), values)
        return out

    @classmethod
    def scatter_min(cls, target, index, values):
        out = np.array(target, copy=True)
        np.minimum.at(out, cls._at(out, index), values)
        return out

    @classmethod
    def scatter_max(cls, target, index, values):
        out = np.array(target, copy=True)
        np.maximum.at(out, cls._at(out, index), values)
        return out

    @staticmethod
    def reduce_min(x):
        return np.min(x)

    @staticmethod
    def reduce_max(x):
        return np.max(x)

    @staticmethod
    def reduce_sum(x):
        return np.sum(x)

    @staticmethod
    def reduce_any(x):
        return bool(np.any(x))

    @staticmethod
    def reduce_all(x):
        return bool(np.all(x))

    @staticmethod
    def inner(a, b):
        return float(np.vdot(np.ravel(a), np.ravel(b)))

    @staticmethod
    def set_at(x, index, value):
        y = np.array(x, copy=True)
        y[index] = value
        return y

    @staticmethod
    def safe_div(a, b, fill):
        ok = b != 0
        return np.where(ok, a / np.where(ok, b, 1.0), fill)

    @staticmethod
    def as_int(x):
        if np.ndim(x) == 0:
            return int(x)
        return np.asarray(x).astype(np.int64)


__all__ = ["NumpyBackend", "NumpyLinearSolvers", "fd_jvp", "cellwise_solve",
           "banded_solve", "band_probe", "as_outputs", "project_nullspace",
           "FD_EPS"]
