"""Eigenvalues of a batch of small real nonsymmetric matrices, computed on the device.

`jnp.linalg.eigvals` on a CUDA GPU runs cuSOLVER `Xgeev` once per matrix of the batch (a
hybrid host / device routine with its own synchronisations): 0.55 ms per 14 x 14 matrix on an
L40S, 67 ms for the 122 face sides of a 60-cell mesh (LAPACK on the host: 8 ms; this module's
Triton kernel: 0.9 ms).  `eigvals_device` follows LAPACK `dgeev`'s path for the eigenvalues (`dgebal`
isolation and power-of-two scaling, `dgehd2` Householder reduction, `dlahqr` double-shift QR
with the Ahues-Tisseur deflation test, `dlanv2` for the converged 2 x 2 blocks).  The
reduction and the QR iteration are written on whole `N x N` blocks (`N` the power of two >=
n) with masks, never an indexed load, so that the same code is one Triton kernel on a CUDA
GPU (Pallas: a program per matrix, the matrix in registers, the loop inside the kernel) and
a vmapped XLA loop elsewhere.  It is LAPACK's algorithm in another order of floating-point
operations: the eigenvalues agree with LAPACK's to round-off, not bitwise, and come in
another order.  `eigvals` keeps `jnp.linalg.eigvals` (LAPACK) on the CPU.

Derivatives (jvp, vjp) of `eigvals_device` are those of `jnp.linalg.eigvals`, evaluated for
them (its primal and ordering).

References:
    Anderson et al. 1999, LAPACK Users' Guide, https://doi.org/10.1137/1.9780898719604
    Francis 1961, https://doi.org/10.1093/comjnl/4.3.265
    Ahues, Tisseur 1997, LAPACK Working Note 122
    Parlett, Reinsch 1969, https://doi.org/10.1007/BF02165404
    Tillet, Kung, Cox 2019, https://doi.org/10.1145/3315508.3329973
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

__all__ = ["eigvals", "eigvals_device"]

#: `dlahqr`'s exceptional-shift period and weights
_KEXSH, _DAT1, _DAT2 = 10, 0.75, -0.4375
#: the largest block a Triton program takes (L40S, 122 matrices: n = 14, 2 warps 0.9 ms; n = 30,
#: 2 warps 5.9 ms; n = 40, 8 warps 34 ms, compiled in 50 s; cuSOLVER 67 / 137 / 180 ms): above,
#: the vmapped XLA loop
_TRITON_MAX = 64


# ── scalar helpers (Triton-lowerable: no copysign / hypot primitives there) ──────────────
def _signbit(x):
    """The sign bit of the float `x` (true for -0.0)."""
    bits = lax.bitcast_convert_type(x, jnp.uint64 if x.dtype == jnp.float64 else jnp.uint32)
    return lax.shift_right_logical(bits, jnp.asarray(8 * x.dtype.itemsize - 1,
                                                     bits.dtype)) != 0


def _copysign(x, y):
    """`|x|` with the sign of `y` (Fortran `SIGN(x, y)`)."""
    return jnp.where(_signbit(y), -jnp.abs(x), jnp.abs(x))


def _hypot(a, b):
    """`dlapy2`: `sqrt(a^2 + b^2)` without unnecessary overflow."""
    a, b = jnp.abs(a), jnp.abs(b)
    w, z = jnp.maximum(a, b), jnp.minimum(a, b)
    ws = jnp.where(w == 0, 1.0, w)
    return jnp.where(z == 0, w, w * jnp.sqrt(1.0 + (z / ws) * (z / ws)))


def _householder(alpha, x1, x2):
    """`dlarfg` of `[alpha; x1; x2]`: `(beta, tau, v1, v2)`."""
    xnorm = jnp.sqrt(x1 * x1 + x2 * x2)
    zero = xnorm == 0
    beta = -_copysign(_hypot(alpha, xnorm), alpha)
    tau = jnp.where(zero, 0.0, (beta - alpha) / jnp.where(zero, 1.0, beta))
    s = jnp.where(zero, 0.0, 1.0 / jnp.where(zero, 1.0, alpha - beta))
    return jnp.where(zero, alpha, beta), tau, x1 * s, x2 * s


def _lanv2(a, b, c, d):
    """`dlanv2`: the eigenvalues `(rt1r, rt1i, rt2r, rt2i)` of `[[a, b], [c, d]]` from its
    standardised Schur factorisation."""
    eps = float(np.finfo(np.float64).eps) if a.dtype == jnp.float64 else float(
        np.finfo(np.float32).eps)
    sgn = lambda x: _copysign(jnp.ones_like(x), x)         # noqa: E731  (SIGN(1, x))
    temp = a - d
    p = 0.5 * temp
    bcmax = jnp.maximum(jnp.abs(b), jnp.abs(c))
    bcmis = jnp.minimum(jnp.abs(b), jnp.abs(c)) * sgn(b) * sgn(c)
    scale = jnp.maximum(jnp.abs(p), bcmax)
    sc = jnp.where(scale == 0, 1.0, scale)
    z = (p / sc) * p + (bcmax / sc) * bcmis
    real = z >= 4.0 * eps
    # real eigenvalues
    zr = p + _copysign(jnp.sqrt(scale) * jnp.sqrt(jnp.where(real, z, 0.0)), p)
    a_r = d + zr
    d_r = d - (bcmax / jnp.where(zr == 0, 1.0, zr)) * bcmis
    # complex or nearly equal eigenvalues: rotate to equal diagonal entries
    sigma = b + c
    tau = _hypot(sigma, temp)
    tau = jnp.where(tau == 0, 1.0, tau)
    cs = jnp.sqrt(0.5 * (1.0 + jnp.abs(sigma) / tau))
    sn = -(p / (tau * cs)) * sgn(sigma)
    aa, bb = a * cs + b * sn, -a * sn + b * cs
    cc, dd = c * cs + d * sn, -c * sn + d * cs
    b2, c2 = bb * cs + dd * sn, -aa * sn + cc * cs
    t = 0.5 * ((aa * cs + cc * sn) + (-bb * sn + dd * cs))
    split = (c2 != 0) & (b2 != 0) & (sgn(b2) == sgn(c2))          # real after all
    pr = _copysign(jnp.sqrt(jnp.abs(b2)) * jnp.sqrt(jnp.abs(c2)), c2)
    a_c, d_c = jnp.where(split, t + pr, t), jnp.where(split, t - pr, t)
    b_c = jnp.where(split, b2 - c2, jnp.where((c2 != 0) & (b2 == 0), -c2, b2))
    c_c = jnp.where(split | (b2 == 0), 0.0, c2)
    # the special cases: triangular, swapped, standard already
    tri = c == 0
    swap = ~tri & (b == 0)
    std = ~tri & ~swap & (temp == 0) & (sgn(b) != sgn(c))
    gen = ~(tri | swap | std)
    A = jnp.where(gen, jnp.where(real, a_r, a_c), jnp.where(swap, d, a))
    D = jnp.where(gen, jnp.where(real, d_r, d_c), jnp.where(swap, a, d))
    Bf = jnp.where(gen, jnp.where(real, b - c, b_c), jnp.where(swap, -c, b))
    C = jnp.where(gen, jnp.where(real, 0.0, c_c), jnp.where(swap, 0.0, c))
    im = jnp.where(C == 0, 0.0, jnp.sqrt(jnp.abs(Bf)) * jnp.sqrt(jnp.abs(C)))
    return A, im, D, -im


# ── whole-block access (masks and reductions only) ──────────────────────────────────────
class _Block:
    """Index grids of an `N x N` block: `r`, `c` (row, column of each entry), `v` (vector)."""

    def __init__(self, N):
        self.N = N
        self.r = lax.broadcasted_iota(jnp.int32, (N, N), 0)
        self.c = lax.broadcasted_iota(jnp.int32, (N, N), 1)
        self.v = lax.broadcasted_iota(jnp.int32, (N,), 0)

    def get(self, H, i, j):
        """`H[i, j]`."""
        return jnp.sum(jnp.where((self.r == i) & (self.c == j), H, 0.0))

    def row(self, H, i):
        """`H[i, :]`."""
        return jnp.sum(jnp.where(self.r == i, H, 0.0), axis=0)

    def col(self, H, j):
        """`H[:, j]`."""
        return jnp.sum(jnp.where(self.c == j, H, 0.0), axis=1)

    def at(self, x, i):
        """`x[i]` of a vector."""
        return jnp.sum(jnp.where(self.v == i, x, 0.0))

    def diag(self, H, offset=0):
        """`d[r] = H[r, r + offset]` (0 outside)."""
        return jnp.sum(jnp.where(self.c == self.r + offset, H, 0.0), axis=1)

    def down(self, x):
        """`y[r] = x[r - 1]` (0 at r = 0)."""
        return jnp.sum(jnp.where(self.r == self.c + 1, x[None, :], 0.0), axis=1)

    def up(self, x):
        """`y[r] = x[r + 1]` (0 at r = N - 1)."""
        return jnp.sum(jnp.where(self.c == self.r + 1, x[None, :], 0.0), axis=1)


def _hessenberg(H, X):
    """`dgehd2`: the upper Hessenberg form of `H` by Householder reflections."""
    N = X.N
    for k in range(N - 2):
        colk = X.col(H, k)
        alpha = X.at(colk, k + 1)
        x = jnp.where(X.v >= k + 2, colk, 0.0)
        xnorm = jnp.sqrt(jnp.sum(x * x))
        zero = xnorm == 0                                        # H = I: column k stays
        beta = -_copysign(_hypot(alpha, xnorm), alpha)
        tau = jnp.where(zero, 0.0, (beta - alpha) / jnp.where(zero, 1.0, beta))
        s = jnp.where(zero, 0.0, 1.0 / jnp.where(zero, 1.0, alpha - beta))
        beta = jnp.where(zero, alpha, beta)
        u = jnp.where(X.v == k + 1, 1.0, x * s)
        right = X.c > k
        Hu = jnp.sum(H * u[None, :], axis=1)                     # from the right
        H = jnp.where(right, H - tau * Hu[:, None] * u[None, :], H)
        uH = jnp.sum(u[:, None] * H, axis=0)                     # from the left
        H = jnp.where(right, H - tau * u[:, None] * uH[None, :], H)
        H = jnp.where((X.c == k) & (X.r == k + 1), beta,
                      jnp.where((X.c == k) & (X.r > k + 1), 0.0, H))
    return H


def _hqr(H, hi0, X):
    """`dlahqr` (eigenvalues only, `wantt = wantz = false`) on the block `[0, hi0]` of the upper
    Hessenberg `H`: `(wr, wi, ok)`, `ok` false when an eigenvalue took more than `dlahqr`'s
    `30 max(10, hi0 + 1)` iterations."""
    N = X.N
    dt = H.dtype
    ulp = float(jnp.finfo(dt).eps)
    smlnum = float(jnp.finfo(dt).tiny) * ((hi0 + 1).astype(dt) / ulp)
    itmax = 30 * jnp.maximum(10, hi0 + 1)
    v = X.v

    def body(state):
        H, i, kdefl, wr, wi, ok = state
        dg, sub, sup = X.diag(H), X.diag(H, -1), X.diag(H, 1)       # H[r,r], H[r,r-1], H[r,r+1]
        dgm = X.down(dg)                                             # H[r-1, r-1]
        # -- deflation: the largest k <= i whose H[k, k-1] is negligible (Ahues-Tisseur test)
        a = jnp.abs(sub)
        hkk = jnp.abs(dg)
        tst = jnp.abs(dgm) + hkk
        below = X.down(jnp.abs(sub))                                 # |H[k-1, k-2]|
        above = X.up(jnp.abs(sub))                                   # |H[k+1, k]|
        tst = jnp.where(tst == 0, tst + below + jnp.where(v + 1 <= hi0, above, 0.0), tst)
        hup = jnp.abs(X.down(sup))                                   # |H[k-1, k]|
        ab, ba = jnp.maximum(a, hup), jnp.minimum(a, hup)
        diff = jnp.abs(dgm - dg)
        aa, bb = jnp.maximum(hkk, diff), jnp.minimum(hkk, diff)
        s = aa + ab
        s = jnp.where(s == 0, 1.0, s)
        neg = (a <= smlnum) | ((a <= ulp * tst)
                               & (ba * (ab / s) <= jnp.maximum(smlnum, ulp * (bb * (aa / s)))))
        l = jnp.max(jnp.where(neg & (v >= 1) & (v <= i), v, 0))
        H = jnp.where((X.r == l) & (X.c == l - 1), 0.0, H)
        one, two = l >= i, l == i - 1
        sweep = ~one & ~two
        # -- converged: one eigenvalue, or the two of a 2 x 2 block
        im1 = jnp.maximum(i - 1, 0)
        hii, hi1i1 = X.get(H, i, i), X.get(H, im1, im1)
        hi1i, hii1 = X.get(H, im1, i), X.get(H, i, im1)
        r1r, r1i, r2r, r2i = _lanv2(hi1i1, hi1i, hii1, hii)
        at_i, at_im1 = v == i, (v == im1) & two
        wr = jnp.where(at_i & one, hii, jnp.where(at_i & two, r2r, jnp.where(at_im1, r1r, wr)))
        wi = jnp.where(at_i & one, 0.0, jnp.where(at_i & two, r2i, jnp.where(at_im1, r1i, wi)))
        kd = kdefl + 1
        # -- shifts: the trailing 2 x 2 block's eigenvalues (exceptional every 10th / 20th)
        ex_bot = (kd % (2 * _KEXSH)) == 0
        ex_top = ~ex_bot & ((kd % _KEXSH) == 0)
        ex = ex_bot | ex_top
        s_ex = jnp.where(ex_bot, jnp.abs(hii1) + jnp.abs(X.get(H, im1, jnp.maximum(i - 2, 0))),
                         jnp.abs(X.get(H, l + 1, l)) + jnp.abs(X.get(H, l + 2, l + 1)))
        h11 = jnp.where(ex, _DAT1 * s_ex + jnp.where(ex_bot, hii, X.get(H, l, l)), hi1i1)
        h21 = jnp.where(ex, s_ex, hii1)
        h12 = jnp.where(ex, _DAT2 * s_ex, hi1i)
        h22 = jnp.where(ex, h11, hii)
        s = jnp.abs(h11) + jnp.abs(h12) + jnp.abs(h21) + jnp.abs(h22)
        zs = s == 0
        ss = jnp.where(zs, 1.0, s)
        h11, h21, h12, h22 = h11 / ss, h21 / ss, h12 / ss, h22 / ss
        tr = (h11 + h22) / 2.0
        det = (h11 - tr) * (h22 - tr) - h12 * h21
        rtdisc = jnp.sqrt(jnp.abs(det))
        cplx = det >= 0
        q1, q2 = tr + rtdisc, tr - rtdisc
        rr = jnp.where(jnp.abs(q1 - h22) <= jnp.abs(q2 - h22), q1, q2) * s
        rt1r = jnp.where(zs, 0.0, jnp.where(cplx, tr * s, rr))
        rt2r = rt1r
        rt1i = jnp.where(zs | ~cplx, 0.0, rtdisc * s)
        rt2i = -rt1i
        # -- the start row m: from i-2 down to l, the first with two small subdiagonals
        hm1m = X.up(sub)                                             # H[m+1, m]
        hm1m1 = X.up(dg)                                             # H[m+1, m+1]
        hm2m1 = X.up(hm1m)                                           # H[m+2, m+1]
        sm = jnp.abs(dg - rt2r) + jnp.abs(rt2i) + jnp.abs(hm1m)
        sm = jnp.where(sm == 0, 1.0, sm)
        h21s = hm1m / sm
        v0 = h21s * sup + (dg - rt1r) * ((dg - rt2r) / sm) - rt1i * (rt2i / sm)
        v1 = h21s * (dg + hm1m1 - rt1r - rt2r)
        v2 = h21s * hm2m1
        sv = jnp.abs(v0) + jnp.abs(v1) + jnp.abs(v2)
        sv = jnp.where(sv == 0, 1.0, sv)
        v0, v1, v2 = v0 / sv, v1 / sv, v2 / sv
        h00 = jnp.abs(sub) * (jnp.abs(v1) + jnp.abs(v2))             # H[m, m-1]
        h01 = jnp.abs(v0) * (jnp.abs(dgm) + jnp.abs(dg) + jnp.abs(hm1m1))
        brk = ((v == l) | (h00 <= ulp * h01)) & (v >= l) & (v <= i - 2)
        m = jnp.max(jnp.where(brk, v, -1))
        vm0, vm1, vm2 = X.at(v0, m), X.at(v1, m), X.at(v2, m)
        # -- the double-shift QR sweep, k = m..i-1, on rows / columns l..i
        def chase(k, Hs):
            """Bulge-chasing step `k`: the reflector of `H[k:k+3, k-1]` (the shift vector at
            `k = m`) from the left on rows k..k+2 and from the right on columns k..k+2."""
            three = k < i - 1                                  # NR = 3, else 2
            first = k == m
            r0, r1, r2 = X.row(Hs, k), X.row(Hs, k + 1), X.row(Hs, k + 2)
            x0 = jnp.where(first, vm0, X.at(r0, k - 1))
            x1 = jnp.where(first, vm1, X.at(r1, k - 1))
            x2 = jnp.where(three, jnp.where(first, vm2, X.at(r2, k - 1)), 0.0)
            beta, t1, w2, w3 = _householder(x0, x1, x2)
            w3 = jnp.where(three, w3, 0.0)
            t2, t3 = t1 * w2, t1 * w3
            colk = X.c == k - 1                                # the bulge column k-1
            Hs = jnp.where(colk & (X.r == k), jnp.where(first, jnp.where(m > l, Hs * (1.0 - t1),
                                                                         Hs), beta),
                           jnp.where(colk & ((X.r == k + 1) | ((X.r == k + 2) & three))
                                     & ~first, 0.0, Hs))
            crow = jnp.where(v == k, t1, jnp.where(v == k + 1, t2,
                                                   jnp.where(v == k + 2, t3, 0.0)))
            # from the left: rows k, k+1 (, k+2) on columns k..i (column k-1 is not among them)
            srow = jnp.where(three, r0 + w2 * r1 + w3 * r2, r0 + w2 * r1)
            cmask = (v >= k) & (v <= i)
            Hs = jnp.where(cmask[None, :] & (crow != 0)[:, None],
                           Hs - crow[:, None] * srow[None, :], Hs)
            # from the right: columns k, k+1 (, k+2) on rows l..min(k+3, i)
            c0, c1, c2 = X.col(Hs, k), X.col(Hs, k + 1), X.col(Hs, k + 2)
            scol = jnp.where(three, c0 + w2 * c1 + w3 * c2, c0 + w2 * c1)
            rmask = (v >= l) & (v <= jnp.minimum(k + 3, i))
            return jnp.where(rmask[:, None] & (crow != 0)[None, :],
                             Hs - scol[:, None] * crow[None, :], Hs)
        Hs = lax.fori_loop(jnp.where(sweep, m, 0), jnp.where(sweep, i, 0), chase, H)
        H = jnp.where(sweep, Hs, H)
        i = jnp.where(one, i - 1, jnp.where(two, i - 2, i))
        kdefl = jnp.where(sweep, kd, 0)
        ok = ok & ~(sweep & (kd > itmax))
        return H, i, kdefl, wr, wi, ok

    z = jnp.zeros((N,), dt)
    _, _, _, wr, wi, ok = lax.while_loop(
        lambda s: (s[1] >= 0) & s[5], body,
        (H, hi0, jnp.zeros((), jnp.int32), z, z, jnp.ones((), jnp.bool_)))
    return wr, wi, ok


def _schur_eigvals(H, hi0):
    """Hessenberg reduction and QR iteration of one `N x N` block, active block `[0, hi0]`."""
    X = _Block(H.shape[-1])
    return _hqr(_hessenberg(H, X), hi0, X)


def _qr_xla(B, hi0):
    """`_schur_eigvals` of every matrix, vmapped XLA."""
    wr, wi, ok = jax.vmap(_schur_eigvals)(B, hi0)
    return wr, wi, jnp.broadcast_to(ok.astype(jnp.int32)[:, None], wr.shape)


def _qr_triton(B, hi0):
    """`_schur_eigvals` of every matrix in one Triton kernel (Pallas), a program per matrix."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt
    nb, N, _ = B.shape
    hiv = jnp.broadcast_to(hi0[:, None], (nb, N)).astype(jnp.int32)

    def kernel(h_ref, hi_ref, wr_ref, wi_ref, ok_ref):
        """One matrix: its eigenvalues and the convergence flag (broadcast over the row)."""
        wr, wi, ok = _schur_eigvals(h_ref[0], jnp.max(hi_ref[0]))
        wr_ref[0] = wr
        wi_ref[0] = wi
        ok_ref[0] = jnp.where(ok, 1, 0) + jnp.zeros((N,), jnp.int32)

    vec = pl.BlockSpec((1, N), lambda b: (b, 0))
    return pl.pallas_call(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((nb, N), B.dtype), jax.ShapeDtypeStruct((nb, N), B.dtype),
                   jax.ShapeDtypeStruct((nb, N), jnp.int32)),
        grid=(nb,),
        in_specs=[pl.BlockSpec((1, N, N), lambda b: (b, 0, 0)), vec],
        out_specs=(vec, vec, vec),
        compiler_params=plt.CompilerParams(num_warps=2 if N <= 32 else 8),  # 1: ptxas fails
    )(B, hiv)


def _isolate(A):
    """`dgebal`'s permutation as a mask: a row (column) of the active principal submatrix whose
    off-diagonal entries inside it are all zero isolates its diagonal entry as an eigenvalue;
    drop such rows and columns until none is left (at most n rounds).  Returns the active mask
    `(..., n)`."""
    n = A.shape[-1]
    off = (A != 0) & ~jnp.eye(n, dtype=bool)
    act = jnp.ones(A.shape[:-1], bool)
    for _ in range(n):              # unrolled: no loop condition to read back on a GPU
        act = act & (off & act[..., None, :]).any(axis=-1)          # a zero row isolates
        act = act & (off & act[..., :, None]).any(axis=-2)          # a zero column isolates
    return act


def _balance(B, sweeps=4):
    """Scale `B` by `D^-1 B D`, `D` diagonal of powers of two (exact): per sweep every row /
    column pair whose off-diagonal 1-norms `r`, `c` are brought closer by `2^k`, `4^k ~ r / c`,
    where that lowers `r + c` by 5 % (Parlett-Reinsch, all pairs of a sweep at once)."""
    n = B.shape[-1]
    offd = jnp.asarray(~np.eye(n, dtype=bool))
    one = jnp.ones((), B.dtype)
    for _ in range(sweeps):
        M = jnp.where(offd, jnp.abs(B), 0.0)
        c, r = M.sum(axis=-2), M.sum(axis=-1)
        ok = (c > 0) & (r > 0)
        _, ec = jnp.frexp(jnp.where(ok, c, 1.0))
        _, er = jnp.frexp(jnp.where(ok, r, 1.0))
        f = jnp.ldexp(one, (er - ec) // 2)
        f = jnp.where(ok & (c * f + r / f < 0.95 * (c + r)), f, 1.0)
        B = B / f[..., :, None] * f[..., None, :]
    return B


@jax.custom_jvp
def eigvals_device(A):
    """The eigenvalues `wr + i wi` (complex, `(..., n)`) of the real matrices `A` `(..., n, n)`
    on the device (module docstring); NaN for a matrix whose QR iteration does not converge, as
    `jnp.linalg.eigvals` gives."""
    batch, n = A.shape[:-2], A.shape[-1]
    A = A.reshape((-1, n, n))
    if A.shape[0] == 0 or n == 0:
        return jnp.zeros(batch + (n,), jnp.result_type(A.dtype, 1j))
    N = max(2, 1 << (n - 1).bit_length())                  # the block: a power of two >= n
    act = _isolate(A)
    perm = jnp.argsort(~act, axis=-1, stable=True)          # the active rows first
    m = jnp.sum(act, axis=-1).astype(jnp.int32)
    idx = jnp.arange(n)
    inside = (idx[:, None] < m[:, None, None]) & (idx[None, :] < m[:, None, None])
    P = jnp.take_along_axis(jnp.take_along_axis(A, perm[:, :, None], 1), perm[:, None, :], 2)
    B = _balance(jnp.where(inside, P, 0.0))
    B = jnp.pad(B, ((0, 0), (0, N - n), (0, N - n)))
    if N <= _TRITON_MAX:
        wr, wi, ok = lax.platform_dependent(B, m - 1, cuda=_qr_triton, default=_qr_xla)
    else:
        wr, wi, ok = _qr_xla(B, m - 1)
    wr, wi, ok = wr[:, :n], wi[:, :n], ok[:, :1] != 0
    diag = jnp.take_along_axis(jnp.diagonal(A, axis1=1, axis2=2), perm, 1)
    wr = jnp.where(idx < m[:, None], wr, diag)                # the isolated eigenvalues
    wi = jnp.where(idx < m[:, None], wi, 0.0)
    nan = jnp.asarray(jnp.nan, A.dtype)
    wr, wi = jnp.where(ok, wr, nan), jnp.where(ok, wi, nan)
    return lax.complex(wr, wi).reshape(batch + (n,))


@eigvals_device.defjvp
def _eigvals_device_jvp(primals, tangents):
    """The derivative of `jnp.linalg.eigvals`, at its own primal."""
    return jax.jvp(jnp.linalg.eigvals, primals, tangents)


def eigvals(A):
    """The eigenvalues (complex, `(..., n)`) of the real matrices `A` `(..., n, n)`: LAPACK
    (`jnp.linalg.eigvals`) on the CPU, `eigvals_device` on an accelerator."""
    A = jnp.asarray(A)
    if not jnp.issubdtype(A.dtype, jnp.floating):
        return jnp.linalg.eigvals(A)
    return lax.platform_dependent(A, cpu=jnp.linalg.eigvals, default=eigvals_device)
