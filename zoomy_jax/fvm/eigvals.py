"""Eigenvalues of a batch of small real nonsymmetric matrices, computed on the device.

`jnp.linalg.eigvals` on a CUDA GPU runs cuSOLVER `Xgeev` once per matrix of the batch (a
hybrid host / device routine with its own synchronisations): ~0.6 ms per 14 x 14 matrix on an
L40S, 37 ms for the 61 cells of a 60-cell mesh, while LAPACK on the host takes 4 ms for the
same batch.  `eigvals_device` follows LAPACK `dgeev`'s path for the eigenvalues (`dgebal`
isolation and power-of-two scaling, `dgehd2` Householder reduction, `dlahqr` double-shift QR
with the Ahues-Tisseur deflation test, `dlanv2` for the converged 2 x 2 blocks), written with
static indices, masks and rank-one updates, so one batched XLA program treats every matrix at
once.  It is the same algorithm with a different order of floating-point operations: the
eigenvalues agree with LAPACK's to round-off, not bitwise, and come in another order.
`eigvals` keeps `jnp.linalg.eigvals` (LAPACK) on the CPU.

Derivatives (jvp, vjp) of `eigvals_device` are those of `jnp.linalg.eigvals`, which are
evaluated for them (its primal and ordering).

References:
    Anderson et al. 1999, LAPACK Users' Guide, https://doi.org/10.1137/1.9780898719604
    Francis 1961, https://doi.org/10.1093/comjnl/4.3.265
    Ahues, Tisseur 1997, LAPACK Working Note 122
    Parlett, Reinsch 1969, https://doi.org/10.1007/BF02165404
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

__all__ = ["eigvals", "eigvals_device"]

#: `dlahqr`'s exceptional-shift period and weights
_KEXSH, _DAT1, _DAT2 = 10, 0.75, -0.4375


def _isolate(A):
    """`dgebal`'s permutation as a mask: a row (column) of the active principal submatrix whose
    off-diagonal entries inside it are all zero isolates its diagonal entry as an eigenvalue;
    drop such rows and columns until none is left.  Returns the active mask `(n,)`."""
    n = A.shape[-1]
    off = (A != 0) & ~jnp.eye(n, dtype=bool)

    def body(_, act):
        act = act & (off & act[None, :]).any(axis=1)        # a zero row isolates
        return act & (off & act[:, None]).any(axis=0)       # a zero column isolates
    return lax.fori_loop(0, n, body, jnp.ones(n, bool))


def _balance(B, sweeps=4):
    """Scale `B` by `D^-1 B D`, `D` diagonal of powers of two (exact): per sweep every row /
    column pair whose off-diagonal 1-norms `r`, `c` are brought closer by `2^k`, `4^k ~ r / c`,
    where that lowers `r + c` by 5 % (Parlett-Reinsch, all pairs of a sweep at once)."""
    n = B.shape[-1]
    offd = jnp.asarray(~np.eye(n, dtype=bool))
    one = jnp.ones((), B.dtype)
    for _ in range(sweeps):
        M = jnp.where(offd, jnp.abs(B), 0.0)
        c, r = M.sum(axis=0), M.sum(axis=1)
        ok = (c > 0) & (r > 0)
        _, ec = jnp.frexp(jnp.where(ok, c, 1.0))
        _, er = jnp.frexp(jnp.where(ok, r, 1.0))
        f = jnp.ldexp(one, (er - ec) // 2)
        f = jnp.where(ok & (c * f + r / f < 0.95 * (c + r)), f, 1.0)
        B = B / f[:, None] * f[None, :]
    return B


def _householder(alpha, x):
    """`dlarfg`: `(beta, tau, v)` with `(I - tau [1; v] [1; v]^T) [alpha; x] = [beta; 0]`."""
    xnorm = jnp.sqrt(jnp.sum(x * x, axis=0))
    zero = xnorm == 0
    beta = -jnp.copysign(jnp.hypot(alpha, xnorm), alpha)
    beta_s = jnp.where(zero, 1.0, beta)
    tau = jnp.where(zero, 0.0, (beta - alpha) / beta_s)
    v = x * jnp.where(zero, 0.0, 1.0 / jnp.where(zero, 1.0, alpha - beta))
    return jnp.where(zero, alpha, beta), tau, v


def _hessenberg(B):
    """`dgehd2`: the upper Hessenberg form of `B` by Householder reflections."""
    n = B.shape[-1]
    rows = np.arange(n)
    for k in range(n - 2):
        beta, tau, v = _householder(B[k + 1, k], B[k + 2:, k])
        u = jnp.concatenate([jnp.zeros((k + 1,), B.dtype), jnp.ones((1,), B.dtype), v])
        B = B - tau * jnp.outer(B @ u, u) * (rows[None, :] > k)        # from the right
        B = B - tau * jnp.outer(u, u @ B) * (rows[None, :] > k)        # from the left
        B = jnp.where((rows[:, None] == k + 1) & (rows[None, :] == k), beta,
                      jnp.where((rows[:, None] > k + 1) & (rows[None, :] == k), 0.0, B))
    return B


def _lanv2(a, b, c, d):
    """`dlanv2`: the eigenvalues `(rt1r, rt1i, rt2r, rt2i)` of `[[a, b], [c, d]]` from its
    standardised Schur factorisation."""
    eps = jnp.finfo(a.dtype).eps
    sgn = lambda x: jnp.copysign(1.0, x)              # noqa: E731  (Fortran SIGN(1, x))
    temp = a - d
    p = 0.5 * temp
    bcmax = jnp.maximum(jnp.abs(b), jnp.abs(c))
    bcmis = jnp.minimum(jnp.abs(b), jnp.abs(c)) * sgn(b) * sgn(c)
    scale = jnp.maximum(jnp.abs(p), bcmax)
    z = (p / jnp.where(scale == 0, 1.0, scale)) * p + (bcmax / jnp.where(scale == 0, 1.0, scale)) * bcmis
    real = z >= 4.0 * eps
    # real eigenvalues
    zr = p + jnp.copysign(jnp.sqrt(scale) * jnp.sqrt(jnp.where(real, z, 0.0)), p)
    a_r = d + zr
    d_r = d - (bcmax / jnp.where(zr == 0, 1.0, zr)) * bcmis
    # complex or nearly equal eigenvalues: rotate to equal diagonal entries
    sigma = b + c
    tau = jnp.hypot(sigma, temp)
    tau = jnp.where(tau == 0, 1.0, tau)
    cs = jnp.sqrt(0.5 * (1.0 + jnp.abs(sigma) / tau))
    sn = -(p / (tau * cs)) * sgn(sigma)
    aa, bb = a * cs + b * sn, -a * sn + b * cs
    cc, dd = c * cs + d * sn, -c * sn + d * cs
    b2, c2 = bb * cs + dd * sn, -aa * sn + cc * cs
    t = 0.5 * ((aa * cs + cc * sn) + (-bb * sn + dd * cs))
    split = (c2 != 0) & (b2 != 0) & (sgn(b2) == sgn(c2))          # real after all
    pr = jnp.copysign(jnp.sqrt(jnp.abs(b2)) * jnp.sqrt(jnp.abs(c2)), c2)
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


def _hqr(H, hi0):
    """`dlahqr` (eigenvalues only, `wantt = wantz = false`) on the block `[0, hi0]` of the upper
    Hessenberg `H`; returns `(wr, wi, ok)`, `ok` false when an eigenvalue took more than
    `dlahqr`'s `30 max(10, hi0 + 1)` iterations."""
    n = H.shape[-1]
    dt = H.dtype
    ulp = jnp.finfo(dt).eps
    smlnum = jnp.finfo(dt).tiny * ((hi0 + 1).astype(dt) / ulp)
    itmax = 30 * jnp.maximum(10, hi0 + 1)
    idx = np.arange(n)
    ks = np.arange(1, n)                           # subdiagonal positions H[k, k-1]
    ms = np.arange(max(n - 2, 0))                  # candidate start rows of a sweep
    eye = np.eye(n)

    def body(state):
        H, i, kdefl, wr, wi, ok = state
        dg, sub, sup = jnp.diagonal(H), jnp.diagonal(H, -1), jnp.diagonal(H, 1)
        # -- deflation: the largest k <= i whose H[k, k-1] is negligible (Ahues-Tisseur test)
        a = jnp.abs(sub)
        hkk = jnp.abs(dg[1:])
        tst = jnp.abs(dg[:-1]) + hkk
        below = jnp.concatenate([jnp.zeros((1,), dt), jnp.abs(sub[:-1])])   # |H[k-1, k-2]|
        above = jnp.concatenate([jnp.abs(sub[1:]), jnp.zeros((1,), dt)])    # |H[k+1, k]|
        tst = jnp.where(tst == 0, tst + below + jnp.where(ks + 1 <= hi0, above, 0.0), tst)
        ab, ba = jnp.maximum(a, jnp.abs(sup)), jnp.minimum(a, jnp.abs(sup))
        diff = jnp.abs(dg[:-1] - dg[1:])
        aa, bb = jnp.maximum(hkk, diff), jnp.minimum(hkk, diff)
        s = aa + ab
        s = jnp.where(s == 0, 1.0, s)
        neg = (a <= smlnum) | ((a <= ulp * tst)
                               & (ba * (ab / s) <= jnp.maximum(smlnum, ulp * (bb * (aa / s)))))
        l = jnp.max(jnp.where(neg & (ks <= i), ks, 0), initial=0)
        H = jnp.where((idx[:, None] == l) & (idx[None, :] == l - 1), 0.0, H)
        one, two = l >= i, l == i - 1
        sweep = ~one & ~two
        # -- converged: one eigenvalue, or the two of a 2 x 2 block
        im1 = jnp.maximum(i - 1, 0)
        r1r, r1i, r2r, r2i = _lanv2(H[im1, im1], H[im1, i], H[i, im1], H[i, i])
        at_i, at_im1 = idx == i, (idx == im1) & two
        wr = jnp.where(at_i & one, H[i, i], jnp.where(at_i & two, r2r,
                                                      jnp.where(at_im1, r1r, wr)))
        wi = jnp.where(at_i & one, 0.0, jnp.where(at_i & two, r2i,
                                                  jnp.where(at_im1, r1i, wi)))
        kd = kdefl + 1
        if n < 3:                           # no sweep: every block converges as is
            return H, jnp.where(one, i - 1, i - 2), kdefl, wr, wi, ok
        # -- shifts: the trailing 2 x 2 block's eigenvalues (exceptional every 10th / 20th)
        ex_bot = (kd % (2 * _KEXSH)) == 0
        ex_top = ~ex_bot & ((kd % _KEXSH) == 0)
        im2 = jnp.maximum(i - 2, 0)
        lp2 = jnp.minimum(l + 2, n - 1)
        s_ex = jnp.where(ex_bot, jnp.abs(H[i, im1]) + jnp.abs(H[im1, im2]),
                         jnp.abs(H[l + 1, l]) + jnp.abs(H[lp2, l + 1]))
        ex = ex_bot | ex_top
        h11 = jnp.where(ex, _DAT1 * s_ex + jnp.where(ex_bot, H[i, i], H[l, l]), H[im1, im1])
        h21 = jnp.where(ex, s_ex, H[i, im1])
        h12 = jnp.where(ex, _DAT2 * s_ex, H[im1, i])
        h22 = jnp.where(ex, h11, H[i, i])
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
        Hmm, Hm1m, Hmm1 = dg[:-2], sub[:-1], sup[:-1]
        Hm1m1, Hm2m1 = dg[1:-1], sub[1:]
        sm = jnp.abs(Hmm - rt2r) + jnp.abs(rt2i) + jnp.abs(Hm1m)
        sm = jnp.where(sm == 0, 1.0, sm)
        h21s = Hm1m / sm
        v0 = h21s * Hmm1 + (Hmm - rt1r) * ((Hmm - rt2r) / sm) - rt1i * (rt2i / sm)
        v1 = h21s * (Hmm + Hm1m1 - rt1r - rt2r)
        v2 = h21s * Hm2m1
        sv = jnp.abs(v0) + jnp.abs(v1) + jnp.abs(v2)
        sv = jnp.where(sv == 0, 1.0, sv)
        v0, v1, v2 = v0 / sv, v1 / sv, v2 / sv
        Hmm_1 = jnp.concatenate([jnp.zeros((1,), dt), sub[:-2]])           # H[m, m-1]
        Hm_1m_1 = jnp.concatenate([jnp.zeros((1,), dt), dg[:-3]])          # H[m-1, m-1]
        h00 = jnp.abs(Hmm_1) * (jnp.abs(v1) + jnp.abs(v2))
        h01 = jnp.abs(v0) * (jnp.abs(Hm_1m_1) + jnp.abs(Hmm) + jnp.abs(Hm1m1))
        brk = ((ms == l) | (h00 <= ulp * h01)) & (ms >= l) & (ms <= i - 2)
        m = jnp.max(jnp.where(brk, ms, -1))
        mm = jnp.maximum(m, 0)
        vm = (v0[mm], v1[mm], v2[mm])
        # -- the double-shift QR sweep, k = m..i-1, on rows / columns l..i
        Hs = H
        for k in range(n - 1):
            act = sweep & (k >= m) & (k <= i - 1)
            three = k < i - 1                                  # NR = 3, else 2
            first = k == m
            has3 = k + 2 <= n - 1
            if k > 0:
                c0, c1 = Hs[k, k - 1], Hs[k + 1, k - 1]
                c2 = jnp.where(three, Hs[k + 2, k - 1], 0.0) if has3 else 0.0 * c0
                x0 = jnp.where(first, vm[0], c0)
                x1 = jnp.where(first, vm[1], c1)
                x2 = jnp.where(first, vm[2], c2)
            else:
                x0, x1, x2 = vm
            x2 = jnp.where(three, x2, 0.0)
            beta, t1, w = _householder(x0, jnp.stack([x1, x2]))
            w2, w3 = w[0], jnp.where(three, w[1], 0.0)
            t2, t3 = t1 * w2, t1 * w3
            if k > 0:                                          # the bulge column k-1
                upd = act & ~first
                keep = act & first & (m > l)
                col = idx[None, :] == k - 1
                Hs = jnp.where(col & (idx[:, None] == k) & upd, beta,
                               jnp.where(col & (idx[:, None] == k) & keep, Hs * (1.0 - t1),
                                         jnp.where(col & (idx[:, None] == k + 1) & upd, 0.0,
                                                   jnp.where(col & (idx[:, None] == k + 2)
                                                             & upd & three, 0.0, Hs))))
            # from the left: rows k, k+1 (, k+2) on columns k..i
            cmask = act & (idx >= k) & (idx <= i)
            r3 = Hs[k + 2] if has3 else jnp.zeros((n,), dt)
            srow = jnp.where(three, Hs[k] + w2 * Hs[k + 1] + w3 * r3, Hs[k] + w2 * Hs[k + 1])
            crow = t1 * eye[k] + t2 * eye[k + 1] + (t3 * eye[k + 2] if has3 else 0.0)
            Hs = jnp.where(cmask[None, :] & (crow != 0)[:, None],
                           Hs - crow[:, None] * srow[None, :], Hs)
            # from the right: columns k, k+1 (, k+2) on rows l..min(k+3, i)
            rmask = act & (idx >= l) & (idx <= jnp.minimum(k + 3, i))
            c3 = Hs[:, k + 2] if has3 else jnp.zeros((n,), dt)
            scol = jnp.where(three, Hs[:, k] + w2 * Hs[:, k + 1] + w3 * c3,
                             Hs[:, k] + w2 * Hs[:, k + 1])
            Hs = jnp.where(rmask[:, None] & (crow != 0)[None, :],
                           Hs - scol[:, None] * crow[None, :], Hs)
        H = jnp.where(sweep, Hs, H)
        i = jnp.where(one, i - 1, jnp.where(two, i - 2, i))
        kdefl = jnp.where(sweep, kd, 0)
        ok = ok & ~(sweep & (kd > itmax))
        return H, i, kdefl, wr, wi, ok

    z = jnp.zeros((n,), dt)
    _, _, _, wr, wi, ok = lax.while_loop(
        lambda s: (s[1] >= 0) & s[5], body,
        (H, hi0, jnp.zeros((), jnp.int32), z, z, jnp.ones((), bool)))
    return wr, wi, ok


def _eigvals_one(A):
    """`(wr, wi)` of one real matrix by `dgeev`'s path (eigenvalues only)."""
    n = A.shape[-1]
    act = _isolate(A)
    perm = jnp.argsort(~act, stable=True)                  # the active rows first
    m = jnp.sum(act).astype(jnp.int32)
    idx = jnp.arange(n)
    inside = (idx[:, None] < m) & (idx[None, :] < m)
    B = jnp.where(inside, A[perm][:, perm], 0.0)
    wr, wi, ok = _hqr(_hessenberg(_balance(B)), m - 1)
    wr = jnp.where(idx < m, wr, jnp.diagonal(A)[perm])     # the isolated eigenvalues
    wi = jnp.where(idx < m, wi, 0.0)
    nan = jnp.asarray(jnp.nan, A.dtype)
    return jnp.where(ok, wr, nan), jnp.where(ok, wi, nan)


@jax.custom_jvp
def eigvals_device(A):
    """The eigenvalues `wr + i wi` (complex, `(..., n)`) of the real matrices `A` `(..., n, n)`
    on the device (module docstring); NaN for a matrix whose QR iteration does not converge, as
    `jnp.linalg.eigvals` gives."""
    batch, n = A.shape[:-2], A.shape[-1]
    flat = A.reshape((-1, n, n))
    if flat.shape[0] == 0 or n == 0:
        return jnp.zeros(batch + (n,), jnp.result_type(A.dtype, 1j))
    wr, wi = jax.vmap(_eigvals_one)(flat)
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
