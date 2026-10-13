"""NumPy runtime of the symbolic solvers: what each name of a printed solver means in plain
numpy, block for block the serial path of `zoomy_jax.symsolver.runtime`.  `While` is a Python
loop, `Cond` an if / else, `Root` a Newton iteration with a finite-difference Jacobian and
`LinSolve` a restarted GMRES (scipy), optionally AMG-preconditioned (pyamg).

References:
    Saad, Schultz 1986, https://doi.org/10.1137/0907058
    Knoll, Keyes 2004, https://doi.org/10.1016/j.jcp.2003.08.010
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from zoomy_core.fvm.reconstruction import DenseDiffusionOperator
from zoomy_core.mesh.geometry_xp import compute_geometry, lsq_derivatives
from zoomy_core.misc.logger_config import logger
from zoomy_core.model.boundary_conditions import boundary_face_branches
from zoomy_core.symsolver.core import contract
from zoomy_core.symsolver.numpy_backend.assembly import cell_pattern, host_amg_solve


class Geometry(NamedTuple):
    """The `mesh` argument of a printed solver: the geometry of `geometry_xp.compute_geometry`."""
    cell_volumes: np.ndarray
    cell_centers: np.ndarray
    cell_inradius: np.ndarray
    face_normals: np.ndarray
    face_volumes: np.ndarray
    face_centers: np.ndarray
    lsq_gradQ: np.ndarray
    lsq_closure: np.ndarray


def geometry_of(topo, X):
    """The solver geometry from the vertex coordinates `X` `(dim, n_vertices)`; also returns
    the full `compute_geometry` dict.
    """
    g = compute_geometry(topo, np.asarray(X, dtype=float))
    return Geometry(*(g[f] for f in Geometry._fields)), g


def runtime(nsm, rt, mesh_np, topo, geom, *, on_write=None, on_save=None, on_load=None,
            implicit=None, systems=None):
    """The name table (dict of callables) a printed solver runs against: the numpy
    implementation of every `zoomy_core.symsolver` block on this mesh, serial.

    Args:
        rt: the `NumpyKernels` of `nsm`
        topo: static topology (`geometry_xp.build_topology`) with the boundary-closure classes
        geom: the `compute_geometry` dict of the mesh
        on_write: `on_write(time, Q, Qaux)` snapshot callback
        on_save, on_load: the restart pair, `on_save(time, Q, Qaux)` and `on_load(time, Q, Qaux)
            -> (time, Q, Qaux)`; None: no restart file / not restarting
        implicit: Newton and linear-solver settings of `root` and `linsolve`
        systems: further `{name: (nsm, NumpyKernels)}` a template uses besides `nsm`
    Returns:
        dict name -> callable
    """
    nc = int(mesh_np.n_inner_cells)
    nf = int(mesh_np.n_faces)
    n_bf = int(mesh_np.n_boundary_faces)
    dim = int(mesh_np.dimension)
    fc0 = np.asarray(mesh_np.face_cells[0])
    fc1 = np.asarray(mesh_np.face_cells[1])
    bf_face = np.asarray(mesh_np.boundary_face_face_indices, dtype=int)
    bf_cells = np.asarray(mesh_np.boundary_face_cells, dtype=int)
    is_bnd = np.zeros(nf, dtype=bool)
    is_bnd[bf_face] = True
    interior = np.flatnonzero(~is_bnd)
    iA, iB, inner = fc0[interior], fc1[interior], fc0[bf_face]
    fc_other = np.where(is_bnd, fc0, fc1)            # inner cell at boundary
    fc_cfl = np.where(fc1 < nc, fc1, fc0)
    branch = np.asarray(boundary_face_branches(mesh_np, getattr(rt.sm, "_bc_source", None)),
                        dtype=int)[:n_bf]
    periodic = bf_cells != inner
    has_periodic = bool(periodic.any())
    # a periodic face's outer trace is its partner face's inner trace
    partner_face = bf_face.copy()
    if has_periodic:
        nrm = np.asarray(mesh_np.face_normals)
        by_cell = {}
        for k, c in enumerate(inner):
            by_cell.setdefault(int(c), []).append(k)
        for i in np.flatnonzero(periodic):
            cand = by_cell[int(bf_cells[i])]
            k = min(cand, key=lambda k: float(nrm[:, bf_face[k]] @ nrm[:, bf_face[i]]))
            partner_face[i] = bf_face[k]
    owner = getattr(nsm, "_symbolic_name", None) or "nsm"
    closure_cells = (topo.lsq_plan.get("closure") or {}).get("cells")
    recon_classes = topo.lsq_closure_class.get((owner, "reconstruction"), ())
    # reconstruction geometry (LSQMUSCLReconstructionJAX on the computed geometry)
    cc_g, fc_g = geom["cell_centers"][:dim], geom["face_centers"][:, :dim].T
    r_A = fc_g[:, interior] - cc_g[:, iA]
    r_B = fc_g[:, interior] - cc_g[:, iB]
    r_bnd = fc_g[:, bf_face] - cc_g[:, inner]
    eps_v2 = (geom["cell_volumes"][:nc] ** (1.0 / max(dim, 1))) ** 2

    # ── kernels on cells and faces ─────────────────────────────────────────
    def cells(k, Q, Qaux, p, time, dt, mesh):
        """Kernel `k` in every cell (with time, dt and the cell centre)."""
        return k(Q, Qaux, p, time=time, dt=dt, position=mesh.cell_centers[:, :Q.shape[-1]])

    def faces(k, QL, QR, AL, AR, p, mesh):
        """Face kernel `k` on every face with its unit normal."""
        return k(QL, QR, AL, AR, p, mesh.face_normals)

    def face_aux(Qaux, mesh):
        """Owner and neighbour aux per face (the inner cell twice at a boundary face)."""
        return Qaux[:, fc0], Qaux[:, fc_other]

    def face_wavespeed(k, Q, Qaux, p, mesh):
        """Max |eigenvalue| per face over the two adjacent cells."""
        n = mesh.face_normals
        evA = np.abs(k(Q[:, fc0], Qaux[:, fc0], p, n))
        evB = np.abs(k(Q[:, fc_cfl], Qaux[:, fc_cfl], p, n))
        if evA.ndim > 1:
            axes = tuple(range(evA.ndim - 1))
            evA, evB = np.max(evA, axis=axes), np.max(evB, axis=axes)
        return np.maximum(evA, evB)

    def face_inradius(mesh):
        """Per-face cell size: min inradius of the two adjacent cells."""
        ir = mesh.cell_inradius
        return np.minimum(ir[fc0], ir[fc_cfl])

    # ── boundary ───────────────────────────────────────────────────────────
    def _bc_at(bc, states, Qaux, p, time, mesh, cells_of_face, ghost=True):
        """The BC kernel `bc` on every boundary face: the ghost kernel at the mirror image of the
        cell centre, or (`ghost=False`) the face kernel at the face centre.
        """
        fc, n = mesh.face_centers, mesh.face_normals
        cc = mesh.cell_centers[:, bf_cells].T
        r = fc[bf_face, :] - cc
        nb = n[:, bf_face].T
        dn = np.sum(r * nb, axis=1)
        pos = cc + dn[:, None] * nb if ghost else fc[bf_face, :]
        dist = dn if ghost else np.linalg.norm(r, axis=1)
        return bc(branch, time, pos.T, dist, states, Qaux[:, cells_of_face], p, n[:, bf_face])

    def boundary_ghosts(ghost, Q, Qaux, p, time, mesh):
        """The GHOST (`boundary_ghost_state`) per boundary face."""
        return _bc_at(ghost, Q[:, bf_cells], Qaux, p, time, mesh, bf_cells)

    def boundary_states(ghost, Q, Qaux, p, time, mesh):
        """The GHOST per boundary face and the inner cell's aux."""
        if n_bf == 0:
            return np.zeros((Q.shape[0], 1)), np.zeros((Qaux.shape[0], 1))
        return _bc_at(ghost, Q[:, bf_cells], Qaux, p, time, mesh, bf_cells), Qaux[:, inner]

    def boundary_traces(bc, Q, QL, QR, Qaux, p, time, mesh):
        """`QR` with the outer FACE state `bc(QL)` on the boundary faces (periodic seam: the
        partner face's `QL`).
        """
        if n_bf == 0:
            return QR
        QRb = _bc_at(bc, QL[:, bf_face], Qaux, p, time, mesh, bf_cells, ghost=False)
        if has_periodic:
            QRb = np.where(periodic[None, :], QL[:, partner_face], QRb)
        QR = QR.copy()
        QR[:, bf_face] = QRb
        return QR

    # ── reconstruction and assembly ────────────────────────────────────────
    def pointwise(k, X, Xaux, p):
        """Kernel `k` in every column of `X`."""
        return k(X, Xaux, p)

    def _traces(WL, WR):
        """A boundary face's outer slot holds the inner trace until `boundary_traces`."""
        WR[:, bf_face] = WL[:, bf_face]
        return WL, WR

    def cell_traces(Q, mesh):
        """First-order traces: owner and neighbour cell value per face."""
        WL, WR = np.zeros((Q.shape[0], nf)), np.zeros((Q.shape[0], nf))
        WL[:, interior], WR[:, interior] = Q[:, iA], Q[:, iB]
        WL[:, bf_face] = Q[:, inner]
        return _traces(WL, WR)

    def _lsq(U, Ug, classes=None):
        """Scaled LSQ coefficients `(rows, cells, monomials)` of `U` (core `lsq_derivatives`),
        the boundary-face samples `Ug` (`None`: the cell value) and the closure `classes`.
        """
        return lsq_derivatives(topo, geom, np.asarray(U, dtype=float),
                               None if Ug is None else np.asarray(Ug, dtype=float),
                               classes if classes else None)

    def gradient(W, Wg, mesh):
        """LSQ gradient of every row of `W` per cell, closed at the boundary by constraints or by
        the ghost samples `Wg`.

        References:
            Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
        """
        D = _lsq(W, Wg, recon_classes)
        return np.transpose(D[:, :, topo.lsq_grad_cols], (0, 2, 1))

    def ghost_bounds(u, ug):
        """Min and max of `u` over the face neighbours and the boundary ghosts `ug`."""
        lo, hi = u.copy(), u.copy()
        for a, b in ((iA, iB), (iB, iA)):
            np.minimum.at(lo, a, u[b])
            np.maximum.at(hi, a, u[b])
        np.minimum.at(lo, inner, ug)
        np.maximum.at(hi, inner, ug)
        return lo, hi

    def _deltas(g):
        """Face increments `G . r` of one row: interior owner / neighbour and boundary."""
        dA, dB, db = np.zeros(len(iA)), np.zeros(len(iB)), np.zeros(len(inner))
        for d in range(dim):
            dA = dA + g[d, iA] * r_A[d]
            dB = dB + g[d, iB] * r_B[d]
            db = db + g[d, inner] * r_bnd[d]
        return dA, dB, db

    def _limit(kind, u, g, lo, hi):
        """Limiter factor of one row (`LSQMUSCLReconstructionJAX._limit_*`).

        References:
            Barth, Jespersen 1989, https://doi.org/10.2514/6.1989-366
            Venkatakrishnan 1995, https://doi.org/10.1006/jcph.1995.1084
            Sweby 1984, https://doi.org/10.1137/0721062
        """
        eps = 1e-30
        phi = np.ones(nc)
        for ids, de in zip((iA, iB, inner), _deltas(g)):
            uc = u[ids]
            dp, dm = hi[ids] - uc, lo[ids] - uc
            with np.errstate(divide="ignore", invalid="ignore"):
                if kind == "vk":
                    e2 = eps_v2[ids]
                    cp = np.clip((dp ** 2 + e2 + 2 * de * dp)
                                 / np.maximum(dp ** 2 + 2 * de ** 2 + de * dp + e2, eps), 0.0, 1.0)
                    cn = np.clip((dm ** 2 + e2 + 2 * de * dm)
                                 / np.maximum(dm ** 2 + 2 * de ** 2 + de * dm + e2, eps), 0.0, 1.0)
                else:                              # minmod and Barth-Jespersen: the same ratio
                    cp = np.clip(dp / np.maximum(de, eps), 0.0, 1.0)
                    cn = np.clip(dm / np.minimum(de, -eps), 0.0, 1.0)
            cand = np.where(de > eps, cp, np.where(de < -eps, cn, 1.0))
            np.minimum.at(phi, ids, cand)
        return np.clip(phi, 0.0, 1.0)

    def limiter(kind):
        """Limiter factory: factors in [0, 1] for the state `rows` (1 elsewhere), bounded by the
        face neighbours and the ghosts.
        """
        def factors(rows, W, Wg, G, mesh):
            """Limiter factors for the state `rows`."""
            phi = np.ones((W.shape[0], nc))
            for v in rows:
                lo, hi = ghost_bounds(W[v], Wg[v])
                phi[v] = _limit(kind, W[v], G[v], lo, hi)
            return phi
        return factors

    def extrapolate(W, G, phi, mesh):
        """Each cell's `W` moved to the face centres by `phi * G` (MUSCL).

        References:
            van Leer 1979, https://doi.org/10.1016/0021-9991(79)90145-1
        """
        phi = np.asarray(phi, dtype=float)
        phi = np.broadcast_to(phi if phi.ndim != 1 else phi[None, :], (W.shape[0], nc))
        WL, WR = np.zeros((W.shape[0], nf)), np.zeros((W.shape[0], nf))
        L, R, B = W[:, iA], W[:, iB], W[:, inner]
        for d in range(dim):
            L = L + phi[:, iA] * G[:, d, iA] * r_A[d][None, :]
            R = R + phi[:, iB] * G[:, d, iB] * r_B[d][None, :]
            B = B + phi[:, inner] * G[:, d, inner] * r_bnd[d][None, :]
        WL[:, interior], WR[:, interior], WL[:, bf_face] = L, R, B
        return _traces(WL, WR)

    nbr = np.where(is_bnd, nc, fc1)          # neighbour, or nc (dropped) at a boundary face

    def divergence(Wm, Wp, mesh):
        """Per cell the sum of `Wm` over the faces it owns and `Wp` over the faces it
        neighbours, times |face| / |cell|.

        References:
            LeVeque 2002, https://doi.org/10.1017/CBO9780511791253
        """
        fv, cv = mesh.face_volumes, mesh.cell_volumes
        out = np.zeros((Wm.shape[0], nc + 1))
        np.add.at(out, (slice(None), fc0), Wm * (fv / cv[fc0]))
        np.add.at(out, (slice(None), nbr), Wp * (fv / cv[np.minimum(nbr, nc - 1)]))
        return out[:, :nc]

    def edge_source(k, Q, QL, QR, Qaux, p, mesh):
        """Per cell the interior nonconservative edge integral of kernel `k` over its faces,
        divided by the volume (order >= 2).
        """
        fv, cv, n = mesh.face_volumes, mesh.cell_volumes, mesh.face_normals
        out = np.zeros((Q.shape[0], nc))
        for c, trace, sgn in ((iA, QL, 1.0), (iB, QR, -1.0)):
            s = k(Q[:, c], trace[:, interior], Qaux[:, c], Qaux[:, c], p, sgn * n[:, interior])
            np.add.at(out, (slice(None), c), s * fv[interior] / cv[c])
        if n_bf:
            s = k(Q[:, inner], QL[:, bf_face], Qaux[:, inner], Qaux[:, inner], p, n[:, bf_face])
            np.add.at(out, (slice(None), inner), s * fv[bf_face] / cv[inner])
        return out

    # ── whole-domain operations ────────────────────────────────────────────
    mis = [tuple(int(k) for k in mi) for mi in topo.lsq_monomial_multi_index]

    def derivative_aux(spec, Q, Qaux, mesh, Qg=None, rows=None):
        """`Qaux` with the derivative rows of `spec` refilled from mesh gradients; with `Qg` and
        `rows` only the derivatives of those state rows, their boundary faces from the ghosts.
        """
        out = np.array(Qaux, dtype=float, copy=True)
        cache = {}
        for e in spec:
            if e["kind"] != "derivative":
                continue
            col = mis.index(tuple(int(k) for k in e["multi_index"]))
            if Qg is None:
                if e["target_kind"] not in ("state", "function"):
                    continue
                field = Q[e["state_index"]] if e["target_kind"] == "state" \
                    else out[e["function_row"]]
                grad = _lsq(field[None, :nc], None)[0, :, col]
            else:
                s = e["state_index"]
                if e["target_kind"] != "state" or s not in rows:
                    continue
                if s not in cache:
                    cache[s] = _lsq(Q[s][None], Qg[s][None], (e["closure"],))[0]
                grad = cache[s][:, col]
            out[e["row"], :grad.shape[0]] = grad
        return out

    def clean_round_off(Q, rows, tol):
        """Entries of the state `rows` in `[-tol, 0)` set to 0."""
        Q = np.array(Q, dtype=float, copy=True)
        r = list(rows)
        X = Q[r]
        Q[r] = np.where((X < 0) & (X >= -tol), 0.0, X)
        return Q

    def put(Q, rows, values):
        """`Q` with the state `rows` replaced by `values`."""
        Q = np.array(Q, dtype=float, copy=True)
        Q[list(rows), :] = values
        return Q

    def zeros(n, mesh):
        """An `(n, cells)` array of zeros."""
        return np.zeros((int(n), nc))

    def inadmissible(k, Q, Qaux, p):
        """Cell mask: an admissibility row of kernel `k` below 0, or a non-finite state."""
        return np.any(k(Q, Qaux, p) < 0.0, axis=0) | ~np.isfinite(Q).all(axis=0)

    def select(mask, A, B):
        """`A` where `mask` else `B`, per cell or per face."""
        mask, A, B = np.asarray(mask), np.asarray(A), np.asarray(B)
        if mask.ndim == 1 and max(A.ndim, B.ndim) == 2:
            mask = mask[None, :]
        return np.where(mask, A, B)

    # ── control ────────────────────────────────────────────────────────────
    def while_loop(c, b, init):
        """A Python loop on the carry."""
        state = tuple(init)
        while bool(c(*state)):
            state = tuple(b(*state))
        return state

    def cond(pred, a, b):
        """Only the taken branch runs."""
        return a() if bool(pred) else b()

    imp = {**dict(tol=1e-8, maxiter=6, gmres_tol=1e-7, gmres_maxiter=30), **(implicit or {})}

    def _newton(f, x0, solve, norm=np.linalg.norm):
        """Newton on `f(x) = 0` from `x0`; `solve(x, r)` returns the update.  Warns when not
        converged after `maxiter` updates.
        """
        x, r, i = x0, f(x0), 0
        while i < imp["maxiter"] and norm(r) > imp["tol"]:
            x = x + solve(x, r)
            r, i = f(x), i + 1
        if not norm(r) <= imp["tol"]:
            logger.warning(f"numpy newton: not converged after {i} of {imp['maxiter']} "
                           f"updates: ||R||_2 = {float(norm(r)):.3e} > tol = {imp['tol']:.3e}")
        return x

    def root(f, x0):
        """Newton on `f(x) = 0` with a per-cell (block-diagonal) Jacobian by central
        differences (step `eps^(1/3)` relative to the cell value).
        """
        x0 = np.asarray(x0, dtype=float)
        n = x0.shape[0]

        def cellwise(x, r):
            """The update: per cell, the `n x n` difference Jacobian solved against `-r`."""
            J = np.empty((x.shape[1], n, n))
            for k in range(n):
                h = 6e-6 * np.maximum(1.0, np.abs(x[k]))
                xp, xm = x.copy(), x.copy()
                xp[k] += h
                xm[k] -= h
                J[:, :, k] = ((f(xp) - f(xm)) / (2.0 * h)).T
            return np.linalg.solve(J, -r.T[:, :, None])[:, :, 0].T
        return _newton(f, x0, cellwise)

    ls = imp.get("linsolve") or dict(tol=imp["gmres_tol"], restart=20,
                                     maxiter=imp["gmres_maxiter"], accept=None)
    pyamg = None
    if ls.get("backend") == "pyamg":
        import pyamg
    patterns, amg_cache = {}, {}
    lsq_nbrs = np.asarray(mesh_np.lsq_neighbors)

    def _gmres(mv, b):
        """Restarted GMRES of the flat system `mv(x) = b`.

        References:
            Saad, Schultz 1986, https://doi.org/10.1137/0907058
        """
        from scipy.sparse.linalg import LinearOperator, gmres
        A = LinearOperator((b.size, b.size), matvec=mv, dtype=float)
        return gmres(A, b, rtol=ls["tol"], atol=0.0, restart=ls["restart"],
                     maxiter=ls["maxiter"])[0]

    def linsolve(A, b):
        """Solve `A x = b` for the affine map `A` (its linear part `A(x) - A(0)`): GMRES, or
        GMRES preconditioned by smoothed-aggregation AMG on the colour-assembled matrix
        (`backend='pyamg'`).  Raises above the accept gate.

        References:
            Saad, Schultz 1986, https://doi.org/10.1137/0907058
            Bell, Olson, Schroder 2022, https://doi.org/10.21105/joss.04142
        """
        b = np.asarray(b, dtype=float)
        shape = b.shape
        a0 = np.asarray(A(np.zeros(shape)), dtype=float)

        def mv(x):
            """The linear part at `x`, evaluated at `x / max|x|` (a tiny iterate added to the
            state would lose its digits)."""
            s = np.abs(x).max()
            if s == 0.0:
                return np.zeros_like(x)
            return s * (np.asarray(A((x / s).reshape(shape)), dtype=float) - a0).reshape(-1)
        bf = b.reshape(-1)
        if pyamg is not None:          # cell-major unknowns, one k x k block per cell
            n, k = bf.size, shape[0]
            to_cell = lambda x: x.reshape(k, -1).T.reshape(-1)       # noqa: E731
            to_row = lambda x: x.reshape(-1, k).T.reshape(-1)        # noqa: E731
            if (n, nc) not in patterns:
                patterns[(n, nc)] = cell_pattern(n, nc, lsq_nbrs)
            rows, cols, colors = patterns[(n, nc)]
            probes = np.arange(int(colors.max()) + 1)[:, None] == colors[None, :]
            AV = np.stack([to_cell(mv(to_row(pr.astype(float)))) for pr in probes])
            vals = AV[colors[cols], rows]
            x = to_row(host_amg_solve(vals, to_cell(bf), rows, cols, n, k, ls, pyamg,
                                      amg_cache.setdefault(n, {})))
        else:
            x = _gmres(mv, bf)
        if ls.get("accept") is not None:
            nb = np.linalg.norm(bf)
            r = np.linalg.norm(mv(x) - bf) / (nb if nb > 0 else 1.0)
            if r > ls["accept"]:
                raise RuntimeError(
                    f"linsolve: relative residual {r:.3e} > accept gate {ls['accept']:.1e} "
                    f"(target {ls['tol']:.1e}, restart {ls['restart']} x maxiter "
                    f"{ls['maxiter']})")
        return x.reshape(shape)

    flux_bf = np.arange(n_bf)
    # the two-point pattern: face neighbours and periodic partners (-1: none)
    pairs = np.concatenate([np.stack([iA, iB]), np.stack([iB, iA]),
                            np.stack([inner[periodic], bf_cells[periodic]])], axis=1)
    pairs = np.unique(pairs[:, pairs[0] != pairs[1]], axis=1)
    deg = np.bincount(pairs[0], minlength=nc)
    face_nbrs = -np.ones((nc, max(int(deg.max(initial=0)), 1)), dtype=int)
    face_nbrs[pairs[0], np.arange(pairs.shape[1]) - np.repeat(np.cumsum(deg) - deg, deg)] = pairs[1]
    two_point, jacobians = {}, {}
    outer_face = np.where(periodic, partner_face, bf_face)
    tpfa = []

    def _operator():
        """The TPFA operator (core `DenseDiffusionOperator`) and the geometry of its interior
        faces: owner, neighbour, normal, unit cell-to-cell vector, distance, floored `|n.e|`,
        `|f| / |V|` of both.
        """
        from zoomy_core.solver.constants import TPFA_COS_FLOOR
        if not tpfa:
            op = DenseDiffusionOperator(mesh_np, dim, rt.n_state)
            cc = np.asarray(mesh_np.cell_centers)[:dim]
            f = np.flatnonzero((fc0 < nc) & (fc1 < nc))
            dx = cc[:, fc1[f]] - cc[:, fc0[f]]
            dist = np.sqrt((dx ** 2).sum(axis=0))
            keep = dist >= 1e-30
            f, dx, dist = f[keep], dx[:, keep], dist[keep]
            e = dx / dist
            n = np.asarray(mesh_np.face_normals)[:dim, f]
            cosf = np.maximum(np.abs((n * e).sum(axis=0)), TPFA_COS_FLOOR)
            fv, cv = np.asarray(mesh_np.face_volumes), np.asarray(mesh_np.cell_volumes)
            tpfa.append((op, dict(ia=fc0[f], ib=fc1[f], n=n, e=e, dist=dist, cosf=cosf,
                                  wa=fv[f] / cv[fc0[f]], wb=fv[f] / cv[fc1[f]])))
        return tpfa[0]

    def _div(k, bc, X, Qaux, p, time, mesh, correction=True):
        """`(D, C)` per cell of `div(A(X) : grad X)`: `D` the two-point fluxes (interior faces,
        to the BC ghost or the periodic partner on boundary faces), `C` the rest of the flux
        `n.A.grad_f` (zero without `correction` or in 1-D), as `zoomy_jax`'s `_div`.

        References:
            Eymard, Gallouet, Herbin 2000, https://doi.org/10.1016/s1570-8659(00)07005-8
        """
        op, geo = _operator()
        A = np.transpose(np.asarray(k(X, Qaux, p)), (0, 1, 4, 2, 3))   # (i, j, nc, d, d)
        D = op._divergence(X, A, bf_grads=None)
        C = np.zeros_like(D)
        correction = correction and dim >= 2
        if not len(flux_bf) and not correction:
            return D, C
        Xg = _bc_at(bc, X[:, bf_cells], Qaux, p, time, mesh, bf_cells) if n_bf else None
        if correction:
            G = np.transpose(_lsq(X, Xg)[:, :, topo.lsq_grad_cols], (0, 2, 1))  # (j, d, nc)
            ia, ib, nI, e = (geo[q] for q in ("ia", "ib", "n", "e"))
            m = np.einsum("ijfde,df->ijfe", 0.5 * (A[:, :, ia] + A[:, :, ib]), nI)
            g = 0.5 * (G[:, :, ia] + G[:, :, ib])
            t = g - np.einsum("jef,ef->jf", g, e)[:, None, :] * e[None]
            s = np.einsum("ijfe,ef->ijf", m, e) - np.einsum("ijfe,ef->ijf", m, nI) / geo["cosf"]
            F = (np.einsum("ijfe,jef->if", m, t)
                 + np.einsum("ijf,jf->if", s, (X[:, ib] - X[:, ia]) / geo["dist"]))
            np.add.at(C, (slice(None), ia), F * geo["wa"])
            np.add.at(C, (slice(None), ib), -F * geo["wb"])
        if not len(flux_bf):
            return D, C
        f, c, o = bf_face[flux_bf], inner[flux_bf], bf_cells[flux_bf]
        Xout = Xg[:, flux_bf]
        fc, cc, nrm = mesh.face_centers, mesh.cell_centers, mesh.face_normals

        def dn(faces_, cells_):
            """Normal distance from the cell centres `cells_` to the faces `faces_`."""
            return np.sum((fc[faces_, :] - cc[:, cells_].T) * nrm[:, faces_].T, axis=1)
        d = dn(f, c) + dn(outer_face[flux_bf], o)
        n = nrm[:dim, f]
        Af = 0.5 * (A[:, :, c] + A[:, :, o])
        T = np.einsum("ijbde,db,eb->ijb", Af, n, n)
        w = mesh.face_volumes[f] / (mesh.cell_volumes[c] * d)
        np.add.at(D, (slice(None), c), np.einsum("ijb,jb->ib", T, Xout - X[:, c]) * w)
        if correction:
            g = 0.5 * (G[:, :, c] + G[:, :, o])
            t = g - np.einsum("jeb,eb->jb", g, n)[:, None, :] * n[None]
            F = np.einsum("ijbde,db,jeb->ib", Af, n, t)
            np.add.at(C, (slice(None), c), F * mesh.face_volumes[f] / mesh.cell_volumes[c])
        return D, C

    def diffusion(k, bc, Q, Qaux, p, time, mesh):
        """Explicit `div(A : grad Q)` per cell (RHS sign, A > 0 diffuses)."""
        D, C = _div(k, bc, Q, Qaux, p, time, mesh)
        return D + C

    def implicit_diffusion(k, bc, Q, Qaux, p, time, dt, mesh):
        """One Crank-Nicolson step of the two-point part of `div(A(X) : grad X)` from `time` to
        `time + dt`, A at the new state, by Newton on a coloured finite-difference Jacobian
        (sparse LU); the correction explicit at `Q`.

        References:
            Knoll, Keyes 2004, https://doi.org/10.1016/j.jcp.2003.08.010
            Crank, Nicolson 1947, https://doi.org/10.1017/s0305004100023197
        """
        D, C = _div(k, bc, Q, Qaux, p, time, mesh)
        b = Q + 0.5 * dt * (D + 2.0 * C)

        def R(X):
            """Crank-Nicolson residual `X - b - dt/2 D(X)` at `time + dt`."""
            return X - b - 0.5 * dt * _div(k, bc, X, Qaux, p, time + dt, mesh,
                                           correction=False)[0]

        def jacobian(x):
            """The sparse Jacobian of `R` at `x` (cell-major unknowns) by central differences,
            one pair of evaluations per colour of the face-neighbour pattern."""
            import scipy.sparse as sps
            k_, n = x.shape[0], x.size
            if n not in two_point:
                two_point[n] = cell_pattern(n, nc, face_nbrs)
            rows, cols, colors = two_point[n]
            xc = x.T.reshape(-1)
            hx = 6e-6 * np.maximum(1.0, np.abs(xc))
            vals = np.empty(len(rows))
            for c in range(int(colors.max()) + 1):
                e = np.where(colors == c, hx, 0.0)
                dR = (R((xc + e).reshape(-1, k_).T) - R((xc - e).reshape(-1, k_).T)).T.reshape(-1)
                pick = colors[cols] == c
                vals[pick] = dR[rows[pick]] / (2.0 * hx[cols[pick]])
            return sps.csc_matrix((vals, (rows, cols)), shape=(n, n))

        last = [None]
        lu = jacobians.setdefault((Q.size, k), {})

        def update(x, r):
            """The Newton update against a factorized Jacobian, kept between the steps (lagged):
            refactorized at a new `dt`, or when an update reduced the residual less than 10x."""
            from scipy.sparse.linalg import splu
            nr = np.linalg.norm(r)
            if (lu.get("dt") != dt or "lu" not in lu
                    or (last[0] is not None and nr > 0.1 * last[0])):
                lu.update(lu=splu(jacobian(x)), dt=dt)
            last[0] = nr
            return lu["lu"].solve(-r.T.reshape(-1)).reshape(-1, x.shape[0]).T
        return _newton(R, np.asarray(Q, dtype=float), update)

    # ── effects ────────────────────────────────────────────────────────────
    warned = []

    def check_cfl(tok, dt_cfl, dt):
        """Warn once when `dt` exceeds the CFL step `dt_cfl`; returns the next order token."""
        if float(dt) > float(dt_cfl) * (1 + 1e-12) and not warned:
            warned.append(True)
            logger.warning(f"time step {float(dt):.3e} exceeds the CFL step "
                           f"{float(dt_cfl):.3e} (warned once)")
        return (tok + 1,)

    def io_write(tok, time, Q, Qaux):
        """Write a snapshot through `on_write`; returns the next order token."""
        if on_write is not None:
            on_write(time, Q, Qaux)
        return (tok + 1,)

    def io_save_state(tok, time, Q, Qaux):
        """Save the march state through `on_save`; the lagged solver caches (AMG hierarchy,
        Jacobian LU) restart here, as in a run restarted from this state."""
        if on_save is not None:
            on_save(time, Q, Qaux)
            amg_cache.clear()
            jacobians.clear()
        return (tok + 1,)

    def io_load_state(tok, time, Q, Qaux):
        """The march state from `on_load` (a restart), else the arguments."""
        if on_load is not None:
            time, Q, Qaux = on_load(time, Q, Qaux)
        return (tok + 1, time, Q, Qaux)

    def with_closure(name, sub):
        """The derivative-aux spec of system `name`, each state entry tagged with its
        boundary-closure class.
        """
        classes = topo.lsq_closure_class.get((name, "state"), ())
        return [dict(e, closure=(classes[e["state_index"]]
                                 if e["target_kind"] == "state" and classes else -1))
                for e in sub.aux_derivatives]

    rt.aux_derivatives = with_closure(owner, nsm)
    named = {owner: rt}
    for name, (sub, sub_rt) in (systems or {}).items():
        sub_rt.aux_derivatives = with_closure(name, sub)
        named[name] = sub_rt
    m = dict(
        **named, cells=cells, faces=faces, face_aux=face_aux,
        face_wavespeed=face_wavespeed, face_inradius=face_inradius,
        boundary_states=boundary_states, boundary_traces=boundary_traces,
        boundary_ghosts=boundary_ghosts, pointwise=pointwise, cell_traces=cell_traces,
        gradient=gradient, minmod=limiter("minmod"), venkatakrishnan=limiter("vk"),
        barth_jespersen=limiter("bj"), extrapolate=extrapolate, divergence=divergence,
        edge_source=edge_source, derivative_aux=derivative_aux,
        inadmissible=inadmissible, reduce_min=np.min, reduce_any=np.any,
        halo=lambda X, mesh: X, put=put, zeros=zeros, clean_round_off=clean_round_off,
        take=lambda Q, rows: np.asarray(Q)[list(rows)], reduce_max=np.max,
        logical_and=np.logical_and, select=select, while_loop=while_loop, cond=cond,
        io_write=io_write, io_save_state=io_save_state, io_load_state=io_load_state,
        root=root, linsolve=linsolve,
        implicit_diffusion=implicit_diffusion, diffusion=diffusion, check_cfl=check_cfl,
        minimum=np.minimum, maximum=np.maximum, abs=np.abs, floor=np.floor)
    return m


def build(src, name, module):
    """Execute printed source against the name table and return the function `name`; raises if
    the source uses a block the table lacks.
    """
    missing = [n for n in contract() if n not in module and n in src]
    if missing:
        raise NotImplementedError(f"numpy runtime lacks blocks {missing}")
    ns = dict(module)
    exec(compile(src, f"<symsolver:{name}>", "exec"), ns)
    return ns[name]
