"""JAX runtime for the symbolic solvers: what each printed name means on JAX.

The printed solver is plain Python calling names from this table.  ``nsm`` is
the JaxRuntime (the NSM's kernels, vmapped); every other entry is a block of
``zoomy_core.symsolver`` implemented with the existing JAX mesh code.
"""
from __future__ import annotations

from typing import NamedTuple

import copy
import os
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.sparse.linalg import gmres as jax_gmres

from zoomy_core.misc.logger_config import logger
from zoomy_core.mesh.geometry_xp import compute_geometry
from zoomy_core.model.boundary_conditions import boundary_face_branches
from zoomy_core.symsolver.core import contract
from zoomy_core.fvm.solver_numpy import HyperbolicSolver as HyperbolicSolverNumpy
from zoomy_jax.fvm.implicit_ad import cellwise_solve, newton_root
from zoomy_jax.mesh.mesh import lsq_gradient_per_field
from zoomy_jax.fvm.reconstruction_jax import (
    ConstantReconstruction, DenseDiffusionOperatorJAX,
    LSQMUSCLReconstructionJAX)
from zoomy_core.mesh.distributed import HALO
from zoomy_jax.fvm.solver_jax import (
    HyperbolicSolver as _Reference)

class Geometry(NamedTuple):
    """The ``mesh`` argument of a printed solver: all geometry, computed from
    the vertex coordinates by ``geometry_xp`` INSIDE the traced function, so
    ``jax.grad`` reaches the vertices.  Topology (indices, stencils) is static
    and lives in the runtime's closures."""
    cell_volumes: jnp.ndarray
    cell_centers: jnp.ndarray
    cell_inradius: jnp.ndarray
    face_normals: jnp.ndarray
    face_volumes: jnp.ndarray
    face_centers: jnp.ndarray
    lsq_gradQ: jnp.ndarray


def geometry_of(topo, X):
    """All solver geometry from vertex coordinates ``X`` ``(dim, n_vertices)``."""
    g = compute_geometry(topo, X)
    return Geometry(*(g[f] for f in Geometry._fields))


def _with_geometry(R, g, mesh_np, nc):
    """A copy of reconstruction ``R`` that reads the traced geometry ``g``
    instead of the arrays it precomputed from the static mesh."""
    if not hasattr(R, "_lsq_gradQ"):              # first order: no geometry
        return R
    R = copy.copy(R)
    dim = R.dim
    cc, fc = g.cell_centers[:dim], g.face_centers[:, :dim].T
    R._r_Af_int = fc[:, R._interior_faces] - cc[:, R._iA_int]
    R._r_Bf_int = fc[:, R._interior_faces] - cc[:, R._iB_int]
    R._r_Af_bnd = fc[:, R._boundary_faces] - cc[:, R._iInner_bnd]
    R._lsq_gradQ = g.lsq_gradQ[:nc]
    R._eps_v2 = (g.cell_volumes[:nc] ** (1.0 / max(dim, 1))) ** 2
    if hasattr(R, "_r_cf"):
        cf = np.asarray(mesh_np.cell_faces)[:, :nc]
        R._r_cf = fc[:, cf] - cc[:, np.newaxis, :nc]
    return R


def runtime(nsm, rt, mesh_np, mesh_j, *, ad_mode="forward", checkpoints=64, on_write=None, couple=None,
            implicit=None, systems=None, partition=None):
    """The name table the printed solver runs against.  ``systems``: further
    ``{name: (nsm, JaxRuntime)}`` a template uses besides ``nsm``.
    ``partition``: this rank's piece of a distributed run
    (:mod:`zoomy_jax.symsolver.parallel`); its ``halo`` and global reductions
    replace the serial ones."""
    sm = rt.sm
    nc = int(mesh_np.n_inner_cells)
    nf = int(mesh_np.n_faces)
    n_bf = int(mesh_np.n_boundary_faces)
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
    # faces a partition cut: the inner state, no BC (nothing owned reads it)
    names = list(mesh_np.boundary_conditions_sorted_names)
    numbers = np.asarray(mesh_np.boundary_face_function_numbers, dtype=int)[:n_bf]
    cut = (numbers == names.index(HALO)) if HALO in names else np.zeros(n_bf, bool)
    phys = np.flatnonzero(~cut)
    branch = jnp.asarray(np.asarray(boundary_face_branches(
        mesh_np, getattr(sm, "_bc_source", None)))[phys], dtype=jnp.int32)
    periodic = bf_cells != inner
    has_periodic = bool(periodic.any())
    j = jnp.asarray
    dim = int(mesh_j.dimension)
    order1 = ConstantReconstruction(mesh_j, dim)
    lsq = (LSQMUSCLReconstructionJAX(mesh_j, dim, limiter="none")
           if nsm.reconstruction.order >= 2 else None)
    if (getattr(rt, "source_explicit", None) is None
            and sm.source_explicit is not None
            and 0 not in tuple(sm.source_explicit.shape)):
        rt.source_explicit = rt._build_slot_operator(
            "source_explicit", sm.source_explicit, squeeze_trailing=True)

    _face_kernel(rt, "numerical_face_sides")
    _face_kernel(rt, "numerical_face_sides_speed")

    # ── kernels on cells and faces ─────────────────────────────────────────
    def cells(k, Q, Qaux, p, time, dt, mesh):
        return k(Q, Qaux, p, time=time, dt=dt,
                 position=mesh.cell_centers[:, :Q.shape[-1]])

    def faces(k, QL, QR, AL, AR, p, mesh):
        return k(QL, QR, AL, AR, p, mesh.face_normals)

    def face_aux(Qaux, mesh):
        return Qaux[:, j(fc0)], Qaux[:, j(fc_other)]

    def face_wavespeed(k, Q, Qaux, p, mesh):
        n = mesh.face_normals
        evA = jnp.abs(k(Q[:, j(fc0)], Qaux[:, j(fc0)], p, n))
        evB = jnp.abs(k(Q[:, j(fc_cfl)], Qaux[:, j(fc_cfl)], p, n))
        if evA.ndim > 1:
            axes = tuple(range(evA.ndim - 1))
            evA, evB = jnp.max(evA, axis=axes), jnp.max(evB, axis=axes)
        return jnp.maximum(evA, evB)

    def face_inradius(mesh):
        ir = mesh.cell_inradius
        return jnp.minimum(ir[j(fc0)], ir[j(fc_cfl)])

    # ── boundary ───────────────────────────────────────────────────────────
    def _bc_at(bc, states, Qaux, p, time, mesh, cells_of_face):
        fc, n = mesh.face_centers, mesh.face_normals
        d_face = jnp.linalg.norm(fc[j(bf_face), :]
                                 - mesh.cell_centers[:, j(bf_cells)].T, axis=1)

        def per_bf(k, i):
            f = j(bf_face)[i]
            return bc(branch[k], time, fc[f, :], d_face[i], states[:, i],
                      Qaux[:, j(cells_of_face)[i]], p, n[:, f])
        if not cut.any():
            return jax.vmap(per_bf)(jnp.arange(n_bf), jnp.arange(n_bf)).T
        if not len(phys):                   # a piece inside the domain
            return states
        out = jax.vmap(per_bf)(jnp.arange(len(phys)), j(phys)).T
        return states.at[:, j(phys)].set(out)

    def boundary_ghosts(bc, Q, Qaux, p, time, mesh):
        return _bc_at(bc, Q[:, j(bf_cells)], Qaux, p, time, mesh, bf_cells)

    def boundary_states(bc, Q, Qaux, p, time, mesh):
        if n_bf == 0:
            return (jnp.zeros((Q.shape[0], 1)), jnp.zeros((Qaux.shape[0], 1)))
        return (_bc_at(bc, Q[:, j(bf_cells)], Qaux, p, time, mesh, bf_cells),
                Qaux[:, j(inner)])

    def boundary_traces(bc, Q, QL, QR, Qaux, p, time, mesh):
        if n_bf == 0:
            return QR
        QRb = _bc_at(bc, QL[:, j(bf_face)], Qaux, p, time, mesh, bf_cells)
        if has_periodic:
            QRb = jnp.where(j(periodic)[None, :], Q[:, j(bf_cells)], QRb)
        return QR.at[:, j(bf_face)].set(QRb)

    # ── reconstruction and assembly ────────────────────────────────────────
    def pointwise(k, X, Xaux, p):
        return k(X, Xaux, p)

    def cell_traces(Q, Qb, mesh):
        return order1(Q, Qb)

    def gradient(W, Wb, mesh):
        R = _with_geometry(lsq, mesh, mesh_np, nc)
        return R._compute_gradients(W, W.shape[0], Wb)

    def limiter(name):
        def factors(rows, W, Wb, G, mesh):
            R = _with_geometry(lsq, mesh, mesh_np, nc)
            fn = getattr(R, name)
            phi = jnp.ones((W.shape[0], nc), dtype=W.dtype)
            for v in rows:
                lo, hi = R._neighbor_bounds(W[v], Wb[v])
                phi = phi.at[v].set(fn(W[v], G[v], lo, hi))
            return phi
        return factors

    def extrapolate(W, Wb, G, phi, mesh):
        R = _with_geometry(lsq, mesh, mesh_np, nc)
        phi = jnp.asarray(phi, dtype=W.dtype)
        phi = jnp.broadcast_to(phi if phi.ndim != 1 else phi[None, :],
                               (W.shape[0], nc))
        return R._reconstruct(W, G, phi, Wb)

    # owner of every face; neighbour, or nc (dropped) on a boundary face
    nbr = np.where(is_bnd, nc, fc1)

    def divergence(Wm, Wp, mesh):
        fv, cv = mesh.face_volumes, mesh.cell_volumes
        out = jnp.zeros((Wm.shape[0], nc), dtype=Wm.dtype)
        out = out.at[:, j(fc0)].add(Wm * (fv / cv[j(fc0)]))
        return out.at[:, j(nbr)].add(Wp * (fv / cv[j(np.minimum(nbr, nc - 1))]),
                                     mode="drop")

    def edge_source(k, Q, QL, QR, Qaux, p, mesh):
        fv, cv, n = mesh.face_volumes, mesh.cell_volumes, mesh.face_normals
        out = jnp.zeros((Q.shape[0], nc), dtype=Q.dtype)
        I, B = j(interior), j(bf_face)
        for cell, trace, sgn in ((iA, QL, 1.0), (iB, QR, -1.0)):
            c = j(cell)
            s = k(Q[:, c], trace[:, I], Qaux[:, c], Qaux[:, c], p, sgn * n[:, I])
            out = out.at[:, c].add(s * fv[I] / cv[c])
        if n_bf:
            c = j(inner)
            s = k(Q[:, c], QL[:, B], Qaux[:, c], Qaux[:, c], p, n[:, B])
            out = out.at[:, c].add(s * fv[B] / cv[c])
        return out

    # ── whole-domain operations ────────────────────────────────────────────
    def derivative_aux(spec, Q, Qaux, mesh, Qb=None, rows=None):
        # stencils are static topology; the LSQ matrices are traced geometry
        traced = dataclasses.replace(mesh_j, lsq_gradQ=mesh.lsq_gradQ)
        if Qb is None:
            return _Reference._walk_derivative_aux(None, Qaux, Q, traced,
                                                   registry=spec)
        for e in spec:            # boundary value -> ghost 2 Qb - Q (face)
            if e["kind"] != "derivative" or e["target_kind"] != "state":
                continue
            s = e["state_index"]
            if s not in rows:
                continue
            ghost = 2.0 * Qb[s] - Q[s, j(bf_cells)]
            grad = lsq_gradient_per_field(traced, Q[s], u_bf=ghost,
                                          multi_index=e["multi_index"])
            Qaux = Qaux.at[e["row"], :grad.shape[0]].set(grad)
        return Qaux

    def clean_round_off(Q, rows, tol):
        r = jnp.asarray(rows)
        X = Q[r]
        return Q.at[r].set(jnp.where((X < 0) & (X >= -tol), 0.0, X))

    def put(Q, rows, values):
        return Q.at[jnp.asarray(rows), :].set(values)

    def zeros(n, mesh):
        return jnp.zeros((int(n), nc), dtype=mesh.cell_volumes.dtype)

    def inadmissible(k, Q, Qaux, p):
        adm = k(Q, Qaux, p)
        return (jnp.any(adm < 0.0, axis=0)
                | ~jnp.isfinite(Q).all(axis=0))

    def select(mask, A, B):
        mask, A, B = jnp.asarray(mask), jnp.asarray(A), jnp.asarray(B)
        if mask.ndim == 1 and max(A.ndim, B.ndim) == 2:
            mask = mask[None, :]
        return jnp.where(mask, A, B)

    # ── control ────────────────────────────────────────────────────────────
    def while_loop(c, b, init):
        init = tuple(jnp.asarray(a, dtype=jnp.asarray(a).dtype) for a in init)
        cf, bf = (lambda s: c(*s)), (lambda s: tuple(b(*s)))
        if ad_mode == "reverse":
            import equinox.internal as eqxi
            return eqxi.while_loop(cf, bf, init, kind="checkpointed",
                                   checkpoints=checkpoints)
        return lax.while_loop(cf, bf, init)

    def cond(p, a, b):
        return lax.cond(p, a, b)

    imp = {**dict(tol=1e-8, maxiter=6, gmres_tol=1e-7, gmres_maxiter=30),
           **(implicit or {})}

    def root(f, x0):
        """Newton on f(x) = 0 with a per-cell (block-diagonal) Jacobian;
        differentiable through the converged root (custom_root)."""
        return newton_root(f, x0, lambda x, v: jax.jvp(f, (x,), (v,))[1],
                           cellwise_solve, tol=imp["tol"], maxiter=imp["maxiter"])

    ls = imp.get("linsolve") or dict(tol=imp["gmres_tol"], restart=20,
                                     maxiter=imp["gmres_maxiter"], accept=None)

    def _ls_gmres(A, b):
        return jax_gmres(A, b, atol=0.0, tol=ls["tol"], restart=ls["restart"],
                         maxiter=ls["maxiter"])[0]

    def _ls_reject(r):
        if float(r) > ls["accept"]:
            raise RuntimeError(
                f"linsolve: relative residual {float(r):.3e} > accept gate "
                f"{ls['accept']:.1e} (target {ls['tol']:.1e}, restart "
                f"{ls['restart']} x maxiter {ls['maxiter']})")

    amgx = _jaxamg() if ls.get("backend") == "amgx" else None
    pyamg = _pyamg() if ls.get("backend") == "pyamg" else None
    amgx_colorings, patterns = {}, {}
    lsq_nbrs = (np.asarray(mesh_np.lsq_neighbors)
                if amgx is not None or pyamg is not None else None)

    def assembled(op, n, ncells=nc):
        """The operator's values on the cell pattern of the first ``ncells``
        cells, by colouring: one evaluation per colour (traced; the pattern
        is static)."""
        if (n, ncells) not in patterns:
            rows, cols = _cell_pattern(n, ncells, lsq_nbrs)
            colors = _greedy_colors(rows, cols, n)
            patterns[(n, ncells)] = (rows, cols, colors)
        rows, cols, colors = patterns[(n, ncells)]
        probes = np.arange(int(colors.max()) + 1)[:, None] == colors[None, :]
        AV = jax.vmap(op)(jnp.asarray(probes, dtype=jnp.float64))  # one per colour
        return AV[colors[cols], rows]

    amg_cache = {}                   # (n, transpose) -> hierarchy and its age

    def host_amg(rows, cols, n, k, transpose):
        """Host solve: GMRES preconditioned by a smoothed-aggregation AMG
        hierarchy of the block-diagonally scaled matrix.  The hierarchy is
        reused (a lagged preconditioner for the current matrix) and rebuilt
        every ``amg_rebuild`` solves, or as soon as the iterations exceed
        ``amg_growth`` x those right after the last rebuild."""
        import scipy.sparse as sps
        from scipy.sparse.linalg import gmres as sp_gmres
        rebuild, growth = int(ls.get("amg_rebuild", 1)), float(ls.get("amg_growth", 2.0))

        def solve(vals, rhs):
            A = sps.csr_matrix((np.asarray(vals), (rows, cols)), shape=(n, n))
            A = (A.T if transpose else A).tobsr(blocksize=(k, k))
            # left-scale by the inverse cell blocks: A ~ M (x) L with a k x k
            # coupling M (indefinite for VAM) becomes ~ I (x) L, AMG-friendly
            A.sort_indices()
            row = np.repeat(np.arange(n // k), np.diff(A.indptr))
            Dinv = sps.block_diag(np.linalg.inv(A.data[A.indices == row]), format="bsr")
            As, bs = (Dinv @ A).tobsr(blocksize=(k, k)), Dinv @ np.asarray(rhs)
            c = amg_cache.setdefault((n, transpose), {"ml": None, "age": 0, "it0": None, "it": 0})
            if (c["ml"] is None or c["age"] >= rebuild
                    or (c["it0"] is not None and c["it"] > growth * max(c["it0"], 1))):
                c.update(ml=pyamg.smoothed_aggregation_solver(As), age=0, it0=None)
            its = [0]
            x, _ = sp_gmres(As, bs, M=c["ml"].aspreconditioner(), rtol=ls["tol"],
                            atol=0.0, restart=ls["restart"], maxiter=ls["maxiter"],
                            callback=lambda _: its.__setitem__(0, its[0] + 1),
                            callback_type="pr_norm")
            c["age"], c["it"] = c["age"] + 1, its[0]
            if c["it0"] is None:
                c["it0"] = its[0]
            return np.asarray(x, dtype=np.asarray(rhs).dtype)
        return solve

    def host_vcycle(rows, cols, n, k, transpose):
        """Host preconditioner: one smoothed-aggregation V-cycle on the
        block-diagonally scaled matrix (a lagged hierarchy, rebuilt once
        ``amg_rebuild`` new matrices have been seen)."""
        import scipy.sparse as sps
        c = amg_cache.setdefault(("vcycle", n, transpose),
                                 {"vals": None, "age": 0, "P": None, "Dinv": None})
        rebuild = int(ls.get("amg_rebuild", 1))

        def apply(vals, rhs):
            vals = np.asarray(vals)
            if c["vals"] is None or not np.array_equal(vals, c["vals"]):
                c["vals"], c["age"] = vals.copy(), c["age"] + 1
            if c["P"] is None or c["age"] >= rebuild:
                A = sps.csr_matrix((vals, (rows, cols)), shape=(n, n))
                A = (A.T if transpose else A).tobsr(blocksize=(k, k))
                A.sort_indices()
                row = np.repeat(np.arange(n // k), np.diff(A.indptr))
                Dinv = sps.block_diag(np.linalg.inv(A.data[A.indices == row]),
                                      format="bsr")
                ml = pyamg.smoothed_aggregation_solver((Dinv @ A).tobsr(blocksize=(k, k)))
                c.update(P=ml.aspreconditioner(cycle="V"), Dinv=Dinv, age=0)
            return np.asarray(c["P"](c["Dinv"] @ np.asarray(rhs)), dtype=np.asarray(rhs).dtype)
        return apply

    if partition is not None:
        from zoomy_jax.symsolver.parallel import krylov, no_halo
        own_cell, gsum = krylov(partition, nc)
        n_own = int(partition.n_own)
        amgx_comm = partition.comm.Clone() if amgx is not None else None

    def _amgx_mpi(A, b):
        """Distributed 'amgx': ONE global AmgX solve across the ranks (AmgX's
        MPI mode through JAX-AMG; the serial solver's FGMRES + AMG, its own
        distributed transpose for AD).  Each rank hands its owned rows, the
        columns numbered globally (halo cells by their owners' ids), the
        values by colouring the operator with the halo off."""
        from jax.experimental import sparse as jsparse
        shape, k = b.shape, b.shape[0]
        with no_halo():
            _, lin_o = jax.linearize(A, jnp.zeros_like(b))
        to_cell = lambda X: X.T.reshape(-1)                         # noqa: E731
        to_row = lambda x: x.reshape(-1, k).T                       # noqa: E731
        n_l, key = k * nc, ("mpi", k)
        if key not in patterns:
            rows, cols = _cell_pattern(n_l, nc, lsq_nbrs)
            keep = rows < k * n_own
            rows, cols = rows[keep], cols[keep]
            colors = _greedy_colors(rows, cols, n_l)
            gids = np.asarray(partition.gids, dtype=np.int64)
            gcol = gids[cols // k] * k + cols % k
            o = np.lexsort((gcol, rows))                # CSR order
            rows, cols, gcol = rows[o], cols[o], gcol[o]
            N = int(partition.n_global)
            off = int(partition.offsets[partition.rank])
            cfg = {"tolerance": ls["tol"], "max_iters": ls["restart"] * ls["maxiter"],
                   **ls["amgx"]}
            # the non-traceable MPI setup (exchange plans), once per pattern,
            # from a concrete matrix of the same structure (all ranks trace
            # together: its host collectives line up)
            import scipy.sparse as sps            # numpy only: inside a trace
            A0 = sps.csr_matrix((np.ones(len(rows)), (rows, gcol)),
                                shape=(k * n_own, k * N))
            meta = amgx.cache_mpi_metadata(cfg, amgx_comm, k * N,
                                           (k * off, k * (off + n_own)), A0,
                                           block_dim=k)
            patterns[key] = (rows, cols, colors, gcol, meta, cfg)
        rows, cols, colors, gcol, meta, cfg = patterns[key]
        probes = jnp.asarray(np.arange(int(colors.max()) + 1)[:, None] == colors[None, :],
                             dtype=b.dtype)
        AV = jax.vmap(lambda x: to_cell(lin_o(to_row(x))))(probes)
        vals = AV[colors[cols], rows]
        N = int(partition.n_global)
        A_sp = amgx.with_cache(jsparse.BCOO((vals, jnp.asarray(np.stack([rows, gcol], 1))),
                                            shape=(k * n_own, k * N)), mpi=meta)
        x_o, info = amgx.solve(A_sp, to_cell(b)[:k * n_own], block_dim=k)  # cfg: in meta
        if os.environ.get("ZOOMY_KRYLOV_LOG"):              # opt-in diagnostics
            jax.debug.print("amgx mpi: {i} iterations, residual {r:.2e}",
                            i=info["iterations"], r=info["residual"])
        x = jnp.zeros(shape, b.dtype).at[:, :n_own].set(x_o.reshape(n_own, k).T)
        if ls["accept"] is not None:            # the global relative residual
            _, lin = jax.linearize(A, jnp.zeros_like(b))
            mask = jnp.broadcast_to(own_cell, shape).astype(b.dtype)
            xs, bs = lax.stop_gradient(x), lax.stop_gradient(b) * mask
            res = mask * lin(xs) - bs
            r = jnp.sqrt(gsum(jnp.sum(res ** 2)) / jnp.maximum(gsum(jnp.sum(bs ** 2)),
                                                              jnp.finfo(b.dtype).tiny))
            lax.cond(r > ls["accept"], lambda: jax.debug.callback(_ls_reject, r),
                     lambda: None)
        return x

    def _dlinsolve(A, b):
        """Distributed 'gmres' / 'pyamg': flexible GMRES over the owned cells
        (global inner products; the operator exchanges the halo itself),
        right preconditioned for 'pyamg' by the rank's own block -- a PyAMG
        V-cycle of the operator assembled with the halo off -- plus a coarse
        correction.  ('amgx': :func:`_amgx_mpi`.)"""
        shape, k = b.shape, b.shape[0]
        _, lin = jax.linearize(A, jnp.zeros_like(b))
        mask = jnp.broadcast_to(own_cell, shape).reshape(-1).astype(b.dtype)
        mv = lambda x: lin(x.reshape(shape)).reshape(-1)          # noqa: E731
        dots = lambda V, w: gsum(V @ (w * mask))                   # noqa: E731
        precond = (lambda transpose: lambda v: v)                  # noqa: E731
        if pyamg is not None:
            with no_halo():                     # the rank's block A_oo
                _, lin_o = jax.linearize(A, jnp.zeros_like(b))
            n_o = k * n_own
            emb = lambda xo: jnp.zeros(shape, b.dtype).at[:, :n_own].set(  # noqa: E731
                xo.reshape(n_own, k).T)
            own = lambda X: X[:, :n_own].T.reshape(-1)             # noqa: E731
            vals = assembled(lambda xo: own(lin_o(emb(xo))), n_o, n_own)
            rows, cols, _ = patterns[(n_o, n_own)]
            out = jax.ShapeDtypeStruct((n_o,), b.dtype)

            def precond(transpose):
                f = host_vcycle(rows, cols, n_o, k, transpose)
                return lambda v: emb(jax.pure_callback(
                    f, out, vals, own(v.reshape(shape)))).reshape(-1)

        if pyamg is not None and ls.get("coarse", True):
            # two levels: + a coarse correction, one unknown per rank and
            # row (constants on the rank's cells; Nicolaides), so the
            # coupling the blocks ignore is solved once, globally
            P = int(partition.size)
            rank = int(partition.rank)
            own_f = own_cell.astype(b.dtype)

            def restrict(v):                    # (k * nc,) -> (P, k)
                loc = jnp.sum(v.reshape(shape) * own_f, axis=1)        # (k,)
                return gsum(jnp.zeros((P, k), b.dtype).at[rank].set(loc))

            def coarse(transpose):
                def column(c, _):
                    q, i = c // k, c % k
                    e = jnp.zeros(shape, b.dtype).at[i].set(
                        jnp.where(rank == q, own_f, 0.0)).reshape(-1)
                    y = mv(e) if not transpose else lin_t(e)
                    return c + 1, restrict(mask * y).reshape(-1)
                _, cols_c = lax.scan(column, 0, None, length=P * k)
                Ac = cols_c.T                                           # (Pk, Pk)
                return lambda v: (jnp.linalg.solve(Ac, restrict(v).reshape(-1))
                                  .reshape(P, k)[rank][:, None] * own_f[None, :]
                                  ).reshape(-1)
            lt = (jax.linear_transpose(mv, b.reshape(-1)) if ad_mode == "reverse"
                  else None)
            lin_t = (lambda e: lt(e)[0]) if lt is not None else None   # noqa: E731
            block = precond
            precond = (lambda transpose: (lambda B, C: (lambda v: B(v) + C(v)))(  # noqa: E731
                block(transpose), coarse(transpose)))

        def solver(transpose):
            M = precond(transpose)
            # Krylov vectors cost memory per rank: a short restart, as many
            # cycles as the serial budget (restart x maxiter iterations)
            m = min(int(ls["restart"]), 60)
            cycles = max(1, int(ls["restart"]) * int(ls["maxiter"]) // m)
            return lambda matvec, rhs: _fgmres(
                lambda v: mask * matvec(v), rhs * mask, dots, M, ls["tol"],
                m, cycles)
        bf = b.reshape(-1) * mask
        x = lax.custom_linear_solve(
            mv, bf, solve=solver(False),
            transpose_solve=solver(True) if ad_mode == "reverse" else None)
        if ls["accept"] is not None:
            xs, bs = lax.stop_gradient(x), lax.stop_gradient(bf)
            nb = jnp.sqrt(dots(bs[None], bs)[0])
            res = mask * mv(xs) - bs
            r = jnp.sqrt(dots(res[None], res)[0]) / jnp.where(nb > 0, nb, 1.0)
            lax.cond(r > ls["accept"], lambda: jax.debug.callback(_ls_reject, r),
                     lambda: None)
        return x.reshape(shape)

    def linsolve(A, b):
        """Matrix-free restarted GMRES on the linear map A (linearized once at
        0, so an affine A is fine); differentiable (custom_linear_solve).
        backend 'amgx': AmgX on the operator's traced sparsity (JAX-AMG,
        differentiable itself).  Above the accept gate the run aborts.
        Distributed: :func:`_dlinsolve`."""
        if partition is not None:
            return _amgx_mpi(A, b) if amgx is not None else _dlinsolve(A, b)
        _, lin = jax.linearize(A, jnp.zeros_like(b))     # primal once per solve
        shape = b.shape                                  # Krylov on flat vectors
        mv = lambda x: lin(x.reshape(shape)).reshape(-1)
        b = b.reshape(-1)
        if amgx is not None:   # cell-major unknowns, one k x k block per cell
            n, k = int(b.shape[0]), shape[0]
            to_cell = lambda x: x.reshape(k, -1).T.reshape(-1)       # noqa: E731
            to_row = lambda x: x.reshape(-1, k).T.reshape(-1)        # noqa: E731
            vals = assembled(lambda x: to_cell(mv(to_row(x))), n)
            rows, cols, _ = patterns[(n, nc)]
            from jax.experimental import sparse as jsparse
            A_sp = jsparse.BCOO((vals, jnp.asarray(np.stack([rows, cols], 1))),
                                shape=(n, n))
            x, info = amgx.solve(A_sp, to_cell(b), **{
                "tolerance": ls["tol"], "max_iters": ls["restart"] * ls["maxiter"],
                "block_dim": k, **ls["amgx"]})
            if os.environ.get("ZOOMY_KRYLOV_LOG"):          # opt-in diagnostics
                jax.debug.print("amgx: {info}", info=info)
            x = to_row(x)
        elif pyamg is not None:   # cell-major, AMG-preconditioned GMRES on host
            n, k = int(b.shape[0]), shape[0]
            to_cell = lambda x: x.reshape(k, -1).T.reshape(-1)       # noqa: E731
            to_row = lambda x: x.reshape(-1, k).T.reshape(-1)        # noqa: E731
            op = lambda x: to_cell(mv(to_row(x)))                    # noqa: E731
            vals = assembled(op, n)
            rows, cols, _ = patterns[(n, nc)]
            out = jax.ShapeDtypeStruct(b.shape, b.dtype)

            def amg(transpose):
                f = host_amg(rows, cols, n, k, transpose)
                return lambda _, rhs: to_row(jax.pure_callback(f, out, vals, to_cell(rhs)))
            x = lax.custom_linear_solve(
                mv, b, solve=amg(False),
                transpose_solve=amg(True) if ad_mode == "reverse" else None)
        else:   # the transposed solve is only traced when a reverse pass needs it
            x = lax.custom_linear_solve(
                mv, b, solve=_ls_gmres,
                transpose_solve=_ls_gmres if ad_mode == "reverse" else None)
        if ls["accept"] is not None:
            xs, bs = lax.stop_gradient(x), lax.stop_gradient(b)
            nb = jnp.linalg.norm(bs)
            r = jnp.linalg.norm(mv(xs) - bs) / jnp.where(nb > 0, nb, 1.0)
            lax.cond(r > ls["accept"], lambda: jax.debug.callback(_ls_reject, r),
                     lambda: None)
        return x.reshape(shape)

    explicit_op = []

    def diffusion(k, Q, Qaux, p, mesh):
        if not explicit_op:
            explicit_op.append(DenseDiffusionOperatorJAX(
                mesh_np, mesh_np.dimension, rt.n_state, state_dependent=False))
        op = explicit_op[0]
        A = op._as_cell_tensor(k(Q, Qaux, p))
        return op._divergence(Q, A, bf_grads=None)

    diffusion_op = []

    def implicit_diffusion(k, Q, Qaux, p, dt, mesh):
        if not diffusion_op:
            _, state_dep, _ = HyperbolicSolverNumpy._classify_diffusion(
                sm.diffusion_matrix, sm)
            diffusion_op.append(DenseDiffusionOperatorJAX(
                mesh_np, mesh_np.dimension, rt.n_state, state_dependent=state_dep))
        op = diffusion_op[0]
        Q_new = op.implicit_solve(
            Q[:, :op.nc], dt, lambda Qs: k(Qs, Qaux[:, :op.nc], p),
            bf_grads=None, tol=imp["gmres_tol"], maxiter=imp["gmres_maxiter"],
            newton_maxiter=imp["maxiter"], newton_tol=imp["tol"])
        return Q.at[:, :op.nc].set(Q_new)

    # ── effects ────────────────────────────────────────────────────────────
    warned = []

    def check_cfl(tok, dt_cfl, dt):
        def host(a, b):
            if float(b) > float(a) * (1 + 1e-12) and not warned:
                warned.append(True)
                logger.warning(f"time step {float(b):.3e} exceeds the CFL step "
                               f"{float(a):.3e} (warned once)")
        a, b = lax.stop_gradient(dt_cfl), lax.stop_gradient(dt)
        lax.cond(b > a * (1 + 1e-12), lambda: jax.debug.callback(host, a, b),
                 lambda: None)
        return (tok + 1,)

    def io_write(tok, time, Q, Qaux):
        if on_write is not None and ad_mode == "forward":
            jax.debug.callback(on_write, time, Q, Qaux, ordered=True)
        return (tok + 1,)

    rt.aux_derivatives = nsm.aux_derivatives
    named = {getattr(nsm, "_symbolic_name", None) or "nsm": rt}
    for name, (sub, sub_rt) in (systems or {}).items():
        sub_rt.aux_derivatives = sub.aux_derivatives
        named[name] = sub_rt
    m = dict(
        **named, cells=cells, faces=faces, face_aux=face_aux,
        face_wavespeed=face_wavespeed, face_inradius=face_inradius,
        boundary_states=boundary_states, boundary_traces=boundary_traces,
        boundary_ghosts=boundary_ghosts,
        pointwise=pointwise, cell_traces=cell_traces, gradient=gradient,
        minmod=limiter("_limit_minmod"),
        venkatakrishnan=limiter("_limit_vk"),
        barth_jespersen=limiter("_limit_bj"),
        extrapolate=extrapolate, divergence=divergence,
        edge_source=edge_source, derivative_aux=derivative_aux,
        inadmissible=inadmissible, reduce_min=jnp.min, reduce_any=jnp.any,
        halo=lambda X, mesh: X,
        put=put, zeros=zeros, clean_round_off=clean_round_off, take=lambda Q, rows: Q[jnp.asarray(rows)],
        reduce_max=jnp.max, logical_and=jnp.logical_and,
        select=select, while_loop=while_loop, cond=cond, io_write=io_write,
        root=root, linsolve=linsolve, implicit_diffusion=implicit_diffusion,
        diffusion=diffusion,
        check_cfl=check_cfl,
        minimum=jnp.minimum, maximum=jnp.maximum, abs=jnp.abs, floor=jnp.floor)
    if couple is not None:
        m.update(_coupling(couple, rt, mesh_np, fc0, bf_face, nf))
    if partition is not None:
        from zoomy_jax.symsolver.parallel import blocks
        m.update(blocks(partition, mesh_np))
    return m


def _fgmres(A, b, dots, M, tol, restart, maxiter):
    """Restarted flexible GMRES, right preconditioner ``M`` (may vary per
    iteration), classical Gram-Schmidt twice; ``dots(V, w)``: the inner
    products of every row of ``V`` with ``w`` in one reduction.  Givens
    rotations track the residual: a cycle stops as soon as ``|r| <= tol |b|``
    (or after ``restart`` iterations); at most ``maxiter`` cycles."""
    from jax.scipy.linalg import solve_triangular
    m, n, dt = int(restart), b.size, b.dtype
    nrm = lambda v: jnp.sqrt(dots(v[None], v)[0])                  # noqa: E731
    bn = nrm(b)
    target = tol * jnp.where(bn > 0, bn, 1.0)

    def cycle(c):
        x, r, it, _ = c
        beta = nrm(r)
        V = jnp.zeros((m + 1, n), dt).at[0].set(r / jnp.where(beta > 0, beta, 1.0))
        g = jnp.zeros(m + 1, dt).at[0].set(beta)

        def arnoldi(st):
            j, V, Z, H, cs, sn, g = st
            z = M(V[j])
            w = A(z)
            act = (jnp.arange(m + 1) <= j).astype(dt)
            h = dots(V, w) * act
            w = w - h @ V
            h2 = dots(V, w) * act
            w = w - h2 @ V
            hn = nrm(w)
            h = (h + h2).at[j + 1].set(hn)

            def rot(i, h):                       # the earlier rotations
                a, b_ = h[i], h[i + 1]
                return h.at[i].set(cs[i] * a + sn[i] * b_).at[i + 1].set(
                    -sn[i] * a + cs[i] * b_)
            h = lax.fori_loop(0, j, rot, h)
            d = jnp.sqrt(h[j] ** 2 + h[j + 1] ** 2)
            ci = jnp.where(d > 0, h[j] / jnp.where(d > 0, d, 1.0), 1.0)
            si = jnp.where(d > 0, h[j + 1] / jnp.where(d > 0, d, 1.0), 0.0)
            h = h.at[j].set(d).at[j + 1].set(0.0)
            g = g.at[j + 1].set(-si * g[j]).at[j].set(ci * g[j])
            return (j + 1, V.at[j + 1].set(w / jnp.where(hn > 0, hn, 1.0)),
                    Z.at[j].set(z), H.at[:, j].set(h), cs.at[j].set(ci),
                    sn.at[j].set(si), g)

        st = (0, V, jnp.zeros((m, n), dt), jnp.zeros((m + 1, m), dt),
              jnp.zeros(m, dt), jnp.zeros(m, dt), g)
        k, V, Z, H, _, _, g = lax.while_loop(
            lambda st: (st[0] < m) & (jnp.abs(st[6][st[0]]) > target), arnoldi, st)
        live = jnp.arange(m) < k                  # the k columns built
        R = jnp.where(live[:, None] & live[None, :], H[:m], jnp.eye(m, dtype=dt))
        y = solve_triangular(R, jnp.where(live, g[:m], 0.0), lower=False)
        x = x + y @ Z
        return x, b - A(x), it + 1, k

    x, r, it, k = lax.while_loop(lambda c: (nrm(c[1]) > target) & (c[2] < maxiter),
                                 cycle, (jnp.zeros_like(b), b, 0, 0))
    if os.environ.get("ZOOMY_KRYLOV_LOG"):              # opt-in diagnostics
        jax.debug.print("fgmres: {c} cycles, last {k} of {m} iterations, "
                        "|r|/|b| = {q:.2e}", c=it, k=k, m=m,
                        q=nrm(r) / jnp.where(bn > 0, bn, 1.0))
    return x


def _jaxamg():
    """The optional JAX-AMG package (Liu, Fan, Wang 2026, SoftwareX 35:102966;
    https://github.com/jx-wang-s-group/JAX-AMG), imported only when asked."""
    try:
        import jaxamg
    except ImportError as e:
        raise ImportError(
            "linear solver 'amgx' needs the optional JAX-AMG package (CUDA GPU "
            "only): build NVIDIA AmgX >= 2.5 from source, set AMGX_ROOT and "
            "CUDA_HOME, then pip install 'jaxamg[cuda12]' -- see "
            "docs/dev/symbolic_solver/AMG.md.  Or use 'gmres'.") from e
    import jax as _jax
    if _jax.default_backend() != "gpu":
        raise RuntimeError("linear solver 'amgx' (AmgX) runs on a CUDA GPU "
                           f"only; JAX uses {_jax.default_backend()!r}")
    return jaxamg


def _cell_pattern(n, nc, nbrs):
    """Sparsity of a cell operator on ``k = n / nc`` unknowns per cell,
    cell-major (index ``c k + r``): every unknown of cell c may couple to
    every unknown of the cells in its LSQ stencil (the derivative aux are LSQ
    fits of the unknowns; boundary ghosts depend on the inner cell)."""
    k = n // nc
    nb = np.asarray(nbrs)[:nc]
    cells = np.concatenate([np.arange(nc)[:, None], nb], axis=1)    # self + stencil
    c_of = np.repeat(np.arange(nc), cells.shape[1])
    d_of = cells.reshape(-1)
    keep = (d_of >= 0) & (d_of < nc)
    pairs = np.unique(np.stack([c_of[keep], d_of[keep]], 1), axis=0)
    r, q = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    rows = (pairs[:, 0, None] * k + r.reshape(1, -1)).reshape(-1)
    cols = (pairs[:, 1, None] * k + q.reshape(1, -1)).reshape(-1)
    order = np.lexsort((cols, rows))
    return rows[order].astype(np.int32), cols[order].astype(np.int32)


def _amgx_coloring(n, nc, nbrs):
    """The pattern with JAX-AMG's column colouring, for ``with_cache``."""
    from jaxamg.sparsity import get_column_coloring
    rows, cols = _cell_pattern(n, nc, nbrs)
    with jax.ensure_compile_time_eval():          # concrete, inside a trace
        colors, n_colors = get_column_coloring(rows, cols, (n, n))
    return rows, cols, np.asarray(colors), int(n_colors), (n, n)


def _greedy_colors(rows, cols, n, seed=0):
    """Column colouring (two columns sharing a row get different colours):
    Jones-Plassmann rounds, each colours an independent set of the column
    conflict graph at once (vectorised)."""
    import scipy.sparse as sps
    P = sps.csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, n))
    C = (P.T @ P).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    w = np.random.default_rng(seed).permutation(n).astype(np.int64)
    colors = -np.ones(n, np.int64)
    has = np.diff(C.indptr) > 0
    starts = C.indptr[:-1][has]
    while (colors < 0).any():
        live = np.where(colors < 0, w, -1)
        nbmax = np.full(n, -1, np.int64)
        nbmax[has] = np.maximum.reduceat(live[C.indices], starts)
        pick = (colors < 0) & (live > nbmax)          # an independent set
        # each picked column takes the smallest colour its neighbours lack
        nc_ = colors[C.indices]
        bits = np.where(nc_ >= 0, np.left_shift(np.uint64(1), np.minimum(nc_, 63).astype(np.uint64)),
                        np.uint64(0))
        used = np.zeros(n, np.uint64)
        used[has] = np.bitwise_or.reduceat(bits, starts)
        free = ~used[pick]
        low = free & (~free + np.uint64(1))           # lowest free bit
        colors[pick] = np.log2(low.astype(np.float64)).round().astype(np.int64)
        if (colors >= 63).any():
            raise RuntimeError("colouring needs more than 63 colours")
    return colors.astype(np.int32)


def _pyamg():
    """PyAMG (Bell, Olson, Schroder 2022, JOSS 7:4142), imported when asked."""
    try:
        import pyamg
    except ImportError as e:
        raise ImportError("linear solver 'pyamg' needs the optional PyAMG "
                          "package: pip install pyamg.  Or use 'gmres'.") from e
    return pyamg


def _face_kernel(rt, name):
    """``rt.<name>`` for a Numerics face function JaxRuntime does not build,
    lambdified the same way as its ``numerical_flux`` (one CSE silo, vmapped
    over the faces)."""
    num = getattr(rt, "_numerics", None)
    if getattr(rt, name, None) is not None or num is None:
        return
    if name not in list(num.functions.keys()):
        setattr(rt, name, None)
        return
    from zoomy_jax.transformation.jax_runtime import (
        _lambdify_array, _to_array_of_exprs)
    sig = (list(num.variables_minus), list(num.variables_plus),
           list(num.aux_variables_minus), list(num.aux_variables_plus),
           rt._p_syms, rt._n_syms)
    per_face = _lambdify_array(_to_array_of_exprs(num.functions[name].definition),
                               sig, rt.module)

    speed = name.endswith("_speed")   # rows (F + D-, -F + D+, s ... s)

    @jax.jit
    def kernel(qL, qR, qauxL, qauxR, parameters, normal):
        def one(a, b, c, d, n):
            out = per_face(a, b, c, d, parameters, n)
            return (out[0], out[1], out[2, 0]) if speed else out
        return jax.vmap(one, in_axes=(1, 1, 1, 1, 1), out_axes=-1)(
            qL, qR, qauxL, qauxR, normal)
    setattr(rt, name, kernel)


# ── preCICE ─────────────────────────────────────────────────────────────────
class PreciceAdapter:
    """Host side of the ``couple_*`` effects: preCICE API calls only."""

    def __init__(self, name, config, mesh_name, write, read, coords):
        import petsc4py  # noqa: F401  (must precede precice)
        import precice
        self.mesh, self.wn, self.rn = mesh_name, list(write), list(read)
        self.p = precice.Participant(name, config, 0, 1)
        self.ids = self.p.set_mesh_vertices(mesh_name, np.asarray(coords))

    def _write(self, q):
        for i, n in enumerate(self.wn):
            self.p.write_data(self.mesh, n, self.ids, np.asarray(q[i]))

    def init(self, q):
        if self.p.requires_initial_data():
            self._write(q)
        self.p.initialize()
        return np.bool_(self.p.is_coupling_ongoing())

    def max_dt(self):
        return np.float64(self.p.get_max_time_step_size())

    def read(self):
        return np.stack([self.p.read_data(self.mesh, n, self.ids, 0.0)
                         for n in self.rn])

    def write(self, q):
        self._write(q)
        return ()

    def advance(self, dt):
        self.p.advance(float(dt))
        return np.bool_(self.p.is_coupling_ongoing())

    def finalize(self):
        self.p.finalize()


def coupled_face_indices(mesh_np, tag):
    """Boundary faces carrying boundary tag ``tag`` (by name)."""
    names = list(mesh_np.boundary_conditions_sorted_names)
    nums = np.asarray(mesh_np.boundary_face_function_numbers, dtype=int)
    bf_face = np.asarray(mesh_np.boundary_face_face_indices, dtype=int)
    n_bf = int(mesh_np.n_boundary_faces)
    return np.array([bf_face[i] for i in range(n_bf)
                     if names[nums[i]] == tag], dtype=int)


def _coupling(cfg, rt, mesh_np, fc0, bf_face, nf):
    """Coupling geometry blocks + the ``couple_*`` effects for one participant.
    Each host call consumes the previous token and returns the next one, so
    XLA cannot reorder the preCICE calls."""
    from jax.experimental import io_callback
    a = cfg["adapter"]
    faces_c = coupled_face_indices(mesh_np, cfg["tag"])
    cells_c = fc0[faces_c]
    mask = np.zeros(nf, dtype=bool)
    mask[faces_c] = True
    n_state = int(rt.n_state)

    def coupled_faces(mesh):
        return jnp.asarray(mask)

    def interface_states(Q, mesh):
        return Q[:, jnp.asarray(cells_c)]

    def interface_flux(k, kaux, side, Q, Qaux, peer, p, mesh):
        # Both participants must pose the IDENTICAL Riemann problem: same
        # states, same aux, same normal.  The aux is therefore rebuilt from
        # each state alone (pointwise formula on a zero template), never taken
        # from one side's cell, whose derivative rows the peer cannot know.
        own = Q[:, jnp.asarray(cells_c)]
        zero = jnp.zeros((Qaux.shape[0], own.shape[1]), dtype=Q.dtype)
        aux = (lambda q: zero) if kaux is False else (lambda q: kaux(q, zero, p))
        n_out = mesh.face_normals[:, jnp.asarray(faces_c)]
        low, high, n = (own, peer, n_out) if int(side) == 0 else (peer, own, -n_out)
        c = k(low, high, aux(low), aux(high), p, n)[int(side)]
        out = jnp.zeros((n_state, nf), dtype=Q.dtype)
        return out.at[:, jnp.asarray(faces_c)].set(c)

    S = jax.ShapeDtypeStruct
    T, F, B = S((), jnp.int64), S((), jnp.float64), S((), jnp.bool_)
    Qe = S((len(a.rn), len(faces_c)), jnp.float64)

    def eff(host, *outs):
        """Effect body: the host call takes the token and returns the next one
        plus its outputs, so XLA cannot reorder the preCICE calls."""
        def body(tok, *args):
            def call(k, *x):
                r = host(*x)
                r = r if isinstance(r, tuple) else (r,)
                return (np.int64(k + 1), *r)
            return io_callback(call, (T, *outs), tok, *args)
        return body

    return dict(coupled_faces=coupled_faces, interface_states=interface_states,
                interface_flux=interface_flux,
                couple_init=eff(a.init, B), couple_max_dt=eff(a.max_dt, F),
                couple_read=eff(a.read, Qe), couple_write=eff(a.write),
                couple_advance=eff(a.advance, B))


def hoisted_jit(fn):
    """``jax.jit(fn)`` with every constant ``fn`` captures (the mesh index
    arrays of the runtime's closures) passed as an argument instead of baked
    into the program: compile time and program size stay independent of the
    mesh.  Traced once per argument shapes; differentiable like ``fn``."""
    cache = {}

    def run(*args):
        flat, tree = jax.tree_util.tree_flatten(args)
        key = (tree, tuple((jnp.shape(x), jnp.result_type(x)) for x in flat))
        if key not in cache:
            closed, out = jax.make_jaxpr(fn, return_shape=True)(*args)
            consts = [jnp.asarray(c) for c in closed.consts]
            body = jax.jit(lambda cs, *xs: jax.core.eval_jaxpr(closed.jaxpr, cs, *xs))
            cache[key] = (body, consts, jax.tree_util.tree_structure(out))
        body, consts, out_tree = cache[key]
        return jax.tree_util.tree_unflatten(out_tree, body(consts, *flat))
    return run


def build(src, name, module):
    """Execute printed source against the name table; return the function."""
    missing = [n for n in contract() if n not in module and n in src]
    if missing:
        raise NotImplementedError(f"JAX runtime lacks blocks {missing}")
    ns = dict(module)
    exec(compile(src, f"<symsolver:{name}>", "exec"), ns)
    return ns[name]
