"""JAX runtime for the symbolic solvers: what each printed name means on JAX.

The printed solver is plain Python calling names from this table.  ``nsm`` is
the JaxRuntime (the NSM's kernels, vmapped); every other entry is a block of
``zoomy_core.symsolver`` implemented with the existing JAX mesh code.
"""
from __future__ import annotations

from typing import NamedTuple

import copy
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
            implicit=None, systems=None):
    """The name table the printed solver runs against.  ``systems``: further
    ``{name: (nsm, JaxRuntime)}`` a template uses besides ``nsm``."""
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
    branch = jnp.asarray(boundary_face_branches(
        mesh_np, getattr(sm, "_bc_source", None)), dtype=jnp.int32)
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

        def per_bf(i):
            f = j(bf_face)[i]
            return bc(branch[i], time, fc[f, :], d_face[i], states[:, i],
                      Qaux[:, j(cells_of_face)[i]], p, n[:, f])
        return jax.vmap(per_bf)(jnp.arange(n_bf)).T

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

    def assembled(op, n):
        """The operator's values on the cell pattern, by colouring: one
        evaluation per colour (traced; the pattern is static)."""
        if n not in patterns:
            rows, cols = _cell_pattern(n, nc, lsq_nbrs)
            colors = _greedy_colors(rows, cols, n)
            patterns[n] = (rows, cols, colors)
        rows, cols, colors = patterns[n]
        probes = np.arange(int(colors.max()) + 1)[:, None] == colors[None, :]
        AV = jax.vmap(op)(jnp.asarray(probes, dtype=jnp.float64))  # one per colour
        return AV[colors[cols], rows]

    def host_amg(rows, cols, n, k, transpose):
        def solve(vals, rhs):
            import scipy.sparse as sps
            A = sps.csr_matrix((np.asarray(vals), (rows, cols)), shape=(n, n))
            A = (A.T if transpose else A).tobsr(blocksize=(k, k))
            # left-scale by the inverse cell blocks: A ~ M (x) L with a k x k
            # coupling M (indefinite for VAM) becomes ~ I (x) L, AMG-friendly
            A.sort_indices()
            row = np.repeat(np.arange(n // k), np.diff(A.indptr))
            diag = A.data[A.indices == row]                      # (cells, k, k)
            Dinv = sps.block_diag(np.linalg.inv(diag), format="bsr")
            As, bs = (Dinv @ A).tobsr(blocksize=(k, k)), Dinv @ np.asarray(rhs)
            ml = pyamg.smoothed_aggregation_solver(As)
            x = ml.solve(bs, tol=ls["tol"], accel="gmres",
                         maxiter=ls["restart"] * ls["maxiter"])
            return np.asarray(x, dtype=np.asarray(rhs).dtype)
        return solve

    def linsolve(A, b):
        """Matrix-free restarted GMRES on the linear map A (linearized once at
        0, so an affine A is fine); differentiable (custom_linear_solve).
        backend 'amgx': AmgX on the operator's traced sparsity (JAX-AMG,
        differentiable itself).  Above the accept gate the run aborts."""
        _, lin = jax.linearize(A, jnp.zeros_like(b))     # primal once per solve
        shape = b.shape                                  # Krylov on flat vectors
        mv = lambda x: lin(x.reshape(shape)).reshape(-1)
        b = b.reshape(-1)
        if amgx is not None:   # cell-major unknowns, one k x k block per cell
            n, k = int(b.shape[0]), shape[0]
            to_cell = lambda x: x.reshape(k, -1).T.reshape(-1)       # noqa: E731
            to_row = lambda x: x.reshape(-1, k).T.reshape(-1)        # noqa: E731
            vals = assembled(lambda x: to_cell(mv(to_row(x))), n)
            rows, cols, _ = patterns[n]
            from jax.experimental import sparse as jsparse
            A_sp = jsparse.BCOO((vals, jnp.asarray(np.stack([rows, cols], 1))),
                                shape=(n, n))
            x, _ = amgx.solve(A_sp, to_cell(b), **{
                "tolerance": ls["tol"], "max_iters": ls["restart"] * ls["maxiter"],
                "block_dim": k, **ls["amgx"]})
            x = to_row(x)
        elif pyamg is not None:   # cell-major, AMG-preconditioned GMRES on host
            n, k = int(b.shape[0]), shape[0]
            to_cell = lambda x: x.reshape(k, -1).T.reshape(-1)       # noqa: E731
            to_row = lambda x: x.reshape(-1, k).T.reshape(-1)        # noqa: E731
            op = lambda x: to_cell(mv(to_row(x)))                    # noqa: E731
            vals = assembled(op, n)
            rows, cols, _ = patterns[n]
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
        put=put, zeros=zeros, clean_round_off=clean_round_off, take=lambda Q, rows: Q[jnp.asarray(rows)],
        reduce_max=jnp.max, logical_and=jnp.logical_and,
        select=select, while_loop=while_loop, cond=cond, io_write=io_write,
        root=root, linsolve=linsolve, implicit_diffusion=implicit_diffusion,
        diffusion=diffusion,
        check_cfl=check_cfl,
        minimum=jnp.minimum, maximum=jnp.maximum, abs=jnp.abs, floor=jnp.floor)
    if couple is not None:
        m.update(_coupling(couple, rt, mesh_np, fc0, bf_face, nf))
    return m


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
    rows, cols = [], []
    for c in range(nc):
        for d in sorted(set(int(j) for j in nbrs[c] if 0 <= j < nc) | {c}):
            for r in range(k):
                for q in range(k):
                    rows.append(c * k + r)
                    cols.append(d * k + q)
    return np.asarray(rows, np.int32), np.asarray(cols, np.int32)


def _amgx_coloring(n, nc, nbrs):
    """The pattern with JAX-AMG's column colouring, for ``with_cache``."""
    from jaxamg.sparsity import get_column_coloring
    rows, cols = _cell_pattern(n, nc, nbrs)
    with jax.ensure_compile_time_eval():          # concrete, inside a trace
        colors, n_colors = get_column_coloring(rows, cols, (n, n))
    return rows, cols, np.asarray(colors), int(n_colors), (n, n)


def _greedy_colors(rows, cols, n):
    """Column colouring: two columns sharing a row get different colours."""
    import scipy.sparse as sps
    P = sps.csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, n))
    conflict = (P.T @ P).tocsr()
    colors = -np.ones(n, np.int32)
    for j in range(n):
        nb = conflict.indices[conflict.indptr[j]:conflict.indptr[j + 1]]
        used = set(colors[nb][colors[nb] >= 0])
        c = 0
        while c in used:
            c += 1
        colors[j] = c
    return colors


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


def build(src, name, module):
    """Execute printed source against the name table; return the function."""
    missing = [n for n in contract() if n not in module and n in src]
    if missing:
        raise NotImplementedError(f"JAX runtime lacks blocks {missing}")
    ns = dict(module)
    exec(compile(src, f"<symsolver:{name}>", "exec"), ns)
    return ns[name]
