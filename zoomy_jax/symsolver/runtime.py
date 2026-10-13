"""JAX runtime of the symbolic solvers: what each name of a printed solver means on JAX.  The
printed template (solver template -> `PyPrinter`) is plain Python calling this name table: `nsm`
is the vmapped JaxRuntime of the kernels, every other entry a `zoomy_core.symsolver` block.

References:
    Bradbury et al. 2018, https://github.com/jax-ml/jax
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
from zoomy_core.misc.options import Option
from zoomy_core.solver.constants import TPFA_COS_FLOOR
from zoomy_core.mesh.geometry_xp import compute_geometry, lsq_closure_derivatives
from zoomy_core.model.boundary_conditions import boundary_face_branches
from zoomy_core.symsolver.core import contract
from zoomy_core.symsolver.numpy_backend.assembly import cell_pattern, host_amg_solve
from zoomy_jax.fvm.implicit_ad import cellwise_solve, newton_root
from zoomy_jax.mesh.mesh import lsq_gradient_per_field
from zoomy_jax.fvm.reconstruction_jax import (
    ConstantReconstruction, DenseDiffusionOperatorJAX,
    LSQMUSCLReconstructionJAX)
from zoomy_core.mesh.distributed import HALO
from zoomy_jax.fvm.solver_jax import (
    HyperbolicSolver as _Reference)

class AdMode(Option):
    """How the JAX march is differentiated."""
    #: jvp / jacfwd through the time loop (tangents carried, constant memory)
    forward = "forward"
    #: jax.grad through a binomially checkpointed time loop
    reverse = "reverse"


class Geometry(NamedTuple):
    """The `mesh` argument of a printed solver, computed from the vertex coordinates inside the
    traced function so that `jax.grad` reaches the vertices.  Topology (indices, stencils) is
    static and lives in the runtime's closures.
    """
    cell_volumes: jnp.ndarray
    cell_centers: jnp.ndarray
    cell_inradius: jnp.ndarray
    face_normals: jnp.ndarray
    face_volumes: jnp.ndarray
    face_centers: jnp.ndarray
    lsq_gradQ: jnp.ndarray
    lsq_closure: jnp.ndarray       # (n_class, n_cc, R, M), ANALYSIS 40


def geometry_of(topo, X):
    """All solver geometry from vertex coordinates `X` `(dim, n_vertices)`."""
    g = compute_geometry(topo, X)
    return Geometry(*(g[f] for f in Geometry._fields))


def _with_geometry(R, g, mesh_np, nc):
    """A copy of reconstruction `R` that reads the traced geometry `g` instead of the arrays
    precomputed from the static mesh.
    """
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


def runtime(nsm, rt, mesh_np, mesh_j, topo, *, ad_mode=AdMode.forward, checkpoints=64,
            on_write=None, on_save=None, on_load=None, couple=None, implicit=None, systems=None,
            partition=None):
    """Build the name table (dict of callables) a printed solver runs against: the JAX
    implementation of every `zoomy_core.symsolver` block on this mesh.  Called by the JAX solver
    driver's `_setup`.

    Args:
        topo: static topology (`geometry_xp.build_topology`) with the boundary-closure classes
        ad_mode: `AdMode.forward`, or `AdMode.reverse` for a time loop with `checkpoints`
            checkpoints
        on_write: `on_write(time, Q, Qaux)` snapshot callback (forward mode)
        on_save, on_load: the restart pair (host), `on_save(time, Q, Qaux)` (forward mode) and
            `on_load(time, Q, Qaux) -> (time, Q, Qaux)`; None: no restart file / not restarting
        couple: preCICE settings `dict(adapter=, tag=, side=)`
        implicit: Newton and linear-solver settings of `root` and `linsolve`
        systems: further `{name: (nsm, JaxRuntime)}` a template uses besides `nsm`
        partition: this rank's piece of a distributed run; its `halo` and global reductions
            replace the serial ones
    Returns:
        dict name -> callable
    """
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
    # a periodic face's outer trace is its partner face's inner trace (the
    # partner cell reconstructed to the seam), not the partner's cell value
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
        """Kernel `k` in every cell (with time, dt and the cell centre)."""
        return k(Q, Qaux, p, time=time, dt=dt,
                 position=mesh.cell_centers[:, :Q.shape[-1]])

    def faces(k, QL, QR, AL, AR, p, mesh):
        """Face kernel `k` on every face with its unit normal."""
        return k(QL, QR, AL, AR, p, mesh.face_normals)

    def face_aux(Qaux, mesh):
        """Owner and neighbour aux per face (the inner cell twice at a boundary face)."""
        return Qaux[:, j(fc0)], Qaux[:, j(fc_other)]

    def face_wavespeed(k, Q, Qaux, p, mesh):
        """Max |eigenvalue| per face over the two adjacent cells."""
        n = mesh.face_normals
        evA = jnp.abs(k(Q[:, j(fc0)], Qaux[:, j(fc0)], p, n))
        evB = jnp.abs(k(Q[:, j(fc_cfl)], Qaux[:, j(fc_cfl)], p, n))
        if evA.ndim > 1:
            axes = tuple(range(evA.ndim - 1))
            evA, evB = jnp.max(evA, axis=axes), jnp.max(evB, axis=axes)
        return jnp.maximum(evA, evB)

    def face_inradius(mesh):
        """Per-face cell size: min inradius of the two adjacent cells."""
        ir = mesh.cell_inradius
        return jnp.minimum(ir[j(fc0)], ir[j(fc_cfl)])

    # ── boundary ───────────────────────────────────────────────────────────
    def _bc_at(bc, states, Qaux, p, time, mesh, cells_of_face, ghost=True):
        """The BC kernel `bc` on every non-cut boundary face: the ghost kernel at the mirror image
        of the cell centre, or (`ghost=False`) the face kernel at the face centre.
        """
        # ``ghost=True``: the GHOST kernel (boundary_ghost_state) of the inner
        # cell, at the mirror image of the cell centre -- evaluated at the
        # foot of the perpendicular, with the normal distance.  ``False``:
        # the FACE kernel (boundary_conditions) of the inner trace, at the
        # face centre.  (Equal on orthogonal cells; on triangles the face
        # centre made data-dependent Dirichlet ghosts O(h) wrong: boundary
        # first derivatives did not converge.)
        fc, n = mesh.face_centers, mesh.face_normals
        cc = mesh.cell_centers[:, j(bf_cells)].T
        r = fc[j(bf_face), :] - cc
        nb = n[:, j(bf_face)].T
        dn = jnp.sum(r * nb, axis=1)
        pos = cc + dn[:, None] * nb if ghost else fc[j(bf_face), :]
        dist = dn if ghost else jnp.linalg.norm(r, axis=1)

        def per_bf(k, i):
            """BC kernel of branch `k` on boundary face `i`."""
            f = j(bf_face)[i]
            return bc(branch[k], time, pos[i], dist[i], states[:, i],
                      Qaux[:, j(cells_of_face)[i]], p, n[:, f])
        if not cut.any():
            return jax.vmap(per_bf)(jnp.arange(n_bf), jnp.arange(n_bf)).T
        if not len(phys):                   # a piece inside the domain
            return states
        out = jax.vmap(per_bf)(jnp.arange(len(phys)), j(phys)).T
        return states.at[:, j(phys)].set(out)

    def boundary_ghosts(ghost, Q, Qaux, p, time, mesh):
        """The GHOST (`boundary_ghost_state`) per boundary face."""
        return _bc_at(ghost, Q[:, j(bf_cells)], Qaux, p, time, mesh, bf_cells)

    def boundary_states(ghost, Q, Qaux, p, time, mesh):
        """The GHOST per boundary face and the inner cell's aux."""
        if n_bf == 0:
            return (jnp.zeros((Q.shape[0], 1)), jnp.zeros((Qaux.shape[0], 1)))
        return (_bc_at(ghost, Q[:, j(bf_cells)], Qaux, p, time, mesh, bf_cells),
                Qaux[:, j(inner)])

    def boundary_traces(bc, Q, QL, QR, Qaux, p, time, mesh):
        """`QR` with the outer FACE state `bc(QL)` on the boundary faces (periodic seam: the
        partner face's `QL`).
        """
        if n_bf == 0:
            return QR
        QRb = _bc_at(bc, QL[:, j(bf_face)], Qaux, p, time, mesh, bf_cells, ghost=False)
        if has_periodic:
            QRb = jnp.where(j(periodic)[None, :], QL[:, j(partner_face)], QRb)
        return QR.at[:, j(bf_face)].set(QRb)

    # ── reconstruction and assembly ────────────────────────────────────────
    def pointwise(k, X, Xaux, p):
        """Kernel `k` in every column of `X`."""
        return k(X, Xaux, p)

    def _inner_outside(WL, WR):
        """A boundary face's outer slot holds the inner trace until `boundary_traces` sets the BC
        state.
        """
        # a boundary face's outer slot holds the inner trace until
        # boundary_traces sets the BC's face state
        return WL, WR.at[:, j(bf_face)].set(WL[:, j(bf_face)])

    def cell_traces(Q, mesh):
        """First-order traces: owner and neighbour cell value per face."""
        return _inner_outside(*order1(Q))

    # the boundary closure (ANALYSIS 40): the class of each row per space,
    # the closure cells; the derivatives there replace the ghost-sample fit's
    owner = getattr(nsm, "_symbolic_name", None) or "nsm"
    closure_cells = (topo.lsq_plan.get("closure") or {}).get("cells")
    recon_classes = topo.lsq_closure_class.get((owner, "reconstruction"), ())

    def closed(D, U, Ug, classes, cols, mesh):
        """`D` `(rows, len(cols), cells)` with the closure fits written at the closure cells for
        the rows that have a class (monomial columns `cols`).
        """
        rows, Dc = lsq_closure_derivatives(topo, mesh.lsq_closure, U, Ug, classes)
        if not rows:
            return D
        r, c = np.asarray(rows), j(closure_cells)
        vals = jnp.transpose(Dc[:, :, np.asarray(cols)], (0, 2, 1))
        return D.at[r[:, None, None], np.arange(len(cols))[None, :, None],
                    c[None, None, :]].set(vals)

    def gradient(W, Wg, mesh):
        """LSQ gradient of every row of `W` per cell, closed at the boundary by constraints or by
        the ghost samples `Wg`.

        References:
            Mavriplis 2003, https://doi.org/10.2514/6.2003-3986
        """
        # ``Wg``: the ghosts; a row the BC fixes closes by its constraint,
        # every other row samples the ghost at the mirror image
        R = _with_geometry(lsq, mesh, mesh_np, nc)
        G = R._compute_gradients(W, W.shape[0], Wg)
        return closed(G, W, Wg, recon_classes, topo.lsq_grad_cols, mesh)

    def ghost_bounds(R, u, ug):
        """Min and max of `u` over the face neighbours and the boundary ghosts `ug` (a ghost
        bounds its face's owner).
        """
        # min/max over the face neighbours and the boundary ghosts ``ug``
        # (as in ``gradient``).  A ghost bounds its face's OWNER (on a
        # periodic seam ``_bf_cells`` is the partner cell)
        lo = hi = u
        for a, b in ((R._iA_int, R._iB_int), (R._iB_int, R._iA_int)):
            lo, hi = lo.at[a].min(u[b]), hi.at[a].max(u[b])
        return lo.at[R._iInner_bnd].min(ug), hi.at[R._iInner_bnd].max(ug)

    def limiter(name):
        """Limiter factory for the reconstruction method `name`; the factors lie in [0, 1],
        bounded by the face neighbours and the ghosts.

        References:
            Barth, Jespersen 1989, https://doi.org/10.2514/6.1989-366
            Venkatakrishnan 1995, https://doi.org/10.1006/jcph.1995.1084
            Sweby 1984, https://doi.org/10.1137/0721062
        """
        def factors(rows, W, Wg, G, mesh):
            """Limiter factors for the state `rows` (1 elsewhere)."""
            R = _with_geometry(lsq, mesh, mesh_np, nc)
            fn = getattr(R, name)
            phi = jnp.ones((W.shape[0], nc), dtype=W.dtype)
            for v in rows:
                lo, hi = ghost_bounds(R, W[v], Wg[v])
                phi = phi.at[v].set(fn(W[v], G[v], lo, hi))
            return phi
        return factors

    def extrapolate(W, G, phi, mesh):
        """Each cell's `W` moved to the face centres by `phi * G` (MUSCL).

        References:
            van Leer 1979, https://doi.org/10.1016/0021-9991(79)90145-1
        """
        R = _with_geometry(lsq, mesh, mesh_np, nc)
        phi = jnp.asarray(phi, dtype=W.dtype)
        phi = jnp.broadcast_to(phi if phi.ndim != 1 else phi[None, :],
                               (W.shape[0], nc))
        return _inner_outside(*R._reconstruct(W, G, phi, None))

    # owner of every face; neighbour, or nc (dropped) on a boundary face
    nbr = np.where(is_bnd, nc, fc1)

    def divergence(Wm, Wp, mesh):
        """Per cell the sum of `Wm` over the faces it owns and `Wp` over the faces it
        neighbours, times |face| / |cell|.

        References:
            LeVeque 2002, https://doi.org/10.1017/CBO9780511791253
        """
        fv, cv = mesh.face_volumes, mesh.cell_volumes
        out = jnp.zeros((Wm.shape[0], nc), dtype=Wm.dtype)
        out = out.at[:, j(fc0)].add(Wm * (fv / cv[j(fc0)]))
        return out.at[:, j(nbr)].add(Wp * (fv / cv[j(np.minimum(nbr, nc - 1))]),
                                     mode="drop")

    def edge_source(k, Q, QL, QR, Qaux, p, mesh):
        """Per cell the interior nonconservative edge integral of kernel `k` over its faces,
        divided by the volume (order >= 2).
        """
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
    def derivative_aux(spec, Q, Qaux, mesh, Qg=None, rows=None):
        """`Qaux` with the derivative rows of `spec` refilled from mesh gradients; with `Qg`
        and `rows` only the derivatives of those state rows, their boundary faces from the ghosts.
        """
        # stencils are static topology; the LSQ matrices are traced geometry
        traced = dataclasses.replace(mesh_j, lsq_gradQ=mesh.lsq_gradQ)
        if Qg is None:
            return _Reference._walk_derivative_aux(None, Qaux, Q, traced,
                                                   registry=spec)
        mis = [tuple(int(k) for k in mi) for mi in topo.lsq_monomial_multi_index]
        for e in spec:            # boundary faces: the ghosts Qg
            if e["kind"] != "derivative" or e["target_kind"] != "state":
                continue
            s = e["state_index"]
            if s not in rows:
                continue
            grad = lsq_gradient_per_field(traced, Q[s], u_bf=Qg[s],
                                          multi_index=e["multi_index"])
            col = mis.index(tuple(int(k) for k in e["multi_index"]))
            grad = closed(grad[None, None], Q[s][None], Qg[s][None],
                          (e["closure"],), (col,), mesh)[0, 0]
            Qaux = Qaux.at[e["row"], :grad.shape[0]].set(grad)
        return Qaux

    def clean_round_off(Q, rows, tol):
        """Entries of the state `rows` in `[-tol, 0)` set to 0."""
        r = jnp.asarray(rows)
        X = Q[r]
        return Q.at[r].set(jnp.where((X < 0) & (X >= -tol), 0.0, X))

    def put(Q, rows, values):
        """`Q` with the state `rows` replaced by `values`."""
        return Q.at[jnp.asarray(rows), :].set(values)

    def zeros(n, mesh):
        """An `(n, cells)` array of zeros."""
        return jnp.zeros((int(n), nc), dtype=mesh.cell_volumes.dtype)

    def inadmissible(k, Q, Qaux, p):
        """Cell mask: an admissibility row of kernel `k` below 0, or a non-finite state."""
        adm = k(Q, Qaux, p)
        return (jnp.any(adm < 0.0, axis=0)
                | ~jnp.isfinite(Q).all(axis=0))

    def select(mask, A, B):
        """`A` where `mask` else `B`, per cell or per face."""
        mask, A, B = jnp.asarray(mask), jnp.asarray(A), jnp.asarray(B)
        if mask.ndim == 1 and max(A.ndim, B.ndim) == 2:
            mask = mask[None, :]
        return jnp.where(mask, A, B)

    # ── control ────────────────────────────────────────────────────────────
    def while_loop(c, b, init):
        """`lax.while_loop`; with `AdMode.reverse` a binomially checkpointed loop that
        `jax.grad` can differentiate.

        References:
            Griewank 1992, https://doi.org/10.1080/10556789208805505
        """
        init = tuple(jnp.asarray(a, dtype=jnp.asarray(a).dtype) for a in init)
        cf, bf = (lambda s: c(*s)), (lambda s: tuple(b(*s)))
        if ad_mode == "reverse":
            import equinox.internal as eqxi
            return eqxi.while_loop(cf, bf, init, kind="checkpointed",
                                   checkpoints=checkpoints)
        return lax.while_loop(cf, bf, init)

    def cond(p, a, b):
        """`lax.cond` of the two branches."""
        return lax.cond(p, a, b)

    imp = {**dict(tol=1e-8, maxiter=6, gmres_tol=1e-7, gmres_maxiter=30),
           **(implicit or {})}

    def root(f, x0):
        """Newton on `f(x) = 0` with a per-cell (block-diagonal) Jacobian; differentiable through
        the converged root (custom_root).
        """
        return newton_root(f, x0, lambda x, v: jax.jvp(f, (x,), (v,))[1],
                           cellwise_solve, tol=imp["tol"], maxiter=imp["maxiter"])

    ls = imp.get("linsolve") or dict(tol=imp["gmres_tol"], restart=20,
                                     maxiter=imp["gmres_maxiter"], accept=None)

    def _ls_gmres(A, b):
        """Matrix-free restarted GMRES of `A x = b` with the configured tolerances.

        References:
            Saad, Schultz 1986, https://doi.org/10.1137/0907058
        """
        return jax_gmres(A, b, atol=0.0, tol=ls["tol"], restart=ls["restart"],
                         maxiter=ls["maxiter"])[0]

    def _ls_reject(r):
        """Raise when the relative residual `r` exceeds the accept gate."""
        if float(r) > ls["accept"]:
            raise RuntimeError(
                f"linsolve: relative residual {float(r):.3e} > accept gate "
                f"{ls['accept']:.1e} (target {ls['tol']:.1e}, restart "
                f"{ls['restart']} x maxiter {ls['maxiter']})")

    amgx = _jaxamg() if ls.get("backend") == "amgx" else None
    pyamg = _pyamg() if ls.get("backend") == "pyamg" else None
    hyp = _hypre() if ls.get("backend") == "hypre" else None
    patterns = {}
    lsq_nbrs = (np.asarray(mesh_np.lsq_neighbors)
                if amgx is not None or pyamg is not None or hyp is not None else None)

    def assembled(op, n, ncells=nc, n_rows=None):
        """Values of the linear operator `op` on the cell pattern of the first `ncells` cells
        (rows: the first `n_rows` cells, default all), by colouring: one traced evaluation per
        colour.

        References:
            Griewank, Walther 2008, https://doi.org/10.1137/1.9780898717761
        """
        if (n, ncells, n_rows) not in patterns:
            patterns[(n, ncells, n_rows)] = cell_pattern(n, ncells, lsq_nbrs, n_rows)
        rows, cols, colors = patterns[(n, ncells, n_rows)]
        probes = np.arange(int(colors.max()) + 1)[:, None] == colors[None, :]
        # one evaluation per colour, a few at a time: all at once holds every
        # probe's intermediates together (33 GiB at 2.95M cells)
        batch = int(np.clip(4e7 // probes.shape[1], 1, len(probes)))
        AV = jax.lax.map(op, jnp.asarray(probes, dtype=jnp.float64), batch_size=batch)
        return AV[colors[cols], rows]

    amg_cache = {}                   # (n, transpose) -> hierarchy and its age

    def host_amg(rows, cols, n, k, transpose):
        """Host solve: GMRES preconditioned by a lagged smoothed-aggregation AMG hierarchy of the
        block-scaled matrix (`numpy_backend.assembly.host_amg_solve`, shared with numpy).

        References:
            Saad, Schultz 1986, https://doi.org/10.1137/0907058
            Bell, Olson, Schroder 2022, https://doi.org/10.21105/joss.04142
        """
        def solve(vals, rhs):
            """Solve with matrix values `vals` and right-hand side `rhs`."""
            return host_amg_solve(vals, rhs, rows, cols, n, k, ls, pyamg,
                                  amg_cache.setdefault((n, transpose), {}), transpose)
        return solve

    def host_vcycle(rows, cols, n, k, transpose):
        """Host preconditioner: one smoothed-aggregation V-cycle on the block-scaled matrix, with a
        lagged hierarchy rebuilt after `amg_rebuild` new matrices.

        References:
            Bell, Olson, Schroder 2022, https://doi.org/10.21105/joss.04142
        """
        import scipy.sparse as sps
        c = amg_cache.setdefault(("vcycle", n, transpose),
                                 {"vals": None, "age": 0, "P": None, "Dinv": None})
        rebuild = int(ls.get("amg_rebuild", 1))

        def apply(vals, rhs):
            """Apply the V-cycle for the matrix values `vals` to `rhs`."""
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

    par = {}                      # distributed: halo + global reductions
    if partition is not None:
        from zoomy_jax.symsolver.parallel import blocks, krylov, no_halo
        own_cell, gsum = krylov(partition, nc)
        n_own = int(partition.n_own)
        amgx_comm = partition.comm.Clone() if amgx is not None else None
        hypre_comm = partition.comm.Clone() if hyp is not None else None
        par = blocks(partition, mesh_np)
    halo = par.get("halo", lambda X, mesh: X)

    def _amgx_mpi(A, b):
        """Distributed 'amgx': one global AmgX solve across the ranks, each handing over its owned
        rows with globally numbered columns.  Called by `linsolve` for a partitioned run.

        References:
            Naumov et al. 2015, https://doi.org/10.1137/140980260
        """
        from jax.experimental import sparse as jsparse
        shape, k = b.shape, b.shape[0]
        with no_halo():
            _, lin_o = jax.linearize(A, jnp.zeros_like(b))
        to_cell = lambda X: X.T.reshape(-1)                         # noqa: E731
        to_row = lambda x: x.reshape(-1, k).T                       # noqa: E731
        n_l, key = k * nc, ("mpi", k)
        if key not in patterns:
            rows, cols, colors = cell_pattern(n_l, nc, lsq_nbrs, n_own)
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

    def _hypre_solve(A, b):
        """'hypre': one host call per solve (`zoomy_jax.symsolver.hypre.BlockRows`,
        collective on all ranks), FlexGMRES + BoomerAMG on the cell-block-scaled rows to the true
        relative residual `tol`; the owned rows with global columns by colouring (as
        `_amgx_mpi`); differentiable via `custom_linear_solve`.

        References:
            Henson, Yang 2002, https://doi.org/10.1016/S0168-9274(01)00115-5
            Saad 1993, https://doi.org/10.1137/0914028
        """
        shape, k = b.shape, b.shape[0]
        dist = partition is not None
        n_r = n_own if dist else nc
        _, lin = jax.linearize(A, jnp.zeros_like(b))
        lin_o = lin
        if dist:
            with no_halo():                     # the rank's rows, halo columns as probed
                _, lin_o = jax.linearize(A, jnp.zeros_like(b))
        to_cell = lambda X: X.T.reshape(-1)                         # noqa: E731
        to_row = lambda x: x.reshape(-1, k).T                       # noqa: E731
        vals = assembled(lambda x: to_cell(lin_o(to_row(x))), k * nc, nc, n_r if dist else None)
        rows, cols, _ = patterns[(k * nc, nc, n_r if dist else None)]
        if ("hypre", k) not in patterns:
            from mpi4py import MPI
            gids = np.asarray(partition.gids, np.int64) if dist else np.arange(nc)
            lo = int(partition.offsets[partition.rank]) if dist else 0
            patterns[("hypre", k)] = hyp.BlockRows(
                hypre_comm if dist else MPI.COMM_SELF, k, n_r, lo, rows,
                gids[cols // k] * k + cols % k, ls["tol"], ls["restart"], ls["maxiter"],
                rebuild=ls.get("amg_rebuild", 1), growth=ls.get("amg_growth", 2.0))
        system = patterns[("hypre", k)]
        maskf = (jnp.broadcast_to(own_cell, shape).reshape(-1).astype(b.dtype) if dist
                 else jnp.ones(b.size, b.dtype))
        mv = lambda x: maskf * lin(x.reshape(shape)).reshape(-1)    # noqa: E731
        out = jax.ShapeDtypeStruct((k * n_r,), b.dtype)

        def solver(transpose):
            """The host solve for `custom_linear_solve` (`transpose`: the adjoint)."""
            def host(v, r, nb2):
                """Solve with matrix values `v`, owned right-hand side `r`, `|r|^2 = nb2`."""
                return system.solve(v, r, transpose, nb=float(np.sqrt(nb2)))

            def solve(_, rhs):
                """Owned rows to the host and back; distributed: between two allreduces (data
                dependencies), so the host collectives keep one place among mpi4jax's ordered
                ones on every rank."""
                r = to_cell(rhs.reshape(shape)[:, :n_r])
                nb2 = gsum(jnp.sum(r ** 2)) if dist else jnp.sum(r ** 2)
                xo = jax.pure_callback(host, out, vals, r, nb2)
                if dist:
                    gsum(jnp.sum(xo[:1]))
                return jnp.zeros(shape, b.dtype).at[:, :n_r].set(to_row(xo)).reshape(-1)
            return solve
        bf = b.reshape(-1) * maskf
        x = lax.custom_linear_solve(
            mv, bf, solve=solver(False),
            transpose_solve=solver(True) if ad_mode == "reverse" else None)
        if ls["accept"] is not None:            # the global relative residual
            xs, bs = lax.stop_gradient(x), lax.stop_gradient(bf)
            res = mv(xs) - bs
            sq = (lambda v: gsum(jnp.sum(v ** 2))) if dist else (lambda v: jnp.sum(v ** 2))
            r = jnp.sqrt(sq(res) / jnp.maximum(sq(bs), jnp.finfo(b.dtype).tiny))
            lax.cond(r > ls["accept"], lambda: jax.debug.callback(_ls_reject, r),
                     lambda: None)
        return x.reshape(shape)

    def _dlinsolve(A, b):
        """Distributed 'gmres' / 'pyamg': flexible GMRES over the owned cells with global inner
        products, right-preconditioned for 'pyamg' by the rank's own PyAMG V-cycle plus a coarse
        correction.

        References:
            Saad 1993, https://doi.org/10.1137/0914028
            Bell, Olson, Schroder 2022, https://doi.org/10.21105/joss.04142
        """
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
            rows, cols, _ = patterns[(n_o, n_own, None)]
            out = jax.ShapeDtypeStruct((n_o,), b.dtype)

            def precond(transpose):
                """Preconditioner of the (transposed) solve: the rank-local V-cycle."""
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
                """Restrict a vector to the coarse space: per-rank sums of each row."""
                loc = jnp.sum(v.reshape(shape) * own_f, axis=1)        # (k,)
                return gsum(jnp.zeros((P, k), b.dtype).at[rank].set(loc))

            def coarse(transpose):
                """Coarse-correction solve on the space of one constant per rank and row."""
                def column(c, _):
                    """Column `c` of the coarse matrix: the operator applied to that coarse basis
                    vector, restricted.
                    """
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
            """FGMRES solve for `custom_linear_solve` (`transpose`: the adjoint)."""
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
        """Solve `A x = b` for the linear map `A` (linearized once at 0) by GMRES, AmgX, hypre
        or AMG-preconditioned GMRES, as configured; differentiable via `custom_linear_solve`.
        Aborts above the accept gate; a partitioned run uses `_hypre_solve`,
        `_dlinsolve` or `_amgx_mpi`.

        References:
            Saad, Schultz 1986, https://doi.org/10.1137/0907058
            Naumov et al. 2015, https://doi.org/10.1137/140980260
        """
        if hyp is not None:
            return _hypre_solve(A, b)
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
            rows, cols, _ = patterns[(n, nc, None)]
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
            rows, cols, _ = patterns[(n, nc, None)]
            out = jax.ShapeDtypeStruct(b.shape, b.dtype)

            def amg(transpose):
                """Host AMG solve (`transpose`: the adjoint)."""
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

    tpfa = []                     # the TPFA geometry of div(A : grad Q)
    # a boundary face's outer value: the partner cell across a periodic seam
    # (its own normal distance to the seam adds), else the ghost; cut faces
    # (distributed) carry no flux
    flux_bf = np.flatnonzero(~cut)
    outer_face = np.where(periodic, partner_face, bf_face)

    def _operator():
        """The TPFA operator and the static geometry of its interior faces: owner, neighbour,
        normal, unit cell-to-cell vector `e`, distance, floored `|n.e|`, `|f| / |V|` of both.
        """
        if not tpfa:
            op = DenseDiffusionOperatorJAX(mesh_np, mesh_np.dimension, rt.n_state)
            cc = np.asarray(mesh_np.cell_centers)[:dim]
            f = np.flatnonzero((fc0 < nc) & (fc1 < nc))
            dx = cc[:, fc1[f]] - cc[:, fc0[f]]
            dist = np.sqrt((dx ** 2).sum(axis=0))
            keep = dist >= 1e-30
            f, dx, dist = f[keep], dx[:, keep], dist[keep]     # the operator's faces
            e = dx / dist
            n = np.asarray(mesh_np.face_normals)[:dim, f]
            cosf = np.maximum(np.abs((n * e).sum(axis=0)), TPFA_COS_FLOOR)
            fv, cv = np.asarray(mesh_np.face_volumes), np.asarray(mesh_np.cell_volumes)
            tpfa.append((op, dict(ia=fc0[f], ib=fc1[f], n=n, e=e, dist=dist, cosf=cosf,
                                  wa=fv[f] / cv[fc0[f]], wb=fv[f] / cv[fc1[f]])))
        return tpfa[0]

    def _state_gradient(X, Xg, mesh):
        """`(rows, dim, cells)` LSQ gradient of the rows of `X`: the derivative-aux fit
        (`lsq_gradQ`), the boundary faces sampled at the ghosts `Xg`.
        """
        traced = dataclasses.replace(mesh_j, lsq_gradQ=mesh.lsq_gradQ)
        mis = [tuple(int(v) for v in topo.lsq_monomial_multi_index[c])
               for c in topo.lsq_grad_cols]
        G = jnp.stack([jnp.stack([lsq_gradient_per_field(
            traced, X[s], u_bf=None if n_bf == 0 else Xg[s], multi_index=mi)
            for mi in mis]) for s in range(X.shape[0])])
        return halo(G.reshape(-1, nc), mesh).reshape(G.shape)

    def _div(k, bc, X, Qaux, p, time, mesh, correction=True):
        """`(D, C)` per cell of `div(A(X) : grad X)`: `D` the two-point fluxes (TPFA on
        interior faces, to the BC ghost of `bc` or the periodic partner on boundary faces),
        `C` the rest of the flux `n.A.grad_f` (zero without `correction` or in 1-D).

        `grad_f` is the mean of the two cells' LSQ gradients plus the normal correction
        `((X_b - X_a) / |d| - mean.e) e`, `e` the unit cell-to-cell vector (`n` on boundary
        faces: the tangential part of the cell gradients).

        References:
            Eymard, Gallouet, Herbin 2000, https://doi.org/10.1016/s1570-8659(00)07005-8
        """
        op, geo = _operator()
        A = op._as_cell_tensor(k(X, Qaux, p))          # (n_eq, n_st, nc, d, d)
        D = op._divergence(X, A, bf_grads=None)
        C = jnp.zeros_like(D)
        correction = correction and dim >= 2
        if not len(flux_bf) and not correction:
            return D, C
        Xg = _bc_at(bc, X[:, j(bf_cells)], Qaux, p, time, mesh, bf_cells) if n_bf else None
        if correction:
            G = _state_gradient(X, Xg, mesh)            # (n_st, d, nc)
            ia, ib, nI, e = (j(geo[k]) for k in ("ia", "ib", "n", "e"))
            m = jnp.einsum("ijfde,df->ijfe", 0.5 * (A[:, :, ia] + A[:, :, ib]), nI)
            g = 0.5 * (G[:, :, ia] + G[:, :, ib])
            t = g - jnp.einsum("jef,ef->jf", g, e)[:, None, :] * e[None]
            s = (jnp.einsum("ijfe,ef->ijf", m, e)
                 - jnp.einsum("ijfe,ef->ijf", m, nI) / j(geo["cosf"]))
            F = (jnp.einsum("ijfe,jef->if", m, t)
                 + jnp.einsum("ijf,jf->if", s, (X[:, ib] - X[:, ia]) / j(geo["dist"])))
            C = C.at[:, ia].add(F * j(geo["wa"])).at[:, ib].add(-F * j(geo["wb"]))
        if not len(flux_bf):
            return D, C
        f, c, o = (j(a[flux_bf]) for a in (bf_face, inner, bf_cells))
        Xout = Xg[:, j(flux_bf)]
        fc, cc, nrm = mesh.face_centers, mesh.cell_centers, mesh.face_normals

        def dn(faces, cells):
            """Normal distance from the cell centres `cells` to the faces `faces`."""
            return jnp.sum((fc[faces, :] - cc[:, cells].T) * nrm[:, faces].T, axis=1)
        d = dn(f, c) + dn(j(outer_face[flux_bf]), o)
        n = nrm[:dim, f]
        Af = 0.5 * (A[:, :, c] + A[:, :, o])
        T = jnp.einsum("ijbde,db,eb->ijb", Af, n, n)
        w = mesh.face_volumes[f] / (mesh.cell_volumes[c] * d)
        D = D.at[:, c].add(jnp.einsum("ijb,jb->ib", T, Xout - X[:, c]) * w)
        if correction:                                  # tangential part of the gradients
            g = 0.5 * (G[:, :, c] + G[:, :, o])
            t = g - jnp.einsum("jeb,eb->jb", g, n)[:, None, :] * n[None]
            F = jnp.einsum("ijbde,db,jeb->ib", Af, n, t)
            C = C.at[:, c].add(F * mesh.face_volumes[f] / mesh.cell_volumes[c])
        return D, C

    def diffusion(k, bc, Q, Qaux, p, time, mesh):
        """Explicit `div(A : grad Q)` per cell (RHS sign, A > 0 diffuses)."""
        D, C = _div(k, bc, Q, Qaux, p, time, mesh)
        return D + C

    def _norm(r):
        """Euclidean norm of `r` (distributed: over the owned cells of all ranks)."""
        if partition is None:
            return jnp.linalg.norm(r)
        return jnp.sqrt(gsum(jnp.sum(jnp.where(own_cell, r, 0.0) ** 2)))

    def implicit_diffusion(k, bc, Q, Qaux, p, time, dt, mesh):
        """One Crank-Nicolson step of the two-point part of `div(A(X) : grad X)` from `time`
        to `time + dt`, A at the new state, solved by Newton with a `linsolve` per update; the
        correction explicit at `Q` (deferred). Differentiable through the root.

        References:
            Knoll, Keyes 2004, https://doi.org/10.1016/j.jcp.2003.08.010
            Crank, Nicolson 1947, https://doi.org/10.1017/s0305004100023197
        """
        D, C = _div(k, bc, Q, Qaux, p, time, mesh)
        b = Q + 0.5 * dt * (D + 2.0 * C)

        def R(X):
            """Crank-Nicolson residual `X - b - dt/2 D(X)` at `time + dt`."""
            X = halo(X, mesh)
            return X - b - 0.5 * dt * _div(k, bc, X, Qaux, p, time + dt, mesh,
                                           correction=False)[0]
        X = newton_root(R, Q, lambda x, v: jax.jvp(R, (x,), (v,))[1], linsolve,
                        tol=imp["tol"], maxiter=imp["maxiter"], norm=_norm)
        return halo(X, mesh)

    # ── effects ────────────────────────────────────────────────────────────
    warned = []

    def check_cfl(tok, dt_cfl, dt):
        """Warn once (host callback) when `dt` exceeds the CFL step `dt_cfl`; returns the next
        order token.
        """
        def host(a, b):
            """Log the one-time warning."""
            if float(b) > float(a) * (1 + 1e-12) and not warned:
                warned.append(True)
                logger.warning(f"time step {float(b):.3e} exceeds the CFL step "
                               f"{float(a):.3e} (warned once)")
        a, b = lax.stop_gradient(dt_cfl), lax.stop_gradient(dt)
        lax.cond(b > a * (1 + 1e-12), lambda: jax.debug.callback(host, a, b),
                 lambda: None)
        return (tok + 1,)

    def io_write(tok, time, Q, Qaux):
        """Forward mode: write a snapshot through `on_write`; returns the next order token."""
        if on_write is not None and ad_mode == "forward":
            jax.debug.callback(on_write, time, Q, Qaux, ordered=True)
        return (tok + 1,)

    def reset_lagged():
        """Restart the lagged AMG hierarchies, as in a run restarted at this state."""
        for key, c in amg_cache.items():
            if key[0] == "vcycle":
                c.update(vals=None, age=0, P=None, Dinv=None)
            else:
                c.clear()

    def io_save_state(tok, time, Q, Qaux):
        """Forward mode: save the march state through `on_save` (host); returns the next token."""
        if on_save is not None and ad_mode == "forward":
            def host(*a):
                """Save, then restart the lagged solver caches."""
                on_save(*a)
                reset_lagged()
            jax.debug.callback(host, *map(lax.stop_gradient, (time, Q, Qaux)), ordered=True)
        return (tok + 1,)

    def io_load_state(tok, time, Q, Qaux):
        """The march state from `on_load` (a restart, ordered host call), else the arguments."""
        if on_load is None:
            return (tok + 1, time, Q, Qaux)
        from jax.experimental import io_callback

        args = tuple(map(lax.stop_gradient, (jnp.asarray(time, Q.dtype), Q, Qaux)))
        shapes = tuple(jax.ShapeDtypeStruct(jnp.shape(x), x.dtype) for x in args)

        def host(*a):
            """Read the saved state, shaped as the arguments."""
            out = on_load(*a)
            reset_lagged()
            return tuple(np.asarray(x, dtype=s.dtype) for x, s in zip(out, shapes))
        return (tok + 1, *io_callback(host, shapes, *args, ordered=True))

    def with_closure(name, sub):
        """The derivative-aux spec of system `name`, each state entry tagged with its
        boundary-closure class.
        """
        # the spec, each state entry with its row's closure class
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
        boundary_ghosts=boundary_ghosts,
        pointwise=pointwise, cell_traces=cell_traces, gradient=gradient,
        minmod=limiter("_limit_minmod"),
        venkatakrishnan=limiter("_limit_vk"),
        barth_jespersen=limiter("_limit_bj"),
        extrapolate=extrapolate, divergence=divergence,
        edge_source=edge_source, derivative_aux=derivative_aux,
        inadmissible=inadmissible, reduce_min=jnp.min, reduce_any=jnp.any,
        halo=halo,
        put=put, zeros=zeros, clean_round_off=clean_round_off, take=lambda Q, rows: Q[jnp.asarray(rows)],
        reduce_max=jnp.max, logical_and=jnp.logical_and,
        select=select, while_loop=while_loop, cond=cond, io_write=io_write,
        io_save_state=io_save_state, io_load_state=io_load_state,
        root=root, linsolve=linsolve, implicit_diffusion=implicit_diffusion,
        diffusion=diffusion,
        check_cfl=check_cfl,
        minimum=jnp.minimum, maximum=jnp.maximum, abs=jnp.abs, floor=jnp.floor)
    if couple is not None:
        m.update(_coupling(couple, rt, mesh_np, fc0, nf, partition))
    m.update(par)
    return m


def _fgmres(A, b, dots, M, tol, restart, maxiter):
    """Restarted flexible GMRES of `A x = b` with right preconditioner `M` (may vary per
    iteration) and global inner products `dots(V, w)`.

    References:
        Saad 1993, https://doi.org/10.1137/0914028
    """
    from jax.scipy.linalg import solve_triangular
    m, n, dt = int(restart), b.size, b.dtype
    nrm = lambda v: jnp.sqrt(dots(v[None], v)[0])                  # noqa: E731
    bn = nrm(b)
    target = tol * jnp.where(bn > 0, bn, 1.0)

    def cycle(c):
        """One restart cycle: Arnoldi steps until `|r| <= tol |b|` or `restart` iterations, then
        the least-squares update.
        """
        x, r, it, _ = c
        beta = nrm(r)
        V = jnp.zeros((m + 1, n), dt).at[0].set(r / jnp.where(beta > 0, beta, 1.0))
        g = jnp.zeros(m + 1, dt).at[0].set(beta)

        def arnoldi(st):
            """One flexible Arnoldi step (Gram-Schmidt twice) with a Givens rotation."""
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
                """Apply the `i`-th earlier Givens rotation to `h`."""
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
    """Import the optional JAX-AMG package (AmgX on a CUDA GPU; Liu, Fan, Wang 2026), with a clear
    error when missing or off the GPU.

    References:
        Naumov et al. 2015, https://doi.org/10.1137/140980260
        Liu, Fan, Wang 2026, https://doi.org/10.1016/j.softx.2026.102966
    """
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


def _pyamg():
    """Import the optional PyAMG package.

    References:
        Bell, Olson, Schroder 2022, https://doi.org/10.21105/joss.04142
    """
    try:
        import pyamg
    except ImportError as e:
        raise ImportError("linear solver 'pyamg' needs the optional PyAMG "
                          "package: pip install pyamg.  Or use 'gmres'.") from e
    return pyamg


def _hypre():
    """Import the hypre binding (`zoomy_jax.symsolver.hypre`) and load its library, with a
    clear error when it is missing.

    References:
        Falgout, Yang 2002, https://doi.org/10.1007/3-540-47789-6_66
    """
    from zoomy_jax.symsolver import hypre
    hypre.load()
    return hypre


def _face_kernel(rt, name):
    """Set `rt.<name>` for a Numerics face function JaxRuntime does not build, lambdified like
    `numerical_flux` and vmapped over the faces.
    """
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
        """The face function vmapped over the faces."""
        def one(a, b, c, d, n):
            """One face (the `_speed` variant returns the two sides and the speed)."""
            out = per_face(a, b, c, d, parameters, n)
            return (out[0], out[1], out[2, 0]) if speed else out
        return jax.vmap(one, in_axes=(1, 1, 1, 1, 1), out_axes=-1)(
            qL, qR, qauxL, qauxR, normal)
    setattr(rt, name, kernel)


# ── preCICE ─────────────────────────────────────────────────────────────────
class PreciceAdapter:
    """Host side of the `couple_*` effects: the preCICE API calls only.  `comm` (mpi4py) makes a
    distributed participant, one preCICE process per rank, so it must span the world (one
    `mpiexec` per participant).

    References:
        Chourdakis et al. 2022, https://doi.org/10.12688/openreseurope.14445.2
    """

    def __init__(self, name, config, mesh_name, write, read, comm=None):
        """Open participant `name` from `config` on mesh `mesh_name`, writing the data
        `write` and reading `read`.
        """
        import petsc4py  # noqa: F401  (must precede precice)
        import precice
        self.mesh, self.wn, self.rn = mesh_name, list(write), list(read)
        rank, size = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
        if size > 1:
            from mpi4py import MPI
            if MPI.Comm.Compare(comm, MPI.COMM_WORLD) not in (MPI.IDENT, MPI.CONGRUENT):
                raise NotImplementedError("preCICE participant on a sub-communicator "
                                          "(run it as its own mpiexec)")
        self.p = precice.Participant(name, config, rank, size)

    def set_vertices(self, coords):
        """The coupling vertices `(n, >= dim)`, in the order of the data."""
        dim = self.p.get_mesh_dimensions(self.mesh)
        self.ids = self.p.set_mesh_vertices(self.mesh, np.asarray(coords)[:, :dim])

    def _write(self, q):
        """Write the rows of `q` as the write data."""
        for i, n in enumerate(self.wn):
            self.p.write_data(self.mesh, n, self.ids, np.asarray(q[i]))

    def init(self, q):
        """Write the initial data if required, initialise; returns whether coupling is ongoing."""
        if self.p.requires_initial_data():
            self._write(q)
        self.p.initialize()
        return np.bool_(self.p.is_coupling_ongoing())

    def max_dt(self):
        """Largest time step left in the window."""
        return np.float64(self.p.get_max_time_step_size())

    def read(self):
        """The read data of the peer, one row per name."""
        return np.stack([self.p.read_data(self.mesh, n, self.ids, 0.0)
                         for n in self.rn])

    def write(self, q):
        """Write `q`; returns an empty tuple."""
        self._write(q)
        return ()

    def advance(self, dt):
        """Advance by `dt`; returns whether coupling is ongoing."""
        self.p.advance(float(dt))
        return np.bool_(self.p.is_coupling_ongoing())

    def finalize(self):
        """Finalize the participant."""
        self.p.finalize()


def coupled_face_indices(mesh_np, tag):
    """Boundary faces carrying boundary tag `tag` (by name)."""
    names = list(mesh_np.boundary_conditions_sorted_names)
    nums = np.asarray(mesh_np.boundary_face_function_numbers, dtype=int)
    bf_face = np.asarray(mesh_np.boundary_face_face_indices, dtype=int)
    n_bf = int(mesh_np.n_boundary_faces)
    return np.array([bf_face[i] for i in range(n_bf)
                     if names[nums[i]] == tag], dtype=int)


def _coupling(cfg, rt, mesh_np, fc0, nf, partition=None):
    """Coupling blocks and `couple_*` effects of one preCICE participant, on the tagged faces of
    the owned cells.  Each host call takes and returns an order token (distributed: bracketed by an
    MPI reduction), so XLA cannot reorder the calls and all ranks meet their collectives in the same
    order.
    """
    from jax.experimental import io_callback
    a = cfg["adapter"]
    if partition is None:
        n_own, order = int(mesh_np.n_inner_cells), (lambda tok: tok)
    else:
        from mpi4py import MPI
        from zoomy_jax.symsolver.parallel import _mpi4jax
        mpi4jax, comm, n_own = _mpi4jax(), partition.comm.Clone(), int(partition.n_own)
        order = lambda tok: mpi4jax.allreduce(tok, MPI.MAX, comm=comm)  # noqa: E731
    faces_c = coupled_face_indices(mesh_np, cfg["tag"])
    faces_c = faces_c[fc0[faces_c] < n_own]
    a.set_vertices(np.asarray(mesh_np.face_centers)[faces_c])
    cells_c = fc0[faces_c]
    mask = np.zeros(nf, dtype=bool)
    mask[faces_c] = True
    n_state = int(rt.n_state)

    def coupled_faces(mesh):
        """Mask of the coupled faces."""
        return jnp.asarray(mask)

    def interface_states(Q, mesh):
        """States of the cells next to the coupled faces."""
        return Q[:, jnp.asarray(cells_c)]

    def interface_flux(k, kaux, side, Q, Qaux, peer, p, mesh):
        """This participant's row of one Riemann problem posed low -> high on the coupled faces,
        zero elsewhere; the aux is rebuilt from each state alone so both sides pose the same
        problem.
        """
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
        """Effect body: the host call takes the token and returns the next one plus its outputs."""
        def body(tok, *args):
            """The ordered `io_callback` of the host call."""
            def call(k, *x):
                """Host call; returns the incremented token and the outputs."""
                r = host(*x)
                r = r if isinstance(r, tuple) else (r,)
                return (np.int64(k + 1), *r)
            out = io_callback(call, (T, *outs), order(tok), *args)
            return (order(out[0]), *out[1:])
        return body

    return dict(coupled_faces=coupled_faces, interface_states=interface_states,
                interface_flux=interface_flux,
                couple_init=eff(a.init, B), couple_max_dt=eff(a.max_dt, F),
                couple_read=eff(a.read, Qe), couple_write=eff(a.write),
                couple_advance=eff(a.advance, B))


def hoisted_jit(fn):
    """`jax.jit(fn)` with the constants `fn` captures (the mesh index arrays) passed as
    arguments, so compile time and program size stay independent of the mesh.
    """
    cache = {}

    def run(*args):
        """Trace once per argument shapes, then call the jitted jaxpr with its constants."""
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
    """Execute printed source against the name table and return the function `name`; raises if the
    source uses a block the table lacks.
    """
    missing = [n for n in contract() if n not in module and n in src]
    if missing:
        raise NotImplementedError(f"JAX runtime lacks blocks {missing}")
    ns = dict(module)
    exec(compile(src, f"<symsolver:{name}>", "exec"), ns)
    return ns[name]
