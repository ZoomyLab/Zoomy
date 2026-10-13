"""JIT-compatible FVM face reconstruction and diffusion for JAX.

Mirrors the NumPy reconstruction classes in zoomy_core.fvm.reconstruction
with the same interface: `recon(Q) → (Q_L, Q_R)`.

All operations use JAX primitives (jnp.at[].add, jnp.where) for full
JIT and autodiff compatibility.

Classes
-------
- `ConstantReconstruction`: 1st-order piecewise-constant.
- `MUSCLReconstruction`: 2nd-order MUSCL with slope limiting.
- `FreeSurfaceMUSCL`: MUSCL with wet-dry fallback and h-positivity.
- `DiffusionOperatorJAX`: Sparse discrete Laplacian with Crank-Nicolson implicit solve.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from zoomy_jax.fvm.implicit_ad import gmres, newton_root


# ── Reconstruction classes ───────────────────────────────────────────────────

class ConstantReconstruction:
    """First-order piecewise-constant reconstruction (JAX).

    Takes inner-cells-only `Q` and BC-provided `bf_face_values` so
    boundary faces get `Q_R = bf_values` (not a stale ghost cell).
    """

    def __init__(self, mesh, dim):
        fc0 = np.asarray(mesh.face_cells[0])
        fc1 = np.asarray(mesh.face_cells[1])
        bf_face_idx = np.asarray(mesh.boundary_face_face_indices)
        n_faces = int(mesh.n_faces)
        bf_set = set(int(f) for f in bf_face_idx)
        interior_faces = np.array(
            [f for f in range(n_faces) if f not in bf_set], dtype=np.int32)

        self._n_faces = n_faces
        self._interior_faces = jnp.asarray(interior_faces)
        self._boundary_faces = jnp.asarray(bf_face_idx)
        self._iA_int = jnp.asarray(fc0[interior_faces])
        self._iB_int = jnp.asarray(fc1[interior_faces])
        self._iInner_bnd = jnp.asarray(fc0[bf_face_idx])

    def __call__(self, Q, bf_face_values=None):
        n_var = Q.shape[0]
        Q_L = jnp.zeros((n_var, self._n_faces))
        Q_R = jnp.zeros((n_var, self._n_faces))
        Q_L = Q_L.at[:, self._interior_faces].set(Q[:, self._iA_int])
        Q_R = Q_R.at[:, self._interior_faces].set(Q[:, self._iB_int])
        Q_L = Q_L.at[:, self._boundary_faces].set(Q[:, self._iInner_bnd])
        if bf_face_values is not None:
            Q_R = Q_R.at[:, self._boundary_faces].set(bf_face_values)
        return Q_L, Q_R


class MUSCLReconstruction:
    """Second-order MUSCL reconstruction with slope limiting (JAX, JIT-compatible).

    Parameters
    ----------
    mesh : MeshJAX
    dim : int
    limiter : "venkatakrishnan", "barth_jespersen", or "minmod"
    """

    def __init__(self, mesh, dim, limiter="venkatakrishnan"):
        self.dim = dim
        self.n_faces = mesh.n_faces
        self.n_cells = mesh.n_cells
        self.nc = mesh.n_inner_cells
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        self.iA = iA
        self.iB = iB
        self._limiter_type = limiter

        centers = mesh.cell_centers[:dim, :]
        face_ctrs = mesh.face_centers[:, :dim].T  # (n_faces, 3) → (dim, n_faces)

        self.r_Af = face_ctrs - centers[:, iA]    # (dim, n_faces)
        self.r_Bf = face_ctrs - centers[:, iB]

        self._normals = mesh.face_normals[:dim, :]
        self._face_volumes = mesh.face_volumes
        self._cell_volumes = mesh.cell_volumes

        # Venkatakrishnan smoothing: eps² = (K·h)²
        h = jnp.zeros(self.n_cells)
        h = h.at[:self.nc].set(
            self._cell_volumes[:self.nc] ** (1.0 / max(dim, 1))
        )
        self._eps_v2 = (1.0 * h) ** 2

        # Select limiter function (resolved once at init, not per call)
        _limiter_map = {
            "venkatakrishnan": self._limit_vk,
            "barth_jespersen": self._limit_bj,
            "minmod": self._limit_minmod,
            "none": lambda u, grad, u_min, u_max: jnp.ones(self.n_cells),
        }
        self._limiter_fn = _limiter_map[limiter]

    def __call__(self, Q):
        """Reconstruct face states. Returns (Q_L, Q_R), each (n_vars, n_faces)."""
        n_vars = Q.shape[0]
        grads, phi = self._compute_limited_gradients(Q, n_vars)
        return self._reconstruct(Q, grads, phi)

    # ── Green-Gauss gradient (JAX) ───────────────────────────────────

    def _green_gauss_gradient(self, u):
        """Cell-center gradient of scalar u. Returns (dim, n_cells)."""
        iA, iB = self.iA, self.iB
        u_face = 0.5 * (u[iA] + u[iB])  # (n_faces,)

        grad = jnp.zeros((self.dim, self.n_cells))
        for d in range(self.dim):
            contrib = u_face * self._normals[d, :] * self._face_volumes
            grad = grad.at[d, :].add(
                jnp.zeros(self.n_cells).at[iA].add(contrib / self._cell_volumes[iA])
            )
            grad = grad.at[d, :].add(
                jnp.zeros(self.n_cells).at[iB].add(-contrib / self._cell_volumes[iB])
            )
        return grad

    # ── Neighbor bounds (JAX) ────────────────────────────────────────

    def _neighbor_bounds(self, u):
        """Per-cell min/max over face-neighbors.

        Each cell starts at its own value and gets max'd / min'd against
        the value at the *other* side of every face it touches. JAX's
        `.at[idx].max(vals)` does a segment-max when `idx` contains
        duplicates — exactly the neighbour-fold we want, without an
        artificial 0-base that previously clamped the bounds and stripped
        MUSCL of its second-order accuracy on fields that cross zero.
        """
        iA, iB = self.iA, self.iB
        u_max = u.at[iA].max(u[iB]).at[iB].max(u[iA])
        u_min = u.at[iA].min(u[iB]).at[iB].min(u[iA])
        return u_min, u_max

    # ── Face deltas ──────────────────────────────────────────────────

    def _face_deltas(self, grad):
        """Reconstructed increment at face centers. Returns (delta_A, delta_B)."""
        dA = jnp.zeros(self.n_faces)
        dB = jnp.zeros(self.n_faces)
        for d in range(self.dim):
            dA = dA + grad[d, self.iA] * self.r_Af[d, :]
            dB = dB + grad[d, self.iB] * self.r_Bf[d, :]
        return dA, dB

    # ── Limiters (JAX, branchless) ───────────────────────────────────

    def _limit_bj(self, u, grad, u_min, u_max):
        """Barth-Jespersen limiter (JAX). Strict DMP."""
        phi = jnp.ones(self.n_cells)
        eps = 1e-30
        dA, dB = self._face_deltas(grad)

        for cell_ids, deltas in [(self.iA, dA), (self.iB, dB)]:
            uc = u[cell_ids]
            # Branchless: use jnp.where instead of boolean indexing
            dm_pos = u_max[cell_ids] - uc
            dm_neg = u_min[cell_ids] - uc
            cand = jnp.where(
                deltas > eps,
                jnp.minimum(1.0, dm_pos / jnp.maximum(deltas, eps)),
                jnp.where(
                    deltas < -eps,
                    jnp.minimum(1.0, dm_neg / jnp.minimum(deltas, -eps)),
                    1.0
                )
            )
            # Scatter min to cells
            phi = jnp.minimum(phi, jnp.ones_like(phi).at[cell_ids].min(cand))

        return jnp.clip(phi, 0.0, 1.0)

    def _limit_vk(self, u, grad, u_min, u_max):
        """Venkatakrishnan limiter (JAX). Smooth, 2nd-order at extrema."""
        phi = jnp.ones(self.n_cells)
        eps = 1e-30
        ev2 = self._eps_v2
        dA, dB = self._face_deltas(grad)

        for cell_ids, deltas in [(self.iA, dA), (self.iB, dB)]:
            uc = u[cell_ids]
            dm_pos = u_max[cell_ids] - uc
            dm_neg = u_min[cell_ids] - uc

            # Positive delta branch
            num_pos = dm_pos ** 2 + ev2[cell_ids] + 2 * deltas * dm_pos
            den_pos = dm_pos ** 2 + 2 * deltas ** 2 + deltas * dm_pos + ev2[cell_ids]
            phi_pos = jnp.minimum(1.0, num_pos / jnp.maximum(den_pos, eps))

            # Negative delta branch
            num_neg = dm_neg ** 2 + ev2[cell_ids] + 2 * deltas * dm_neg
            den_neg = dm_neg ** 2 + 2 * deltas ** 2 + deltas * dm_neg + ev2[cell_ids]
            phi_neg = jnp.minimum(1.0, num_neg / jnp.maximum(den_neg, eps))

            cand = jnp.where(
                deltas > eps, phi_pos,
                jnp.where(deltas < -eps, phi_neg, 1.0)
            )
            phi = jnp.minimum(phi, jnp.ones_like(phi).at[cell_ids].min(cand))

        return jnp.clip(phi, 0.0, 1.0)

    def _limit_minmod(self, u, grad, u_min, u_max):
        """Minmod limiter (JAX). Most diffusive TVD limiter."""
        phi = jnp.ones(self.n_cells)
        eps = 1e-30
        dA, dB = self._face_deltas(grad)

        for cell_ids, deltas in [(self.iA, dA), (self.iB, dB)]:
            uc = u[cell_ids]
            dm_pos = u_max[cell_ids] - uc
            dm_neg = u_min[cell_ids] - uc
            r_pos = dm_pos / jnp.maximum(deltas, eps)
            r_neg = dm_neg / jnp.minimum(deltas, -eps)
            cand = jnp.where(
                deltas > eps,
                jnp.clip(r_pos, 0.0, 1.0),
                jnp.where(deltas < -eps, jnp.clip(r_neg, 0.0, 1.0), 1.0)
            )
            phi = jnp.minimum(phi, jnp.ones_like(phi).at[cell_ids].min(cand))

        return jnp.clip(phi, 0.0, 1.0)

    # ── Core pipeline ────────────────────────────────────────────────

    def _compute_limited_gradients(self, Q, n_vars):
        """Gradient + limiter for all variables."""
        limiter_fn = self._limiter_fn

        grads = jnp.zeros((n_vars, self.dim, self.n_cells))
        phi = jnp.ones((n_vars, self.n_cells))

        for v in range(n_vars):
            u = Q[v, :]
            g = self._green_gauss_gradient(u)
            grads = grads.at[v].set(g)
            u_min, u_max = self._neighbor_bounds(u)
            p = limiter_fn(u, g, u_min, u_max)
            phi = phi.at[v].set(p)

        return grads, phi

    def _reconstruct(self, Q, grads, phi):
        """Linear reconstruction at face centers."""
        iA, iB = self.iA, self.iB
        Q_L = Q[:, iA]
        Q_R = Q[:, iB]
        for d in range(self.dim):
            Q_L = Q_L + phi[:, iA] * grads[:, d, iA] * self.r_Af[d, :][jnp.newaxis, :]
            Q_R = Q_R + phi[:, iB] * grads[:, d, iB] * self.r_Bf[d, :][jnp.newaxis, :]
        return Q_L, Q_R


class LSQMUSCLReconstructionJAX:
    """LSQ-stencil MUSCL reconstruction (JAX) — mirrors the NumPy
    `zoomy_core.fvm.reconstruction.LSQMUSCLReconstruction`.

    The LSQ stencil + boundary-face augmentation come from the mesh
    (`lsq_gradQ`, `lsq_neighbors`, `lsq_boundary_face_neighbors`,
    `lsq_scale_factors`).  `Q` has shape `(n_var, n_inner_cells)`
    (no ghost cells).  Boundary face values are passed in by the caller
    (typically the flux operator evaluates them via the BC kernel
    before calling `reconstruct`).

    The class returns `(Q_L, Q_R)` of shape `(n_var, n_faces)`:

    - Interior faces: both sides reconstructed from inner cells.
    - Boundary faces: `Q_L` = inner cell reconstructed at the face,
      `Q_R` = the BC-provided face value.

    Limiter coefficients come from neighbor min/max (interior +
    boundary-face values).  Currently supports Venkatakrishnan,
    Barth-Jespersen, and minmod.
    """

    def __init__(self, mesh, dim, limiter="venkatakrishnan",
                 unlimited_indices=None):
        self.dim = dim
        self._limiter_type = limiter
        self._unlimited_indices = (
            tuple(unlimited_indices) if unlimited_indices else ())
        nc = int(mesh.n_inner_cells)
        self._nc = nc
        self._n_faces = int(mesh.n_faces)
        self._n_bf = int(mesh.n_boundary_faces)

        fc0 = np.asarray(mesh.face_cells[0])
        fc1 = np.asarray(mesh.face_cells[1])

        # Split faces into interior and boundary.
        bf_face_set = set(int(f) for f in np.asarray(
            mesh.boundary_face_face_indices))
        interior_faces = np.array(
            [f for f in range(self._n_faces) if f not in bf_face_set],
            dtype=np.int32)
        boundary_faces = np.asarray(
            mesh.boundary_face_face_indices, dtype=np.int32)

        self._interior_faces = jnp.asarray(interior_faces)
        self._boundary_faces = jnp.asarray(boundary_faces)
        self._iA_int = jnp.asarray(fc0[interior_faces])
        self._iB_int = jnp.asarray(fc1[interior_faces])
        # Inner cell at boundary face = face_cells[0, boundary_face].
        self._iInner_bnd = jnp.asarray(fc0[boundary_faces])

        # Cell→face displacement vectors for reconstruction extrapolation.
        centers = np.asarray(mesh.cell_centers)[:dim, :]
        face_ctrs = np.asarray(mesh.face_centers)[:, :dim].T
        self._r_Af_int = jnp.asarray(
            face_ctrs[:, interior_faces] - centers[:, fc0[interior_faces]])
        self._r_Bf_int = jnp.asarray(
            face_ctrs[:, interior_faces] - centers[:, fc1[interior_faces]])
        self._r_Af_bnd = jnp.asarray(
            face_ctrs[:, boundary_faces] - centers[:, fc0[boundary_faces]])

        # Boundary face cell list (for limiter bounds via BC values).
        self._bf_cells = jnp.asarray(mesh.boundary_face_cells, dtype=jnp.int32)

        # LSQ stencil data — pulled straight from the mesh.
        self._lsq_gradQ = jnp.asarray(mesh.lsq_gradQ[:nc])
        self._lsq_neighbors = jnp.asarray(
            mesh.lsq_neighbors[:nc], dtype=jnp.int32)
        # Restrict neighbor indices to valid inner cells (clip out-of-range
        # to a self-loop so the JAX gather is well-defined; the
        # corresponding `delta_u` rows then vanish by construction).
        self._lsq_neighbors = jnp.minimum(self._lsq_neighbors, nc - 1)
        self._lsq_scale = jnp.asarray(mesh.lsq_scale_factors)

        bdy_nbr = getattr(mesh, "lsq_boundary_face_neighbors", None)
        has_bdy = bdy_nbr is not None and np.asarray(bdy_nbr).size > 0
        self._lsq_has_bdy = bool(has_bdy)
        if has_bdy:
            self._lsq_bdy_nbr = jnp.asarray(
                np.asarray(bdy_nbr[:nc], dtype=np.int32))
        else:
            self._lsq_bdy_nbr = None

        # ``mesh.lsq_monomial_multi_index`` lists the polynomial
        # monomial each column of ``lsq_gradQ`` produces.  For degree-1
        # LSQ the multi-indices are the unit vectors ``e_0 = (1, 0, …)``
        # (∂/∂x), ``e_1 = (0, 1, …)`` (∂/∂y), and so on — but their
        # *ordering* in ``lsq_monomial_multi_index`` is **not**
        # guaranteed to be ``[e_0, e_1, …]``.  Empirically in
        # ``LSQMesh.create_2d`` it comes out as ``[(0, 1), (1, 0)]`` —
        # i.e. column 0 of ``lsq_gradQ`` is ``∂/∂y`` and column 1 is
        # ``∂/∂x``.  The NumPy ``compute_derivatives`` consults the
        # multi-index via ``find_derivative_indices`` and permutes;
        # the JAX port was just slicing ``coeffs[:dim]`` and so
        # silently swapped x and y on 2D meshes — making the
        # reconstruction extrapolate the y-slope in the x-direction
        # and vice versa, killing convergence to anything physical.
        # Build the permutation once here.
        mon_idx = np.asarray(mesh.lsq_monomial_multi_index)
        eye = np.eye(dim, dtype=int)
        grad_perm = np.empty(dim, dtype=np.int32)
        for d in range(dim):
            matches = np.where(
                (mon_idx == eye[d]).all(axis=1))[0]
            if matches.size == 0:
                raise RuntimeError(
                    f"LSQ stencil has no monomial for ∂/∂x_{d}; "
                    f"available multi-indices: {mon_idx.tolist()}")
            grad_perm[d] = int(matches[0])
        self._grad_perm = jnp.asarray(grad_perm)

        # Venkatakrishnan smoothing: eps² = h².
        cell_vols = np.asarray(mesh.cell_volumes[:nc])
        h = cell_vols ** (1.0 / max(dim, 1))
        self._eps_v2 = jnp.asarray(h ** 2)

        _limiter_map = {
            "venkatakrishnan": self._limit_vk,
            "barth_jespersen": self._limit_bj,
            "minmod": self._limit_minmod,
            "none": lambda u, grad_v, u_min, u_max: jnp.ones_like(u),
        }
        if limiter not in _limiter_map:
            raise ValueError(
                f"Unknown limiter {limiter!r}; pick one of "
                f"{list(_limiter_map)}.")
        self._limiter_fn = _limiter_map[limiter]

    def __call__(self, Q, bf_face_values):
        """Reconstruct face states.

        Parameters
        ----------
        Q : jnp.ndarray, shape `(n_var, n_inner_cells)`
        bf_face_values : jnp.ndarray, shape `(n_var, n_boundary_faces)`
            BC-provided ghost values (the flux operator computes these
            from the indexed BC kernel before calling reconstruct).

        Returns
        -------
        Q_L, Q_R : each shape `(n_var, n_faces)`.  `Q_R` at
        boundary faces is set to `bf_face_values` (placeholder —
        the flux operator typically overwrites it with
        `BC(Q_L_at_boundary_face)` for consistency with the limiter).
        """
        n_var = Q.shape[0]
        grads = self._compute_gradients(Q, n_var, bf_face_values)
        phi = self._compute_phi(Q, n_var, bf_face_values, grads)
        return self._reconstruct(Q, grads, phi, bf_face_values)

    def reconstruct_with_grad(self, Q, bf_face_values):
        """Like `__call__` but additionally returns the LIMITED cell
        gradient `phi·grads` of shape `(n_var, dim, n_cells)` —
        required by the order ≥ 2 cell-interior non-conservative
        integral (mirrors NumPy `LSQMUSCLReconstruction._limited_grad`)."""
        n_var = Q.shape[0]
        grads = self._compute_gradients(Q, n_var, bf_face_values)
        phi = self._compute_phi(Q, n_var, bf_face_values, grads)
        Q_L, Q_R = self._reconstruct(Q, grads, phi, bf_face_values)
        return Q_L, Q_R, phi[:, jnp.newaxis, :] * grads

    # ── LSQ gradient via vmap over cells ────────────────────────────

    def _lsq_gradient_scalar(self, u, u_bf):
        """Cell-center gradient for a single scalar field via the LSQ
        stencil.  vmap'd over the cell axis inside `_compute_gradients`.

        **Ghost-distance convention: prescribed face values enter the LSQ
        row as `2·(u_face - u_cell)`.**

        The LSQ stencil places its virtual boundary point at the GHOST-CELL
        offset `2·(face - cell)` — the symmetric image of the inner cell
        through the boundary face (`zoomy_core/mesh/base_mesh.py:318,323`
        for the ghost centers; `zoomy_core/mesh/lsq_reconstruction.py:299-319`
        for the stencil rationale, which explicitly rejects face placement as
        over-constraining Neumann-zero rows).  BC kernels, however, evaluate
        at the FACE CENTER.  The bridge between the two conventions is
        `u_ghost = 2·u_face - u_cell`, so the delta carried at the ghost
        position is `2·(u_face - u_cell)`.

        Using the bare face delta places a face-valued sample at the ghost
        position: the boundary-cell gradient is HALVED, boundary cells drop
        to 1st order, and the global order-2 convergence rate is lost.  This
        is invisible to Wall/Extrapolation/periodic BCs and to flat
        lake-at-rest suites (delta ≈ 0 or pure state reflection) — it only
        shows up for a Dirichlet/inflow-prescribed field with a NONZERO
        boundary slope.

        Mirrors the numpy twin `zoomy_core/fvm/reconstruction.py:571`
        (REQ-46, commit 341ccd8) and ``zoomy_core/mesh/
        lsq_reconstruction.py:425`` (`_resolve_u_boundary_face`).  The
        factor is UNCONDITIONAL on `bf >= 0`: the extrapolation /
        Neumann-zero case self-cancels because `u_bf_i = u_i` there makes
        the delta 0, and `2·0 = 0` is still the correct ghost = inner-cell
        value."""
        nc = self._nc
        dim = self.dim
        A_glob = self._lsq_gradQ
        neighbors = self._lsq_neighbors
        scale = self._lsq_scale
        has_bdy = self._lsq_has_bdy
        bdy_nbr = self._lsq_bdy_nbr

        grad_perm = self._grad_perm

        def per_cell(i):
            A_loc = A_glob[i]
            nbr_idx = neighbors[i]
            u_i = u[i]
            u_cells = u[nbr_idx] - u_i
            if has_bdy:
                bf = bdy_nbr[i]
                # ``bf < 0`` means "no boundary face on this side"; ``u_bf is
                # None`` means extrapolation (Neumann-zero) — both collapse to
                # the inner cell value, i.e. a zero delta (numpy twin:
                # ``np.full_like(bf, u_i)``).
                if u_bf is None:
                    u_bf_i = jnp.full(bf.shape, u_i, dtype=u.dtype)
                else:
                    u_bf_i = jnp.where(
                        bf >= 0,
                        u_bf[jnp.maximum(bf, 0)],
                        u_i,
                    )
                # The virtual point sits at the ghost offset 2·(face - cell),
                # so the delta it carries is u_ghost - u_cell — and ``u_bf`` is
                # ALREADY a ghost value: the BC kernels return ghost semantics
                # (Wall mirrors q -> -q; FromData returns 2·interp - Q). So it
                # enters DIRECTLY, with no second lift.
                #
                # Lifting it again by 2·(u_bf - u_i) asserts q_ghost = -3·q at a
                # wall instead of -q. That double lift is measurable and does
                # NOT converge: it freezes a 7/6 (+16.67%) error into the
                # boundary-cell wall-normal slope at every resolution, which is
                # geometrically equivalent to placing the wall a quarter cell
                # inside the domain. Fixed in the numpy twin at
                # zoomy_core/fvm/reconstruction.py:591; matching it here.
                # (Only a kernel that returns a FACE value would need the lift —
                # that is what zoomy_core/mesh/lsq_reconstruction.py:431 does,
                # and it applies the lift itself.)
                u_bf_delta = jnp.where(bf >= 0, u_bf_i - u_i, 0.0)
                delta_u = jnp.concatenate([u_cells, u_bf_delta])
            else:
                delta_u = u_cells
            coeffs = scale * (A_loc.T @ delta_u)
            # Permute from monomial-multi-index order to
            # ``(∂/∂x, ∂/∂y, …)`` so downstream consumers can dot
            # ``grads[:, d, :]`` against ``r[d]`` for face
            # extrapolation without worrying about the LSQ stencil's
            # internal ordering.
            return coeffs[grad_perm]

        return jax.vmap(per_cell)(jnp.arange(nc)).T   # (dim, nc)

    def _compute_gradients(self, Q, n_var, bf_face_values):
        nc = self._nc
        grads = jnp.zeros((n_var, self.dim, nc))
        for v in range(n_var):
            u_bf = (bf_face_values[v]
                    if bf_face_values is not None else None)
            grads = grads.at[v].set(self._lsq_gradient_scalar(Q[v, :], u_bf))
        return grads

    # ── Limiter bounds (interior + boundary passes) ─────────────────

    def _neighbor_bounds(self, u, bf_values):
        """Per-cell min/max over face-neighbors.

        Interior faces contribute the OTHER cell's centre value `u[iA]` /
        `u[iB]` — a sample a full cell spacing away.  Boundary faces must
        contribute a sample at the SAME distance, i.e. at the GHOST-CELL
        centre, not at the face centre.

        **Ghost-distance convention** (identical to the one enforced on the
        LSQ gradient row in `_lsq_gradient_scalar`, see its docstring):
        BC kernels evaluate at the FACE CENTRE, so the bound carried at the
        ghost position is `u_ghost = 2·u_face − u_cell`.

        Using the bare face value halves the admissible increment at every
        boundary cell relative to an interior cell seeing the same smooth
        field.  Both the Venkatakrishnan numerator and denominator are then
        O(h), so the resulting φ deficit is GRID-INDEPENDENT — it never
        relaxes under refinement.  Measured on a smooth Dirichlet SME(0)
        with a nonzero boundary slope (N = 50…800): φ at the clipped
        boundary cell converged to 0.953 instead of 1, and
        `|limited_grad − exact|` DIVERGED (3.1e-3 → 2.1e-2, rates −1.86 →
        −0.11) to a permanent ~4.8 % gradient deficit, while the raw LSQ
        gradient converged cleanly (rates 0.84 → 0.98).  With the ghost
        conversion below, φ at that cell goes 0.9522 → 0.9995 and the
        limited-gradient error collapses onto the raw one.

        Self-cancels for extrapolation / Neumann-zero (`u_face = u_cell`
        ⇒ `u_ghost = u_cell`) and for reflective walls evaluated at the
        face (`u_face = 0` ⇒ `u_ghost = −u_cell`), so it is a no-op on
        every BC family that does not prescribe a value.

        NumPy twin carries the SAME defect at
        `zoomy_core/fvm/reconstruction.py:600-601` — reported, not fixed
        here (that tree is owned elsewhere).
        """
        u_max = u
        u_min = u
        u_max = u_max.at[self._iA_int].max(u[self._iB_int])
        u_max = u_max.at[self._iB_int].max(u[self._iA_int])
        u_min = u_min.at[self._iA_int].min(u[self._iB_int])
        u_min = u_min.at[self._iB_int].min(u[self._iA_int])
        u_bf_ghost = 2.0 * bf_values - u[self._bf_cells]
        u_max = u_max.at[self._bf_cells].max(u_bf_ghost)
        u_min = u_min.at[self._bf_cells].min(u_bf_ghost)
        return u_min, u_max

    # ── Face deltas (interior + boundary, NumPy-style split) ────────

    def _face_deltas_interior(self, grad_v):
        """Single-variable face delta contributions for interior faces.

        Returns (dA, dB) each shape (n_interior_faces,)."""
        dA = jnp.zeros(self._iA_int.shape[0])
        dB = jnp.zeros(self._iB_int.shape[0])
        for d in range(self.dim):
            dA = dA + grad_v[d, self._iA_int] * self._r_Af_int[d]
            dB = dB + grad_v[d, self._iB_int] * self._r_Bf_int[d]
        return dA, dB

    def _face_deltas_boundary(self, grad_v):
        """Single-variable A-side face delta for boundary faces."""
        dA = jnp.zeros(self._iInner_bnd.shape[0])
        for d in range(self.dim):
            dA = dA + grad_v[d, self._iInner_bnd] * self._r_Af_bnd[d]
        return dA

    # ── Limiters (Venkatakrishnan, Barth-Jespersen, minmod) ─────────

    def _limit_vk(self, u, grad_v, u_min, u_max):
        nc = self._nc
        eps = 1e-30
        ev2 = self._eps_v2
        dA_int, dB_int = self._face_deltas_interior(grad_v)
        dA_bnd = self._face_deltas_boundary(grad_v)

        # Apply face-by-face: φ_cell = min over all face contributions.
        def _apply(cell_ids, deltas):
            uc = u[cell_ids]
            dm_pos = u_max[cell_ids] - uc
            dm_neg = u_min[cell_ids] - uc
            num_p = dm_pos ** 2 + ev2[cell_ids] + 2 * deltas * dm_pos
            den_p = dm_pos ** 2 + 2 * deltas ** 2 + deltas * dm_pos + ev2[cell_ids]
            cand_p = jnp.clip(num_p / jnp.maximum(den_p, eps), 0.0, 1.0)
            num_n = dm_neg ** 2 + ev2[cell_ids] + 2 * deltas * dm_neg
            den_n = dm_neg ** 2 + 2 * deltas ** 2 + deltas * dm_neg + ev2[cell_ids]
            cand_n = jnp.clip(num_n / jnp.maximum(den_n, eps), 0.0, 1.0)
            cand = jnp.where(
                deltas > eps, cand_p,
                jnp.where(deltas < -eps, cand_n, 1.0),
            )
            # Segment-min over duplicate cell_ids via .at[].min.
            phi_face = jnp.ones(nc).at[cell_ids].min(cand)
            return phi_face

        phi = jnp.minimum(_apply(self._iA_int, dA_int),
                          _apply(self._iB_int, dB_int))
        phi = jnp.minimum(phi, _apply(self._iInner_bnd, dA_bnd))
        return jnp.clip(phi, 0.0, 1.0)

    def _limit_bj(self, u, grad_v, u_min, u_max):
        nc = self._nc
        eps = 1e-30
        dA_int, dB_int = self._face_deltas_interior(grad_v)
        dA_bnd = self._face_deltas_boundary(grad_v)

        def _apply(cell_ids, deltas):
            uc = u[cell_ids]
            dm_pos = u_max[cell_ids] - uc
            dm_neg = u_min[cell_ids] - uc
            cand_p = jnp.clip(dm_pos / jnp.maximum(deltas, eps), 0.0, 1.0)
            cand_n = jnp.clip(dm_neg / jnp.minimum(deltas, -eps), 0.0, 1.0)
            cand = jnp.where(
                deltas > eps, cand_p,
                jnp.where(deltas < -eps, cand_n, 1.0),
            )
            return jnp.ones(nc).at[cell_ids].min(cand)

        phi = jnp.minimum(_apply(self._iA_int, dA_int),
                          _apply(self._iB_int, dB_int))
        phi = jnp.minimum(phi, _apply(self._iInner_bnd, dA_bnd))
        return jnp.clip(phi, 0.0, 1.0)

    def _limit_minmod(self, u, grad_v, u_min, u_max):
        nc = self._nc
        eps = 1e-30
        dA_int, dB_int = self._face_deltas_interior(grad_v)
        dA_bnd = self._face_deltas_boundary(grad_v)

        def _apply(cell_ids, deltas):
            uc = u[cell_ids]
            dm_pos = u_max[cell_ids] - uc
            dm_neg = u_min[cell_ids] - uc
            r_pos = dm_pos / jnp.maximum(deltas, eps)
            r_neg = dm_neg / jnp.minimum(deltas, -eps)
            cand = jnp.where(
                deltas > eps, jnp.clip(r_pos, 0.0, 1.0),
                jnp.where(deltas < -eps, jnp.clip(r_neg, 0.0, 1.0), 1.0),
            )
            return jnp.ones(nc).at[cell_ids].min(cand)

        phi = jnp.minimum(_apply(self._iA_int, dA_int),
                          _apply(self._iB_int, dB_int))
        phi = jnp.minimum(phi, _apply(self._iInner_bnd, dA_bnd))
        return jnp.clip(phi, 0.0, 1.0)

    # ── Compute φ per variable, skip ``unlimited_indices`` ──────────

    def _compute_phi(self, Q, n_var, bf_face_values, grads):
        phi = jnp.ones((n_var, self._nc))
        for v in range(n_var):
            if v in self._unlimited_indices:
                continue
            u = Q[v, :]
            bf_v = (bf_face_values[v]
                    if bf_face_values is not None
                    else jnp.zeros(self._n_bf))
            u_min, u_max = self._neighbor_bounds(u, bf_v)
            phi = phi.at[v].set(self._limiter_fn(u, grads[v], u_min, u_max))
        return phi

    # ── Reconstruction with interior + boundary face split ──────────

    def _reconstruct(self, Q, grads, phi, bf_face_values):
        n_var = Q.shape[0]
        n_faces = self._n_faces
        Q_L = jnp.zeros((n_var, n_faces))
        Q_R = jnp.zeros((n_var, n_faces))

        # Interior faces: both sides from inner cells with limited slopes.
        Q_L_int = Q[:, self._iA_int]
        Q_R_int = Q[:, self._iB_int]
        for d in range(self.dim):
            Q_L_int = Q_L_int + (
                phi[:, self._iA_int] * grads[:, d, self._iA_int]
                * self._r_Af_int[d][jnp.newaxis, :])
            Q_R_int = Q_R_int + (
                phi[:, self._iB_int] * grads[:, d, self._iB_int]
                * self._r_Bf_int[d][jnp.newaxis, :])
        Q_L = Q_L.at[:, self._interior_faces].set(Q_L_int)
        Q_R = Q_R.at[:, self._interior_faces].set(Q_R_int)

        # Boundary faces: inner cell reconstructed at the face,
        # BC value on the outer side.
        Q_inner_recon = Q[:, self._iInner_bnd]
        for d in range(self.dim):
            Q_inner_recon = Q_inner_recon + (
                phi[:, self._iInner_bnd] * grads[:, d, self._iInner_bnd]
                * self._r_Af_bnd[d][jnp.newaxis, :])
        Q_L = Q_L.at[:, self._boundary_faces].set(Q_inner_recon)
        if bf_face_values is not None:
            Q_R = Q_R.at[:, self._boundary_faces].set(bf_face_values)

        return Q_L, Q_R


class FreeSurfaceLSQMUSCLJAX(LSQMUSCLReconstructionJAX):
    """LSQ-MUSCL with wet-dry fallback for free-surface flows (JAX) —
    mirrors NumPy `FreeSurfaceLSQMUSCL`.

    In dry cells (`h < eps_wet`) drops to first-order (`φ = 0`).
    Clamps `h ≥ 0` at face states and zeros the corresponding
    momentum components so `hu/h` doesn't blow the flux to NaN at
    the dry front.
    """

    def __init__(self, mesh, dim, h_index, eps_wet=1e-3,
                 limiter="venkatakrishnan", unlimited_indices=None,
                 momentum_indices=None):
        super().__init__(mesh, dim, limiter=limiter,
                         unlimited_indices=unlimited_indices)
        self._h_idx = h_index
        self._eps_wet = eps_wet
        self._mom_idx = (
            tuple(momentum_indices) if momentum_indices is not None
            else tuple(range(h_index + 1, h_index + 1 + dim))
        )

    # NOT a bare ``supports_grad_recon = True`` opt-in: the inherited
    # ``LSQMUSCLReconstructionJAX.reconstruct_with_grad`` is NOT correct for
    # this class.  It skips BOTH wet/dry treatments this class exists for —
    # the dry-cell ``φ = 0`` demotion and ``_wet_dry_clamp`` — so an
    # NCP-bearing model at order ≥ 2 would take face states with no wet/dry
    # handling at all (the exact defeat the ``solver_jax.get_flux_operator``
    # guard was written to catch: measured min h_face = −3.1e-03, NEGATIVE
    # DEPTH, on the sibling ``PositivityPreservingLSQMUSCLJAX``).  The class
    # therefore provides its OWN implementation below and ``__call__``
    # delegates to it, so the two entry points cannot drift apart.

    def reconstruct_with_grad(self, Q, bf_face_values):
        """Wet/dry-aware twin of `__call__` that also returns the
        LIMITED cell gradient for the order ≥ 2 cell-interior NCP integral.

        The gradient returned is `φ_dry · ∇Q` — the SAME dry-demoted
        limiter the face states are built from, so the interior NCP
        `B(Q_c)·s_c` is consistent with the face fluctuations (in a dry
        cell both are first order, slope ≡ 0).  `_wet_dry_clamp` is a
        nonlinear clip of the FACE states only and has no slope counterpart;
        it is applied to the faces here exactly as in `__call__`.
        """
        # Drop to first order in dry cells: zero φ for h < eps_wet.
        # ``phi`` is sized for inner cells only (``self._nc``); the
        # SPMD path may pass a full local mesh ``Q`` of shape
        # ``(n_var, n_inner + n_halo)``, so slice ``dry`` to match.
        n_var = Q.shape[0]
        grads = self._compute_gradients(Q, n_var, bf_face_values)
        phi = self._compute_phi(Q, n_var, bf_face_values, grads)
        dry = Q[self._h_idx, :self._nc] < self._eps_wet
        phi = jnp.where(dry[jnp.newaxis, :], 0.0, phi)
        Q_L, Q_R = self._reconstruct(Q, grads, phi, bf_face_values)
        Q_L = self._wet_dry_clamp(Q_L)
        Q_R = self._wet_dry_clamp(Q_R)
        return Q_L, Q_R, phi[:, jnp.newaxis, :] * grads

    def __call__(self, Q, bf_face_values):
        # Single body, single source of truth — see the note above.
        Q_L, Q_R, _ = self.reconstruct_with_grad(Q, bf_face_values)
        return Q_L, Q_R

    def _wet_dry_clamp(self, Q_face):
        h = jnp.maximum(Q_face[self._h_idx, :], 0.0)
        Q_face = Q_face.at[self._h_idx, :].set(h)
        dry = h < self._eps_wet
        for mi in self._mom_idx:
            Q_face = Q_face.at[mi, :].set(
                jnp.where(dry, 0.0, Q_face[mi, :]))
        return Q_face


class _ZhangShuPP:
    """Xing-Zhang-Shu 2010 a-priori cell-mean positivity limiter (shared).

    Scales the reconstruction *deviation* by a per-cell θ ∈ [0, 1] so the
    reconstructed depth at every face midpoint of the cell stays ≥ 0.  The cell
    mean (hence conservation) is untouched, and θ = 1 (no-op) wherever the
    reconstruction is already non-negative — in particular at still water, so
    well-balancing is preserved.  Mixed into the conservative-h
    (`PositivityPreservingLSQMUSCLJAX`) and η=h+b
    (`EtaWellBalancedLSQMUSCLJAX`) reconstructions.
    """

    def _build_cell_face_rays(self, mesh, dim):
        """`r_cf[:, f, c] = face_centers[cell_faces[f, c]] − cell_centers[c]`,
        shape `(dim, n_faces_per_cell, n_inner_cells)` — the rays from each
        cell centre to its own face midpoints, for evaluating the
        reconstruction at the cell's faces."""
        nc = self._nc
        cell_faces = np.asarray(mesh.cell_faces)[:, :nc]
        face_ctrs = np.asarray(mesh.face_centers)[:, :dim].T  # (dim, n_faces)
        cell_ctrs = np.asarray(mesh.cell_centers)[:dim, :nc]   # (dim, nc)
        self._r_cf = jnp.asarray(
            face_ctrs[:, cell_faces] - cell_ctrs[:, np.newaxis, :])

    def _pp_theta(self, h_bar, h_slope, eps=1e-14):
        """Per-cell θ so `min_face(h_bar + h_slope·r_cf) ≥ 0` (θ=1 if already
        ≥ 0).  `h_slope` is the LIMITED conservative depth slope `(dim, nc)`
        — `φ_h·∇h` for the plain reconstruction, `φ_η·∇η − φ_b·∇b` for η.
        Rescales the deviation, never the mean, so conservation is exact."""
        delta = jnp.einsum("dc,dfc->fc", h_slope, self._r_cf)
        h_min = jnp.min(h_bar[jnp.newaxis, :] + delta, axis=0)
        denom = jnp.maximum(h_bar - h_min, eps)
        return jnp.where(h_min < 0.0,
                         jnp.clip(h_bar / denom, 0.0, 1.0),
                         jnp.ones_like(h_bar))


class PositivityPreservingLSQMUSCLJAX(_ZhangShuPP, LSQMUSCLReconstructionJAX):
    """LSQ-MUSCL with Xing–Zhang 2013 cell-mean positivity preservation.

    Mathematical recipe
    -------------------
    Given a cell `K` with cell-mean `h̄ = Q[h, K]` and the standard
    limiter `φ ∈ [0, 1]^{n_var × n_inner}` already applied, evaluate
    the reconstructed depth at every face midpoint of `K`:

    ```
        h_f = h̄ + φ_h · grad_h · (x_f − x_K),   f ∈ faces(K).
    ```

    If `min_f h_f < 0`, scale the **deviation polynomial** uniformly
    by `θ_K ∈ [0, 1)` so the new face minimum is exactly zero:

    ```
        θ_K  =  h̄ / (h̄ − min_f h_f).
    ```

    Apply `θ_K` not just to `φ_h` but to **every momentum
    component** `φ_{hu_i}`: scaling the linear polynomial uniformly
    preserves the cell-mean of every variable AND keeps the
    reconstructed velocity `u_f = (hu)_f / h_f` bounded as
    `h_f → 0`.

    Why this is the right fix for cell-mean positivity over a CFL-
    bounded step (Xing & Zhang, J. Sci. Comput. 57(1):19–41, 2013,
    Theorem 3.1): given a Rusanov / HLL flux with HR, the
    instantaneous flux through a face depends only on the face-state
    values `h_L, h_R`.  If those are non-negative and the time step
    satisfies the standard CFL `dt · α_max / h_inradius ≤ 1/(2k+1)`
    for piecewise-degree-`k` reconstruction (so `1/3` for our
    linear LSQ-MUSCL), the FV update ``h̄_K^{n+1} = h̄_K^n − dt/|K|
    Σ_f F_f · area_f`` stays `≥ 0`.  Bathymetry `b` is **not**
    rescaled (stationary by the model contract).

    Difference from `FreeSurfaceLSQMUSCLJAX`: that class
    clamps `h_face ≥ 0` **after** reconstruction (a face-state
    safety net) and drops to 1st order in cells with
    `h̄ < eps_wet`.  Neither of those guarantees the cell-mean
    update stays positive over a finite `dt` — for Malpasset over
    T = 100 s the wet/dry front drains thin cells to `h̄ < 0`.
    The Xing–Zhang clip prevents the drain at source by ensuring the
    high-side face value can't exceed the cell's mass budget over the
    CFL-bounded step.

    Parameters
    ----------
    mesh, dim, limiter, unlimited_indices : see `LSQMUSCLReconstructionJAX`.
    h_index : int
        Row index of `h` in the state vector.
    momentum_indices : sequence of int, optional
        Row indices of momentum components scaled by the same θ_K as
        `h`.  Defaults to `range(h_index+1, h_index+1+dim)` (the
        `[h, hu, hv]` convention).
    eps_positivity : float
        Floating-point regularisation in the θ denominator.  Should be
        ≪ the model's wet/dry threshold so it only fires at strictly
        zero `h̄`.
    """

    def __init__(self, mesh, dim, h_index, momentum_indices=None,
                 eps_positivity=1e-14,
                 limiter="venkatakrishnan", unlimited_indices=None):
        super().__init__(mesh, dim, limiter=limiter,
                         unlimited_indices=unlimited_indices)
        self._h_idx = int(h_index)
        self._mom_idx = (
            tuple(momentum_indices) if momentum_indices is not None
            else tuple(range(h_index + 1, h_index + 1 + dim))
        )
        self._eps_positivity = float(eps_positivity)
        self._build_cell_face_rays(mesh, dim)

    def _xing_zhang_scale_phi(self, Q, grads, phi):
        """Scale the h + momentum rows by the Xing-Zhang-Shu θ so the
        reconstructed depth stays ≥ 0 at every face of the cell."""
        h_idx = self._h_idx
        h_slope = grads[h_idx] * phi[h_idx][jnp.newaxis, :]   # φ_h ∇h, (dim, nc)
        theta = self._pp_theta(
            Q[h_idx, :self._nc], h_slope, self._eps_positivity)
        new_phi = phi.at[h_idx, :].multiply(theta)
        for mi in self._mom_idx:
            new_phi = new_phi.at[mi, :].multiply(theta)
        return new_phi

    # ``supports_grad_recon`` is the EXPLICIT opt-in the flux operator tests
    # before routing the order≥2 cell-interior NCP integral through
    # ``reconstruct_with_grad``.  ``hasattr`` is NOT a valid test: it succeeds
    # on an INHERITED base-class method and so silently routes an NCP-bearing
    # model through a reconstruction whose positivity/WB treatment was never
    # applied (see ``solver_jax.get_flux_operator``).
    supports_grad_recon = True

    def reconstruct_with_grad(self, Q, bf_face_values):
        """Positivity-preserving twin of `__call__` that also returns
        the LIMITED cell gradient for the order≥2 interior NCP integral.

        MUST exist on this class.  Without it the inherited
        `LSQMUSCLReconstructionJAX.reconstruct_with_grad` runs, which skips
        `_xing_zhang_scale_phi` entirely — so an NCP-bearing model at
        order ≥ 2 gets face states with NO a-priori positivity cap even
        though `__call__` has one.  Measured divergence between the two
        paths on a 1-D draining bed (state `[b, h, q]`, b = 0.5x,
        h = max(0.25 − 0.5x, 0)):

            max|Q_L(__call__) − Q_L(reconstruct_with_grad)| = 3.125e-03
            min h_face via __call__              = −2.07e-25  (θ fired)
            min h_face via reconstruct_with_grad = −3.125e-03  (NEGATIVE
                                                                DEPTH)
            max|lim_grad − XZ-scaled lim_grad|   = 0.125

        The path is reachable for the DERIVED model: SME(level=0, dim=2)
        derives a nonzero `nonconservative_matrix` (B[q, b] = g·h), so the
        solver turns the interior NCP on at order ≥ 2.

        Scope note: the defect is POSITIVITY only, not well-balancing.  The
        base method returns the conservative-h limited slope, which IS the
        slope the conservative / xz face reconstruction is consistent with;
        the WB weakness of those modes is inherent to reconstructing
        conservative `h` (which is why `EtaWellBalancedLSQMUSCLJAX`
        exists), not a leak from this missing override.
        """
        n_var = Q.shape[0]
        grads = self._compute_gradients(Q, n_var, bf_face_values)
        phi = self._compute_phi(Q, n_var, bf_face_values, grads)
        phi = self._xing_zhang_scale_phi(Q, grads, phi)
        Q_L, Q_R = self._reconstruct(Q, grads, phi, bf_face_values)
        return Q_L, Q_R, phi[:, jnp.newaxis, :] * grads

    def __call__(self, Q, bf_face_values):
        # Single body, single source of truth: the two entry points CANNOT
        # drift apart again (that drift is exactly what this class shipped).
        Q_L, Q_R, _ = self.reconstruct_with_grad(Q, bf_face_values)
        return Q_L, Q_R


class EmittedWLSQMUSCLJAX(_ZhangShuPP, LSQMUSCLReconstructionJAX):
    """Primitive-variable LSQ-MUSCL on the **emitted** reconstruction
    variables `W = reconstruction_variables(Q, Qaux, p)`, inverted by the
    **emitted** `state_from_reconstruction` (Audusse-Bouchut 2005,
    Kurganov-Petrova 2007).

    For SWE / SME core emits `W = (b, b+h, hinv·q)` — free surface plus
    PRIMITIVE VELOCITY, with `hinv` the KP-desingularized inverse depth at
    core's canonical `wet_dry_eps`.  This is the identical pair OpenFOAM
    consumes in `zoomy_foam/numerics_o2.H`, and it is why foam needs no
    wet/dry threshold in its reconstruction.

    MANDATE 6a — WHAT THIS CLASS REPLACED, AND WHY IT MATTERED.  This used to
    be `EtaWellBalancedLSQMUSCLJAX`, which hand-coded its own variable set
    `(b, η, hu, hv)` — free surface but CONSERVATIVE momentum — and then
    needed a class-default `eps_wet = 1e-3` literal to survive `hu/h` at a
    dry front.  That literal was five orders of magnitude larger than core's
    canonical `1e-8`, and 20% of the SWASHES Ritter reservoir depth.  It was
    used twice: to demote the limiter in cells with `h < 1e-3`, and to ZERO
    THE FACE MOMENTUM wherever `h_f < 1e-3`.  The second deleted the
    advective flux that drives the Ritter front, so the front stalled on the
    exact solution's `h = 1e-3` contour and the entire tail beyond it was
    erased.  The resulting order-2 L1 error was a FIXED PHYSICAL REGION that
    could not shrink under refinement — the "convergence rate" of 0.155 was
    never a convergence rate.

    Reconstructing the emitted primitive velocity removes the need for the
    threshold rather than retuning it: `u = hinv·q` is bounded as `h -> 0`,
    and the inverse recovers `q_f = u_f·h_f`, which vanishes exactly where
    `h_f` does and nowhere else.  There is no epsilon in this class.

    The reconstruction problem on conservative `(b, h, hu, hv)` is
    ill-conditioned at a wet/dry shoreline on non-flat bathymetry:
    `h` drops to zero discontinuously at the shore line while `b`
    rises out of the water, so a linear LSQ fit through `h` sees a
    huge slope that the limiter cannot fully tame — face values
    extrapolated from a wet cell into a dry cell land well below zero
    (we measured `h_face = -0.8` in a 1D probe at slope = -0.2).

    Reconstructing the **free surface** `η = h + b` instead is the
    standard remedy.  At lake-at-rest `η` is exactly constant
    (slope = 0 everywhere — well-balanced by construction).  At a
    wet/dry shore `η` is continuous up to the shoreline, so the
    slope it sees is finite and the limiter clips it cleanly.  After
    the linear extrapolation gives face states ``(b_f, η_f, hu_f,
    hv_f)``, the depth is recovered as:

    ```
        h_f = max(η_f - b_f, 0)
    ```

    The outer `max(·, 0)` is the Audusse-Bouchut face-state
    positivity clip and is conservation-safe — at the face we use
    `h_f = 0` to compute the flux, and the upstream cell still
    drains at the rate determined by `α` and the cell's own face
    value.  Combined with the HR step that
    `PositiveNonconservativeHLL` performs on these face
    states, the discrete scheme is provably positivity-preserving
    and well-balanced under the standard CFL bound.

    Parameters
    ----------
    mesh, dim, limiter, unlimited_indices : see
        `LSQMUSCLReconstructionJAX`.
    b_index : int
        Row index of bathymetry `b` (defaults to 0 for the
        `[b, h, hu, hv]` convention).
    h_index : int
        Row index of depth `h` (defaults to 1).  This row carries
        `η = h + b` during the reconstruction pass and is converted
        back to `h` at the face afterwards.
    """

    # This reconstruction honours an external per-cell ``force_o1`` mask
    # (zero the slope → constant reconstruction) — used by the solver's
    # a-posteriori MOOD corrector to demote troubled cells.  The flux
    # operator checks this flag before threading the mask through.
    supports_force_o1 = True

    def __init__(self, mesh, dim, w_of_q, q_of_w, aux_of_q=None,
                 eps_wet=None, b_index=0, h_index=1,
                 momentum_indices=None, positivity=None,
                 front_tol=None, piecewise_constant=False,
                 limiter="venkatakrishnan", unlimited_indices=None):
        super().__init__(mesh, dim, limiter=limiter,
                         unlimited_indices=unlimited_indices)
        # MANDATE 6a: the reconstruction VARIABLES and their inverse are the
        # EMITTED model-map pair, not a set hand-coded here.  ``w_of_q`` is
        # ``reconstruction_variables(Q, Qaux, p)`` and ``q_of_w`` is
        # ``state_from_reconstruction(W, Qaux, p)`` — the same pair OpenFOAM
        # consumes.  There is deliberately NO ``eps_wet`` argument: the emitted
        # forward map divides the momentum by core's KP-desingularized ``hinv``
        # (canonical ``wet_dry_eps``), so the primitive velocity is bounded at a
        # dry front by construction and the backend needs no threshold to
        # survive it.  A float literal here would be exactly the defect.
        if w_of_q is None or q_of_w is None:
            raise ValueError(
                "EmittedWLSQMUSCLJAX requires the EMITTED reconstruction map "
                "pair (reconstruction_variables / state_from_reconstruction). "
                "The runtime returned None for at least one — the model "
                "declares no reconstruction map, so there is nothing to "
                "reconstruct in primitive variables.")
        self._w_of_q = w_of_q
        self._q_of_w = q_of_w
        # The EMITTED local aux formula (``update_aux_variables``).  Needed
        # because the RK stages do NOT refresh Qaux: SSP-RK2 evaluates stage 2
        # at ``Q1`` while still passing stage 1's ``Qaux``
        # (``solver_jax._rk2``).  That staleness was harmless while ``W``
        # depended only on ``Q`` (η = h + b), but the emitted ``W`` carries
        # ``u = hinv·q`` — pairing a stage-2 momentum with a stage-1 ``hinv``
        # at a fast-moving front gives a velocity that is simply wrong, and it
        # runs away.  Re-deriving the aux from the CURRENT ``Q`` through the
        # model's own emitted map is what core's numpy solver does
        # (``_apply_local_aux_formula`` / ``_ghost_aux``); the non-local
        # derivative-aux rows pass through from the supplied ``Qaux``.
        self._aux_of_q = aux_of_q
        # EMITTED wet/dry threshold, or None.  NOT a class default — the whole
        # point of this rewrite is that no float literal lives here.
        #
        # Why a demotion is still needed at all (MEASURED): the emitted W puts
        # PRIMITIVE VELOCITY on the momentum row, and the face momentum is
        # recovered as ``q_f = u_f·h_f``.  In a near-dry cell the η-row
        # extrapolation can hand back a face depth much LARGER than the cell's
        # own mean (h_f/h_cell ~ 1e4 measured at the Ritter front, cell h =
        # 1.2e-7 against h_f ~ 1.2e-3), so a large ``u`` from ``q/h`` in that
        # cell is AMPLIFIED onto the face instead of damped.  Demoting such a
        # cell to piecewise-constant removes the amplification.
        self._eps_wet = None if eps_wet is None else float(eps_wet)
        self._b_idx = int(b_index)
        self._h_idx = int(h_index)
        self._mom_idx = (
            tuple(momentum_indices) if momentum_indices is not None
            else tuple(range(h_index + 1, h_index + 1 + dim))
        )
        # ``positivity="zhang_shu"`` adds the a-priori Xing-Zhang-Shu cell-mean
        # cap on top of the η reconstruction → a-priori h≥0 under CFL ≤ 1/(2k+1)
        # with NO a-posteriori step.  ``None`` → WB + Audusse face clip only.
        self._positivity = positivity
        # ``front_tol`` ∈ (0, 1]: a-priori wet/dry-front PRE-DETECTOR.  Demote a
        # cell to genuine 1st order (φ:=0) wherever the positivity indicator
        # θ_front < front_tol — i.e. wherever the η reconstruction WOULD push h
        # negative at a face (the wet/dry shoreline + steep h extrema).  Reuses
        # the same _pp_theta as the a-priori cap, but instead of *scaling* the
        # deviation it *zeroes* it, so the cell is exactly piecewise-constant
        # (Gallardo-Parés-Castro: 1st order near wet/dry transitions).  ``None``
        # → off.  Lets the run use the linear-stability CFL instead of 1/(2k+1).
        self._front_tol = front_tol
        # ``piecewise_constant=True`` → ORDER-1 Audusse η/HR reconstruction: zero
        # the reconstruction slope so ``W`` is piecewise-constant per cell, but
        # STILL apply the free-surface face-state machinery — the ``η→h`` W-space
        # Audusse clip ``η_f := max(η_f, b_f)`` (giving ``h_f = max(0, η_f−b_f) ≥
        # 0``) and the emitted inverse ``q_f = u_f·h_f`` (dry-face momentum
        # vanishes as a matched pair, no threshold).  This is Audusse-Bouchut
        # 2005 hydrostatic reconstruction at first order, positivity-preserving
        # by construction under the Audusse CFL — the piece the plain
        # :class:`ConstantReconstruction` order-1 fallback omitted, which let the
        # draining wet/dry front push ``h < 0`` and NaN (2-D Thacker planar O1).
        self._piecewise_constant = bool(piecewise_constant)
        self._build_cell_face_rays(mesh, dim)

    def _eta_core(self, Q, bf_face_values, force_o1=None,
                  Qaux=None, parameters=None, bf_aux=None):
        """Shared reconstruction core for `__call__` and
        `reconstruct_with_grad` — returns `(Q_L, Q_R, grads, phi)` where
        `grads` / `phi` are the slopes + tied limiter on the EMITTED
        `W = reconstruction_variables(Q, Qaux, p)`.  Exposing them lets the
        order≥2 interior NCP integral reuse the *same* limited slopes the face
        flux was built from (well-balanced + wet/dry-safe)."""
        if Qaux is None or parameters is None:
            raise ValueError(
                "EmittedWLSQMUSCLJAX needs Qaux and parameters: the EMITTED "
                "reconstruction map reads the desingularized ``hinv`` AUX. "
                "The caller passed None — it is still on the old hand-coded "
                "signature.")
        # Q -> W through the EMITTED map (SWE/SME: [b, b+h, hinv·q]).  The
        # momentum row is the PRIMITIVE VELOCITY, which is what removes the
        # need for a wet/dry threshold: u = hinv·q stays bounded as h -> 0
        # (KP), where the old hand-coded conservative ``hu`` row did not.
        # Re-derive the LOCAL aux from THIS stage's Q before forming W (see
        # ``_aux_of_q``): the solver hands us the previous stage's Qaux.
        if self._aux_of_q is not None:
            Qaux = self._aux_of_q(Q, Qaux, parameters)
        W = self._w_of_q(Q, Qaux, parameters)
        bf_W = bf_face_values
        if bf_face_values is not None:
            if bf_aux is None:
                raise ValueError(
                    "EmittedWLSQMUSCLJAX needs ``bf_aux`` whenever "
                    "``bf_face_values`` is given: the boundary face state must "
                    "go through the SAME emitted map, and that map reads aux. "
                    "Feeding the inner cell's aux to a ghost state desyncs "
                    "``hinv`` from the ghost ``h`` (core's numpy solver calls "
                    "this the periodic mass leak).")
            bf_W = self._w_of_q(bf_face_values, bf_aux, parameters)

        # Compute slope + limiter per variable on the emitted W.
        n_var = Q.shape[0]
        grads = self._compute_gradients(W, n_var, bf_W)
        phi = self._compute_phi(W, n_var, bf_W, grads)

        # ORDER-1 Audusse η/HR: zero every slope so ``W`` reconstructs
        # piecewise-constant.  φ=0 makes the tie / front / zhang_shu / o1 blocks
        # below no-ops (all key on φ·∇W = 0 ⇒ θ=1, no front), and ``grads`` is
        # then dead (XLA DCEs the LSQ gradient).  The Audusse face clip +
        # emitted inverse further down still run — that is the whole point.
        if self._piecewise_constant:
            phi = jnp.zeros_like(phi)

        # Dry-cell demotion, threshold from the EMITTED ``wet_dry_eps`` ONLY.
        # ``None`` => no demotion at all (the model emitted no threshold); this
        # class never invents one.  See ``_eps_wet`` for why it is needed.
        if self._eps_wet is not None:
            dry = Q[self._h_idx, :self._nc] < self._eps_wet
            phi = jnp.where(dry[jnp.newaxis, :], 0.0, phi)

        # NOTE: there is deliberately NO dry-cell ``φ:=0`` demotion here.  The
        # old code demoted every cell with ``h < eps_wet = 1e-3`` — 20% of the
        # SWASHES Ritter reservoir depth — which is what froze the front at the
        # exact solution's h = 1e-3 contour.  The emitted W needs no such
        # guard: ``η`` is continuous across the shoreline and ``u = hinv·q`` is
        # bounded, so the limiter sees finite slopes on both.  The genuine
        # positivity guards (``front_tol`` pre-detector, ``zhang_shu`` cap)
        # remain below and are threshold-FREE — they key on the positivity
        # indicator θ, not on an absolute depth.

        # **Critical: tie phi_b to phi_η.**  At the shoreline the η-row
        # limiter clips to 0 (η has the wet/dry discontinuity), but b
        # is globally smooth so the b-row limiter would leave phi_b=1.
        # The back-transform ``h_face = η_face − b_face`` then sees an
        # inconsistent pair: ``η_face = η_cell`` (no slope, from
        # phi_η=0) but ``b_face = b_cell + grad_b·r`` (full slope,
        # phi_b=1), so ``h_face = h_cell − grad_b·r`` — a spurious
        # non-zero h at the dry-side face of the shoreline cell that
        # bleeds mass into the dry region over many steps.
        # Audusse-Bouchut 2005 §3 uses a single per-cell ``φ_K`` across
        # all reconstruction variables; here we tie ``phi_b`` to the
        # smaller of (its own minmod result, phi_η) so smooth-bath
        # regions away from the shoreline keep their full O2 b-slope
        # while shoreline cells consistently drop b's slope alongside
        # η's.  Same reasoning applies to the momentum rows.
        phi_eta = phi[self._h_idx]
        for var_idx in (self._b_idx, *self._mom_idx):
            phi = phi.at[var_idx, :].set(
                jnp.minimum(phi[var_idx, :], phi_eta))

        # Per-cell 1st-order demotion mask = (a-priori front pre-detector) ∪
        # (external a-posteriori MOOD ``force_o1``).  Computed from the
        # PRE-positivity tied slopes (the raw "would h go negative at a face"
        # indicator), so it is independent of whether the zhang_shu cap below
        # also runs.  Applied as φ:=0 just before reconstruction → genuine
        # constant reconstruction (the Audusse h_f = max(η_f−b_f, 0) clip then
        # makes such a cell positivity-preserving by the XZS 1st-order lemma,
        # and the shared face flux keeps it conservative).
        o1_mask = force_o1
        if self._front_tol is not None:
            bslope0 = phi[self._b_idx][jnp.newaxis, :] * grads[self._b_idx]
            sh0 = phi[self._h_idx][jnp.newaxis, :] * grads[self._h_idx] - bslope0
            theta_front = self._pp_theta(Q[self._h_idx, :self._nc], sh0)
            front = theta_front < self._front_tol
            o1_mask = front if o1_mask is None else (o1_mask[:self._nc] | front)

        # A-priori Xing-Zhang-Shu cell-mean positivity (optional): cap the
        # CONSERVATIVE depth deviation so the reconstructed h stays ≥ 0 at every
        # face — while keeping the BED reconstruction FAITHFUL (b is fixed
        # topography; scaling it corrupts b_face and the Audusse h*).  Net depth
        # slope is sh = φ_η·∇η − φ_b·∇b; we want the capped slope θ·sh with the
        # bed keeping φ_b·∇b, i.e. the η slope becomes φ_b·∇b + θ·sh.  We fold
        # the limiter into grad and set φ_η=1 so the back-transform
        # h_f = η_f − b_f sees exactly h̄ + θ·sh·r.  Dry cells already have φ=0 ⇒
        # sh=0 ⇒ untouched; at still water ∇η=0 and h̄≥0 ⇒ θ=1 ⇒ WB preserved.
        # Branchless ⇒ GPU-friendly; the interior-NCP gradient inherits θ·sh.
        if self._positivity == "zhang_shu":
            bslope = phi[self._b_idx][jnp.newaxis, :] * grads[self._b_idx]
            sh = phi[self._h_idx][jnp.newaxis, :] * grads[self._h_idx] - bslope
            theta = self._pp_theta(Q[self._h_idx, :self._nc], sh)
            grads = grads.at[self._h_idx].set(bslope + theta[jnp.newaxis, :] * sh)
            phi = phi.at[self._h_idx, :].set(jnp.ones_like(phi[self._h_idx]))
            for mi in self._mom_idx:
                phi = phi.at[mi, :].multiply(theta)

        # Force flagged cells to genuine 1st order (constant reconstruction):
        # zero EVERY row's limiter, including the h-row whose slope the
        # zhang_shu block folded into ``grads`` with φ_h=1 — φ_h:=0 kills that
        # contribution too, so the back-transform gives h_f = h̄ on all faces.
        if o1_mask is not None:
            phi = jnp.where(o1_mask[jnp.newaxis, :self._nc], 0.0, phi)

        W_L, W_R = self._reconstruct(W, grads, phi, bf_W)

        # AUDUSSE-BOUCHUT 2005 §4 face-state positivity, applied in W SPACE:
        # clip the reconstructed FREE SURFACE to the reconstructed bed,
        # ``η_f := max(η_f, b_f)``.  The emitted inverse then yields
        # ``h_f = η_f − b_f >= 0`` AND ``q_f = u_f·h_f = 0`` as a matched pair,
        # so the momentum needs no separate rule (the old code zeroed it on
        # ``h_f < eps_wet`` — that literal is what deleted the Ritter tail).
        #
        # This is an exact ``max(·, 0)`` on a FACE state, not an epsilon and not
        # a floor on the evolved cell mean: no constant enters, and the cell
        # average is never truncated (MOOD repairs a cell whose AVERAGE goes
        # negative, exactly as foam's numerics_o2.H documents).  It is NOT
        # optional — the MOOD detector's own correctness argument cites it, and
        # measured without it the Ritter march reaches h = −2.6e-02 at t = 0.7
        # and NaNs by t = 0.75.
        eta_L = jnp.maximum(W_L[self._h_idx, :], W_L[self._b_idx, :])
        eta_R = jnp.maximum(W_R[self._h_idx, :], W_R[self._b_idx, :])
        W_L = W_L.at[self._h_idx, :].set(eta_L)
        W_R = W_R.at[self._h_idx, :].set(eta_R)

        # W_face -> Q_face through the EMITTED INVERSE
        # ``state_from_reconstruction`` (SWE/SME: h = η−b, q = u·(η−b)).
        #
        # This is where the deleted-tail bug dies.  The old code zeroed the
        # face momentum wherever ``h_f < eps_wet = 1e-3``, which deleted the
        # advective flux that drives the Ritter front.  The emitted inverse
        # needs no such rule and has no threshold: the momentum is RECOVERED as
        # ``q_f = u_f · h_f``, so it vanishes exactly where ``h_f`` vanishes and
        # nowhere else.  The Riemann solver never sees a ``hu/h`` blow-up
        # because ``u_f`` was reconstructed as a bounded primitive, not divided
        # out at the face.
        #
        # The aux passed to the inverse is the CELL aux broadcast to faces.
        # That is only sound while the inverse does not actually read aux —
        # asserted at construction by the solver via
        # ``state_from_reconstruction_uses_aux``.
        Q_L = self._q_of_w(W_L, self._face_aux(W_L, Qaux), parameters)
        Q_R = self._q_of_w(W_R, self._face_aux(W_R, Qaux), parameters)
        return Q_L, Q_R, grads, phi

    @staticmethod
    def _face_aux(W_face, Qaux):
        """Aux array shaped to the FACE axis for the emitted inverse.

        Zeros, not the cell aux: the inverse is asserted aux-free (see
        `state_from_reconstruction_uses_aux`), so this array is never read.
        Building it explicitly — rather than passing the cell-shaped `Qaux`
        and relying on broadcasting — makes the shape mismatch a loud error
        the day a model DOES put aux in its inverse, instead of a silent
        wrong-length read.
        """
        return jnp.zeros((Qaux.shape[0], W_face.shape[1]), dtype=W_face.dtype)

    def __call__(self, Q, bf_face_values, force_o1=None,
                 Qaux=None, parameters=None, bf_aux=None):
        Q_L, Q_R, _grads, _phi = self._eta_core(
            Q, bf_face_values, force_o1, Qaux, parameters, bf_aux)
        return Q_L, Q_R

    def reconstruct_with_grad(self, Q, bf_face_values, force_o1=None,
                              Qaux=None, parameters=None, bf_aux=None):
        """η-consistent limited cell gradient for the order≥2 cell-interior
        NCP integral `B(Q_c)·s_c` (well-balanced + positivity-safe on
        wet/dry beds).

        The base `reconstruct_with_grad` returns the slope of the
        *conservative* `h` row, whose limiter sees the wet/dry
        discontinuity (a huge slope, only partly tamed) — inconsistent with
        the `η = h + b` face flux.  That mismatch (a) breaks order-2
        lake-at-rest well-balancing and (b) injects a large spurious
        bed-slope momentum at the shoreline that drains thin cells to
        `h < 0` (the failure shared by all three face variants, since they
        all reused this base gradient).

        Returning the W-consistent slope instead, pushed back to CONSERVATIVE
        variables through the EMITTED inverse: at lake-at-rest (`∂η = 0`) the
        depth slope reduces to `−∂b` and the interior NCP exactly cancels the
        hydrostatic face flux.

        The push-back is a JVP of `state_from_reconstruction`, NOT a
        hand-written chain rule.  `B(Q_c)·s_c` contracts against the
        CONSERVATIVE state gradient, but the limited slopes above live on the
        emitted `W` — and now that the momentum row of `W` is the PRIMITIVE
        velocity `u` rather than `q`, `∂q = h·∂u + u·∂h ≠ ∂u`.  Writing
        that transform out by hand here would re-introduce exactly the kind of
        backend-local duplicate of the model map this change exists to delete
        (and would be silently wrong for any model whose inverse is not SWE's).
        `jax.jvp` differentiates the emitted inverse itself, so the chain rule
        is whatever the model says it is.

        For SWE/SME this is not merely cosmetic bookkeeping: it keeps the h row
        at `∂η − ∂b` (identical to the old hand-written line, verified
        numerically) while giving the momentum rows the conservative slope the
        old code only got by accident, because it reconstructed `q` directly.
        """
        Q_L, Q_R, grads, phi = self._eta_core(
            Q, bf_face_values, force_o1, Qaux, parameters, bf_aux)
        lim_grad = phi[:, jnp.newaxis, :] * grads     # slopes on emitted W
        # Linearize about the CELL-CENTRE W, sliced to the owned cells so it
        # matches the slope array's cell axis exactly (no silent broadcast).
        W_c = self._w_of_q(Q, Qaux, parameters)[:, :self._nc]
        aux0 = jnp.zeros((Qaux.shape[0], self._nc), dtype=W_c.dtype)

        def _push(dW):
            """W-slope -> conservative-state slope, per direction."""
            return jax.jvp(
                lambda w: self._q_of_w(w, aux0, parameters),
                (W_c,), (dW,))[1]

        # vmap over the DIRECTION axis of lim_grad (n_state, dim, n_cells).
        lim_grad = jax.vmap(_push, in_axes=1, out_axes=1)(
            lim_grad[:, :, :self._nc])
        return Q_L, Q_R, lim_grad


class FreeSurfaceMUSCL(MUSCLReconstruction):
    """[LEGACY] Green-Gauss MUSCL with wet-dry fallback (JAX).

    Superseded by `FreeSurfaceLSQMUSCLJAX` for the new
    NSM-routed JAX HyperbolicSolver — kept temporarily for tests
    that pin against the old reconstruction.

    In dry cells (h < eps_wet), falls back to 1st order (φ = 0).
    Clamps h ≥ 0 at face states after reconstruction, and zeros the
    corresponding momentum components so dry-face velocities never become
    infinite (without this, `hu/h` blows the HLLC flux to NaN at the
    dry front of a Ritter dam-break).
    """

    def __init__(self, mesh, dim, h_index, eps_wet=1e-3, limiter="venkatakrishnan",
                 momentum_indices=None):
        super().__init__(mesh, dim, limiter=limiter)
        self._h_idx = h_index
        self._eps_wet = eps_wet
        # Indices of the momentum components (hu, hv, …). If left as None,
        # assume they immediately follow h in the state vector — matches the
        # standard ``[h, hu]`` and ``[h, hu, hv]`` SWE conventions.
        self._mom_idx = (
            tuple(momentum_indices) if momentum_indices is not None
            else tuple(range(h_index + 1, h_index + 1 + dim))
        )

    def __call__(self, Q):
        n_vars = Q.shape[0]
        grads, phi = self._compute_limited_gradients(Q, n_vars)
        Q_L, Q_R = self._reconstruct(Q, grads, phi)
        Q_L = self._wet_dry_clamp(Q_L)
        Q_R = self._wet_dry_clamp(Q_R)
        return Q_L, Q_R

    def _wet_dry_clamp(self, Q_face):
        h = jnp.maximum(Q_face[self._h_idx, :], 0.0)
        Q_face = Q_face.at[self._h_idx, :].set(h)
        # If h was clamped to ~0 the cell is dry-side: also zero its
        # momentum components so u = hu/h doesn't blow up downstream.
        dry = h < self._eps_wet
        for mi in self._mom_idx:
            Q_face = Q_face.at[mi, :].set(jnp.where(dry, 0.0, Q_face[mi, :]))
        return Q_face

    def _compute_limited_gradients(self, Q, n_vars):
        grads, phi = super()._compute_limited_gradients(Q, n_vars)
        dry = Q[self._h_idx, :] < self._eps_wet
        phi = jnp.where(dry[jnp.newaxis, :], 0.0, phi)
        return grads, phi


# ── Diffusion operator (JAX, JIT-compatible) ────────────────────────────────

class DiffusionOperatorJAX:
    """Sparse discrete diffusion operator for JAX: L(u) = nabla . (nu nabla u).

    Assembled once per mesh + viscosity as a dense (nc, nc) matrix.
    Provides:
    - explicit(u): L @ u  (for explicit stepping)
    - implicit_solve(u_star, dt): Crank-Nicolson solve

    The dense matrix approach is JIT-compatible and works inside
    jax.lax.while_loop. For typical 1D/2D FVM grids (up to ~1000 cells)
    this is efficient; for larger grids a matrix-free GMRES variant is
    also available via `implicit_solve_gmres`.
    """

    def __init__(self, mesh, dim, nu=1.0):
        nc = mesh.n_inner_cells
        n_cells = mesh.n_cells

        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        centers = mesh.cell_centers[:dim, :]
        normals = mesh.face_normals[:dim, :]
        face_vol = mesh.face_volumes
        cell_vol = mesh.cell_volumes

        # Build the Laplacian as a dense (nc, nc) matrix using NumPy
        # then convert to JAX.  This is done once at setup time.
        import numpy as np
        L = np.zeros((nc, nc), dtype=float)

        n_faces = mesh.n_faces
        for f in range(n_faces):
            a = int(iA[f])
            b = int(iB[f])
            dx = centers[:, b] - centers[:, a]
            dist = np.linalg.norm(dx)
            if dist < 1e-30:
                continue

            n = normals[:, f]
            n_dot_e = np.dot(n, dx / dist)
            n_dot_e = max(abs(n_dot_e), 0.1) * np.sign(n_dot_e + 1e-30)
            coeff = nu * float(face_vol[f]) / dist / abs(n_dot_e)

            if a < nc and b < nc:
                L[a, b] += coeff / float(cell_vol[a])
                L[a, a] -= coeff / float(cell_vol[a])
                L[b, a] += coeff / float(cell_vol[b])
                L[b, b] -= coeff / float(cell_vol[b])

        self.L = jnp.array(L)
        self.nc = nc
        self.n_cells = n_cells

        # Precompute boundary ghost -> inner cell mapping for post-solve copy
        self._boundary_ghosts = jnp.array(mesh.boundary_face_ghosts)
        self._boundary_cells = jnp.array(mesh.boundary_face_cells)

    def explicit(self, u):
        """Compute L @ u[:nc] (for explicit stepping). Returns shape (nc,)."""
        return self.L @ u[:self.nc]

    def implicit_solve(self, u_star, dt):
        """Crank-Nicolson: (I - dt/2 * L) u^{n+1} = (I + dt/2 * L) u*.

        Second-order in time for diffusion. Uses dense linear solve
        (jnp.linalg.solve) which is fully JIT-compatible.

        Parameters
        ----------
        u_star : jnp.ndarray, shape (n_cells,)
            State after explicit advection step (includes ghost cells).
        dt : scalar
            Time step.

        Returns
        -------
        u_new : jnp.ndarray, shape (n_cells,)
            Updated state with ghost cells copied from inner neighbors.
        """
        nc = self.nc
        L = self.L
        I = jnp.eye(nc)

        # Right-hand side: (I + dt/2 * L) @ u*[:nc]
        rhs = (I + 0.5 * dt * L) @ u_star[:nc]

        # System matrix: (I - dt/2 * L)
        A = I - 0.5 * dt * L

        # Solve
        sol = jnp.linalg.solve(A, rhs)

        # Assemble full solution with ghost cells
        result = u_star.at[:nc].set(sol)

        # Copy inner cell values to boundary ghosts (Neumann-like)
        result = result.at[self._boundary_ghosts].set(result[self._boundary_cells])
        return result

    def implicit_solve_gmres(self, u_star, dt, tol=1e-8, maxiter=100):
        """Crank-Nicolson via matrix-free GMRES. JIT-compatible.

        Use this for larger grids where the dense solve becomes expensive.
        """
        from jax.scipy.sparse.linalg import gmres as jax_gmres

        nc = self.nc
        L = self.L

        rhs = u_star[:nc] + 0.5 * dt * (L @ u_star[:nc])

        def matvec(x):
            return x - 0.5 * dt * (L @ x)

        sol, info = jax_gmres(matvec, rhs, x0=u_star[:nc],
                              tol=tol, atol=0.0, maxiter=maxiter)

        result = u_star.at[:nc].set(sol)
        result = result.at[self._boundary_ghosts].set(result[self._boundary_cells])
        return result


class DenseDiffusionOperatorJAX:
    r"""JAX mirror of core `DenseDiffusionOperator` (REQ-109): dense,
    state-dependent implicit diffusion `div(A : grad Q)`.

    Assembles the two-point (TPFA) FVM divergence of the model's GENERAL
    diffusive flux `F_diff[i,d] = Σ_{j,e} A[i,j,d,e] ∂_e Q[j]` from the
    rank-4 `diffusion_matrix` `A(Q, Qaux, p)`.  Unlike the scalar
    `DiffusionOperatorJAX` (`ν` × constant Laplacian, per-variable
    diagonal) this couples DIFFERENT variables (dense `A[i,j]`, `i≠j`) and
    lets `A` depend on the state, so a state-dependent cross-modal viscosity
    (e.g. the μ(I) in-plane ν-moments) is treated FULLY IMPLICITLY — no
    parabolic CFL.

    Discretisation mirrors core `DenseDiffusionOperator` bit-for-bit: the
    same ν-free geometry factors, the same face-normal contraction
    `T[i,j] = Σ_{d,e} A[i,j,d,e]·n_d·n_e` (`Ā = ½(A_a+A_b)`), and
    Crank–Nicolson (θ=½).  A constant diagonal `A = ν·I·δ_de` reduces to
    `ν·L` (unit normals ⇒ `T = ν·δ_ij`), so the jax dense path matches both
    the jax scalar path and the numpy reference on that regression.

    Boundaries: like the scalar jax path, the implicit half is interior-only
    (`bf_grads=None`) and ghosts are refilled by the solver's `bc_op` after
    the solve — exact for the periodic / Neumann cases the jax IMEX targets.

    Improvement over the numpy reference: the state-dependent Newton inner
    Jacobian action is EXACT automatic differentiation (`jax.jvp`), not a
    finite-difference JVP — no `fd_eps` to tune.
    """

    def __init__(self, mesh, dim, n_vars, state_dependent=False):
        import numpy as np
        self.dim = int(dim)
        self.n_vars = int(n_vars)
        self.state_dependent = bool(state_dependent)
        nc = mesh.n_inner_cells
        self.nc = nc
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        centers = mesh.cell_centers[:dim, :]
        normals = mesh.face_normals[:dim, :]
        face_vol = mesh.face_volumes
        cell_vol = mesh.cell_volumes

        # Interior-face geometry (ν-free) + unit normal — the SAME per-face
        # factors DiffusionOperatorJAX bakes into ``L``.
        ia, ib, gA, gB, n_int = [], [], [], [], []
        for f in range(mesh.n_faces):
            a, b = int(iA[f]), int(iB[f])
            if not (a < nc and b < nc):
                continue
            dx = centers[:, b] - centers[:, a]
            dist = float(np.linalg.norm(dx))
            if dist < 1e-30:
                continue
            n = normals[:, f]
            n_dot_e = float(np.dot(n, dx / dist))
            n_dot_e = max(abs(n_dot_e), 0.1) * np.sign(n_dot_e + 1e-30)
            g = float(face_vol[f]) / dist / abs(n_dot_e)
            ia.append(a); ib.append(b)
            gA.append(g / float(cell_vol[a])); gB.append(g / float(cell_vol[b]))
            n_int.append(np.asarray(n, dtype=float).copy())
        self._ia = jnp.asarray(ia, dtype=int)
        self._ib = jnp.asarray(ib, dtype=int)
        self._gA = jnp.asarray(gA, dtype=float)
        self._gB = jnp.asarray(gB, dtype=float)
        self._n_int = (jnp.asarray(np.asarray(n_int, dtype=float).T)
                       if n_int else jnp.zeros((dim, 0)))
        self._has_int = len(ia) > 0

        # Boundary-face geometry + unit normal (inner-cell tensor).
        bf_inner, bf_g, n_bf = [], [], []
        for i_bf in range(mesh.n_boundary_faces):
            fidx = int(mesh.boundary_face_face_indices[i_bf])
            inner = int(mesh.boundary_face_cells[i_bf])
            a, b = int(iA[fidx]), int(iB[fidx])
            dx = centers[:, b] - centers[:, a]
            dist = float(np.linalg.norm(dx))
            n = normals[:, fidx]
            if dist < 1e-30:
                bf_inner.append(inner); bf_g.append(0.0)
                n_bf.append(np.asarray(n, dtype=float).copy())
                continue
            n_dot_e = float(np.dot(n, dx / dist))
            n_dot_e = max(abs(n_dot_e), 0.1) * np.sign(n_dot_e + 1e-30)
            g = float(face_vol[fidx]) / abs(n_dot_e) / float(cell_vol[inner])
            bf_inner.append(inner); bf_g.append(g)
            n_bf.append(np.asarray(n, dtype=float).copy())
        self._bf_inner = jnp.asarray(bf_inner, dtype=int)
        self._bf_g = jnp.asarray(bf_g, dtype=float)
        self._n_bf = (jnp.asarray(np.asarray(n_bf, dtype=float).T)
                      if n_bf else jnp.zeros((dim, 0)))
        self._has_bf = len(bf_inner) > 0

    def _as_cell_tensor(self, A):
        """Reshape the runtime `diffusion_matrix` return to core's per-cell
        layout `(n_eq, n_st, nc, d, d)`.

        The jax runtime vmaps over cells with `out_axes=-1` ⇒ a full-grid
        call yields `(n_eq, n_st, d, d, nc)`; transpose the trailing cell
        axis into the middle so the divergence einsum mirrors the numpy
        reference 1:1.  A purely-constant tensor (no cell axis) is broadcast."""
        A = jnp.asarray(A)
        d = self.dim
        if A.ndim == 5:                      # (n_eq, n_st, d, d, nc)
            return jnp.transpose(A, (0, 1, 4, 2, 3))
        if A.ndim == 4:                      # (n_eq, n_st, d, d) — constant
            return jnp.broadcast_to(
                A[:, :, None, :, :],
                (A.shape[0], A.shape[1], self.nc, d, d))
        raise ValueError(
            f"diffusion_matrix returned ndim={A.ndim}; expected 4 or 5.")

    def _divergence(self, Q, A_cells, bf_grads=None):
        """Discrete `∇·F_diff` — mirror of core `_divergence` (TPFA
        face-normal contraction + scatter-add).  `Q` and the output are
        inner-cell only (`(n_vars, nc)` / `(n_eq, nc)`)."""
        n_eq = A_cells.shape[0]
        D = jnp.zeros((n_eq, self.nc), dtype=Q.dtype)
        if self._has_int:
            A_a = A_cells[:, :, self._ia]           # (n_eq,n_st,nf,d,d)
            A_b = A_cells[:, :, self._ib]
            Aface = 0.5 * (A_a + A_b)
            T = jnp.einsum('ijfde,df,ef->ijf', Aface, self._n_int, self._n_int)
            dQ = Q[:, self._ib] - Q[:, self._ia]    # (n_st,nf)
            flux = jnp.einsum('ijf,jf->if', T, dQ)  # (n_eq,nf)
            D = D.at[:, self._ia].add(flux * self._gA)
            D = D.at[:, self._ib].add(-flux * self._gB)
        if bf_grads is not None and self._has_bf:
            A_in = A_cells[:, :, self._bf_inner]    # (n_eq,n_st,nbf,d,d)
            T_bf = jnp.einsum('ijbde,db,eb->ijb', A_in, self._n_bf, self._n_bf)
            fluxb = jnp.einsum('ijb,jb->ib', T_bf, jnp.asarray(bf_grads))
            D = D.at[:, self._bf_inner].add(fluxb * self._bf_g)
        return D

    def implicit_solve(self, Q_star, dt, A_fn, bf_grads=None,
                       tol=1e-8, maxiter=100, newton_maxiter=8,
                       newton_tol=1e-8):
        r"""One Crank–Nicolson implicit-diffusion sub-step (mirror of core).

        Solves `(Q^{n+1}−Q*)/dt = ½[𝒟_impl(Q^{n+1}) + 𝒟_expl]` with
        `𝒟_expl = interior(Q*) [+ boundary(Q*)]` frozen.  ``A_fn(Q) ->
        diffusion_matrix`` closes over `Qaux`/`p` (inner cells).  Constant
        `A` ⇒ one linear `jax_gmres` solve; state-dependent ⇒ a Newton
        loop with an EXACT `jax.jvp` Jacobian action and a GMRES inner solve,
        differentiable through its root (implicit function theorem).
        `Q_star` is inner-cell `(n_vars, nc)`.
        """
        from jax.scipy.sparse.linalg import gmres as jax_gmres

        Q_star = jnp.asarray(Q_star)
        shape = Q_star.shape

        A_star = self._as_cell_tensor(A_fn(Q_star))
        D_expl = self._divergence(Q_star, A_star, bf_grads=bf_grads)

        if not self.state_dependent:
            rhs = (Q_star + 0.5 * dt * D_expl).reshape(-1)

            def matvec(x_flat):
                X = x_flat.reshape(shape)
                DX = self._divergence(X, A_star, bf_grads=None)
                return (X - 0.5 * dt * DX).reshape(-1)

            sol, _ = jax_gmres(matvec, rhs, x0=Q_star.reshape(-1),
                               tol=tol, atol=0.0, maxiter=maxiter)
            return sol.reshape(shape)

        # State-dependent: Newton on R(Q) = Q − Q* − dt/2 (𝒟_impl(Q) + 𝒟_expl),
        # A re-evaluated at each iterate (i.e. at Q^{n+1}).  Exact FD-free
        # Jacobian action via jax.jvp; GMRES for the inner linear solve.  The
        # root is differentiated by the implicit function theorem
        # (``newton_root`` = ``lax.custom_root``), not through the iterations.
        def residual(Q):
            D_impl = self._divergence(Q, self._as_cell_tensor(A_fn(Q)),
                                      bf_grads=None)
            return Q - Q_star - 0.5 * dt * (D_impl + D_expl)

        def jvp(Q, V):
            return jax.jvp(residual, (Q,), (V,))[1]

        def linear_solve(matvec, b):
            return gmres(matvec, b, tol=tol, maxiter=maxiter)

        return newton_root(residual, Q_star, jvp, linear_solve,
                           tol=newton_tol, maxiter=newton_maxiter)

    def residual_norm(self, Q_np1, Q_star, dt, A_fn, bf_grads=None):
        r"""Norm of the Crank–Nicolson residual the Newton solve drives to
        zero (mirror of core `residual_norm`):

        ```
            R = Q^{n+1} − Q* − dt/2 ( 𝒟_int(Q^{n+1}) + 𝒟_int(Q*) + bf ).
        ```

        `→0` ⇔ the implicit-diffusion update is satisfied."""
        D_expl = self._divergence(
            Q_star, self._as_cell_tensor(A_fn(Q_star)), bf_grads=bf_grads)
        D_impl = self._divergence(
            Q_np1, self._as_cell_tensor(A_fn(Q_np1)), bf_grads=None)
        R = Q_np1 - Q_star - 0.5 * dt * (D_impl + D_expl)
        return float(jnp.linalg.norm(R))
