"""FVM face reconstruction and diffusion operators.

Reconstruction classes
---------------------
All reconstruction classes share the same interface: ``recon(Q) → (Q_L, Q_R)``.
The solver calls this once per flux evaluation — no if-clauses in the flux loop.

- ``ConstantReconstruction``: 1st-order piecewise-constant (identity).
- ``MUSCLReconstruction``: 2nd-order piecewise-linear with slope limiting
  (Barth-Jespersen or Venkatakrishnan).
- ``FreeSurfaceMUSCL``: MUSCL with wet-dry fallback and h-positivity.

Diffusion operators
-------------------
- ``FaceGradient``: corrected face-normal gradient for diffusion.
- ``DiffusionOperator``: sparse discrete Laplacian (explicit + implicit).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres

from zoomy_core.solver.constants import TPFA_COS_FLOOR


# ── Reconstruction classes ───────────────────────────────────────────────────

class ConstantReconstruction:
    """First-order piecewise-constant reconstruction.

    Returns cell-center values at each face: Q_L = Q[:, iA], Q_R = Q[:, iB].
    """

    def __init__(self, mesh, dim):
        self.iA = mesh.face_cells[0]
        self.iB = mesh.face_cells[1]

    def __call__(self, Q):
        return Q[:, self.iA], Q[:, self.iB]


class MUSCLReconstruction:
    """Second-order MUSCL reconstruction with slope limiting.

    Green-Gauss gradients + slope limiter.
    Call signature matches ``ConstantReconstruction``: ``recon(Q) → (Q_L, Q_R)``.

    Parameters
    ----------
    mesh : FVMMesh or LSQMesh
    dim : int
    limiter : str
        "venkatakrishnan" — smooth, 2nd-order at extrema (default).
        "barth_jespersen" — strict DMP, clips to 1st order at extrema.
        "minmod" — classic TVD, most diffusive, clips to 1st order at extrema.
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

        centers = mesh.cell_centers[:dim, :]       # (dim, n_cells)
        face_ctrs = mesh.face_centers[:, :dim].T   # (n_faces, 3) → (dim, n_faces)

        # Cell-center → face-center displacement vectors
        self.r_Af = face_ctrs - centers[:, iA]     # (dim, n_faces)
        self.r_Bf = face_ctrs - centers[:, iB]

        # Mesh data for Green-Gauss
        self._normals = mesh.face_normals[:dim, :]
        self._face_volumes = mesh.face_volumes
        self._cell_volumes = mesh.cell_volumes

        # Venkatakrishnan smoothing: eps² = (K·h)²
        # Quadratic scaling preserves 2nd-order accuracy at smooth extrema.
        # (Cubic scaling (K·h)³ from the original paper gives φ→0 at extrema.)
        h = np.zeros(self.n_cells)
        h[:self.nc] = self._cell_volumes[:self.nc] ** (1.0 / max(dim, 1))
        self._eps_v2 = (1.0 * h) ** 2

    def __call__(self, Q):
        """Reconstruct face states. Returns (Q_L, Q_R), each (n_vars, n_faces)."""
        n_vars = Q.shape[0]
        grads, phi = self._compute_limited_gradients(Q, n_vars)
        return self._reconstruct(Q, grads, phi)

    # ── Green-Gauss gradient ─────────────────────────────────────────

    def _green_gauss_gradient(self, u):
        """Cell-center gradient of scalar u via Green-Gauss. Returns (dim, n_cells)."""
        dim = self.dim
        grad = np.zeros((dim, self.n_cells))
        iA, iB = self.iA, self.iB
        u_face = 0.5 * (u[iA] + u[iB])
        for d in range(dim):
            contrib = u_face * self._normals[d, :] * self._face_volumes
            np.add.at(grad[d], iA, contrib / self._cell_volumes[iA])
            np.subtract.at(grad[d], iB, contrib / self._cell_volumes[iB])
        return grad

    # ── Neighbor bounds ──────────────────────────────────────────────

    def _neighbor_bounds(self, u):
        """Per-cell min/max over immediate face-neighbors."""
        u_max = u.copy()
        u_min = u.copy()
        np.maximum.at(u_max, self.iA, u[self.iB])
        np.maximum.at(u_max, self.iB, u[self.iA])
        np.minimum.at(u_min, self.iA, u[self.iB])
        np.minimum.at(u_min, self.iB, u[self.iA])
        return u_min, u_max

    # ── Slope limiters ───────────────────────────────────────────────

    def _face_deltas(self, grad):
        """Reconstructed increment at face center. Returns (delta_A, delta_B)."""
        dA = np.zeros(self.n_faces)
        dB = np.zeros(self.n_faces)
        for d in range(self.dim):
            dA += grad[d, self.iA] * self.r_Af[d, :]
            dB += grad[d, self.iB] * self.r_Bf[d, :]
        return dA, dB

    def _limit_bj(self, u, grad, u_min, u_max):
        """Barth-Jespersen limiter → φ ∈ [0, 1] per cell."""
        phi = np.ones(self.n_cells)
        eps = 1e-30
        dA, dB = self._face_deltas(grad)

        for cell_ids, deltas in [(self.iA, dA), (self.iB, dB)]:
            uc = u[cell_ids]
            pos = deltas > eps
            neg = deltas < -eps
            cand = np.ones(self.n_faces)
            cand[pos] = np.minimum(1.0, (u_max[cell_ids[pos]] - uc[pos]) / deltas[pos])
            cand[neg] = np.minimum(1.0, (u_min[cell_ids[neg]] - uc[neg]) / deltas[neg])
            np.minimum.at(phi, cell_ids, cand)

        return np.clip(phi, 0.0, 1.0)

    def _limit_vk(self, u, grad, u_min, u_max):
        """Venkatakrishnan limiter → φ ∈ [0, 1] per cell (smooth)."""
        phi = np.ones(self.n_cells)
        eps = 1e-30
        ev2 = self._eps_v2
        dA, dB = self._face_deltas(grad)

        for cell_ids, deltas in [(self.iA, dA), (self.iB, dB)]:
            uc = u[cell_ids]
            pos = deltas > eps
            neg = deltas < -eps

            if pos.any():
                dm = u_max[cell_ids[pos]] - uc[pos]
                df = deltas[pos]
                num = dm ** 2 + ev2[cell_ids[pos]] + 2 * df * dm
                den = dm ** 2 + 2 * df ** 2 + df * dm + ev2[cell_ids[pos]]
                np.minimum.at(phi, cell_ids[pos],
                              np.minimum(1.0, num / np.maximum(den, eps)))

            if neg.any():
                dm = u_min[cell_ids[neg]] - uc[neg]
                df = deltas[neg]
                num = dm ** 2 + ev2[cell_ids[neg]] + 2 * df * dm
                den = dm ** 2 + 2 * df ** 2 + df * dm + ev2[cell_ids[neg]]
                np.minimum.at(phi, cell_ids[neg],
                              np.minimum(1.0, num / np.maximum(den, eps)))

        return np.clip(phi, 0.0, 1.0)

    def _limit_minmod(self, u, grad, u_min, u_max):
        """Minmod limiter → φ ∈ [0, 1] per cell. Most diffusive TVD limiter."""
        phi = np.ones(self.n_cells)
        eps = 1e-30
        dA, dB = self._face_deltas(grad)

        for cell_ids, deltas in [(self.iA, dA), (self.iB, dB)]:
            uc = u[cell_ids]
            pos = deltas > eps
            neg = deltas < -eps

            # Upwind slope ratio r = (u_upwind - u_cell) / delta
            # minmod(1, r) = max(0, min(1, r))
            cand = np.ones(self.n_faces)
            if pos.any():
                # delta > 0 → the reconstruction goes up → compare against neighbor max
                r = (u_max[cell_ids[pos]] - uc[pos]) / deltas[pos]
                cand[pos] = np.maximum(0.0, np.minimum(1.0, r))
            if neg.any():
                r = (u_min[cell_ids[neg]] - uc[neg]) / deltas[neg]
                cand[neg] = np.maximum(0.0, np.minimum(1.0, r))

            np.minimum.at(phi, cell_ids, cand)

        return np.clip(phi, 0.0, 1.0)

    # ── Core pipeline ────────────────────────────────────────────────

    def _compute_limited_gradients(self, Q, n_vars):
        """Gradient + limiter for all variables. Override for wet-dry."""
        _limiter_map = {
            "barth_jespersen": self._limit_bj,
            "venkatakrishnan": self._limit_vk,
            "minmod": self._limit_minmod,
        }
        limiter_fn = _limiter_map[self._limiter_type]

        grads = np.zeros((n_vars, self.dim, self.n_cells))
        phi = np.ones((n_vars, self.n_cells))

        for v in range(n_vars):
            u = Q[v, :]
            grads[v] = self._green_gauss_gradient(u)
            u_min, u_max = self._neighbor_bounds(u)
            phi[v] = limiter_fn(u, grads[v], u_min, u_max)

        return grads, phi

    def _reconstruct(self, Q, grads, phi):
        """Linear reconstruction at face centers."""
        iA, iB = self.iA, self.iB
        Q_L = Q[:, iA].copy()
        Q_R = Q[:, iB].copy()
        for d in range(self.dim):
            Q_L += phi[:, iA] * grads[:, d, iA] * self.r_Af[d, :][np.newaxis, :]
            Q_R += phi[:, iB] * grads[:, d, iB] * self.r_Bf[d, :][np.newaxis, :]
        return Q_L, Q_R


class FreeSurfaceMUSCL(MUSCLReconstruction):
    """MUSCL with wet-dry fallback for free-surface flows.

    In dry cells (h < eps_wet), falls back to 1st order (φ = 0).
    Clamps h ≥ 0 at face states after reconstruction.

    Parameters
    ----------
    mesh, dim, limiter : same as MUSCLReconstruction
    h_index : int
        Index of water depth h in the Q array.
    eps_wet : float
        Dry-cell threshold.
    """

    def __init__(self, mesh, dim, h_index, eps_wet=1e-3, limiter="venkatakrishnan"):
        super().__init__(mesh, dim, limiter=limiter)
        self._h_idx = h_index
        self._eps_wet = eps_wet

    def __call__(self, Q):
        n_vars = Q.shape[0]
        grads, phi = self._compute_limited_gradients(Q, n_vars)
        Q_L, Q_R = self._reconstruct(Q, grads, phi)
        # Clamp water depth at faces
        np.maximum(Q_L[self._h_idx, :], 0.0, out=Q_L[self._h_idx, :])
        np.maximum(Q_R[self._h_idx, :], 0.0, out=Q_R[self._h_idx, :])
        return Q_L, Q_R

    def _compute_limited_gradients(self, Q, n_vars):
        grads, phi = super()._compute_limited_gradients(Q, n_vars)
        dry = Q[self._h_idx, :] < self._eps_wet
        phi[:, dry] = 0.0
        return grads, phi


# ── Ghost-cell-free reconstruction classes ───────────────────────────────────
#
# These classes operate on Q of shape (n_vars, n_inner_cells) — no ghost cells.
# Boundary-face states come from BC.face_value(), passed in as bf_face_values.
# Loops are split into interior faces and boundary faces — no if-branches.


class ConstantReconstructionV2:
    """First-order piecewise-constant reconstruction (ghost-cell-free).

    Q has shape (n_vars, n_inner_cells). Boundary face values are supplied
    externally via ``bf_face_values``.
    """

    def __init__(self, mesh, dim):
        nc = mesh.n_inner_cells
        fc0 = mesh.face_cells[0]
        fc1 = mesh.face_cells[1]

        # Identify boundary faces via mesh metadata
        bf_face_set = set(mesh.boundary_face_face_indices)

        # Split faces into interior and boundary — no mixing
        self._interior_faces = np.array(
            [f for f in range(mesh.n_faces) if f not in bf_face_set], dtype=int)
        self._boundary_faces = mesh.boundary_face_face_indices.copy()
        self._iA_int = fc0[self._interior_faces]
        self._iB_int = fc1[self._interior_faces]
        # Inner cell at boundary faces: use face_cells[0] which is always the
        # THIS-side cell (BaseMesh.__init__ normalizes this). Do NOT use
        # mesh.boundary_face_cells which may be remapped for periodic BCs.
        self._iInner_bnd = mesh.face_cells[0, mesh.boundary_face_face_indices].copy()
        self._n_faces = mesh.n_faces

    def __call__(self, Q, bf_face_values=None, phi=None, force_o1=None):
        """Reconstruct face states.

        Parameters
        ----------
        Q : ndarray, shape (n_vars, n_inner_cells)
        bf_face_values : ndarray, shape (n_vars, n_boundary_faces)
            BC-provided boundary face values.
        phi : ignored
            Accepted for interface parity with the MUSCL classes;
            piecewise-constant reconstruction has no limiter.
        force_o1 : ignored
            Interface parity with the MUSCL classes.  This reconstruction
            IS the order-1 demotion target, so an a-posteriori MOOD mask is
            exactly a no-op here — not a silently dropped instruction.
            ``supports_force_o1`` stays absent (falsy) so the cascade knows
            there is nothing below this to demote to.  Declaring the
            argument matters because :class:`SurfaceReconstruction` wraps
            *any* base and forwards the full protocol.

        Returns (Q_L, Q_R) each (n_vars, n_faces).
        """
        n_vars = Q.shape[0]
        Q_L = np.zeros((n_vars, self._n_faces))
        Q_R = np.zeros((n_vars, self._n_faces))

        # Interior faces: both sides from Q
        Q_L[:, self._interior_faces] = Q[:, self._iA_int]
        Q_R[:, self._interior_faces] = Q[:, self._iB_int]

        # Boundary faces: L = inner cell (face_cells[0]), R = BC value
        Q_L[:, self._boundary_faces] = Q[:, self._iInner_bnd]
        if bf_face_values is not None:
            Q_R[:, self._boundary_faces] = bf_face_values

        # Piecewise-constant → zero slope: no cell-interior gradient.
        self._limited_grad = None
        return Q_L, Q_R

    def compute_phi(self, Q, bf_face_values=None):
        """No limiter — piecewise-constant reconstruction.  Returns
        ``None`` (interface parity with the MUSCL classes)."""
        return None


class LSQMUSCLReconstruction:
    """Second-order MUSCL reconstruction with LSQ gradients (ghost-cell-free).

    Uses precomputed LSQ stencil from ``LSQMesh`` for gradient computation.
    Only interior cells participate in the LSQ stencil — boundary cells grow
    their stencil inward automatically.

    Q has shape (n_vars, n_inner_cells). Boundary face values are supplied
    externally. All loops are split into interior and boundary — no if-branches.

    Parameters
    ----------
    mesh : LSQMesh
    dim : int
    limiter : str
        "venkatakrishnan" (default), "barth_jespersen", or "minmod".
    unlimited_indices : sequence of int, optional
        State-row indices whose slope is used **un-limited** (``φ = 1``).
        Use for static fields like bathymetry ``b`` — limiting them would
        smear the topography and reduce well-balanced accuracy.
        The unlimited gradient is still the LSQ slope from cell averages
        (no special slope computation), it just isn't clipped by the
        neighbour min/max bounds.
    """

    # Honours an external per-cell ``force_o1`` mask: the solver's a-posteriori
    # MOOD corrector passes a boolean array over inner cells; flagged cells get
    # their limiter zeroed → genuine constant reconstruction (order-1 demotion).
    supports_force_o1 = True

    def __init__(self, mesh, dim, limiter="venkatakrishnan",
                 unlimited_indices=None):
        self.dim = dim
        self._limiter_type = limiter
        self._unlimited_indices = (tuple(unlimited_indices)
                                    if unlimited_indices else ())
        nc = mesh.n_inner_cells
        self._nc = nc
        self._n_faces = mesh.n_faces

        fc0 = mesh.face_cells[0]
        fc1 = mesh.face_cells[1]

        # Identify boundary faces via mesh metadata
        bf_face_set = set(mesh.boundary_face_face_indices)

        # Split faces into interior and boundary
        self._interior_faces = np.array(
            [f for f in range(mesh.n_faces) if f not in bf_face_set], dtype=int)
        self._boundary_faces = mesh.boundary_face_face_indices.copy()

        # Interior face connectivity (both cells are valid inner cells)
        self._iA_int = fc0[self._interior_faces]
        self._iB_int = fc1[self._interior_faces]

        # Inner cell at boundary faces: use face_cells[0] which is always the
        # THIS-side cell. Do NOT use mesh.boundary_face_cells which may be
        # remapped for periodic BCs (pointing to the opposite boundary).
        self._iInner_bnd = mesh.face_cells[0, mesh.boundary_face_face_indices].copy()

        # Cell→face displacement vectors for reconstruction extrapolation
        centers = mesh.cell_centers[:dim, :]
        face_ctrs = mesh.face_centers[:, :dim].T  # (dim, n_faces)

        # Interior faces: both A and B displacements
        self._r_Af_int = face_ctrs[:, self._interior_faces] - centers[:, self._iA_int]
        self._r_Bf_int = face_ctrs[:, self._interior_faces] - centers[:, self._iB_int]

        # Boundary faces: displacement from inner cell to face center
        self._r_Af_bnd = face_ctrs[:, self._boundary_faces] - centers[:, self._iInner_bnd]

        # Face normals and volumes for face_deltas (only interior + boundary A-side)
        self._normals_int = mesh.face_normals[:dim, self._interior_faces]
        self._fv_int = mesh.face_volumes[self._interior_faces]

        # Boundary face metadata (for limiter bounds)
        self._bf_cells = mesh.boundary_face_cells
        self._n_bf = mesh.n_boundary_faces

        # LSQ gradient data (precomputed by LSQMesh)
        self._lsq_gradQ = mesh.lsq_gradQ          # (n_cells, max_nbr, n_monomials)
        self._lsq_neighbors = mesh.lsq_neighbors    # (n_cells, max_nbr)
        self._lsq_scale = mesh.lsq_scale_factors    # (n_monomials,)

        # Columns of the monomial coefficient vector that hold the
        # first-order derivatives ``∂_{x_d}``, axis by axis.  The
        # monomial order from ``build_monomial_indices`` is
        # reverse-lexicographic (e.g. 2D degree-1 → ``[(0,1),(1,0)]``),
        # so the gradient is NOT the leading ``dim`` coefficients — that
        # slice would swap the axes (∂_y where ∂_x is expected).  Map
        # ``e_d`` to its column explicitly.
        from zoomy_core.mesh.lsq_reconstruction import find_derivative_indices
        mon = mesh.lsq_monomial_multi_index
        grad_multi = [tuple(1 if k == d else 0 for k in range(dim))
                      for d in range(dim)]
        self._lsq_grad_cols = np.asarray(
            find_derivative_indices(mon, grad_multi), dtype=int)
        if np.any(self._lsq_grad_cols < 0):
            raise ValueError(
                "LSQMUSCLReconstruction: first-order derivative monomials "
                f"missing from the mesh stencil {list(mon)} — rebuild the "
                "LSQ mesh at degree >= 1.")

        # Boundary-face neighbours for LSQ — when present, ``A_loc`` is
        # sized ``(n_cell_nbr + n_bf_nbr, n_monomials)`` so ``delta_u``
        # must also include the boundary-face values to match.  When
        # absent, ``A_loc`` only covers interior cells.  See
        # ``zoomy_core.mesh.lsq_reconstruction.compute_derivatives``
        # for the partner code that does the boundary-aware extension
        # in the off-line derivative case.
        self._lsq_bf_neighbors = getattr(mesh,
                                         "lsq_boundary_face_neighbors", None)
        self._lsq_has_bdy = (
            self._lsq_bf_neighbors is not None
            and self._lsq_bf_neighbors.size > 0
        )

        # Venkatakrishnan smoothing: eps² = (K·h)²
        cell_vols = mesh.cell_volumes[:nc]
        h = cell_vols ** (1.0 / max(dim, 1))
        self._eps_v2 = h ** 2

    def __call__(self, Q, bf_face_values, phi=None, force_o1=None):
        """Reconstruct face states.

        Parameters
        ----------
        Q : ndarray, shape (n_vars, n_inner_cells)
        bf_face_values : ndarray, shape (n_vars, n_boundary_faces)
            BC-provided boundary face values (for limiter bounds and Q_R at boundary).
        phi : ndarray, shape (n_vars, n_inner_cells), optional
            Pre-computed limiter coefficients.  When supplied the
            limiter is *frozen* — only the (smooth, linear-in-Q) LSQ
            gradient is recomputed — so the reconstruction is an
            affine, hence smooth, function of ``Q``.  This lets an
            implicit solver call it inside a Newton iteration with a
            clean finite-difference Jacobian.  When ``None`` (default)
            φ is computed from ``Q``.
        force_o1 : ndarray of bool, shape (n_inner_cells,), optional
            A-posteriori MOOD demotion mask.  Flagged cells have EVERY
            row's limiter zeroed → genuine piecewise-constant
            reconstruction (Q_L = Q_R = cell mean).  This is a slope
            demotion, NOT a clamp — no state row is truncated.

        Returns (Q_L, Q_R) each (n_vars, n_faces).
        Q_R at boundary faces is set to bf_face_values (placeholder; overwritten
        by face_value(Q_L) in the flux operator).
        """
        n_vars = Q.shape[0]
        grads = self._compute_gradients(Q, n_vars, bf_face_values=bf_face_values)
        if phi is None:
            phi = self._compute_phi(Q, n_vars, bf_face_values, grads)
        phi = self._apply_force_o1(phi, force_o1)
        # Limited cell-centre gradient (n_vars, dim, nc) — consumed by
        # the solver's cell-interior non-conservative integral.
        self._limited_grad = phi[:, None, :] * grads
        return self._reconstruct(Q, grads, phi, bf_face_values)

    def _apply_force_o1(self, phi, force_o1):
        """Zero the limiter of every ``force_o1`` cell → order-1 (constant)
        reconstruction there.  Identity when ``force_o1`` is ``None``."""
        if force_o1 is None:
            return phi
        phi = np.array(phi, copy=True)
        phi[:, np.asarray(force_o1, dtype=bool)[:self._nc]] = 0.0
        return phi

    def compute_phi(self, Q, bf_face_values):
        """Limiter coefficients φ (n_vars, nc) for the given state.

        Exposed so a caller can compute φ once — e.g. from a lagged
        state — and pass it back into :meth:`__call__` to freeze the
        limiter through an implicit-solver Newton iteration."""
        n_vars = Q.shape[0]
        grads = self._compute_gradients(Q, n_vars, bf_face_values=bf_face_values)
        return self._compute_phi(Q, n_vars, bf_face_values, grads)

    # ── LSQ gradient ────────────────────────────────────────────────

    def _lsq_gradient(self, u, u_bf=None):
        """Cell-center gradient via precomputed LSQ stencil.

        When the mesh carries boundary-face neighbours
        (``lsq_boundary_face_neighbors``), the LSQ ``A`` matrix was
        built including them — so ``delta_u`` must also include the
        boundary-face values to match ``A_loc``'s row count.  This
        mirrors :func:`zoomy_core.mesh.lsq_reconstruction.compute_derivatives`.

        Parameters
        ----------
        u : ndarray, shape (n_cells,)
            Inner-cell field values.
        u_bf : ndarray, shape (n_boundary_faces,), optional
            GHOST values from the BC kernel (the kernel applied to the
            inner cell state).  NOT face values — see the note in the
            body.  When the mesh has boundary-face neighbours and
            ``u_bf`` is ``None``, extrapolation is used (ghost = inner
            cell).

        Returns (dim, nc).
        """
        nc = self._nc
        dim = self.dim
        grad = np.zeros((dim, nc))
        A_glob = self._lsq_gradQ
        neighbors = self._lsq_neighbors
        scale = self._lsq_scale
        has_bdy = self._lsq_has_bdy
        bdy_nbr = self._lsq_bf_neighbors

        for i in range(nc):
            A_loc = A_glob[i]                     # (max_nbr, n_monomials)
            nbr_idx = neighbors[i]                # (n_cell_nbr,)
            u_i = u[i]
            u_cells = u[nbr_idx] - u_i            # (n_cell_nbr,)
            if has_bdy:
                bf = bdy_nbr[i]
                # ``bf < 0`` means "no boundary face on this side" —
                # use the inner cell value (extrapolation).
                u_bf_i = (u_bf[np.maximum(bf, 0)] if u_bf is not None
                          else np.full_like(bf, u_i, dtype=float))
                # The virtual boundary neighbour sits at the GHOST-CELL
                # offset ``2·(face - cell)`` (see
                # ``least_squares_reconstruction_local``), so the value
                # carried there must be the GHOST value — and that is
                # exactly what ``u_bf`` is on this path: the FVM flux
                # operator fills ``bf_values`` by applying the BC kernel
                # to the INNER CELL state, and every kernel returns a
                # ghost state (Wall/CharacteristicWall return the mirror
                # ``q → −q``; FromData returns ``2·u_face − u_cell``;
                # Extrapolation returns the inner cell).  The delta is
                # therefore the bare ``u_ghost − u_cell``.
                #
                # Lifting it a SECOND time by ``2·(u_bf − u_i)`` — the
                # "ghost = 2·face − cell" lift, correct only for FACE
                # values — put ``−3·q_0`` at the ghost instead of the
                # true mirror ``−q_0``, freezing the wall-normal
                # momentum slope at a 7/6 error that did not converge
                # (rate 0.00 at N=20..320).  ``zoomy_jax`` (mesh.py:113)
                # already consumes ghost values undoubled; this is the
                # numpy path catching up with the shared convention.
                #
                # NOTE: ``lsq_reconstruction._resolve_u_boundary_face``
                # is a DIFFERENT entry point whose documented contract
                # is FACE values, so it keeps its ``2·u_face − u_cell``
                # lift.  For extrapolation (``u_bf is None``)
                # ``u_bf_i = u_i``, so this stays 0 either way.
                u_bf_delta = np.where(bf >= 0, u_bf_i - u_i, 0.0)
                delta_u = np.concatenate([u_cells, u_bf_delta])
            else:
                delta_u = u_cells
            coeffs = scale * (A_loc.T @ delta_u)  # (n_monomials,)
            grad[:, i] = coeffs[self._lsq_grad_cols]

        return grad

    # ── Neighbor bounds (split loops) ───────────────────────────────

    def _neighbor_bounds(self, u, bf_values):
        """Per-cell min/max over face-neighbors.

        Two separate passes: interior faces (both neighbors valid),
        boundary faces (use BC-provided values).
        """
        u_max = u.copy()
        u_min = u.copy()

        # Pass 1: interior faces
        iA, iB = self._iA_int, self._iB_int
        np.maximum.at(u_max, iA, u[iB])
        np.maximum.at(u_max, iB, u[iA])
        np.minimum.at(u_min, iA, u[iB])
        np.minimum.at(u_min, iB, u[iA])

        # Pass 2: boundary faces — use BC face values
        bf_cells = self._bf_cells
        np.maximum.at(u_max, bf_cells, bf_values)
        np.minimum.at(u_min, bf_cells, bf_values)

        return u_min, u_max

    # ── Face deltas (split loops) ───────────────────────────────────

    def _face_deltas_interior(self, grad):
        """Reconstructed increment at interior face centers.

        Returns (delta_A, delta_B) each shape (n_interior_faces,).
        """
        dA = np.zeros(len(self._interior_faces))
        dB = np.zeros(len(self._interior_faces))
        for d in range(self.dim):
            dA += grad[d, self._iA_int] * self._r_Af_int[d]
            dB += grad[d, self._iB_int] * self._r_Bf_int[d]
        return dA, dB

    def _face_deltas_boundary(self, grad):
        """Reconstructed increment at boundary face centers (A-side only).

        Returns delta_A shape (n_boundary_faces,).
        """
        dA = np.zeros(len(self._boundary_faces))
        for d in range(self.dim):
            dA += grad[d, self._iInner_bnd] * self._r_Af_bnd[d]
        return dA

    # ── Slope limiters ──────────────────────────────────────────────

    def _limit_vk(self, u, grad, u_min, u_max):
        """Venkatakrishnan limiter (smooth)."""
        nc = self._nc
        phi = np.ones(nc)
        eps = 1e-30
        ev2 = self._eps_v2

        # Interior face deltas
        dA_int, dB_int = self._face_deltas_interior(grad)
        # Boundary face deltas (A-side only)
        dA_bnd = self._face_deltas_boundary(grad)

        def _apply_vk(cell_ids, deltas):
            uc = u[cell_ids]
            pos = deltas > eps
            neg = deltas < -eps
            if pos.any():
                dm = u_max[cell_ids[pos]] - uc[pos]
                df = deltas[pos]
                num = dm ** 2 + ev2[cell_ids[pos]] + 2 * df * dm
                den = dm ** 2 + 2 * df ** 2 + df * dm + ev2[cell_ids[pos]]
                np.minimum.at(phi, cell_ids[pos],
                              np.minimum(1.0, num / np.maximum(den, eps)))
            if neg.any():
                dm = u_min[cell_ids[neg]] - uc[neg]
                df = deltas[neg]
                num = dm ** 2 + ev2[cell_ids[neg]] + 2 * df * dm
                den = dm ** 2 + 2 * df ** 2 + df * dm + ev2[cell_ids[neg]]
                np.minimum.at(phi, cell_ids[neg],
                              np.minimum(1.0, num / np.maximum(den, eps)))

        # Interior faces: constrain both A and B sides
        _apply_vk(self._iA_int, dA_int)
        _apply_vk(self._iB_int, dB_int)
        # Boundary faces: constrain A-side only
        _apply_vk(self._iInner_bnd, dA_bnd)

        return np.clip(phi, 0.0, 1.0)

    def _limit_bj(self, u, grad, u_min, u_max):
        """Barth-Jespersen limiter (strict DMP)."""
        nc = self._nc
        phi = np.ones(nc)
        eps = 1e-30

        dA_int, dB_int = self._face_deltas_interior(grad)
        dA_bnd = self._face_deltas_boundary(grad)

        def _apply_bj(cell_ids, deltas):
            uc = u[cell_ids]
            pos = deltas > eps
            neg = deltas < -eps
            cand = np.ones(len(deltas))
            cand[pos] = np.minimum(1.0, (u_max[cell_ids[pos]] - uc[pos]) / deltas[pos])
            cand[neg] = np.minimum(1.0, (u_min[cell_ids[neg]] - uc[neg]) / deltas[neg])
            np.minimum.at(phi, cell_ids, cand)

        _apply_bj(self._iA_int, dA_int)
        _apply_bj(self._iB_int, dB_int)
        _apply_bj(self._iInner_bnd, dA_bnd)

        return np.clip(phi, 0.0, 1.0)

    def _limit_minmod(self, u, grad, u_min, u_max):
        """Minmod limiter (most diffusive TVD)."""
        nc = self._nc
        phi = np.ones(nc)
        eps = 1e-30

        dA_int, dB_int = self._face_deltas_interior(grad)
        dA_bnd = self._face_deltas_boundary(grad)

        def _apply_minmod(cell_ids, deltas):
            uc = u[cell_ids]
            pos = deltas > eps
            neg = deltas < -eps
            cand = np.ones(len(deltas))
            if pos.any():
                r = (u_max[cell_ids[pos]] - uc[pos]) / deltas[pos]
                cand[pos] = np.maximum(0.0, np.minimum(1.0, r))
            if neg.any():
                r = (u_min[cell_ids[neg]] - uc[neg]) / deltas[neg]
                cand[neg] = np.maximum(0.0, np.minimum(1.0, r))
            np.minimum.at(phi, cell_ids, cand)

        _apply_minmod(self._iA_int, dA_int)
        _apply_minmod(self._iB_int, dB_int)
        _apply_minmod(self._iInner_bnd, dA_bnd)

        return np.clip(phi, 0.0, 1.0)

    def _limit_va(self, u, grad, u_min, u_max):
        """Van Albada limiter — smooth, strictly bounded φ ∈ [0, 1).

        Per-face candidate ``(r² + r) / (r² + r + 1)`` with
        ``r = headroom / reconstructed-increment`` (always ≥ 0 in both
        the positive and negative branches).  This is a smooth,
        monotone, strictly-bounded rational function — unlike the
        Venkatakrishnan / Barth-Jespersen candidates it needs **no**
        ``min(1, ·)`` clamp, so the implicit-Newton finite-difference
        Jacobian stays well-behaved.  (The per-cell ``min`` over faces
        remains; the dominant non-smoothness — the unit clamp — is
        what this removes.)
        """
        nc = self._nc
        phi = np.ones(nc)
        eps = 1e-30

        dA_int, dB_int = self._face_deltas_interior(grad)
        dA_bnd = self._face_deltas_boundary(grad)

        def _apply_va(cell_ids, deltas):
            uc = u[cell_ids]
            pos = deltas > eps
            neg = deltas < -eps
            cand = np.ones(len(deltas))
            if pos.any():
                r = (u_max[cell_ids[pos]] - uc[pos]) / deltas[pos]
                cand[pos] = (r * r + r) / (r * r + r + 1.0)
            if neg.any():
                r = (u_min[cell_ids[neg]] - uc[neg]) / deltas[neg]
                cand[neg] = (r * r + r) / (r * r + r + 1.0)
            np.minimum.at(phi, cell_ids, cand)

        _apply_va(self._iA_int, dA_int)
        _apply_va(self._iB_int, dB_int)
        _apply_va(self._iInner_bnd, dA_bnd)

        return np.clip(phi, 0.0, 1.0)

    # ── Core pipeline ───────────────────────────────────────────────

    def _compute_gradients(self, Q, n_vars, bf_face_values=None):
        """Raw LSQ cell-centre gradients (n_vars, dim, nc) — a fixed
        linear stencil, hence smooth (C∞) in ``Q``.

        ``bf_face_values`` (n_vars, n_boundary_faces) come from the
        BC kernel applied to the INNER CELL state, i.e. they are GHOST
        states (see ``_lsq_gradient``), and are threaded into the LSQ
        stencil whenever the mesh carries boundary-face neighbours.
        Without it the LSQ falls back to inner-cell extrapolation."""
        grads = np.zeros((n_vars, self.dim, self._nc))
        for v in range(n_vars):
            u_bf = (bf_face_values[v] if bf_face_values is not None
                    else None)
            grads[v] = self._lsq_gradient(Q[v, :], u_bf=u_bf)
        return grads

    def _compute_phi(self, Q, n_vars, bf_face_values, grads):
        """Slope-limiter coefficients φ (n_vars, nc) — the non-smooth
        part of the reconstruction (neighbour min/max, sign branch,
        per-cell face-min)."""
        phi = np.ones((n_vars, self._nc))
        if self._limiter_type == "none":
            # Unlimited: full LSQ slope on every variable.  Used for smooth
            # verification (MMS, manufactured solutions) where limiter clipping
            # at extrema would mask the true reconstruction order.
            return phi
        _limiter_map = {
            "barth_jespersen": self._limit_bj,
            "venkatakrishnan": self._limit_vk,
            "minmod": self._limit_minmod,
            "van_albada": self._limit_va,
        }
        limiter_fn = _limiter_map[self._limiter_type]
        for v in range(n_vars):
            if v in self._unlimited_indices:
                # Skip limiting: full LSQ slope.  Used for static fields
                # (bathymetry) so the topography isn't smeared.
                continue
            u = Q[v, :]
            u_min, u_max = self._neighbor_bounds(u, bf_face_values[v, :])
            phi[v] = limiter_fn(u, grads[v], u_min, u_max)
        return phi

    def _compute_limited_gradients(self, Q, n_vars, bf_face_values):
        """LSQ gradient + slope limiter for all variables."""
        grads = self._compute_gradients(Q, n_vars)
        phi = self._compute_phi(Q, n_vars, bf_face_values, grads)
        return grads, phi

    def _reconstruct(self, Q, grads, phi, bf_face_values):
        """Linear reconstruction at face centers (split loops)."""
        n_vars = Q.shape[0]
        Q_L = np.zeros((n_vars, self._n_faces))
        Q_R = np.zeros((n_vars, self._n_faces))

        # Interior faces: both sides reconstructed from Q
        Q_L[:, self._interior_faces] = Q[:, self._iA_int]
        Q_R[:, self._interior_faces] = Q[:, self._iB_int]
        for d in range(self.dim):
            Q_L[:, self._interior_faces] += (
                phi[:, self._iA_int] * grads[:, d, self._iA_int]
                * self._r_Af_int[d][np.newaxis, :])
            Q_R[:, self._interior_faces] += (
                phi[:, self._iB_int] * grads[:, d, self._iB_int]
                * self._r_Bf_int[d][np.newaxis, :])

        # Boundary faces: inner cell reconstructed, BC on the other side
        # Compute the reconstructed inner state at each boundary face
        Q_inner_recon = Q[:, self._iInner_bnd].copy()
        for d in range(self.dim):
            Q_inner_recon += (
                phi[:, self._iInner_bnd] * grads[:, d, self._iInner_bnd]
                * self._r_Af_bnd[d][np.newaxis, :])

        # L = reconstructed inner cell, R = BC value
        Q_L[:, self._boundary_faces] = Q_inner_recon
        Q_R[:, self._boundary_faces] = bf_face_values

        return Q_L, Q_R


class _ZhangShuPPMixin:
    """Xing–Zhang–Shu 2010 a-priori cell-mean positivity limiter (numpy).

    Mirror of the jax ``_ZhangShuPP`` mixin.  Scales the reconstruction
    *deviation* of the depth row by a per-cell ``θ ∈ [0, 1]`` so the
    reconstructed depth at every face midpoint of the cell stays ``≥ 0``.  The
    cell mean (hence conservation) is untouched, and ``θ = 1`` (no-op) wherever
    the reconstruction is already non-negative — in particular at still water,
    so well-balancing is preserved.
    """

    def _build_cell_face_rays(self, mesh, dim):
        """``r_cf[:, f, c] = face_centers[cell_faces[f, c]] − cell_centers[c]``,
        shape ``(dim, n_faces_per_cell, n_inner_cells)`` — the rays from each
        cell centre to its own face midpoints, for evaluating the
        reconstruction at the cell's faces."""
        nc = self._nc
        cell_faces = np.asarray(mesh.cell_faces)[:, :nc]        # (nf_pc, nc)
        face_ctrs = np.asarray(mesh.face_centers)[:, :dim].T     # (dim, n_faces)
        cell_ctrs = np.asarray(mesh.cell_centers)[:dim, :nc]     # (dim, nc)
        self._r_cf = face_ctrs[:, cell_faces] - cell_ctrs[:, np.newaxis, :]

    def _pp_theta(self, h_bar, h_slope, eps=1e-14):
        """Per-cell ``θ`` so ``min_face(h_bar + h_slope·r_cf) ≥ 0`` (``θ=1`` if
        already ``≥ 0``).  ``h_slope`` is the LIMITED depth slope ``(dim, nc)``.
        Rescales the deviation, never the mean, so conservation is exact."""
        delta = np.einsum("dc,dfc->fc", h_slope, self._r_cf)    # (nf_pc, nc)
        h_min = np.min(h_bar[np.newaxis, :] + delta, axis=0)     # (nc,)
        denom = np.maximum(h_bar - h_min, eps)
        theta = np.where(h_min < 0.0,
                         np.clip(h_bar / denom, 0.0, 1.0),
                         np.ones_like(h_bar))
        return theta


class PositivityPreservingLSQMUSCL(_ZhangShuPPMixin, LSQMUSCLReconstruction):
    """LSQ-MUSCL with a-priori Xing–Zhang–Shu cell-mean positivity (numpy).

    Mirror of the jax ``PositivityPreservingLSQMUSCLJAX``: after the standard
    slope limiter produces ``φ``, cap the DEPTH-row deviation by the per-cell
    XZS ``θ`` so the reconstructed depth stays ``≥ 0`` at every face midpoint.
    Only the depth row is scaled; the momentum rows keep their limited slope.
    Conservative (mean untouched) and well-balanced (``θ=1`` at still water).
    """

    def __init__(self, mesh, dim, h_index, eps_positivity=1e-14, **kw):
        super().__init__(mesh, dim, **kw)
        self._h_idx = int(h_index)
        self._eps_positivity = float(eps_positivity)
        self._build_cell_face_rays(mesh, dim)

    def __call__(self, Q, bf_face_values, phi=None, force_o1=None):
        n_vars = Q.shape[0]
        grads = self._compute_gradients(Q, n_vars, bf_face_values=bf_face_values)
        if phi is None:
            phi = self._compute_phi(Q, n_vars, bf_face_values, grads)
        else:
            phi = np.array(phi, copy=True)
        # A-priori XZS cap on the depth row: scale φ_h so the reconstructed
        # depth is ≥ 0 at every face of the cell.  ``h_slope`` is the already
        # slope-limited depth gradient (φ_h·∇h).
        h_idx = self._h_idx
        h_slope = phi[h_idx][np.newaxis, :] * grads[h_idx]
        theta = self._pp_theta(Q[h_idx], h_slope, self._eps_positivity)
        phi = np.array(phi, copy=True)
        phi[h_idx] = phi[h_idx] * theta
        # A-posteriori MOOD demotion on top of the a-priori cap (order-1 in the
        # flagged cells).  Applied last so it wins over the θ-scaled slope.
        phi = self._apply_force_o1(phi, force_o1)
        self._limited_grad = phi[:, None, :] * grads
        return self._reconstruct(Q, grads, phi, bf_face_values)


class FreeSurfaceLSQMUSCL(LSQMUSCLReconstruction):
    """LSQ MUSCL with wet-dry fallback for free-surface flows (ghost-cell-free).

    In dry cells (h < eps_wet), falls back to 1st order (phi = 0).
    Clamps h >= 0 at face states after reconstruction.
    """

    def __init__(self, mesh, dim, h_index, eps_wet=1e-3,
                 limiter="venkatakrishnan", unlimited_indices=None):
        super().__init__(mesh, dim, limiter=limiter,
                         unlimited_indices=unlimited_indices)
        self._h_idx = h_index
        self._eps_wet = eps_wet

    def __call__(self, Q, bf_face_values, phi=None, force_o1=None):
        n_vars = Q.shape[0]
        grads, phi = self._compute_limited_gradients(Q, n_vars, bf_face_values)
        # Wet-dry: zero limiter in dry cells — even on unlimited rows
        # (so bathymetry's slope stops contributing once h is dry).
        dry = Q[self._h_idx, :] < self._eps_wet
        phi[:, dry] = 0.0
        # This class inherits ``supports_force_o1 = True`` from the base but
        # used to override ``__call__`` WITHOUT the kwarg, so the solver's
        # a-posteriori MOOD pass raised ``unexpected keyword argument
        # 'force_o1'`` on every order-2 free-surface run.  The demotion is the
        # same operation as the dry fallback above: zero the limiter there.
        phi = self._apply_force_o1(phi, force_o1)
        # PUBLISH the limited cell-centre gradient, exactly as the parent does.
        # This override computes grads/phi itself and used to drop them on the
        # floor; the solver's cell-interior non-conservative integral guards on
        # ``getattr(reconstruct, "_limited_grad", None)`` and so was SILENTLY
        # skipped for every free-surface order-2 run -- no error, no NaN, just a
        # scheme quietly missing the interior source term that Noelle, Pankratz,
        # Puppo & Natvig (2006) eq. (15) requires of a 2nd-order Audusse-type
        # well-balanced scheme (1st order needs only the face jump, Audusse 2004
        # eq. 2.5, which is why order 1 was always right).
        self._limited_grad = phi[:, None, :] * grads
        Q_L, Q_R = self._reconstruct(Q, grads, phi, bf_face_values)
        # Clamp h >= 0
        np.maximum(Q_L[self._h_idx, :], 0.0, out=Q_L[self._h_idx, :])
        np.maximum(Q_R[self._h_idx, :], 0.0, out=Q_R[self._h_idx, :])
        return Q_L, Q_R


class SurfaceReconstruction:
    """Well-balanced reconstruction in surface-elevation variables.

    Wraps a base reconstruction (``ConstantReconstructionV2`` or
    ``LSQMUSCLReconstruction``).  Instead of reconstructing the water
    depth ``h`` directly, it reconstructs the free-surface elevation
    ``η = h + b`` together with the bathymetry ``b``, then recovers the
    per-side face depth ``h = η − b``.

    At lake-at-rest ``η`` is constant, so the limiter sees a flat field,
    produces a zero slope, and ``h = η − b`` is recovered with the full
    un-clipped ``b`` variation — which is what keeps the limiter
    well-balanced.  For piecewise-constant reconstruction the whole
    transform is an exact no-op.

    The depth / bathymetry locations are resolved **once** by the caller
    (the symbolic ``Numerics.find_field`` → ``FieldHandle``, or the
    solver's numpy analogue) and passed in here as plain indices — this
    class never searches for them, so there is a single source of truth
    for "which row is h, which field is b".

    Note: reconstructing ``η`` makes the *limiter* well-balanced, but at
    order ≥ 2 the flux + NCP discretization still needs the cell-interior
    non-conservative integral to telescope exactly.

    Parameters
    ----------
    base : reconstruction instance
        Underlying ``ConstantReconstructionV2`` or
        ``LSQMUSCLReconstruction``.
    h_index : int
        Row of the water depth in ``Q``.
    b_index : int
        Index of the bathymetry — a row of ``Q`` when ``b_in_state``,
        else a row of ``Qaux``.
    b_in_state : bool
        ``True`` when ``b`` lives in the conservative state.
    """

    def __init__(self, base, h_index, b_index, b_in_state):
        self.base = base
        self.h_index = int(h_index)
        self.b_index = int(b_index)
        self.b_in_state = bool(b_in_state)
        # Delegates the a-posteriori MOOD ``force_o1`` mask to the base
        # reconstruction (which zeroes the slope of flagged cells on the
        # η-extended array → constant h = η − b there).
        self.supports_force_o1 = bool(
            getattr(base, "supports_force_o1", False))

    def _extend(self, Q, Qaux, bf_Q):
        """Build the extended arrays the base reconstruction operates
        on: ``η = h + b`` in the depth slot, plus (when ``b`` lives in
        ``Qaux``) an appended ``b`` row.  Returns ``(Q_ext, bf_ext)``."""
        h_idx = self.h_index
        n_state = Q.shape[0]
        n_bf = bf_Q.shape[1]
        if self.b_in_state:
            b_idx = self.b_index
            b_cells = Q[b_idx, :]
            Q_ext = Q.copy()
            Q_ext[h_idx, :] = Q[h_idx, :] + b_cells          # η in h-slot
            bf_ext = bf_Q.copy()
            bf_ext[h_idx, :] = bf_Q[h_idx, :] + bf_Q[b_idx, :]
        else:
            b_cells = Qaux[self.b_index, :]
            i_inner = self.base._iInner_bnd
            Q_ext = np.vstack([Q, b_cells[None, :]])
            Q_ext[h_idx, :] = Q[h_idx, :] + b_cells          # η in h-slot
            bf_ext = np.zeros((n_state + 1, n_bf))
            bf_ext[:n_state, :] = bf_Q
            bf_ext[h_idx, :] = bf_Q[h_idx, :] + b_cells[i_inner]
            bf_ext[n_state, :] = b_cells[i_inner]
        return Q_ext, bf_ext

    def compute_phi(self, Q, Qaux, bf_Q):
        """Limiter coefficients of the extended (η-in-depth-slot) array
        — for freezing the limiter through an implicit Newton
        iteration.  ``None`` when the base reconstruction has no
        limiter."""
        Q_ext, bf_ext = self._extend(Q, Qaux, bf_Q)
        return self.base.compute_phi(Q_ext, bf_ext)

    def __call__(self, Q, Qaux, bf_Q, phi=None, force_o1=None):
        """Reconstruct face states in surface-elevation variables.

        Parameters
        ----------
        Q : ndarray, shape (n_state, n_inner_cells)
        Qaux : ndarray, shape (n_aux, n_inner_cells)
        bf_Q : ndarray, shape (n_state, n_boundary_faces)
            BC-provided cell-centre values at boundary faces.
        phi : ndarray, optional
            Pre-computed limiter coefficients of the *extended* array
            (see :meth:`compute_phi`).  When supplied the limiter is
            frozen — passed straight through to the base reconstruction
            — so the result is a smooth function of ``Q``.
        force_o1 : ndarray of bool, shape (n_inner_cells,), optional
            A-posteriori MOOD demotion mask forwarded to the base
            reconstruction: flagged cells go piecewise-constant on the
            η-extended array, so ``h = η − b`` collapses to the cell
            mean (order-1) there — no depth clamp.

        Returns
        -------
        Q_L, Q_R : ndarray, shape (n_state, n_faces)
            Face states with the depth row holding the recovered ``h``.
        b_L, b_R : ndarray, shape (n_faces,)
            Per-side reconstructed bathymetry — the caller writes these
            into the per-face ``Qaux`` (or they already sit in ``Q_L`` /
            ``Q_R`` when ``b_in_state``) so the hydrostatic
            reconstruction sees a consistent ``h + b``.
        """
        h_idx = self.h_index
        n_state = Q.shape[0]
        Q_ext, bf_ext = self._extend(Q, Qaux, bf_Q)
        QE_L, QE_R = self.base(Q_ext, bf_ext, phi=phi, force_o1=force_o1)

        if self.b_in_state:
            b_L = QE_L[self.b_index, :]
            b_R = QE_R[self.b_index, :]
            Q_L = QE_L.copy()
            Q_R = QE_R.copy()
        else:
            b_L = QE_L[n_state, :]
            b_R = QE_R[n_state, :]
            Q_L = QE_L[:n_state, :].copy()
            Q_R = QE_R[:n_state, :].copy()

        Q_L[h_idx, :] = QE_L[h_idx, :] - b_L                 # h = η − b
        Q_R[h_idx, :] = QE_R[h_idx, :] - b_R

        # Limited gradient of the *actual* state Q.  The base
        # reconstructed the extended (η-in-depth-slot) array, so the
        # depth slope is recovered as ∂_x h = ∂_x η − ∂_x b.
        g_ext = self.base._limited_grad
        if g_ext is None:
            self._limited_grad = None
        elif self.b_in_state:
            grad_Q = g_ext.copy()
            grad_Q[h_idx, :, :] = (
                g_ext[h_idx, :, :] - g_ext[self.b_index, :, :])
            self._limited_grad = grad_Q
        else:
            grad_Q = g_ext[:n_state, :, :].copy()
            grad_Q[h_idx, :, :] = g_ext[h_idx, :, :] - g_ext[n_state, :, :]
            self._limited_grad = grad_Q

        return Q_L, Q_R, b_L, b_R


# ── Diffusion operators ──────────────────────────────────────────────────────

class FaceGradient:
    """Precomputes face geometry and provides face gradient reconstruction.

    This class is instantiated once per mesh and reused across timesteps.
    """

    def __init__(self, mesh, dim):
        self.mesh = mesh
        self.dim = dim
        nc = mesh.n_inner_cells
        n_cells = mesh.n_cells
        n_faces = mesh.n_faces

        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        centers = mesh.cell_centers[:dim, :]
        normals = mesh.face_normals[:dim, :]

        self.iA = iA
        self.iB = iB
        self.n_faces = n_faces
        self.n_cells = n_cells
        self.nc = nc

        self.d_AB = np.zeros(n_faces)
        self.e_AB = np.zeros((dim, n_faces))

        for f in range(n_faces):
            dx = centers[:, iB[f]] - centers[:, iA[f]]
            dist = np.linalg.norm(dx)
            self.d_AB[f] = max(dist, 1e-30)
            self.e_AB[:, f] = dx / self.d_AB[f]

        self.n_dot_e = np.zeros(n_faces)
        for f in range(n_faces):
            self.n_dot_e[f] = np.dot(normals[:, f], self.e_AB[:, f])
            self.n_dot_e[f] = max(abs(self.n_dot_e[f]), TPFA_COS_FLOOR) * np.sign(self.n_dot_e[f] + 1e-30)

    def compute_cell_gradients(self, u, cell_volumes, face_volumes, normals):
        """Green-Gauss cell-center gradients."""
        dim = self.dim
        grad = np.zeros((dim, self.n_cells))
        for f in range(self.n_faces):
            a, b = self.iA[f], self.iB[f]
            u_face = 0.5 * (u[a] + u[b])
            for d in range(dim):
                contrib = u_face * normals[d, f] * face_volumes[f]
                grad[d, a] += contrib / cell_volumes[a]
                grad[d, b] -= contrib / cell_volumes[b]
        return grad

    def face_normal_gradient(self, u):
        """Corrected face-normal gradient: (u_B - u_A) / d_AB / (n·e).

        Returns shape (n_faces,).
        """
        grad_n = np.zeros(self.n_faces)
        for f in range(self.n_faces):
            grad_n[f] = (u[self.iB[f]] - u[self.iA[f]]) / self.d_AB[f] / self.n_dot_e[f]
        return grad_n


class DiffusionOperator:
    """Sparse discrete diffusion operator: L(u) = nabla·(nu nabla u).

    Assembled once per mesh + viscosity. Can be used for:
    - Explicit diffusion: dQ += dt * L @ Q
    - Implicit diffusion: solve (I - dt * L) @ Q^{n+1} = Q^*
    """

    def __init__(self, mesh, dim, nu=1.0):
        self.mesh = mesh
        self.dim = dim
        nc = mesh.n_inner_cells
        n_cells = mesh.n_cells

        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        centers = mesh.cell_centers[:dim, :]
        normals = mesh.face_normals[:dim, :]
        face_vol = mesh.face_volumes
        cell_vol = mesh.cell_volumes

        L = lil_matrix((nc, nc), dtype=float)

        for f in range(mesh.n_faces):
            a, b = iA[f], iB[f]
            dx = centers[:, b] - centers[:, a]
            dist = np.linalg.norm(dx)
            if dist < 1e-30:
                continue

            n = normals[:, f]
            n_dot_e = np.dot(n, dx / dist)
            n_dot_e = max(abs(n_dot_e), TPFA_COS_FLOOR) * np.sign(n_dot_e + 1e-30)
            coeff = nu * face_vol[f] / dist / abs(n_dot_e)

            if a < nc and b < nc:
                L[a, b] += coeff / cell_vol[a]
                L[a, a] -= coeff / cell_vol[a]
                L[b, a] += coeff / cell_vol[b]
                L[b, b] -= coeff / cell_vol[b]
            elif a < nc:
                pass
            elif b < nc:
                pass

        self.L = csr_matrix(L)
        self.nc = nc
        self.n_cells = n_cells

    def explicit(self, u):
        """Compute L @ u (for explicit stepping). Returns shape (nc,)."""
        return self.L @ u[:self.nc]

    def implicit_solve(self, u_star, dt, tol=1e-8, maxiter=100):
        """Crank-Nicolson: (I - dt/2 * L) u^{n+1} = (I + dt/2 * L) u*.

        Second-order in time for diffusion (backward Euler was first-order).
        """
        nc = self.nc
        rhs = u_star[:nc] + 0.5 * dt * (self.L @ u_star[:nc])

        def matvec(x):
            return x - 0.5 * dt * (self.L @ x)

        A = LinearOperator((nc, nc), matvec=matvec, dtype=float)
        sol, info = gmres(A, rhs, x0=u_star[:nc], atol=0.0, rtol=tol, maxiter=maxiter)

        result = np.zeros(self.n_cells)
        result[:nc] = sol
        mesh = self.mesh
        for i_bf in range(mesh.n_boundary_faces):
            ghost = mesh.boundary_face_ghosts[i_bf]
            inner = mesh.boundary_face_cells[i_bf]
            result[ghost] = result[inner]
        return result


class DiffusionOperatorV2:
    """Sparse discrete diffusion with boundary face contributions (ghost-cell-free).

    Interior faces are assembled into the sparse matrix L (same as DiffusionOperator).
    Boundary faces are stored separately — their contributions come from
    ``BC.face_gradient()`` and are added as an explicit RHS correction.

    Parameters
    ----------
    mesh : FVMMesh or LSQMesh
    dim : int
    nu : float
        Kinematic viscosity.
    """

    def __init__(self, mesh, dim, nu=1.0):
        nc = mesh.n_inner_cells
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        centers = mesh.cell_centers[:dim, :]
        normals = mesh.face_normals[:dim, :]
        face_vol = mesh.face_volumes
        cell_vol = mesh.cell_volumes
        self.nu = nu

        L = lil_matrix((nc, nc), dtype=float)
        # GEOMETRY (nu-free) per face for the STATE-DEPENDENT diffusivity path:
        # interior face → (a, b, g_a, g_b) with g_x = fv/dist/|n_dot_e|/cv_x.
        ia, ib, gA, gB = [], [], [], []

        # Interior faces only
        for f in range(mesh.n_faces):
            a, b = iA[f], iB[f]
            if not (a < nc and b < nc):
                continue
            dx = centers[:, b] - centers[:, a]
            dist = np.linalg.norm(dx)
            if dist < 1e-30:
                continue
            n = normals[:, f]
            n_dot_e = np.dot(n, dx / dist)
            n_dot_e = max(abs(n_dot_e), TPFA_COS_FLOOR) * np.sign(n_dot_e + 1e-30)
            g = face_vol[f] / dist / abs(n_dot_e)
            coeff = nu * g
            L[a, b] += coeff / cell_vol[a]
            L[a, a] -= coeff / cell_vol[a]
            L[b, a] += coeff / cell_vol[b]
            L[b, b] -= coeff / cell_vol[b]
            ia.append(a); ib.append(b)
            gA.append(g / cell_vol[a]); gB.append(g / cell_vol[b])

        self.L = csr_matrix(L)
        self.nc = nc
        self._tol = 1e-8
        self._ia = np.asarray(ia, dtype=int); self._ib = np.asarray(ib, dtype=int)
        self._gA = np.asarray(gA, dtype=float); self._gB = np.asarray(gB, dtype=float)

        # Boundary face data: (inner_cell, coefficient, dist) + nu-free geometry
        # coefficient = nu * face_vol / |n_dot_e| / cell_vol
        self._bf_data = []
        bf_inner, bf_g = [], []
        for i_bf in range(mesh.n_boundary_faces):
            fidx = mesh.boundary_face_face_indices[i_bf]
            inner = mesh.boundary_face_cells[i_bf]
            a, b = iA[fidx], iB[fidx]
            dx = centers[:, b] - centers[:, a]
            dist = np.linalg.norm(dx)
            if dist < 1e-30:
                self._bf_data.append((inner, 0.0, dist))
                bf_inner.append(inner); bf_g.append(0.0)
                continue
            n = normals[:, fidx]
            n_dot_e = np.dot(n, dx / dist)
            n_dot_e = max(abs(n_dot_e), TPFA_COS_FLOOR) * np.sign(n_dot_e + 1e-30)
            g = face_vol[fidx] / abs(n_dot_e) / cell_vol[inner]
            self._bf_data.append((inner, nu * g, dist))
            bf_inner.append(inner); bf_g.append(g)
        self._bf_inner = np.asarray(bf_inner, dtype=int)
        self._bf_g = np.asarray(bf_g, dtype=float)

    def explicit(self, u):
        """Interior-only diffusion: L @ u. Returns shape (nc,)."""
        return self.L @ u[:self.nc]

    def explicit_with_bc(self, u, bf_face_normal_grads, nu_cell=None):
        """Diffusion with boundary face contributions.

        Parameters
        ----------
        u : ndarray
            Scalar field (only the first ``nc`` inner-cell entries are used).
        bf_face_normal_grads : ndarray, shape (n_boundary_faces,)
            Face-normal gradient dQ/dn at each boundary face, from
            ``BC.face_gradient()``.
        nu_cell : ndarray, optional
            PER-CELL diffusivity (state-dependent, e.g. ``ν/h²`` = ``−A[v,v,d,d]``).
            When given, each face uses the average of its two cells' diffusivity
            and the boundary face uses the inner cell's — i.e. the actual rank-4
            ``diffusion_matrix`` coefficient, not a constant ``nu``.  ``None``
            falls back to the constant-``nu`` operator built in ``__init__``.

        Returns
        -------
        ndarray, shape (nc,)
        """
        if nu_cell is not None:
            u = np.asarray(u, dtype=float)[:self.nc]
            nu_cell = np.asarray(nu_cell, dtype=float)
            Lu = np.zeros(self.nc)
            if self._ia.size:
                fnu = 0.5 * (nu_cell[self._ia] + nu_cell[self._ib])
                du = u[self._ib] - u[self._ia]
                np.add.at(Lu, self._ia, fnu * self._gA * du)
                np.add.at(Lu, self._ib, -fnu * self._gB * du)
            if self._bf_inner.size:
                np.add.at(Lu, self._bf_inner,
                          nu_cell[self._bf_inner] * self._bf_g
                          * np.asarray(bf_face_normal_grads, dtype=float))
            return Lu
        Lu = self.L @ u[:self.nc]
        for i_bf, (inner, coeff, dist) in enumerate(self._bf_data):
            # Diffusive flux contribution: coeff * (du/dn) * dist
            # coeff = nu * fv / |n_dot_e| / cv, times dist gives
            # nu * fv * dist / |n_dot_e| / cv * (du/dn)
            # But face_gradient already returns (Q_face - Q_inner) / d_face,
            # and the standard FVM contribution is nu * (Q_B - Q_A) / dist * fv / cv / |n_dot_e|
            # = coeff * (Q_B - Q_A) = coeff * face_gradient * dist
            # Wait — let me re-derive:
            # Interior: contribution = nu * fv * (u_B - u_A) / dist / |n_dot_e| / cv
            #         = (nu * fv / |n_dot_e| / cv) * (u_B - u_A) / dist
            #         = coeff * (u_B - u_A) / dist
            # But coeff already includes / dist? No:
            # Original code: coeff = nu * face_vol[f] / dist / abs(n_dot_e)
            # then: L[a,b] += coeff / cell_vol[a], L[a,a] -= coeff / cell_vol[a]
            # So L[a,b] = nu * fv / dist / |n_dot_e| / cv
            # L @ u gives: sum_j L[i,j] * u[j]
            # For face f: L[a,b]*u[b] + L[a,a]*u[a] = coeff/cv * (u[b] - u[a])
            # = nu * fv / dist / |n_dot_e| / cv * (u_B - u_A)
            #
            # For boundary: same formula but u_B is not a DOF.
            # contribution = nu * fv / dist / |n_dot_e| / cv * (u_face - u_inner)
            # face_gradient = (u_face - u_inner) / d_face
            # So contribution = (nu * fv / |n_dot_e| / cv) * face_gradient
            # Our stored coeff = nu * fv / |n_dot_e| / cv  (no /dist this time)
            Lu[inner] += coeff * bf_face_normal_grads[i_bf]
        return Lu

    def implicit_solve_with_bc(self, u_star, dt, bf_face_normal_grads,
                               tol=1e-8, maxiter=100):
        """Crank-Nicolson with boundary contributions in RHS.

        Boundary terms are treated explicitly (in the RHS), not in the operator.

        Parameters
        ----------
        u_star : ndarray, shape (nc,) or larger (only [:nc] used)
            State after explicit step.
        dt : float
        bf_face_normal_grads : ndarray, shape (n_boundary_faces,)
            From ``BC.face_gradient()``.

        Returns
        -------
        ndarray, shape (nc,)
        """
        nc = self.nc
        u = u_star[:nc]

        # Boundary flux contribution (explicit)
        bf_contrib = np.zeros(nc)
        for i_bf, (inner, coeff, dist) in enumerate(self._bf_data):
            bf_contrib[inner] += coeff * bf_face_normal_grads[i_bf]

        # RHS: (I + dt/2 * L) u* + dt/2 * bf_contrib
        rhs = u + 0.5 * dt * (self.L @ u + bf_contrib)

        def matvec(x):
            return x - 0.5 * dt * (self.L @ x)

        A = LinearOperator((nc, nc), matvec=matvec, dtype=float)
        sol, info = gmres(A, rhs, x0=u, atol=0.0, rtol=tol, maxiter=maxiter)

        return sol


class DenseDiffusionOperator:
    r"""Dense, state-dependent implicit diffusion: ``div(A : grad Q)``.

    Assembles the two-point (normal-gradient) FVM divergence of the model's
    GENERAL diffusive flux ``F_diff[i,d] = Σ_{j,e} A[i,j,d,e] ∂_e Q[j]`` from
    the rank-4 ``diffusion_matrix`` ``A(Q, Qaux, p)`` (shape
    ``(n_eq, n_state, n_dim, n_dim)``).  Unlike :class:`DiffusionOperatorV2`
    (scalar ``ν`` × constant Laplacian, per-variable diagonal) this couples
    variables (dense ``A[i,j]``, ``i≠j``) and lets ``A`` depend on the state.

    Discretisation.  At each face the full gradient is approximated by its
    face-NORMAL two-point difference (TPFA), ``grad Q_j ≈ (∂Q_j/∂n)·n`` with
    the unit face normal ``n``.  Contracting the tensor against ``n`` on both
    the flux and the gradient index collapses ``A`` to a scalar variable
    coupling ::

        T[i,j] = Σ_{d,e} A[i,j,d,e] · n_d · n_e

    (face-averaged, ``Ā = ½(A_a + A_b)``), so the interior face-normal flux for
    equation ``i`` is ``Σ_j T[i,j]·(Q_{j,b}−Q_{j,a})/(n·Δx)·|f|``.  The
    geometry factors (``g = |f|/dist/|n·ê|`` interior, ``|f|/|n·ê|/V`` boundary)
    are IDENTICAL to :class:`DiffusionOperatorV2`, so a constant diagonal
    ``A = ν·I·δ_de`` reduces to ``ν·L`` bit-for-bit (unit normals ⇒
    ``T = ν·δ_ij``).  This is the choice the jax mirror must match.

    Time discretisation is Crank–Nicolson (θ=½): unconditionally stable, so no
    parabolic CFL.  ``state_dependent=False`` ⇒ ``A`` is constant ⇒ a SINGLE
    linear GMRES solve; ``state_dependent=True`` ⇒ ``A`` is evaluated at the
    unknown ``Q^{n+1}`` ⇒ the implicit half is NONLINEAR and is solved by
    Newton with a matrix-free (finite-difference JVP) GMRES inner solve.
    """

    def __init__(self, mesh, dim, n_vars, state_dependent=False):
        self.mesh = mesh
        self.dim = dim
        self.n_vars = int(n_vars)
        self.state_dependent = bool(state_dependent)
        nc = mesh.n_inner_cells
        iA = mesh.face_cells[0]
        iB = mesh.face_cells[1]
        centers = mesh.cell_centers[:dim, :]
        normals = mesh.face_normals[:dim, :]
        face_vol = mesh.face_volumes
        cell_vol = mesh.cell_volumes
        self.nc = nc

        # Interior-face geometry (nu-free) + unit normal per face — the SAME
        # per-face factors DiffusionOperatorV2 bakes into ``L``.
        ia, ib, gA, gB, n_int = [], [], [], [], []
        for f in range(mesh.n_faces):
            a, b = iA[f], iB[f]
            if not (a < nc and b < nc):
                continue
            dx = centers[:, b] - centers[:, a]
            dist = np.linalg.norm(dx)
            if dist < 1e-30:
                continue
            n = normals[:, f]
            n_dot_e = np.dot(n, dx / dist)
            n_dot_e = max(abs(n_dot_e), TPFA_COS_FLOOR) * np.sign(n_dot_e + 1e-30)
            g = face_vol[f] / dist / abs(n_dot_e)
            ia.append(a); ib.append(b)
            gA.append(g / cell_vol[a]); gB.append(g / cell_vol[b])
            n_int.append(n.copy())
        self._ia = np.asarray(ia, dtype=int)
        self._ib = np.asarray(ib, dtype=int)
        self._gA = np.asarray(gA, dtype=float)
        self._gB = np.asarray(gB, dtype=float)
        # (dim, n_int) unit normals for the T = Σ A n_d n_e contraction.
        self._n_int = (np.asarray(n_int, dtype=float).T
                       if n_int else np.zeros((dim, 0)))

        # Boundary-face geometry + unit normal (inner-cell tensor, per V2).
        bf_inner, bf_g, n_bf = [], [], []
        for i_bf in range(mesh.n_boundary_faces):
            fidx = mesh.boundary_face_face_indices[i_bf]
            inner = mesh.boundary_face_cells[i_bf]
            a, b = iA[fidx], iB[fidx]
            dx = centers[:, b] - centers[:, a]
            dist = np.linalg.norm(dx)
            if dist < 1e-30:
                bf_inner.append(inner); bf_g.append(0.0)
                n_bf.append(normals[:, fidx].copy())
                continue
            n = normals[:, fidx]
            n_dot_e = np.dot(n, dx / dist)
            n_dot_e = max(abs(n_dot_e), TPFA_COS_FLOOR) * np.sign(n_dot_e + 1e-30)
            g = face_vol[fidx] / abs(n_dot_e) / cell_vol[inner]
            bf_inner.append(inner); bf_g.append(g)
            n_bf.append(n.copy())
        self._bf_inner = np.asarray(bf_inner, dtype=int)
        self._bf_g = np.asarray(bf_g, dtype=float)
        self._n_bf = (np.asarray(n_bf, dtype=float).T
                      if n_bf else np.zeros((dim, 0)))

    # -- tensor normalisation ------------------------------------------

    def _as_cell_tensor(self, A):
        """Broadcast a lambdified ``diffusion_matrix`` return to the per-cell
        rank-4 layout ``(n_eq, n_state, nc, n_dim, n_dim)``.

        The numpy runtime stacks along a trailing axis, so a full-grid call
        yields ``(n_eq, n_state, nc, dim, dim)`` while a purely-constant tensor
        collapses to ``(n_eq, n_state, dim, dim)`` — both handled here."""
        A = np.asarray(A, dtype=float)
        d = self.dim
        if A.ndim == 5:
            return A
        if A.ndim == 4:  # (n_eq, n_st, dim, dim) — constant, no cell axis
            return np.broadcast_to(
                A[:, :, None, :, :],
                (A.shape[0], A.shape[1], self.nc, d, d))
        raise ValueError(
            f"diffusion_matrix returned ndim={A.ndim}; expected 4 or 5.")

    # -- discrete divergence  D[i] = (∇·F_diff)[i] ---------------------

    def _divergence(self, Q, A_cells, bf_grads=None):
        """Discrete ``∇·F_diff`` for state ``Q`` and per-cell tensor
        ``A_cells`` (``(n_eq, n_st, nc, dim, dim)``).

        When ``bf_grads`` (``(n_vars, n_bf)`` face-normal gradients from the BC
        kernel) is given the boundary diffusive flux is added; ``None`` returns
        the interior-only operator (used inside the implicit matvec / JVP)."""
        n_eq = A_cells.shape[0]
        D = np.zeros((n_eq, self.nc), dtype=float)
        if self._ia.size:
            A_a = A_cells[:, :, self._ia]           # (n_eq,n_st,nf,d,d)
            A_b = A_cells[:, :, self._ib]
            Aface = 0.5 * (A_a + A_b)
            # T[i,j,f] = Σ_{d,e} Ā[i,j,f,d,e] n_d n_e
            T = np.einsum('ijfde,df,ef->ijf', Aface,
                          self._n_int, self._n_int)
            dQ = Q[:, self._ib] - Q[:, self._ia]    # (n_st,nf)
            flux = np.einsum('ijf,jf->if', T, dQ)   # (n_eq,nf)
            np.add.at(D, (slice(None), self._ia),  flux * self._gA)
            np.add.at(D, (slice(None), self._ib), -flux * self._gB)
        if bf_grads is not None and self._bf_inner.size:
            A_in = A_cells[:, :, self._bf_inner]    # (n_eq,n_st,nbf,d,d)
            T_bf = np.einsum('ijbde,db,eb->ijb', A_in,
                             self._n_bf, self._n_bf)
            fluxb = np.einsum('ijb,jb->ib', T_bf,
                              np.asarray(bf_grads, dtype=float))
            np.add.at(D, (slice(None), self._bf_inner), fluxb * self._bf_g)
        return D

    # -- Crank–Nicolson implicit solve ---------------------------------

    def implicit_solve(self, Q_star, dt, A_fn, bf_grads,
                       tol=1e-8, maxiter=100, newton_maxiter=8,
                       newton_tol=1e-8, fd_eps=1e-7):
        r"""One implicit-diffusion sub-step.

        Solves the Crank–Nicolson update of ``∂_t Q = ∇·F_diff(Q)``::

            (Q^{n+1} − Q*)/dt = ½ [ 𝒟_impl(Q^{n+1}) + 𝒟_expl ],

        with ``𝒟_impl`` the interior divergence (``A`` at the current iterate)
        and ``𝒟_expl = interior(Q*) + boundary(Q*)`` the frozen explicit half
        (boundary treated explicitly from ``bf_grads``, exactly as
        :meth:`DiffusionOperatorV2.implicit_solve_with_bc`).

        ``A_fn(Q) -> diffusion_matrix`` closes over ``Qaux``/``p``.  For a
        state-independent operator this is a single linear GMRES solve; for a
        state-dependent one a Newton loop with a matrix-free (FD-JVP) GMRES
        inner solve."""
        Q_star = np.asarray(Q_star, dtype=float)
        n_vars, nc = Q_star.shape
        shape = Q_star.shape

        A_star = self._as_cell_tensor(A_fn(Q_star))
        # Explicit half (constant): interior(Q*) + boundary(Q*).
        D_expl = self._divergence(Q_star, A_star, bf_grads=bf_grads)

        if not self.state_dependent:
            # Linear CN: (I − dt/2 D_int) Q = Q* + dt/2 (D_int Q* + bf).
            rhs = (Q_star + 0.5 * dt * D_expl).reshape(-1)

            def matvec(x_flat):
                X = x_flat.reshape(shape)
                DX = self._divergence(X, A_star, bf_grads=None)
                return (X - 0.5 * dt * DX).reshape(-1)

            LinOp = LinearOperator((Q_star.size, Q_star.size),
                                   matvec=matvec, dtype=float)
            sol, _ = gmres(LinOp, rhs, x0=Q_star.reshape(-1),
                           atol=0.0, rtol=tol, maxiter=maxiter)
            return sol.reshape(shape)

        # State-dependent: Newton on R(Q) = Q − Q* − dt/2 (D_impl(Q) + D_expl).
        def residual(Q):
            D_impl = self._divergence(Q, self._as_cell_tensor(A_fn(Q)),
                                      bf_grads=None)
            return Q - Q_star - 0.5 * dt * (D_impl + D_expl)

        Q = np.array(Q_star, copy=True)
        for _ in range(newton_maxiter):
            R = residual(Q)
            if float(np.linalg.norm(R)) < newton_tol:
                break

            def jvp(v_flat, _Q=Q, _R=R):
                V = v_flat.reshape(shape)
                Rp = residual(_Q + fd_eps * V)
                return ((Rp - _R) / fd_eps).reshape(-1)

            J = LinearOperator((Q.size, Q.size), matvec=jvp, dtype=float)
            # Use the GMRES iterate even when it stops on maxiter (info>0):
            # the diffusion Jacobian ``I − dt/2 ∂𝒟`` is well-conditioned, so a
            # non-"converged" iterate is still a good Newton step — the outer
            # Newton loop absorbs the residual.  (Replacing it with ``−R`` would
            # be a catastrophic step.)
            delta, _info = gmres(J, (-R).reshape(-1), atol=0.0,
                                 rtol=tol, maxiter=maxiter)
            Q = Q + delta.reshape(shape)
        return Q

    def residual_norm(self, Q_np1, Q_star, dt, A_fn, bf_grads):
        r"""Norm of the Crank–Nicolson residual the Newton solve drives to zero::

            R = Q^{n+1} − Q* − dt/2 ( 𝒟_int(Q^{n+1}) + 𝒟_int(Q*) + bf ).

        This is the θ=½ analogue of the task's backward-Euler target
        ``(Q^{n+1}−Q*)/dt = ∇·F_diff(Q^{n+1})`` (θ=1); CN is used so the
        constant-diagonal case reduces to :class:`DiffusionOperatorV2`
        bit-for-bit.  ``→0`` ⇔ the implicit diffusion update is satisfied."""
        D_expl = self._divergence(
            Q_star, self._as_cell_tensor(A_fn(Q_star)), bf_grads=bf_grads)
        D_impl = self._divergence(
            Q_np1, self._as_cell_tensor(A_fn(Q_np1)), bf_grads=None)
        R = Q_np1 - Q_star - 0.5 * dt * (D_impl + D_expl)
        return float(np.linalg.norm(R))


class PrimitiveReconstruction:
    """Wraps a base ghost-cell-free reconstruction with pre/post
    transforms to the model's primitive well-balanced variables.

    Pre  (per cell / boundary face):  ``state → wb``   via ``forward_fn``.
    Limit (per cell):                 base reconstruction on the WB values.
    Post (per face, L and R side):    ``wb → state``    via ``inverse_fn``.

    The point of the WB transform is to limit the *physically*
    bounded quantities (``η = h+b``, ``u = q/h``, …) instead of the
    conservative state, killing momentum overshoot at the wet/dry
    front (Audusse-Bouchut-Bristeau and follow-ups; see
    ``thesis/chapters/30_numerics.md``).  The smooth-limiter
    choice (Venkatakrishnan, Van Albada) becomes harmless because
    overshoots happen on ``η`` (nearly flat near LAR) and ``u``
    (physically bounded), and the recovered ``q_face = h_face · u_face``
    is positivity-consistent by construction.

    ``forward_fn`` / ``inverse_fn`` are the model's OWN
    ``reconstruction_variables`` / ``state_from_reconstruction`` operators,
    lowered through :class:`~zoomy_core.transformation.to_numpy.
    NumpyRuntimeModel` exactly like ``flux`` / ``source`` / every other
    operator — so they carry the FULL declared signature
    (``OPERATOR_ARG_SLOTS["reconstruction_variables"] == (variables,
    aux_variables, parameters)``, ``["state_from_reconstruction"] ==
    (reconstruction, aux_variables, parameters)``, see
    ``systemmodel.system_model``) instead of a hand-rolled state-only
    lambdify.  A model whose forward/inverse map references an algebraic
    aux (e.g. the KP-desingularised ``hinv = 1/h`` that ``u = q·hinv``
    depends on) needs that ``Qaux`` argument to even evaluate — dropping it
    raised a bare ``NameError`` the first time an order-2 VAM case exercised
    this path.  A model that inlines its own regularization instead (no
    aux row at all) simply has an empty ``aux_variables`` group here and the
    lowered kernel ignores it — nothing in this class ever assumes a
    particular aux row exists, by name or position.

    Aux is genuinely PER-CELL / PER-FACE data (unlike ``parameters``, which
    are compile-time constants the caller binds once), so it cannot be
    baked in at construction — ``Qaux`` (cells) / ``Qaux_bf`` (boundary
    faces) are supplied fresh on every call, mirroring the ``Q`` /
    ``bf_face_values`` pair.  A model with zero aux rows still works: the
    caller passes ``(0, n)`` arrays and the lowered kernel's (empty)
    aux argument group contributes nothing.
    """

    #: Opt-in flag read by ``solver_numpy.get_flux_operator``'s flux
    #: operator (mirrors ``supports_force_o1``): only when set does the
    #: caller bother building/passing ``Qaux`` / ``Qaux_bf``.
    needs_aux = True

    def __init__(self, base, forward_fn, inverse_fn, n_aux, parameters):
        self.base = base
        self._forward = forward_fn      # rt.reconstruction_variables:  (Q, Qaux, p)  -> WB
        self._inverse = inverse_fn      # rt.state_from_reconstruction: (WB, Qaux, p) -> Q
        self._n_aux = n_aux
        self._parameters = parameters   # constant (n_params,) array — OPERATOR_ARG_SLOTS 'parameters' group

    def __call__(self, Q, bf_face_values, Qaux=None, Qaux_bf=None):
        # ``Q``: (n_state, n_inner_cells).  ``bf_face_values``: (n_state, n_boundary_faces).
        # ``Qaux`` / ``Qaux_bf``: (n_aux, n_inner_cells) / (n_aux, n_boundary_faces);
        # default to the (possibly empty) zero-aux case so a bare model — or a
        # caller that hasn't opted into ``needs_aux`` wiring — still works.
        if Qaux is None:
            Qaux = np.zeros((self._n_aux, Q.shape[1]))
        if Qaux_bf is None:
            Qaux_bf = np.zeros((self._n_aux, bf_face_values.shape[1]))

        wb_Q  = self._apply(self._forward, Q, Qaux)
        wb_bf = self._apply(self._forward, bf_face_values, Qaux_bf)
        wb_L, wb_R = self.base(wb_Q, wb_bf)

        # The inverse map is evaluated on the FULL face axis (interior +
        # boundary — ``self.base``'s own face count, ``Q_L``/``Q_R`` shape).
        # Aux is PIECEWISE-CONSTANT per cell everywhere else in this solver —
        # the Riemann dissipation reads the OWNING cell's aux, never a
        # reconstructed one (``get_flux_operator``'s ``qauxA``/``qauxB`` /
        # ``_ghost_aux``) — so the L/R aux fed to the inverse map at a face is
        # likewise the owning cell's (or the boundary ghost's), matching that
        # same convention instead of inventing a face-limited aux profile.
        aux_L, aux_R = self._face_aux(Qaux, Qaux_bf)
        Q_L = self._apply(self._inverse, wb_L, aux_L)
        Q_R = self._apply(self._inverse, wb_R, aux_R)
        # Publish a CONSERVATIVE-space limited gradient for the solver's
        # cell-interior non-conservative integral (B(Q)@dQ, Noelle, Pankratz,
        # Puppo & Natvig 2006 eq. 15) — it needs d/dx of the CONSERVATIVE
        # state, exactly what ``self.base`` already publishes correctly when
        # used bare (no WB wrapper).  Proxying ``self.base._limited_grad``
        # straight off the face-reconstruction call above (the previous
        # implementation) is NOT that gradient: that call ran on ``wb_Q``
        # (η=b+h, u=q·hinv, …), so both its slope AND its dry-cell demotion
        # are in WB/primitive space, not conservative space —
        #   * d(u)/dx = hinv·[dq/dx − u·dh/dx] carries an EXTRA hinv the
        #     model's ``nonconservative_matrix`` does not expect: B's own
        #     hinv-scaled entries are already the normal "advective
        #     velocity" of a quasilinear system (correct, matches every
        #     other well-balanced model) — paired with an ALSO-hinv-scaled
        #     WB slope they compound to hinv² as h → 0.  Verified on the
        #     Escalante bump: B's entries scale as a clean 1/h from
        #     h=0.14 down to h=1e-4 (eps_wet=1e-8 never becomes relevant),
        #     so an extra hinv factor from the gradient is the difference
        #     between the term staying physically-normal (~1/h, a velocity
        #     gradient) and blowing up (~1/h²) as a cell dries.
        #   * ``self.base``'s dry check compares ``Q[h_idx] < eps_wet``, and
        #     on the WB call ``Q`` IS ``wb_Q`` whose h_idx slot holds
        #     η = b+h (``reconstruction_rows = {h: b+h, …}``) — b is the
        #     static bed, so η tracks the topography height regardless of
        #     how thin h itself gets, and the demotion that is supposed to
        #     zero this very term in a drying cell never fires.
        # A second call on the TRUE conservative ``Q`` (this method's own,
        # untransformed argument) reuses the SAME LSQ + limiter + dry-
        # fallback + MOOD machinery with the RIGHT input instead of adding a
        # Jacobian or a new threshold: ``self.base`` then sees the real
        # depth, exactly as it does for every model that is NOT WB-wrapped.
        # Its Q_L/Q_R (discarded here) are NOT the well-balanced face states
        # returned below — those still come from the WB call above,
        # unchanged, so face reconstruction keeps its momentum-overshoot
        # protection at the wet/dry front.
        self.base(Q, bf_face_values)
        self._limited_grad = getattr(self.base, "_limited_grad", None)
        return Q_L, Q_R

    def _face_aux(self, Qaux, Qaux_bf):
        """Scatter cell / boundary aux onto ``self.base``'s full face axis,
        aligned with its own ``Q_L``/``Q_R`` convention
        (``LSQMUSCLReconstruction._reconstruct``): L = the face's owning
        inner cell, R = the neighbour cell (interior faces) or the boundary
        ghost ``Qaux_bf`` (boundary faces)."""
        base = self.base
        n_faces = base._n_faces
        aux_L = np.zeros((self._n_aux, n_faces))
        aux_R = np.zeros((self._n_aux, n_faces))
        aux_L[:, base._interior_faces] = Qaux[:, base._iA_int]
        aux_R[:, base._interior_faces] = Qaux[:, base._iB_int]
        aux_L[:, base._boundary_faces] = Qaux[:, base._iInner_bnd]
        aux_R[:, base._boundary_faces] = Qaux_bf
        return aux_L, aux_R

    def _apply(self, fn, arr, aux):
        """Apply the model's declared ``(state_like, Qaux, parameters)``
        operator to a WHOLE array of columns (cells or faces) at once — the
        same broadcasting convention every other lowered operator uses
        (``sm.flux(Q, Qaux, p)`` etc.), so this works identically whether the
        model declares 0, 1, or many aux rows."""
        return np.asarray(fn(arr, aux, self._parameters), dtype=float)


__all__ = (__all__ if "__all__" in dir() else []) + [
    "PrimitiveReconstruction",
]
