"""LSQMesh — FVMMesh with precomputed least-squares reconstruction stencils.

This is the highest-fidelity mesh class: it caches the LSQ derivative
operators so that ``compute_derivatives`` is one batched matrix–vector
product over all cells instead of building the stencil on every call.
"""

from __future__ import annotations

import numpy as np
import param
from scipy.sparse import lil_matrix, csr_matrix

from zoomy_core.mesh.base_mesh import BaseMesh
from zoomy_core.mesh.fvm_mesh import FVMMesh
from zoomy_core.mesh.lsq_reconstruction import (
    least_squares_reconstruction_local,
    scale_lsq_derivative,
    find_derivative_indices,
    build_monomial_indices,
)


def _build_face_neighbors(face_cells, cell_neighbors, n_faces, n_faces_per_cell, n_cells):
    """Build face neighbor stencil for flux reconstruction (vectorised):
    per face its two cells, then the other neighbours of both, sorted."""
    polynomial_degree = 1
    n_face_neighbors = (2 * (n_faces_per_cell + 1) - 2) * polynomial_degree
    sentinel = n_cells + 1
    c0, c1 = np.asarray(face_cells[0]), np.asarray(face_cells[1])
    cn = np.asarray(cell_neighbors)
    pad = np.full(cn.shape[1], sentinel)
    nb = lambda c: np.where((c < len(cn))[:, None], cn[np.minimum(c, len(cn) - 1)], pad)
    cand = np.concatenate([nb(c0), nb(c1)], axis=1)
    cand = np.where((cand < n_cells) & (cand != c0[:, None]) & (cand != c1[:, None]),
                    cand, sentinel)
    cand = np.sort(cand, axis=1)
    dup = np.zeros_like(cand, dtype=bool)
    dup[:, 1:] = cand[:, 1:] == cand[:, :-1]
    cand = np.sort(np.where(dup, sentinel, cand), axis=1)
    face_neighbors = np.concatenate([c0[:, None], c1[:, None], cand], axis=1)
    face_neighbors = face_neighbors[:, :n_face_neighbors]
    if face_neighbors.shape[1] < n_face_neighbors:
        face_neighbors = np.pad(face_neighbors, ((0, 0), (0, n_face_neighbors
                                - face_neighbors.shape[1])), constant_values=sentinel)
    return face_neighbors


class LSQMesh(FVMMesh):
    """FVMMesh + precomputed LSQ derivative operators."""

    _lsq_gradQ = param.Array(default=None, allow_None=True)
    _lsq_neighbors = param.Array(default=None, allow_None=True)
    _lsq_boundary_face_neighbors = param.Array(default=None, allow_None=True)
    _lsq_monomial_multi_index = param.Parameter(default=None)
    _lsq_scale_factors = param.Array(default=None, allow_None=True)
    _face_neighbors = param.Array(default=None, allow_None=True)

    @property
    def lsq_gradQ(self):
        return self._lsq_gradQ

    @property
    def lsq_neighbors(self):
        return self._lsq_neighbors

    @property
    def lsq_boundary_face_neighbors(self):
        return self._lsq_boundary_face_neighbors

    @property
    def lsq_monomial_multi_index(self):
        return self._lsq_monomial_multi_index

    @property
    def lsq_scale_factors(self):
        return self._lsq_scale_factors

    @property
    def face_neighbors(self):
        return self._face_neighbors

    def compute_derivatives(self, u: np.ndarray, degree: int = 1,
                            derivatives_multi_index=None, *,
                            u_boundary_face) -> np.ndarray:
        """Compute derivatives using precomputed LSQ stencil.

        Parameters
        ----------
        u_boundary_face : ndarray | ``'extrapolation'``
            **Required.**  Either ``(n_boundary_faces,)`` array of face
            values (from a prescribed BC kernel), or the string
            ``'extrapolation'`` for Neumann-zero / face = inner-cell.
            See :func:`zoomy_core.mesh.lsq_reconstruction.compute_derivatives`
            for the rationale: silent extrapolation was deprecated
            because it masks Dirichlet BCs as Neumann-zero.
        """
        from zoomy_core.mesh.lsq_reconstruction import (
            _resolve_u_boundary_face,
        )
        A_glob = self._lsq_gradQ
        neighbors = self._lsq_neighbors
        bdy_neighbors = self._lsq_boundary_face_neighbors
        mon_indices = self._lsq_monomial_multi_index
        sf = self._lsq_scale_factors

        has_bdy = bdy_neighbors is not None and bdy_neighbors.size > 0
        if has_bdy:
            u_boundary_face = _resolve_u_boundary_face(
                u, u_boundary_face, self)

        if derivatives_multi_index is None:
            derivatives_multi_index = mon_indices
        indices = find_derivative_indices(mon_indices, derivatives_multi_index)

        # Batched over all cells: gather the stencil values, then one
        # (n_mon, R) @ (R,) product per cell.  Ragged stencils are already
        # padded by the builder (own index i in ``neighbors`` -> delta_u = 0,
        # -1 in ``bdy_neighbors`` -> u[i] -> delta_u = 0) and the matching
        # rows of ``A_glob`` are exactly 0.0, so padded slots contribute 0.0.
        # The batched matmul runs the same per-cell gemv (same accumulation
        # order) as the old ``A_loc.T @ delta_u`` loop -> bit-exact; einsum
        # is not.
        u = np.asarray(u)
        n_cells = A_glob.shape[0]
        if u.shape[0] < n_cells:
            # The old loop read u[i] for every cell (ghosts included);
            # keep that contract and its error class.
            raise IndexError(
                f"u has length {u.shape[0]} but the stencil covers "
                f"{n_cells} cells (inner + ghost)")
        u_self = u[:n_cells]                                   # (n_cells,)
        u_full = u[neighbors]                                  # (n_cells, max_nbr)
        if has_bdy:
            u_bdy = np.where(
                bdy_neighbors >= 0,
                u_boundary_face[np.maximum(bdy_neighbors, 0)],
                u_self[:, None],
            )                                                  # (n_cells, max_bdy)
            u_full = np.concatenate([u_full, u_bdy], axis=1)   # (n_cells, R)
        delta_u = u_full - u_self[:, None]                     # (n_cells, R)
        coeffs = (A_glob.transpose(0, 2, 1) @ delta_u[:, :, None])[:, :, 0]
        # C-contiguous like the old ``np.zeros`` output buffer.
        return np.ascontiguousarray((sf * coeffs)[:, indices], dtype=float)

    def derivative_operator(self, multi_index) -> csr_matrix:
        """Sparse-matrix realisation of the LSQ derivative stencil.

        Returns a ``(n_inner_cells, n_inner_cells)`` sparse matrix
        ``D`` such that ``D @ u`` equals the cell-wise estimate of the
        derivative ``∂^multi_index u`` — the *same* quantity
        :meth:`compute_derivatives` produces for that ``multi_index``,
        but as an explicit **linear operator**.

        ``compute_derivatives`` applies the stencil to a *known* field;
        ``derivative_operator`` exposes the stencil itself, for
        assembling implicit / elliptic systems where the field is the
        unknown (e.g. the Chorin pressure-projection ``A·P = rhs``,
        where ``∂_xx P`` must enter the matrix, not be evaluated).

        Per cell ``i`` the stencil is
        ``deriv[i] = Σ_j w_j·(u[nbr_j] − u[i])`` with
        ``w_j = sf[idx]·A_loc[j, idx]`` — so
        ``D[i, nbr_j] += w_j`` and ``D[i, i] -= w_j``.

        Parameters
        ----------
        multi_index : tuple[int]
            Spatial-derivative orders per axis, e.g. ``(1,)`` for
            ``∂_x`` or ``(2,)`` for ``∂_xx`` in 1D — the same
            convention as :meth:`compute_derivatives` and the
            ``aux_registry`` ``multi_index`` field.

        Raises
        ------
        ValueError
            If ``multi_index`` is not in the mesh's monomial set —
            i.e. the mesh was built with too low an ``lsq_degree``.
        """
        multi_index = tuple(int(o) for o in multi_index)
        idx = int(find_derivative_indices(
            self._lsq_monomial_multi_index, [multi_index])[0])
        if idx < 0:
            raise ValueError(
                f"multi_index {multi_index} is not in the LSQ monomial "
                f"set {self._lsq_monomial_multi_index} — rebuild the "
                f"mesh with a higher lsq_degree."
            )
        A_glob = self._lsq_gradQ
        neighbors = self._lsq_neighbors
        sf = self._lsq_scale_factors
        nc = self.n_inner_cells
        D = lil_matrix((nc, nc), dtype=float)
        for i in range(nc):
            w = sf[idx] * A_glob[i][:, idx]          # stencil weights
            for j, n in enumerate(neighbors[i]):
                if n >= nc:
                    continue
                D[i, n] += w[j]
                D[i, i] -= w[j]
        return csr_matrix(D)

    @classmethod
    def from_fvm(cls, fvm: FVMMesh) -> "LSQMesh":
        """Build an LSQMesh shell from an FVMMesh, populating the LSQ
        stencil at degree 1 (the minimum that supports any derivative
        reconstruction).

        The LSQ polynomial degree is **not** a hand-adjustable knob
        here — use :func:`zoomy_core.mesh.ensure_lsq_mesh(mesh, model)`
        for any solver setup, which sizes the stencil from the model's
        NumericalSystemModel.  This factory exists for low-level mesh
        construction; the degree is set by the model in the higher
        layers."""
        lsq = cls(
            dimension=fvm.dimension,
            type=fvm.type,
            n_cells=fvm.n_cells,
            n_inner_cells=fvm.n_inner_cells,
            n_faces=fvm.n_faces,
            n_vertices=fvm.n_vertices,
            n_boundary_faces=fvm.n_boundary_faces,
            n_faces_per_cell=fvm.n_faces_per_cell,
            vertex_coordinates=fvm.vertex_coordinates,
            cell_vertices=fvm.cell_vertices,
            cell_faces=fvm.cell_faces,
            face_cells=fvm.face_cells,
            cell_neighbors=fvm.cell_neighbors,
            boundary_face_cells=fvm.boundary_face_cells,
            boundary_face_ghosts=fvm.boundary_face_ghosts,
            boundary_face_function_numbers=fvm.boundary_face_function_numbers,
            boundary_face_physical_tags=fvm.boundary_face_physical_tags,
            boundary_face_face_indices=fvm.boundary_face_face_indices,
            boundary_conditions_sorted_physical_tags=fvm.boundary_conditions_sorted_physical_tags,
            boundary_conditions_sorted_names=fvm.boundary_conditions_sorted_names,
            z_ordering=fvm.z_ordering,
            _cell_centers=fvm._cell_centers,
            _cell_volumes=fvm._cell_volumes,
            _cell_inradius=fvm._cell_inradius,
            _face_normals=fvm._face_normals,
            _face_volumes=fvm._face_volumes,
            _face_centers=fvm._face_centers,
        )

        lsq._build_lsq_stencil(degree=1)

        lsq._face_neighbors = _build_face_neighbors(
            fvm.face_cells, fvm.cell_neighbors,
            fvm.n_faces, fvm.n_faces_per_cell, fvm.n_cells,
        )

        return lsq

    def _build_lsq_stencil(self, degree: int) -> None:
        """Rebuild the LSQ stencil at the given polynomial ``degree``.

        Called from :meth:`from_fvm` (with ``degree=1``) and from
        :func:`zoomy_core.mesh.ensure_lsq_mesh` (with the model-derived
        degree).  Mutates the LSQ-cache slots on ``self`` in place.

        The ``degree`` parameter is **internal**: user code never sets
        it.  The right entry point for solver setup is
        ``ensure_lsq_mesh(mesh, model)``."""
        from zoomy_core.mesh import geometry_xp
        if str(self.type) in geometry_xp._SUPPORTED:      # vectorised
            topo = geometry_xp.build_topology(self, lsq_degree=degree)
            g = geometry_xp.compute_geometry(
                topo, np.asarray(self.vertex_coordinates)[:self.dimension])
            self._lsq_gradQ = np.asarray(g["lsq_gradQ"])
            self._lsq_neighbors = topo.lsq_neighbors
            self._lsq_boundary_face_neighbors = topo.lsq_boundary_face_neighbors
            self._lsq_monomial_multi_index = build_monomial_indices(
                degree, self.dimension)
            self._lsq_scale_factors = scale_lsq_derivative(
                self._lsq_monomial_multi_index)
            return
        dim = self.dimension
        centers = self.cell_centers_computed()
        face_centers = self.face_centers_computed()
        bdy_face_centers = face_centers[
            self.boundary_face_face_indices, :dim
        ]
        cell_boundary_faces = [[] for _ in range(self.n_cells)]
        for i_bf, inner_cell in enumerate(self.boundary_face_cells):
            cell_boundary_faces[int(inner_cell)].append(i_bf)

        (self._lsq_gradQ,
         self._lsq_neighbors,
         self._lsq_boundary_face_neighbors,
         self._lsq_monomial_multi_index) = (
            least_squares_reconstruction_local(
                self.n_cells, dim, self.cell_neighbors,
                centers[:dim, :].T, degree,
                n_inner_cells=self.n_inner_cells,
                boundary_face_centers=bdy_face_centers,
                cell_boundary_faces=cell_boundary_faces,
                cell_vertices=self.cell_vertices,
            )
        )
        self._lsq_scale_factors = scale_lsq_derivative(
            self._lsq_monomial_multi_index)

    def _current_lsq_degree(self) -> int:
        """Polynomial degree currently represented by the cached
        stencil.  Used by ``ensure_lsq_mesh`` to decide if a rebuild
        is needed."""
        if self._lsq_monomial_multi_index is None:
            return 0
        return max(sum(mi) for mi in self._lsq_monomial_multi_index)

    @classmethod
    def from_msh(cls, filepath: str) -> "LSQMesh":
        fvm = FVMMesh.from_msh(filepath)
        return cls.from_fvm(fvm)

    @classmethod
    def from_hdf5(cls, filepath: str) -> "LSQMesh":
        fvm = FVMMesh.from_hdf5(filepath)
        return cls.from_fvm(fvm)

    @classmethod
    def create_1d(cls, domain: tuple, n_inner_cells: int) -> "LSQMesh":
        fvm = FVMMesh.create_1d(domain, n_inner_cells)
        return cls.from_fvm(fvm)

    @classmethod
    def create_2d(cls, domain: tuple, nx: int, ny: int) -> "LSQMesh":
        fvm = FVMMesh.create_2d(domain, nx, ny)
        return cls.from_fvm(fvm)

    @classmethod
    def create_3d(cls, domain: tuple, nx: int, ny: int, nz: int) -> "LSQMesh":
        fvm = FVMMesh.create_3d(domain, nx, ny, nz)
        return cls.from_fvm(fvm)
