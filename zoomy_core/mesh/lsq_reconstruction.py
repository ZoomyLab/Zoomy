"""Least-squares polynomial reconstruction for FVM derivative estimation.

Pure-numpy utilities: monomial construction, Vandermonde matrices,
weighted least-squares stencils, and cell-wise derivative computation.
"""

from __future__ import annotations

from itertools import product
from math import comb, factorial

import numpy as np


# ── Monomial helpers ──────────────────────────────────────────────────────────

def build_monomial_indices(degree: int, dim: int):
    """Build monomial multi-indices up to total degree (excluding constant)."""
    mon_indices = []
    for powers in product(range(degree + 1), repeat=dim):
        if 0 < sum(powers) <= degree:
            mon_indices.append(powers)
    return mon_indices


def scale_lsq_derivative(mon_indices):
    """Factorial scaling so LSQ coefficients equal derivatives."""
    return np.array(
        [np.prod([factorial(k) for k in mi]) for mi in mon_indices]
    )


def find_derivative_indices(full_monomials_arr, requested_derivs_arr):
    """Map requested derivative multi-indices to positions in monomial list.

    Returns shape (M,) array; -1 where a requested derivative is not found.
    """
    full_monomials_arr = np.array(full_monomials_arr)
    requested_derivs_arr = np.array(requested_derivs_arr)
    matches = np.all(
        full_monomials_arr[:, None, :] == requested_derivs_arr[None, :, :],
        axis=-1,
    )
    found = np.any(matches, axis=0)
    indices = np.argmax(matches, axis=0)
    return np.where(found, indices, -1)


def get_polynomial_degree(mon_indices):
    return max(sum(m) for m in mon_indices)


def get_required_monomials_count(degree, dim):
    return comb(int(degree + dim), int(dim))


# ── Vandermonde & weights ─────────────────────────────────────────────────────

def build_vandermonde(cell_diffs, mon_indices):
    n_neighbors, dim = cell_diffs.shape
    n_monomials = len(mon_indices)
    V = np.zeros((n_neighbors, n_monomials))
    for i, powers in enumerate(mon_indices):
        V[:, i] = np.prod(cell_diffs ** powers, axis=1)
    return V


def expand_neighbors(neighbors_list, initial_neighbors, n_valid=None):
    """Expand neighbor set by one ring.  Skips sentinel/invalid indices."""
    expanded = set(initial_neighbors)
    for n in initial_neighbors:
        if n_valid is not None and n >= n_valid:
            continue
        if n < len(neighbors_list):
            nbrs = neighbors_list[n]
            for nb in nbrs:
                if n_valid is None or nb < n_valid:
                    expanded.add(nb)
    return list(expanded)


def build_vertex_to_cells(cell_vertices, n_inner_cells):
    """Inverse of ``cell_vertices``: for each vertex, the list of inner
    cells that touch it.  ``cell_vertices`` has shape
    ``(verts_per_cell, n_cells)``.  Ghost cells (``ic >= n_inner_cells``)
    are excluded — the LSQ stencil only samples interior cells."""
    n_cells = cell_vertices.shape[1]
    # Vertex ids are dense from 0..n_vertices-1.  Use the max ID as size.
    n_v = int(cell_vertices[:, :n_inner_cells].max()) + 1 if n_inner_cells > 0 else 0
    v2c = [[] for _ in range(n_v)]
    for ic in range(min(n_cells, n_inner_cells)):
        for v in cell_vertices[:, ic]:
            v_int = int(v)
            if 0 <= v_int < n_v:
                v2c[v_int].append(ic)
    return v2c


def vertex_one_ring(ic, cell_vertices, vertex_to_cells, n_inner_cells):
    """All inner cells sharing at least one vertex with ``ic``.
    Excludes ``ic`` itself.  On a structured 2D quad/triangle mesh
    this is the geometrically symmetric stencil around ``ic``."""
    ring = set()
    for v in cell_vertices[:, ic]:
        v_int = int(v)
        if v_int < 0 or v_int >= len(vertex_to_cells):
            continue
        for jc in vertex_to_cells[v_int]:
            if jc != ic and jc < n_inner_cells:
                ring.add(jc)
    return sorted(ring)


def compute_gaussian_weights(dX, sigma=1.0):
    distances = np.linalg.norm(dX, axis=1)
    return np.exp(-((distances / sigma) ** 2))


# ── Core LSQ reconstruction ──────────────────────────────────────────────────

def least_squares_reconstruction_local(
    n_cells, dim, neighbors_list, cell_centers, lsq_degree,
    n_inner_cells=None,
    boundary_face_centers=None,
    cell_boundary_faces=None,
    cell_vertices=None,
):
    """Build per-cell LSQ gradient operators.

    Interior cells (index < ``n_inner_cells``) are used as neighbors.
    Boundary cells additionally include the boundary-face center(s)
    they touch as **virtual sampling positions** — the LSQ row at a
    boundary face contributes the BC-prescribed face value, so the
    cell-centered derivative is boundary-aware.  This is the
    ghost-cell-free analogue of "apply boundary_operator then take
    the gradient": the boundary face is a first-class entity in the
    mesh.

    The virtual sampling POSITION is the GHOST-CELL offset
    ``2·(face_center − cell_center)`` (= dx in 1-D) — see the ``dX``
    assembly below — so the VALUE carried there must be a GHOST value.
    (An earlier revision of this docstring claimed the sample sat at
    ``|face_center − cell_center|`` = dx/2; the code has always used
    2·, and a caller trusting the dx/2 claim placed a face value at a
    ghost position.)

    Stencil selection
    -----------------
    * When ``cell_vertices`` is provided (recommended for ≥2D),
      the per-cell stencil is the **full vertex-1-ring**: every
      inner cell sharing at least one vertex with the current cell.
      On a structured quad/triangle mesh this is geometrically
      symmetric around the cell centre (offset sum ≈ 0), giving
      O(h²) gradient accuracy at the cell centre on smooth fields.
    * When ``cell_vertices`` is None, the legacy face-neighbour +
      ``expand_neighbors`` iteration is used to fill the stencil.
      That path is symmetric in 1D but biased in 2D/3D — only kept
      for backward compatibility with callers that lack vertex
      topology.

    Stencil selection
    -----------------
    * When ``cell_vertices`` is provided (recommended for ≥2D),
      the per-cell stencil is the **full vertex-1-ring**: every
      inner cell sharing at least one vertex with the current cell.
      On a structured quad/triangle mesh this is geometrically
      symmetric around the cell centre (offset sum ≈ 0), giving
      O(h²) gradient accuracy at the cell centre on smooth fields.
    * When ``cell_vertices`` is None, the legacy face-neighbour +
      ``expand_neighbors`` iteration is used to fill the stencil.
      That path is symmetric in 1D but biased in 2D/3D — only kept
      for backward compatibility with callers that lack vertex
      topology.

    Parameters
    ----------
    boundary_face_centers : ndarray, shape ``(n_boundary_faces, dim)``, optional
        Face centers of all boundary faces.  Required when
        ``cell_boundary_faces`` provides any non-empty entry.
    cell_boundary_faces : sequence, length ``n_cells``, optional
        For each cell, the list of boundary-face indices touching that
        cell.  Empty list for interior cells.  When omitted, the
        stencil reduces to the legacy interior-only form.
    cell_vertices : ndarray, shape ``(verts_per_cell, n_cells)``, optional
        Vertex topology used to build a symmetric vertex-1-ring
        stencil.  Strongly recommended for 2D/3D meshes — without
        it the stencil is biased and gradient convergence drops to
        O(h).

    Returns
    -------
    A_glob : ``(n_cells, max_neighbors + max_bdy_per_cell, n_monomials)``
    neighbors_array : ``(n_cells, max_neighbors)`` int
    boundary_face_neighbors_array : ``(n_cells, max_bdy_per_cell)`` int
        Boundary-face indices used by each cell's stencil, padded with
        ``-1`` in unused slots.  Empty array (shape ``(n_cells, 0)``)
        when no boundary-face augmentation was requested.
    mon_indices : list of tuples
    """
    if n_inner_cells is None:
        n_inner_cells = n_cells
    if cell_boundary_faces is None:
        cell_boundary_faces = [[] for _ in range(n_cells)]
    has_bdy = any(len(b) > 0 for b in cell_boundary_faces)
    if has_bdy and boundary_face_centers is None:
        raise ValueError(
            "cell_boundary_faces lists boundary faces but "
            "boundary_face_centers is None."
        )
    max_bdy_per_cell = (max(len(b) for b in cell_boundary_faces)
                        if has_bdy else 0)

    mon_indices = build_monomial_indices(lsq_degree, dim)
    degree = get_polynomial_degree(mon_indices)
    required_neighbors = get_required_monomials_count(degree, dim)

    use_vertex_ring = cell_vertices is not None and dim >= 2
    if use_vertex_ring:
        cell_vertices = np.asarray(cell_vertices)
        # ``cell_vertices`` covers the inner cells only on most
        # FVMMesh-derived meshes (shape[1] == n_inner_cells); the
        # vertex-ring branch is bypassed for indices that fall
        # outside that range.
        n_inner_for_vring = min(n_inner_cells, cell_vertices.shape[1])
        vertex_to_cells = build_vertex_to_cells(
            cell_vertices, n_inner_for_vring)

    neighbors_all = []
    for i_c in range(n_cells):
        if use_vertex_ring and i_c < n_inner_for_vring:
            # Symmetric vertex-1-ring stencil.
            current_neighbors = vertex_one_ring(
                i_c, cell_vertices, vertex_to_cells, n_inner_cells)
            # Top-up with face-neighbour expansion only if the ring
            # itself is somehow too small (degenerate corner cells
            # on irregular meshes); on a structured grid this branch
            # is never taken for interior cells.
            if len(current_neighbors) < required_neighbors:
                fill = [n for n in neighbors_list[i_c]
                        if n < n_inner_cells and n != i_c
                        and n not in current_neighbors]
                current_neighbors = current_neighbors + fill
                while len(current_neighbors) < required_neighbors:
                    extended = expand_neighbors(
                        neighbors_list, current_neighbors,
                        n_valid=n_inner_cells)
                    new = [n for n in extended
                           if n != i_c and n not in current_neighbors]
                    if not new:
                        break
                    current_neighbors.extend(new)
        else:
            # Legacy path: face-neighbour expansion.
            current_neighbors = [n for n in neighbors_list[i_c]
                                 if n < n_inner_cells]
            while len(current_neighbors) < required_neighbors:
                new_neighbors = expand_neighbors(
                    neighbors_list, current_neighbors, n_valid=n_inner_cells)
                current_neighbors = list(set(new_neighbors) - {i_c})
                if len(current_neighbors) == 0:
                    break
        neighbors_all.append(current_neighbors)

    # legacy path: the width of the INNER stencils -- a ghost row (its inner
    # cell's neighbours, the inner cell itself included) is cut to it, never
    # widens the interior stencils one-sidedly
    max_neighbors = max(len(nbrs) for nbrs in (
        neighbors_all if use_vertex_ring else neighbors_all[:n_inner_cells]))
    A_glob = []
    neighbors_array = np.empty((n_cells, max_neighbors), dtype=int)
    bdy_neighbors_array = -np.ones((n_cells, max_bdy_per_cell), dtype=int)

    # REQ-174b: axis → rows of the *pure-normal* monomials for that axis
    # (``e_axis``, ``2·e_axis``, …).  The boundary ghost is kept coupled ONLY
    # to these pure boundary-normal derivatives at ≥2-D boundary cells; the
    # tangential and mixed derivatives are fit from the interior stencil alone
    # (see the pinv block below).
    pure_axis_rows = {}
    for k, mi in enumerate(mon_indices):
        nz = [a for a in range(dim) if mi[a] != 0]
        if len(nz) == 1:
            pure_axis_rows.setdefault(nz[0], []).append(k)

    for i_c in range(n_cells):
        current_neighbors = list(neighbors_all[i_c])
        n_nbr = len(current_neighbors)

        if use_vertex_ring:
            # Keep the stencil exactly as the vertex-1-ring (symmetric
            # by construction).  Pad shorter stencils with the cell
            # itself — those rows have zero offset and are zeroed
            # out in V below, contributing nothing to the LSQ fit.
            # Do NOT expand via expand_neighbors here: that would
            # re-introduce the centroid bias the vertex ring removed.
            pass
        else:
            while n_nbr < max_neighbors:
                extended_neighbors = expand_neighbors(
                    neighbors_list, current_neighbors, n_valid=n_inner_cells)
                extended_neighbors = list(set(extended_neighbors) - {i_c})
                new_neighbors = [n for n in extended_neighbors
                                 if n not in current_neighbors and n < n_inner_cells]
                current_neighbors.extend(new_neighbors)
                n_nbr = len(current_neighbors)
                if n_nbr >= max_neighbors:
                    break

        if len(current_neighbors) >= max_neighbors:
            trimmed_neighbors = current_neighbors[:max_neighbors]
        else:
            padding = [i_c] * (max_neighbors - len(current_neighbors))
            trimmed_neighbors = current_neighbors + padding

        neighbors_array[i_c, :] = trimmed_neighbors

        # Boundary-face virtual neighbors for this cell.
        bdy_faces = list(cell_boundary_faces[i_c])
        n_bdy_here = len(bdy_faces)
        bdy_neighbors_array[i_c, :n_bdy_here] = bdy_faces

        # Build dX with [cell rows | boundary-ghost rows].
        # Boundary points are placed at the GHOST-CELL position (offset
        # = 2·(face - cell)) — the symmetric image of the inner cell
        # through the boundary face.  This matches the FV ghost-cell
        # convention used by OpenFOAM, JAX legacy VAM Poisson, and the
        # standard FV literature.  Placing the boundary point at the
        # FACE (offset dx/2) over-constrains the LSQ for Neumann-zero
        # BCs (which is the common case for h, b, P): two stencil
        # points with the same value at separated x positions force the
        # boundary slope to be too close to zero, distorting the
        # elliptic-block coefficients.  The ghost-cell placement at
        # offset dx makes the slope between ghost and cell_0 = 0 only at
        # the MIDPOINT — i.e. exactly at the boundary face — giving the
        # correct Neumann-zero discretisation.  For prescribed-value
        # BCs the caller supplies ``u_ghost = 2·u_face - u_cell_0`` via
        # ``_resolve_u_boundary_face``, recovering linear extrapolation
        # through the face.
        total_rows = max_neighbors + max_bdy_per_cell
        dX = np.zeros((total_rows, dim), dtype=float)
        for j, neighbor in enumerate(trimmed_neighbors):
            dX[j, :] = cell_centers[neighbor] - cell_centers[i_c]
        for j, bf in enumerate(bdy_faces):
            dX[max_neighbors + j, :] = (
                2 * (boundary_face_centers[bf] - cell_centers[i_c])
            )
        # Pad unused boundary-face rows by repeating the cell itself
        # (zero offset, contributes nothing to the LSQ fit).
        # ``dX`` rows with zero distance get zero Gaussian weight via
        # the eps check below; but to keep them harmless we leave them
        # at zero (cell-center offset) and let the weight handle it.

        V = build_vandermonde(dX, mon_indices)
        # Apples-to-apples with the JAX-legacy LSQ (``/tmp/vam_legacy_jax.py``):
        # unweighted Moore–Penrose pseudoinverse, no Tikhonov
        # regularisation.  The previous Gaussian-weighted +
        # α-regularised form gave subtly different gradients at boundary
        # / shock cells (chain h_x ≈ 1.5× JAX h_x at cell 0 / cell 50),
        # which contaminated the elliptic-block coefficients in the
        # pressure projection.  Switch to plain ``pinv(V)`` so chain LSQ
        # numerically equals JAX LSQ given the same stencil + values.
        # Zero out the padding boundary-face rows by replacing those
        # rows of V with zero (so they contribute nothing to the
        # pseudoinverse).
        for j in range(n_bdy_here, max_bdy_per_cell):
            V[max_neighbors + j, :] = 0.0

        # REQ-174b — boundary-ghost / tangential-derivative decoupling.
        #
        # The extrapolation/Neumann ghost is placed at the mirror point
        # 2·(face − cell); with an extrapolation value it encodes the ∂_n = 0
        # (zeroGradient / Neumann) boundary condition along its own normal
        # axis, and with a prescribed value the Dirichlet BC.  That is genuine
        # boundary information for the *pure boundary-normal* derivatives
        # (∂_n, ∂_nn, …): the Chorin pressure Poisson NEEDS ∂_nn kept — it is
        # the discrete Neumann coupling at the lateral walls, and dropping it
        # makes the elliptic operator gain a near-null transverse mode
        # (measured: exact solve then blows the extruded VAM case).
        #
        # But when the SAME ghost is left in the coupled multivariate fit it
        # also corrupts the TANGENTIAL and MIXED derivatives via cross terms:
        # on an exact y-invariant quadratic f=1+2x+3x² a boundary cell gets
        # d_yy≈21, d_xy≈14 (vs exact 0), which seeds spurious y-variation into
        # the elliptic (∂_yy P) and corrector (∂_y P) and destabilises 2-D VAM.
        #
        # Fix: at ≥2-D boundary cells, fit the tangential and mixed
        # derivatives from the INTERIOR stencil only (already complete and
        # symmetric there), and restore each PURE-NORMAL derivative (∂_n,
        # ∂_nn, …) from a fit carrying ONLY its own aligned ghost — so the
        # elliptic Neumann coupling is preserved and corner cells (two ghosts)
        # do not cross-contaminate.  A boundary edge cell's pure-normal
        # derivatives are therefore bit-identical to the pre-fix coupled fit,
        # preserving the REQ-174 BC-aware gradient, well-balancing and the
        # elliptic well-posedness.  In 1-D there is no tangential direction ⇒
        # this branch is skipped and the fit is bit-identical.
        if dim >= 2 and n_bdy_here > 0:
            V_int = V.copy()
            V_int[max_neighbors:, :] = 0.0          # drop every ghost row
            A_loc = np.linalg.pinv(V_int)           # interior-only base
            for j in range(n_bdy_here):
                n_axis = int(np.argmax(np.abs(dX[max_neighbors + j])))
                rows = pure_axis_rows.get(n_axis)
                if not rows:
                    continue
                V_one = V_int.copy()
                V_one[max_neighbors + j, :] = V[max_neighbors + j, :]
                A_one = np.linalg.pinv(V_one)
                for row in rows:
                    A_loc[row, :] = A_one[row, :]
        else:
            A_loc = np.linalg.pinv(V)
        A_glob.append(A_loc.T)

    A_glob = np.array(A_glob)
    return A_glob, neighbors_array, bdy_neighbors_array, mon_indices


# ── Derivative computation ────────────────────────────────────────────────────

_BC_EXTRAPOLATION = "extrapolation"


def _resolve_u_boundary_face(u, u_boundary_face, mesh):
    """Resolve ``u_boundary_face`` → values at the boundary virtual-
    neighbour position.

    The LSQ stencil places boundary points at the **ghost-cell**
    position (symmetric image of the inner cell through the boundary
    face — see ``least_squares_reconstruction_local``).  The value
    assigned to that ghost point is:

    * ``'extrapolation'`` (Neumann-zero / ``zeroGradient``): ghost =
      inner cell value.  Slope between ghost and cell is zero, and by
      symmetry the slope at the midpoint (= boundary face) is zero —
      the exact Neumann-zero discretisation.
    * ndarray of face values (Dirichlet / Lambda prescribed): the
      virtual point sits at the GHOST-CELL offset ``2·(face - cell)``,
      so the value carried there is the linear extrapolation through
      the face, ``u_ghost = 2·u_face - u_cell``.  Returning the bare
      face value would place a face-valued sample at the ghost
      position — an inconsistency that caps the boundary gradient at
      1st order for a linear field at any finite ``dx``.  The
      ``2·u_face - u_cell`` form reproduces ``(b, c)`` exactly for
      ``u = a + b·x + c·y``.

    Passing ``None`` or any other type raises — callers must be
    explicit about the BC interpretation.
    """
    if isinstance(u_boundary_face, np.ndarray):
        return 2.0 * u_boundary_face - u[mesh.boundary_face_cells]
    if isinstance(u_boundary_face, str) and u_boundary_face == _BC_EXTRAPOLATION:
        return u[mesh.boundary_face_cells]
    raise ValueError(
        "compute_derivatives: u_boundary_face must be either an ndarray "
        "of shape (n_boundary_faces,) of face values, or the literal "
        f"string 'extrapolation'.  Got: {u_boundary_face!r}.  "
        "Pass 'extrapolation' explicitly for Neumann-zero / "
        "zeroGradient at boundary faces (e.g. the pressure-projection "
        "variable in Chorin), or supply BC-evaluated face values "
        "(e.g. from sm.boundary_conditions(...)) for prescribed BCs."
    )


def compute_derivatives(u, mesh, derivatives_multi_index=None, *,
                        u_boundary_face):
    """Cell-wise derivative estimates using the mesh LSQ stencil.

    Parameters
    ----------
    u : ndarray, shape ``(n_cells,)``
        Field values at cell centers (interior + sentinel/ghost slots).
    u_boundary_face : ndarray | ``'extrapolation'``
        **Required.**  Either an ``(n_boundary_faces,)`` array of values
        at boundary face centers (typically from the SystemModel's
        ``boundary_conditions`` runtime kernel applied to the inner-
        cell state) — for prescribed-value BCs (Dirichlet, Lambda) — or
        the literal string ``'extrapolation'`` to request Neumann-zero
        / face = inner-cell-value treatment (the Chorin pressure
        projection's natural P BC, and the safe fallback for
        intrinsically-Neumann fields like static bathymetry).
    """
    A_glob = mesh.lsq_gradQ
    neighbors = mesh.lsq_neighbors
    mon_indices = mesh.lsq_monomial_multi_index
    scale_factors = mesh.lsq_scale_factors
    bdy_neighbors = getattr(mesh, "lsq_boundary_face_neighbors", None)
    has_bdy = bdy_neighbors is not None and bdy_neighbors.size > 0
    if has_bdy:
        u_boundary_face = _resolve_u_boundary_face(
            u, u_boundary_face, mesh)

    if derivatives_multi_index is None:
        derivatives_multi_index = mon_indices
    indices = find_derivative_indices(mon_indices, derivatives_multi_index)

    out = np.zeros((A_glob.shape[0], len(derivatives_multi_index)), dtype=float)
    for i in range(A_glob.shape[0]):
        A_loc = A_glob[i]
        neighbor_idx = neighbors[i]
        u_i = u[i]
        u_cells = u[neighbor_idx]
        if has_bdy:
            bf = bdy_neighbors[i]
            u_bdy = np.where(bf >= 0, u_boundary_face[np.maximum(bf, 0)], u_i)
            u_full = np.concatenate([u_cells, u_bdy])
        else:
            u_full = u_cells
        delta_u = u_full - u_i
        out[i, :] = (scale_factors * (A_loc.T @ delta_u))[indices]

    return out


def get_physical_boundary_labels(filepath):
    """Extract physical boundary labels from a .msh file via meshio."""
    import meshio
    mesh = meshio.read(filepath)
    return {key: value[0] for key, value in mesh.field_data.items()}


def compute_inradius_generic(cell_center, face_centers, face_normals):
    """Inradius of a cell: shortest normal distance from center to any face."""
    inradius = np.inf
    for center, normal in zip(face_centers, face_normals):
        distance = np.abs(np.dot(center - cell_center, normal))
        inradius = min(inradius, distance)
    if inradius <= 0:
        inradius = np.array(face_centers - cell_center).min()
    return inradius
