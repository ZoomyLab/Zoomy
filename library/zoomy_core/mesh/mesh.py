import os

try:
    import h5py

    _HAVE_H5PY = True
except ImportError:
    _HAVE_H5PY = False


try:
    from petsc4py import PETSc

    _HAVE_PETSC = True
except ImportError:
    _HAVE_PETSC = False


try:
    import meshio

    _HAVE_MESHIO = True
except:
    _HAVE_MESHIO = False
from copy import deepcopy
from itertools import product
from typing import Union

import numpy as np
from attr import define

import zoomy_core.mesh.mesh_extrude as extrude
import zoomy_core.mesh.mesh_util as mesh_util
from zoomy_core.mesh.mesh_util import compute_subvolume, get_extruded_mesh_type
from zoomy_core.misc.custom_types import CArray, FArray, IArray
from zoomy_core.model.boundary_conditions import Periodic

# petsc4py.init(sys.argv)


def build_monomial_indices(degree: int, dim: int):
    """
    Build the full set of monomial multi-indices for a given degree and dimension.

    Parameters:
    - degree (int): Maximum total polynomial degree
    - dim (int): Number of dimensions

    Returns:
    - List of tuples: each tuple is a multi-index (α₁, ..., α_dim)
    """
    mon_indices = []
    for powers in product(range(degree + 1), repeat=dim):
        if sum(powers) <= degree:
            if sum(powers) == 0:
                continue
            mon_indices.append(powers)
    return mon_indices


def scale_lsq_derivative(mon_indices):
    from math import factorial

    import numpy as np

    scale_factors = np.array(
        [np.prod([factorial(k) for k in mi]) for mi in mon_indices]
    )  # shape (n_monomials,)
    return scale_factors


def find_derivative_indices(full_monomials_arr, requested_derivs_arr):
    """
    Args:
        full_monomials_arr: shape (N, dim)
        requested_derivs_arr: shape (M, dim)

    Returns:
        indices: shape (M,), where indices[i] gives the index of requested_derivs_arr[i]
                 in full_monomials_arr, or -1 if not found
    """
    # Ensure arrays
    full_monomials_arr = np.array(full_monomials_arr)
    requested_derivs_arr = np.array(requested_derivs_arr)

    # Compare all requested derivatives with all monomials
    matches = np.all(
        full_monomials_arr[:, None, :] == requested_derivs_arr[None, :, :], axis=-1
    )  # (N, M)

    # Which monomials match each requested derivative?
    found = np.any(matches, axis=0)  # (M,)
    indices = np.argmax(matches, axis=0)  # (M,) — returns 0 even if not found

    # Replace unfound indices with -1
    indices = np.where(found, indices, -1)
    return indices


def compute_derivatives(u, mesh, derivatives_multi_index=None):
    A_glob = mesh.lsq_gradQ  # shape (n_cells, n_neighbors, n_monomials)
    neighbors = mesh.lsq_neighbors  # list of neighbors per cell
    mon_indices = mesh.lsq_monomial_multi_index  # shape (n_monomials, dim)
    # scale_factors = scale_lsq_derivative(mon_indices)
    scale_factors = mesh.lsq_scale_factors

    if derivatives_multi_index is None:
        derivatives_multi_index = mon_indices
    indices = find_derivative_indices(mon_indices, derivatives_multi_index)

    def reconstruct_cell(A_loc, neighbor_idx, u_i):
        u_neighbors = u[neighbor_idx]
        delta_u = u_neighbors - u_i
        return (scale_factors * (A_loc.T @ delta_u)).T  # shape (n_monomials,)

    out = np.zeros((A_glob.shape[0], len(
        derivatives_multi_index)), dtype=float)
    for i in range(A_glob.shape[0]):
        A_loc = A_glob[i]
        neighbor_idx = neighbors[i]
        u_i = u[i]
        out[i, :] = reconstruct_cell(A_loc, neighbor_idx, u_i)[indices]

    return out


def get_polynomial_degree(mon_indices):
    return max(sum(m) for m in mon_indices)


def get_required_monomials_count(degree, dim):
    from math import comb

    return comb(int(degree + dim), int(dim))


def build_vandermonde(cell_diffs, mon_indices):
    n_neighbors, dim = cell_diffs.shape
    n_monomials = len(mon_indices)
    V = np.zeros((n_neighbors, n_monomials))
    for i, powers in enumerate(mon_indices):
        V[:, i] = np.prod(cell_diffs**powers, axis=1)
    return V


def expand_neighbors(neighbors_list, initial_neighbors):
    # Expand to the next ring: union of neighbors of neighbors
    expanded = set(initial_neighbors)
    for n in initial_neighbors:
        expanded.update(neighbors_list[n])
    return list(expanded)


def compute_gaussian_weights(dX, sigma=1.0):
    distances = np.linalg.norm(dX, axis=1)
    weights = np.exp(-((distances / sigma) ** 2))
    return weights


def least_squares_reconstruction_local(
    n_cells, dim, neighbors_list, cell_centers, lsq_degree
):
    """
    Build the full set of monomial multi-indices for a given degree and dimension.

    Parameters:
    - degree (int): Maximum total polynomial degree
    - dim (int): Number of dimensions

    Returns:
    - List of tuples: each tuple is a multi-index (α₁, ..., α_dim)
    """
    mon_indices = build_monomial_indices(lsq_degree, dim)
    A_glob = []
    neighbors_all = []  # to store expanded neighbors per cell
    degree = get_polynomial_degree(mon_indices)
    required_neighbors = get_required_monomials_count(degree, dim)

    # First, expand neighborhoods and store them
    for i_c in range(n_cells):
        current_neighbors = list(neighbors_list[i_c])

        # Expand neighbor rings until we have enough
        while len(current_neighbors) < required_neighbors:
            new_neighbors = expand_neighbors(neighbors_list, current_neighbors)
            current_neighbors = list(set(new_neighbors) - {i_c})

        neighbors_all.append(current_neighbors)

    # Compute max neighbors after expansion
    max_neighbors = max(len(nbrs) for nbrs in neighbors_all)

    A_glob = []
    neighbors_array = np.empty((n_cells, max_neighbors), dtype=int)

    for i_c in range(n_cells):
        current_neighbors = list(neighbors_all[i_c])  # previously expanded
        n_nbr = len(current_neighbors)

        # Expand further if still under-resolved
        while n_nbr < max_neighbors:
            extended_neighbors = expand_neighbors(
                neighbors_list, current_neighbors)
            extended_neighbors = list(
                set(extended_neighbors) - {i_c})  # Remove self
            # Keep only new neighbors not already present
            new_neighbors = [
                n for n in extended_neighbors if n not in current_neighbors
            ]
            current_neighbors.extend(new_neighbors)
            n_nbr = len(current_neighbors)

            # Stop if enough neighbors collected
            if n_nbr >= max_neighbors:
                break

        # Trim or pad neighbor list to exact max_neighbors
        if len(current_neighbors) >= max_neighbors:
            trimmed_neighbors = current_neighbors[:max_neighbors]
        else:
            padding = [i_c] * (max_neighbors - len(current_neighbors))
            trimmed_neighbors = current_neighbors + padding

        neighbors_array[i_c, :] = trimmed_neighbors

        # Build dX matrix for least squares
        dX = np.zeros((max_neighbors, dim), dtype=float)
        for j, neighbor in enumerate(trimmed_neighbors):
            dX[j, :] = cell_centers[neighbor] - cell_centers[i_c]

        # shape (max_neighbors, n_monomials)
        V = build_vandermonde(dX, mon_indices)

        # Compute weights
        weights = compute_gaussian_weights(dX)  # shape (n_neighbors,)
        W = np.diag(weights)  # shape (n_neighbors, n_neighbors)
        VW = W @ V  # shape (n_neighbors, n_monomials)
        alpha = dX.min() * 10 ** (-8)
        A_loc = (
            np.linalg.pinv(VW.T @ VW + alpha * np.eye(VW.shape[1])) @ VW.T @ W
        )  # shape (n_monomials, n_neighbors)
        # A_loc = np.linalg.pinv(V).T  # shape (n_monomials, max_neighbors)
        A_glob.append(A_loc.T)

    A_glob = np.array(A_glob)  # shape (n_cells, n_monomials, max_neighbors)
    return A_glob, neighbors_array, mon_indices


def get_physical_boundary_labels(filepath):
    if not _HAVE_MESHIO:
        raise RuntimeError(
            "_write_to_vtk_from_vertices_edges requires meshio, which is not available."
        )
    mesh = meshio.read(filepath)
    boundary_dict = {key: value[0] for key, value in mesh.field_data.items()}
    return boundary_dict


def _compute_inradius_generic(cell_center, face_centers, face_normals):
    """
    strategy: find the shortest path from the center to each side (defined by face center and normal)
    use the minimum of all these shortest paths
    For a distorted element, the inradius might be zero. In this case, use minimal distance of center to face_centers
    """
    inradius = np.inf
    for center, normal in zip(face_centers, face_normals):
        distance = np.abs(np.dot(center - cell_center, normal))
        inradius = min(inradius, distance)
    if inradius <= 0:
        inradius = np.array(face_centers - cell_center).min()
    return inradius


def compute_cell_inradius(dm):
    """
    Error in petsc4py? dm.getMinRadius() always returns 0
    strategy: find the shortest path from the center to each side (defined by face center and normal)
    use the minimum of all these shortest paths
    """
    (cStart, cEnd) = dm.getHeightStratum(0)
    (vStart, vEnd) = dm.getDepthStratum(0)
    (eStart, eEnd) = dm.getDepthStratum(1)
    inradius = []
    for c in range(cStart, cEnd):
        _, cell_center, _ = dm.computeCellGeometryFVM(c)
        face_normals = []
        face_centers = []
        faces = dm.getCone(c)
        for f in faces:
            _, center, normal = dm.computeCellGeometryFVM(f)
            face_normals.append(normal)
            face_centers.append(center)
        inradius.append(
            _compute_inradius_generic(cell_center, face_centers, face_normals)
        )
    return np.array(inradius, dtype=float)


def get_mesh_type_from_dm(num_faces_per_cell, dim):
    if dim == 1:
        if num_faces_per_cell == 2:
            return "line"
        else:
            assert False
    elif dim == 2:
        if num_faces_per_cell == 3:
            return "triangle"
        elif num_faces_per_cell == 4:
            return "quad"
        else:
            assert False
    elif dim == 3:
        if num_faces_per_cell == 4:
            return "tetra"
        elif num_faces_per_cell == 6:
            return "edge"
        elif num_faces_per_cell == 8:
            return "hexahedron"
        else:
            assert False
    assert False


def _boundary_dict_to_list(d):
    l = []
    for i, vs in enumerate(d.values()):
        l += vs
    return l


def _boundary_dict_indices(d):
    indices = []
    index = 0
    for values in d.values():
        indices += [index for v in values]
        index += 1
    return np.array(indices, dtype=int)


def _get_neighberhood(dm, cell, cStart=0):
    neighbors = (
        np.array(
            [dm.getSupport(f)[dm.getSupport(f) != cell][0]
             for f in dm.getCone(cell)],
            dtype=int,
        )
        - cStart
    )
    return neighbors


def _fill_neighborhood(dm, neighbors, max_neighbors, cStart=0):
    new_potential_neighbors = []
    for cell in neighbors:
        new_potential_neighbors += list(_get_neighberhood(dm,
                                        cell, cStart=cStart))
    new_potential_neighbors = np.array(new_potential_neighbors)
    new_neighbors = np.setdiff1d(new_potential_neighbors, neighbors)
    neighbors = np.concatenate((neighbors, new_neighbors))[:max_neighbors]
    if len(neighbors) == max_neighbors:
        return neighbors
    else:
        return _fill_neighborhood(dm, neighbors, max_neighbors, cStart=cStart)


@define(slots=True, frozen=True)
class Mesh:
    dimension: int
    type: str
    n_cells: int
    n_inner_cells: int
    n_faces: int
    n_vertices: int
    n_boundary_faces: int
    n_faces_per_cell: int
    vertex_coordinates: FArray
    cell_vertices: IArray
    cell_faces: IArray
    cell_volumes: FArray
    cell_centers: FArray
    cell_inradius: FArray
    cell_neighbors: IArray
    boundary_face_cells: IArray
    boundary_face_ghosts: IArray
    boundary_face_function_numbers: IArray
    boundary_face_physical_tags: IArray
    boundary_face_face_indices: IArray
    face_cells: IArray
    # face_cell_face_index: IArray
    face_normals: FArray
    face_volumes: FArray
    face_centers: FArray
    face_subvolumes: FArray
    face_neighbors: IArray
    boundary_conditions_sorted_physical_tags: IArray
    boundary_conditions_sorted_names: CArray
    lsq_gradQ: FArray
    lsq_neighbors: FArray
    lsq_monomial_multi_index: int
    lsq_scale_factors: FArray
    z_ordering: IArray

    def resolve_periodic_bcs(self, bcs):
        """
        Goal: if 'apply_boundary_condition' is called, the ghost cell value is computed, given an input cell value funtion.
              In case of a periodic BC, this is NOT the adjacent cell. So I need to change the 'boundary_face_cell' for the periodic
              cells to point to the right data location!
              This is why we only overwrite the 'mesh.boundary_face_cell' in the end.
              Furthermore, I CANNOT alter any ordering! However, I need to sort the two boundaries such that the e.g.
              left and right border are mapped correctly, as the boundary cells are not ordered.
              As ghost/inner cells is confusing here, I rather like to use 'from' and 'to', as 'from' data from which bc is copied and 'to' stands for the boundary where it is copied to. As we only change the cells where the data is taken from (from the adjacent to the periodic cell), we only alter the 'boundary_face_cell' in the end.
        """
        dict_physical_name_to_index = {
            v: i for i, v in enumerate(self.boundary_conditions_sorted_names)
        }
        dict_function_index_to_physical_tag = {
            i: v for i, v in enumerate(self.boundary_conditions_sorted_physical_tags)
        }

        boundary_face_cells_copy = deepcopy(self.boundary_face_cells)

        for i_bc, bc in enumerate(bcs._boundary_conditions):
            if type(bc) == Periodic:
                from_physical_tag = dict_function_index_to_physical_tag[
                    dict_physical_name_to_index[bc.periodic_to_physical_tag]
                ]
                to_physical_tag = dict_function_index_to_physical_tag[
                    dict_physical_name_to_index[bc.tag]
                ]

                mask_face_from = self.boundary_face_physical_tags == from_physical_tag
                mask_face_to = self.boundary_face_physical_tags == to_physical_tag

                # I want to copy from boundary_face_cells to boundary_face_ghosts!
                from_cells = self.boundary_face_cells[mask_face_from]
                to_cells = self.boundary_face_ghosts[mask_face_to]

                from_coords = self.cell_centers[:, from_cells]
                to_coords = self.cell_centers[:, to_cells]

                # sort not dimension by dimension, but most significant dimension to least significant dimension
                # determine significance by max difference
                significance_per_dimension = [
                    from_coords[d, :].max() - from_coords[d, :].min()
                    for d in range(self.dimension)
                ]
                _significance_per_dimension = [
                    to_coords[d, :].max() - to_coords[d, :].min()
                    for d in range(self.dimension)
                ]
                # reverse the order of lexsort such that the most important is first IS NOT NEEDED, since lexsort starts sorting by the last entry in the list
                sort_order_significance = np.lexsort(
                    [significance_per_dimension])

                from_cells_sort_order = np.lexsort(
                    [from_coords[d, :] for d in sort_order_significance]
                )
                to_cells_sort_order = np.lexsort(
                    [to_coords[d, :] for d in sort_order_significance]
                )

                # generates indices from 0 to number of ghost_cells (total)
                indices = np.array(list(range(mask_face_to.shape[0])))
                # masks away all cells that do not belong to this tag
                indices_to = indices[mask_face_to]
                # sort the indices
                indices_to_sort = indices_to[to_cells_sort_order]

                indices_from = indices[mask_face_from]
                indices_from_sort = indices_from[from_cells_sort_order]

                self.boundary_face_cells[indices_to_sort] = boundary_face_cells_copy[
                    indices_from_sort
                ]

    @classmethod
    def create_1d(cls, domain: tuple[float, float], n_inner_cells: int, lsq_degree=1):
        xL = domain[0]
        xR = domain[1]

        n_cells = n_inner_cells + 2
        n_vertices = n_inner_cells + 1
        dimension = 1
        n_faces_per_cell = 2
        n_faces = n_inner_cells + 1
        n_boundary_faces = 2
        dx = (xR - xL) / n_inner_cells
        vertex_coordinates = np.zeros((n_vertices, 1))
        vertex_coordinates[:, 0] = np.linspace(xL, xR, n_vertices, dtype=float)
        cell_vertices = np.zeros((n_inner_cells, n_faces_per_cell), dtype=int)
        cell_vertices[:, 0] = list(range(0, n_vertices - 1))
        cell_vertices[:, 1] = list(range(1, n_vertices))
        cell_volumes = dx * np.ones(n_cells, dtype=float)
        cell_inradius = dx / 2 * np.ones(n_cells, dtype=float)

        face_normals = np.zeros((n_faces, dimension), dtype=float)
        face_volumes = np.ones((n_faces), dtype=float)

        cell_centers = np.zeros((n_cells, 3), dtype=float)
        cell_centers[:n_inner_cells, 0] = np.arange(xL + dx / 2, xR, dx)
        cell_centers[n_inner_cells, 0] = xL - dx / 2
        cell_centers[n_inner_cells + 1, 0] = xR + dx / 2
        cell_neighbors = (n_cells + 1) * \
            np.ones((n_cells, n_faces_per_cell), dtype=int)

        cell_faces = np.empty((n_inner_cells, n_faces_per_cell), dtype=int)
        cell_faces[:, 0] = list(range(0, n_faces - 1))
        cell_faces[:, 1] = list(range(1, n_faces))

        # inner cells
        for i_cell in range(0, n_cells):
            cell_neighbors[i_cell, :] = [i_cell - 1, i_cell + 1]
        # left neighbor of 0is the first ghost
        cell_neighbors[0, 0] = n_inner_cells
        # right neighbor of n_inner_cell is the second ghost
        cell_neighbors[n_inner_cells - 1, 1] = n_inner_cells + 1
        # left neighbor of first ghost is empty, but we add the neighbor of the neighbor
        cell_neighbors[n_inner_cells, 0] = 1
        # right neighbor of first ghost is first cell
        cell_neighbors[n_inner_cells, 1] = 0
        # left neighbor of second ghost is last inner
        cell_neighbors[n_inner_cells + 1, 0] = n_inner_cells - 1
        # right neighbor of second ghost is empty, but we add the neighbor of the neighbor
        cell_neighbors[n_inner_cells + 1, 1] = n_inner_cells - 2

        for i_face in range(0, n_faces):
            face_normals[i_face, 0] = 1.0

        boundary_face_cells = np.array([0, n_inner_cells - 1], dtype=int)
        boundary_face_ghosts = np.array(
            [n_inner_cells, n_inner_cells + 1], dtype=int)
        boundary_face_function_numbers = np.empty(
            (n_boundary_faces), dtype=int)
        boundary_face_function_numbers[0] = 0
        boundary_face_function_numbers[1] = 1
        boundary_face_physical_tags = np.array([0, 1], dtype=int)
        boundary_face_face_indices = np.array([0, n_faces - 1], dtype=int)

        face_cells = np.empty((n_faces, 2), dtype=int)
        # face_cell_face_index = (n_faces + 1)*np.ones((n_faces, 2), dtype=int)
        face_cells[1: n_faces - 1, 0] = list(range(0, n_inner_cells - 1))
        face_cells[1: n_faces - 1, 1] = list(range(1, n_inner_cells))
        # face_cell_face_index[1:n_faces-1, 0] = 1
        # face_cell_face_index[1:n_faces-1, 1] = 0
        face_cells[0, 0] = n_inner_cells
        # face_cell_face_index[0, 0] = 0
        face_cells[0, 1] = 0
        # face_cell_face_index[0, 1] = 0
        face_cells[-1, 0] = n_inner_cells - 1
        # face_cell_face_index[-1, 0] = 1
        face_cells[-1, 1] = n_inner_cells + 1
        # face_cell_face_index[-1, 1] = 0
        face_centers = 0.5 * (
            cell_centers[face_cells[:, 0]] + cell_centers[face_cells[:, 1]]
        )

        face_subvolumes = np.empty((n_faces, 2), dtype=float)
        face_subvolumes[:, 0] = dx / 2
        face_subvolumes[:, 1] = dx / 2

        boundary_conditions_sorted_physical_tags = np.array([0, 1], dtype=int)
        boundary_conditions_sorted_names = np.array(["left", "right"])

        lsq_gradQ = np.zeros((n_cells, dimension, n_cells), dtype=float)
        deltaQ = np.zeros((n_cells, n_faces_per_cell, n_cells), dtype=float)
        cell_centers = cell_centers[:, :dimension]

        polynomial_degree = 1
        n_neighbors = n_faces_per_cell * polynomial_degree
        dim = 1
        lsq_gradQ, lsq_neighbors, lsq_monomial_multi_index = (
            least_squares_reconstruction_local(
                n_cells, dim, cell_neighbors, cell_centers, lsq_degree
            )
        )
        lsq_scale_factors = scale_lsq_derivative(lsq_monomial_multi_index)

        n_face_neighbors = 2
        face_neighbors = (n_cells + 1) * \
            np.ones((n_faces, n_face_neighbors), dtype=int)

        for i_f, neighbors in enumerate(face_cells):
            face_neighbors[i_f] = neighbors

        z_ordering = np.array([-1], dtype=float)

        # return cls(dimension, 'line', n_cells, n_cells + 1, 2, n_faces_per_element, vertex_coordinates, element_vertices, element_face_areas, element_centers, element_volume, element_inradius, element_face_normals, element_n_neighbors, element_neighbors, element_neighbors_face_index, boundary_face_vertices, boundary_face_corresponding_element, boundary_face_element_face_index, boundary_face_tag, boundary_tag_names)
        return cls(
            dimension,
            "line",
            n_cells,
            n_inner_cells,
            n_faces,
            n_vertices,
            n_boundary_faces,
            n_faces_per_cell,
            vertex_coordinates.T,
            cell_vertices.T,
            cell_faces.T,
            cell_volumes,
            cell_centers.T,
            cell_inradius,
            cell_neighbors,
            boundary_face_cells.T,
            boundary_face_ghosts.T,
            boundary_face_function_numbers,
            boundary_face_physical_tags,
            boundary_face_face_indices.T,
            face_cells.T,
            face_normals.T,
            face_volumes,
            face_centers,
            face_subvolumes,
            face_neighbors,
            boundary_conditions_sorted_physical_tags,
            boundary_conditions_sorted_names,
            lsq_gradQ,
            lsq_neighbors,
            lsq_monomial_multi_index,
            lsq_scale_factors,
            z_ordering,
        )

    def _compute_ascending_order_structured_axis(self):
        cell_centers = self.cell_centers.T
        dimension = self.dimension
        if dimension == 1:
            order = np.lexsort((cell_centers[:, 0],))
        elif dimension == 2:
            order = np.lexsort(
                (
                    cell_centers[:, 0],
                    cell_centers[:, 1],
                )
            )
        elif dimension == 3:
            order = np.lexsort(
                (
                    cell_centers[:, 0],
                    cell_centers[:, 1],
                    cell_centers[:, 2],
                )
            )
        else:
            assert False

        # find number of cells in z (Nz) from first column
        Nz = 1
        while cell_centers[order[Nz], 2] > cell_centers[order[Nz - 1], 2]:
            Nz += 1
        Nz += 1

        # partition order into [(Nx x Ny) , (Nz)]
        N = int(cell_centers.shape[0] / Nz)
        assert cell_centers.shape[0] == N * Nz
        order = order.reshape((N, Nz))

        coords_z = order[0]
        for o in order:
            assert o == coords_z

        return order

    @classmethod
    def from_gmsh(cls, filepath, allow_z_integration=False, lsq_degree=1):
        if not _HAVE_PETSC:
            raise RuntimeError(
                "Mesh.from_gmsh() requires petsc4py, which is not available."
            )
        dm = PETSc.DMPlex().createFromFile(filepath, comm=PETSc.COMM_WORLD)
        boundary_dict = get_physical_boundary_labels(filepath)
        (cStart, cEnd) = dm.getHeightStratum(0)
        (vStart, vEnd) = dm.getDepthStratum(0)
        # (eStart, eEnd) = dm.getDepthStratum(1)
        (eStart, eEnd) = dm.getHeightStratum(1)
        gdm = dm.clone()
        gdm.constructGhostCells()
        (cgStart, cgEnd) = gdm.getHeightStratum(0)
        (vgStart, vgEnd) = gdm.getDepthStratum(0)
        # (egStart, egEnd) = gdm.getDepthStratum(1)
        (egStart, egEnd) = gdm.getHeightStratum(1)

        dim = dm.getDimension()
        n_faces_per_cell = len(gdm.getCone(cgStart))
        n_vertices_per_cell = n_faces_per_cell
        if dim > 2:
            transitive_closure_points, transitive_closure_orientation = (
                dm.getTransitiveClosure(cStart, useCone=True)
            )
            n_vertices_per_cell = transitive_closure_points[
                np.logical_and(
                    transitive_closure_points >= vStart,
                    transitive_closure_points < vEnd,
                )
            ].shape[0]
        n_cells = cgEnd - cgStart
        n_inner_cells = cEnd - cStart
        n_faces = egEnd - egStart
        n_vertices = vEnd - vStart
        cell_vertices = np.zeros(
            (n_inner_cells, n_vertices_per_cell), dtype=int)
        cell_faces = np.zeros((n_inner_cells, n_faces_per_cell), dtype=int)
        cell_centers = np.zeros((n_cells, 3), dtype=float)
        # I create cell_volumes of size n_cells because then I can avoid an if clause in the numerical flux computation. The values will be delted after using apply_boundary_conditions anyways
        cell_volumes = np.ones((n_cells), dtype=float)
        cell_inradius = np.zeros((n_cells), dtype=float)
        cell_inradius[:n_inner_cells] = compute_cell_inradius(dm)
        for i_c, c in enumerate(range(cStart, cEnd)):
            cell_volume, cell_center, cell_normal = dm.computeCellGeometryFVM(
                c)
            transitive_closure_points, transitive_closure_orientation = (
                dm.getTransitiveClosure(c, useCone=True)
            )
            _cell_vertices = transitive_closure_points[
                np.logical_and(
                    transitive_closure_points >= vStart,
                    transitive_closure_points < vEnd,
                )
            ]
            # _cell_vertices_orientation = transitive_closure_orientation[np.logical_and(transitive_closure_points >= vStart, transitive_closure_points <vEnd)]
            assert _cell_vertices.shape[0] == cell_vertices.shape[1]
            # assert (_cell_vertices_orientation == 0).all()
            cell_vertices[i_c, :] = _cell_vertices - vStart
            cell_centers[i_c, :dim] = cell_center[:dim]
            cell_volumes[i_c] = cell_volume

        vertex_coordinates = np.array(gdm.getCoordinates()).reshape((-1, dim))
        boundary_face_cells = {k: [] for k in boundary_dict.values()}
        boundary_face_ghosts = {k: [] for k in boundary_dict.values()}
        boundary_face_face_indices = {k: [] for k in boundary_dict.values()}
        boundary_face_physical_tags = {k: [] for k in boundary_dict.values()}
        face_cells = []
        # face_cell_face_index = np.zeros((n_faces, 2), dtype=int)
        face_normals = []
        face_volumes = []
        face_centers = []
        face_subvolumes = []
        allowed_keys = []
        vertex_coordinates = np.array(dm.getCoordinates()).reshape((-1, dim))

        def get_face_vertices(dim, gdm, vgStart, e):
            if dim == 2:
                return gdm.getCone(e) - vgStart
            elif dim == 3:
                face_vertices = set()
                face_edges = gdm.getCone(e)
                for edge in face_edges:
                    face_vertices.update(gdm.getCone(edge))
                return np.array(list(face_vertices), dtype=int) - vgStart
            else:
                assert False

        for e in range(egStart, egEnd):
            label = gdm.getLabelValue("Face Sets", e)
            face_volume, face_center, face_normal = gdm.computeCellGeometryFVM(
                e)
            face_vertices = get_face_vertices(dim, gdm, vgStart, e)
            face_vertices_coords = vertex_coordinates[face_vertices]
            _face_cells = gdm.getSupport(e)

            _, _cell_center, _ = gdm.computeCellGeometryFVM(_face_cells[0])
            _face_subvolume = np.zeros(2, dtype=float)
            _face_subvolume[0] = compute_subvolume(
                face_vertices_coords, _cell_center, dim
            )
            if _face_cells[1] < n_inner_cells:
                _, _cell_center, _ = gdm.computeCellGeometryFVM(_face_cells[1])
                _face_subvolume[1] = compute_subvolume(
                    face_vertices_coords, _cell_center, dim
                )

            # 2 cells support an face. Ghost cell is the one with the higher number
            if label > -1:
                allowed_keys.append(label)
                boundary_cell = gdm.getSupport(e).min() - cStart
                boundary_ghost = gdm.getSupport(e).max() - cStart
                boundary_face_cells[label].append(boundary_cell)
                boundary_face_ghosts[label].append(boundary_ghost)
                boundary_face_face_indices[label].append(e - egStart)
                boundary_face_physical_tags[label].append(label)
                # boundary_face_vertices[label].append(gdm.getCone(e)-vStart)
                # for periodic boudnary conditions, I need the ghost cell to have a cell_center. I copy the one from the related inner cell.
                _face_cell = gdm.getSupport(e).min()
                _face_ghost = gdm.getSupport(e).max()
                cell_centers[_face_ghost, :dim] = (
                    cell_centers[_face_cell, :dim]
                    + 2
                    * ((face_center - cell_centers[_face_cell, :dim]) @ face_normal)
                    * face_normal
                )
                cell_inradius[_face_ghost] = cell_inradius[_face_cell]
                cell_volumes[_face_ghost] = cell_volumes[_face_cell]
                # subvolumes of the ghost cell are computes wrongly. In this case, copy the value from the inner cell.
                if _face_cells[0] > _face_cells[1]:
                    _face_subvolume[0] = _face_subvolume[1]
                else:
                    _face_subvolume[1] = _face_subvolume[0]
                

            face_centers.append(face_center)
            _face_cells = gdm.getSupport(e) - cStart
            face_volumes.append(face_volume)
            face_normals.append(face_normal)
            face_cells.append(_face_cells)
            face_subvolumes.append(_face_subvolume)

        # I only want to iterate over the inner cells, but in the ghosted mesh
        for i_c, c in enumerate(range(cgStart, cgStart + n_inner_cells)):
            faces = gdm.getCone(c) - egStart
            cell_faces[i_c, :] = faces

        # least squares liner reconstruction
        # Consider 2d, three points, scalar field, then the reconstruction is for a quad mesh
        # DU = [u_{i+1, j} - u_{i,j}, u_{i-1, j} - u_{i,j}, u_{i, j+1} - u_{i,j}, u_{i, j-1} - u_{i,j}]  \in \mathbb{R}^{4}
        # DX = [[x_{i+1, j} - x_{i,j}, x_{i-1, j} - x_{i,j}, x_{i, j+1} - x_{i,j}, x_{i, j-1} - x_{i,j}],
        #       [y_{i+1, j} - y_{i,j}, y_{i-1, j} - y_{i,j}, y_{i, j+1} - y_{i,j}, y_{i, j-1} - y_{i,j}]]  \in \mathbb{R}^{4x2}
        # S = [S_x, S_y] \mathbb{R}^{2}
        # solve DU = DX \dot S via normal equation
        # A = (DX.T \dot DX)^{-1} DX.T \mathbb{R}^{2x4}
        # so the solution is given by S = A \dot DU
        # the vectorized version is more complicated. I have the vectorization (...) in the first dimension of size n_cells
        # However, I do not want to apply the matrix vector product on DU (such that I can get the reconstruction with a single matrix vector product in the solver), but rather on a scalar field q \in \mathbb{R}^{n_cells}. This requires the need of a discretization matrix D \in \mathbb{R}^{n_cells x 4 x n_cells}, such that Dq = DU \in \mathbb{n_cells x 4}, where the first dimension is the dimension of vectorization

        # cell_neighbors = np.zeros(1)
        lsq_gradQ = np.zeros(1)
        # NON_VECTORIZED CASE
        polynomial_degree = 1
        n_neighbors = n_faces_per_cell * polynomial_degree
        cell_neighbors = (n_cells + 1) * \
            np.ones((n_cells, n_neighbors), dtype=int)

        for i_c, c in enumerate(range(cgStart, cgEnd)):
            # GET NEIGHBORHOOD
            neighbors = _get_neighberhood(gdm, c, cStart=cgStart)
            assert not (i_c == neighbors).any()
            _n_neighbors = neighbors.shape[0]
            if _n_neighbors == 1:
                neighbors_of_neighbor = _get_neighberhood(
                    gdm, neighbors[0] + cgStart, cStart=cgStart
                )
                assert len(neighbors_of_neighbor) == n_faces_per_cell
                neighbors = np.setdiff1d(
                    np.union1d(neighbors_of_neighbor, neighbors), [c]
                )
            cell_neighbors[i_c, :] = neighbors

        lsq_gradQ, lsq_neighbors, lsq_monomial_multi_index = (
            least_squares_reconstruction_local(
                n_cells, dim, cell_neighbors, cell_centers[:, :dim], lsq_degree
            )
        )
        lsq_scale_factors = scale_lsq_derivative(lsq_monomial_multi_index)

        n_face_neighbors = (2 * (n_faces_per_cell + 1) - 2) * polynomial_degree
        face_neighbors = (n_cells + 1) * \
            np.ones((n_faces, n_face_neighbors), dtype=int)

        for i_f, f in enumerate(range(egStart, egEnd)):
            # GET NEIGHBORHOOD
            neighbors = _fill_neighborhood(
                gdm, gdm.getSupport(f), n_face_neighbors, cgStart
            )
            face_neighbors[i_f, :] = neighbors

        face_volumes = np.array(face_volumes, dtype=float)
        _face_centers = np.array(face_centers, dtype=float)
        face_centers = np.zeros((n_faces, 3), dtype=float)
        face_centers[:, :dim] = _face_centers[:, :dim]
        face_normals = np.array(face_normals, dtype=float)
        face_subvolumes = np.array(face_subvolumes, dtype=float)

        face_cells = np.array(face_cells, dtype=int)
        # face_cell_face_index = np.array(face_cell_face_index, dtype=int)
        boundary_face_function_numbers = _boundary_dict_indices(
            boundary_face_cells)

        # get rid of empty keys in the boundary_dict (e.g. no surface values in 2d)
        boundary_dict_inverted = {v: k for k, v in boundary_dict.items()}
        boundary_dict_reduced = {
            k: boundary_dict_inverted[k] for k in allowed_keys}

        # sort the dict by the values
        sorted_keys = np.array(list(boundary_dict_reduced.keys()), dtype=int)
        sorted_keys.sort()
        boundary_dict = {k: boundary_dict_reduced[k] for k in sorted_keys}
        boundary_face_cells = {k: boundary_face_cells[k] for k in sorted_keys}
        boundary_face_ghosts = {
            k: boundary_face_ghosts[k] for k in sorted_keys}
        boundary_face_face_indices = {
            k: boundary_face_face_indices[k] for k in sorted_keys
        }

        boundary_conditions_sorted_physical_tags = np.array(
            list(boundary_dict.keys()), dtype=int
        )
        boundary_conditions_sorted_names = np.array(
            list(boundary_dict.values()), dtype="str"
        )
        boundary_face_cells = np.array(
            _boundary_dict_to_list(boundary_face_cells), dtype=int
        )
        boundary_face_ghosts = np.array(
            _boundary_dict_to_list(boundary_face_ghosts), dtype=int
        )
        boundary_face_physical_tags = np.array(
            _boundary_dict_to_list(boundary_face_physical_tags), dtype=int
        )
        boundary_face_face_indices = np.array(
            _boundary_dict_to_list(boundary_face_face_indices), dtype=int
        )
        n_boundary_faces = boundary_face_cells.shape[0]

        mesh_type = get_mesh_type_from_dm(n_faces_per_cell, dim)

        z_ordering = np.array([-1], dtype=float)

        obj = cls(
            dim,
            mesh_type,
            n_cells,
            n_inner_cells,
            n_faces,
            n_vertices,
            n_boundary_faces,
            n_faces_per_cell,
            vertex_coordinates.T,
            cell_vertices.T,
            cell_faces.T,
            cell_volumes,
            cell_centers.T,
            cell_inradius,
            cell_neighbors,
            boundary_face_cells.T,
            boundary_face_ghosts.T,
            boundary_face_function_numbers,
            boundary_face_physical_tags,
            boundary_face_face_indices.T,
            face_cells.T,
            face_normals.T,
            face_volumes,
            face_centers,
            face_subvolumes,
            face_neighbors,
            boundary_conditions_sorted_physical_tags,
            boundary_conditions_sorted_names,
            lsq_gradQ,
            lsq_neighbors,
            lsq_monomial_multi_index,
            lsq_scale_factors,
            z_ordering,
        )

        if allow_z_integration:
            obj._compute_ascending_order_structured_axis()

        return obj

    @classmethod
    def extrude_mesh(cls, msh, n_layers=10):
        Z = np.linspace(0, 1, n_layers + 1)
        dimension = msh.dimension + 1
        mesh_type = get_extruded_mesh_type(msh.type)
        n_cells = msh.n_cells * n_layers
        n_inner_cells = msh.n_inner_cells * n_layers
        n_vertices = msh.n_cells * (n_layers + 1)
        n_boundary_faces = msh.n_boundary_faces * n_layers + 2 * msh.n_cells
        n_faces_per_cell = mesh_util._get_faces_per_element(mesh_type)
        n_faces = n_inner_cells * n_faces_per_cell
        vertex_coordinates = extrude.extrude_points(
            msh.vertex_coordinates.T, Z).T
        cell_vertices = extrude.extrude_element_vertices(
            msh.cell_vertices.T, msh.n_vertices, n_layers
        ).T

        cell_centers = np.empty((n_cells, 3), dtype=float)
        cell_volumes = np.empty((n_cells), dtype=float)
        cell_inradius = np.empty((n_cells), dtype=float)
        cell_face_areas = np.empty((n_cells, n_faces_per_cell), dtype=float)
        cell_face_normals = np.zeros(
            (n_cells, n_faces_per_cell, 3), dtype=float)
        cell_n_neighbors = np.empty((n_cells), dtype=int)
        cell_neighbors = np.empty((n_cells, n_faces_per_cell), dtype=int)
        cell_neighbors_face_index = np.empty(
            (n_cells, n_faces_per_cell), dtype=int)
        for i_elem, elem in enumerate(cell_vertices.T):
            cell_centers[i_elem, :dimension] = mesh_util.center(
                vertex_coordinates.T, elem
            )

        # truncate normals and positions from 3d to dimendion-d
        vertex_coordinates = vertex_coordinates.T[:, :dimension].T
        cell_centers = cell_centers[:, :]
        # cell_face_normals = cell_face_normals[:, :, :dimension]

        # TODO
        # empty fields
        cell_faces = np.empty((n_inner_cells, n_faces_per_cell), dtype=int)
        boundary_face_cells = np.array([0, n_inner_cells - 1], dtype=int)
        boundary_face_ghosts = np.array(
            [n_inner_cells, n_inner_cells + 1], dtype=int)
        boundary_face_function_numbers = np.empty(
            (n_boundary_faces), dtype=int)
        boundary_face_physical_tags = np.array([0, 1], dtype=int)
        boundary_face_face_indices = np.array([0, n_faces - 1], dtype=int)
        face_cells = np.empty((n_faces, 2), dtype=int)
        face_subvolumes = np.empty((n_faces, 2), dtype=float)
        boundary_conditions_sorted_physical_tags = np.array([0, 1], dtype=int)
        boundary_conditions_sorted_names = np.array(["left", "right"])
        face_normals = np.zeros((n_faces, 3), dtype=float)
        face_volumes = np.zeros((n_faces), dtype=float)
        face_centers = np.zeros((n_faces, 3), dtype=float)

        # hard coded guess
        n_face_neighbors = 0
        face_neighbors = (n_cells + 1) * \
            np.ones((n_faces, n_face_neighbors), dtype=int)
        lsq_gradQ = np.zeros((n_cells, dimension, 0), dtype=float)
        lsq_neighbors = np.zeros(1)
        lsq_monomial_multi_index = np.zeros(1)
        lsq_scale_factors = np.zeros(1)
        z_ordering = np.array([-1], dtype=float)

        return cls(
            msh.dimension + 1,
            mesh_type,
            n_cells,
            n_inner_cells,
            n_faces,
            n_vertices,
            n_boundary_faces,
            n_faces_per_cell,
            vertex_coordinates,
            cell_vertices,
            cell_faces.T,
            cell_volumes,
            cell_centers.T,
            cell_inradius,
            cell_neighbors,
            boundary_face_cells.T,
            boundary_face_ghosts.T,
            boundary_face_function_numbers,
            boundary_face_physical_tags,
            boundary_face_face_indices.T,
            face_cells.T,
            face_normals.T,
            face_volumes,
            face_centers,
            face_subvolumes,
            face_neighbors,
            boundary_conditions_sorted_physical_tags,
            boundary_conditions_sorted_names,
            lsq_gradQ,
            lsq_neighbors,
            lsq_monomial_multi_index,
            lsq_scale_factors,
            z_ordering,
        )

    def write_to_hdf5(self, filepath: str):
        if not _HAVE_H5PY:
            raise RuntimeError(
                "Mesh.write_to_hdf5() requires h5py, which is not available."
            )
        main_dir = os.getenv("ZOOMY_DIR")
        with h5py.File(os.path.join(main_dir, filepath), "w") as f:
            mesh = f.create_group("mesh")
            mesh.create_dataset("dimension", data=self.dimension)
            mesh.create_dataset("type", data=self.type)
            mesh.create_dataset("n_cells", data=self.n_cells)
            mesh.create_dataset("n_inner_cells", data=self.n_inner_cells)
            mesh.create_dataset("n_faces", data=self.n_faces)
            mesh.create_dataset("n_vertices", data=self.n_vertices)
            mesh.create_dataset("n_boundary_faces", data=self.n_boundary_faces)
            mesh.create_dataset("n_faces_per_cell", data=self.n_faces_per_cell)
            mesh.create_dataset("vertex_coordinates",
                                data=self.vertex_coordinates)
            mesh.create_dataset("cell_vertices", data=self.cell_vertices)
            mesh.create_dataset("cell_faces", data=self.cell_faces)
            mesh.create_dataset("cell_volumes", data=self.cell_volumes)
            mesh.create_dataset("cell_centers", data=self.cell_centers)
            mesh.create_dataset("cell_inradius", data=self.cell_inradius)
            mesh.create_dataset("cell_neighbors", data=self.cell_neighbors)
            mesh.create_dataset("boundary_face_cells",
                                data=self.boundary_face_cells)
            mesh.create_dataset("boundary_face_ghosts",
                                data=self.boundary_face_ghosts)
            mesh.create_dataset(
                "boundary_face_function_numbers",
                data=self.boundary_face_function_numbers,
            )
            mesh.create_dataset(
                "boundary_face_physical_tags", data=self.boundary_face_physical_tags
            )
            mesh.create_dataset(
                "boundary_face_face_indices", data=self.boundary_face_face_indices
            )
            mesh.create_dataset("face_cells", data=self.face_cells)
            # mesh.create_dataset("face_cell_face_index", data=self.face_cell_face_index)
            mesh.create_dataset("face_normals", data=self.face_normals)
            mesh.create_dataset("face_volumes", data=self.face_volumes)
            mesh.create_dataset("face_centers", data=self.face_centers)
            mesh.create_dataset("face_subvolumes", data=self.face_subvolumes)
            mesh.create_dataset("face_neighbors", data=self.face_neighbors)
            mesh.create_dataset(
                "boundary_conditions_sorted_physical_tags",
                data=np.array(self.boundary_conditions_sorted_physical_tags),
            )
            mesh.create_dataset(
                "boundary_conditions_sorted_names",
                data=np.array(
                    self.boundary_conditions_sorted_names, dtype="S"),
            )
            mesh.create_dataset("lsq_gradQ", data=np.array(self.lsq_gradQ))
            mesh.create_dataset(
                "lsq_neighbors", data=np.array(self.lsq_neighbors))
            mesh.create_dataset(
                "lsq_monomial_multi_index", data=(self.lsq_monomial_multi_index)
            )
            mesh.create_dataset("lsq_scale_factors",
                                data=(self.lsq_scale_factors))
            mesh.create_dataset("z_ordering", data=np.array(self.z_ordering))

    @classmethod
    def from_hdf5(cls, filepath: str):
        if not _HAVE_H5PY:
            raise RuntimeError(
                "Mesh.from_hdf5() requires h5py, which is not available."
            )
        with h5py.File(filepath, "r") as file:
            file = file
            mesh = cls(
                file["mesh"]["dimension"][()],
                (file["mesh"]["type"][()]).decode("utf-8"),
                file["mesh"]["n_cells"][()],
                file["mesh"]["n_inner_cells"][()],
                file["mesh"]["n_faces"][()],
                file["mesh"]["n_vertices"][()],
                file["mesh"]["n_boundary_faces"][()],
                file["mesh"]["n_faces_per_cell"][()],
                file["mesh"]["vertex_coordinates"][()],
                file["mesh"]["cell_vertices"][()],
                file["mesh"]["cell_faces"][()],
                file["mesh"]["cell_volumes"][()],
                file["mesh"]["cell_centers"][()],
                file["mesh"]["cell_inradius"][()],
                file["mesh"]["cell_neighbors"][()],
                file["mesh"]["boundary_face_cells"][()],
                file["mesh"]["boundary_face_ghosts"][()],
                file["mesh"]["boundary_face_function_numbers"][()],
                file["mesh"]["boundary_face_physical_tags"][()],
                file["mesh"]["boundary_face_face_indices"][()],
                file["mesh"]["face_cells"][()],
                # file["mesh"]["face_cell_face_index"][()],
                file["mesh"]["face_normals"][()],
                file["mesh"]["face_volumes"][()],
                file["mesh"]["face_centers"][()],
                file["mesh"]["face_subvolumes"][()],
                file["mesh"]["face_neighbors"][()],
                file["mesh"]["boundary_conditions_sorted_physical_tags"][()],
                np.array(
                    file["mesh"]["boundary_conditions_sorted_names"][()
                                                                     ], dtype="str"
                ),
                file["mesh"]["lsq_gradQ"][()],
                file["mesh"]["lsq_neighbors"][()],
                file["mesh"]["lsq_monomial_multi_index"][()],
                file["mesh"]["lsq_scale_factors"][()],
                file["mesh"]["z_ordering"][()],
            )
        return mesh

    def write_to_vtk(
        self,
        filepath: str,
        fields: Union[FArray, None] = None,
        field_names: Union[list[str], None] = None,
        point_data: dict = {},
    ):
        if not _HAVE_MESHIO:
            raise RuntimeError(
                "_write_to_vtk_from_vertices_edges requires meshio, which is not available."
            )
        d_fields = {}
        vertex_coords_3d = np.zeros((3, self.vertex_coordinates.shape[1]))
        vertex_coords_3d[: self.vertex_coordinates.shape[0], :] = (
            self.vertex_coordinates
        )
        if fields is not None:
            if field_names is None:
                field_names = [str(i) for i in range(fields.shape[0])]
            for i_fields, field in enumerate(fields):
                d_fields[field_names[i_fields]] = [field]
        # the brackets around the second argument (cells) indicate that I have only mesh type of mesh element of type self.type, with corresponding vertices.
        meshout = meshio.Mesh(
            vertex_coords_3d.T,
            [
                (
                    mesh_util.convert_mesh_type_to_meshio_mesh_type(self.type),
                    self.cell_vertices.T,
                )
            ],
            cell_data=d_fields,
            point_data=point_data,
        )
        path, _ = os.path.split(filepath)
        filepath, file_ext = os.path.splitext(filepath)
        if not os.path.exists(path) and path != "":
            os.mkdir(path)
        meshout.write(filepath + ".vtk", binary=False)


if __name__ == "__main__":
    path = "/home/ingo/Git/sms/meshes/quad_2d/mesh_coarse.msh"
    path2 = "/home/ingo/Git/sms/meshes/quad_2d/mesh_fine.msh"
    path3 = "/home/ingo/Git/sms/meshes/quad_2d/mesh_finest.msh"
    path4 = "/home/ingo/Git/sms/meshes/triangle_2d/mesh_coarse.msh"
    labels = get_physical_boundary_labels(path)

    # dm, boundary_dict, ghost_cells_dict = load_gmsh(path)

    mesh = Mesh.from_gmsh(path)
    assert mesh.cell_faces.max() == mesh.n_faces - 1
    assert mesh.cell_faces.min() == 0
    assert mesh.face_cells.max() == mesh.n_cells - 1
    assert mesh.face_cells.min() == 0
    assert mesh.cell_vertices.max() == mesh.n_vertices - 1
    assert mesh.cell_vertices.min() == 0

    mesh.write_to_hdf5("./test.h5")
    mesh = Mesh.from_hdf5("./test.h5")
    # mesh.write_to_vtk('./test.vtk')
    mesh.write_to_vtk(
        "./test.vtk",
        fields=np.ones((2, mesh.n_inner_cells), dtype=float),
        field_names=["A", "B"],
    )
