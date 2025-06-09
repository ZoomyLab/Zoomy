import os
import h5py
from petsc4py import PETSc
import numpy as np
import meshio
import jax.numpy as jnp
import jax

import attr
from attr import define
from typing import Union, Any

from library.misc.custom_types import IArray, FArray, CArray
from library.mesh.mesh_util import compute_subvolume, get_extruded_mesh_type
import library.mesh.mesh_extrude as extrude
import library.mesh.mesh_util as mesh_util
from library.misc.static_class import register_static_pytree


# petsc4py.init(sys.argv)


def compute_gradient(u, A_glob, neighbors):
    def grad_single_cell(A_loc, neighbor_idx, u_i):
        u_neighbors = u[neighbor_idx]  # shape (n_neighbors,)
        delta_u = u_neighbors - u_i  # shape (n_neighbors,)
        return A_loc @ delta_u  # shape (dim,)

    # returns (n_cells, dim)
    return jax.vmap(grad_single_cell)(
        A_glob,  # shape (n_cells, dim, n_neighbors)
        neighbors,  # shape (n_cells, n_neighbors)
        u,  # shape (n_cells,)
    )


def least_squares_reconstruction(
    n_cells, dim, n_neighbors, neighbors, cell_centers, polynomial_degree=1
):
    A_glob = np.zeros((n_cells, dim, n_cells), dtype=float)
    R_glob = np.zeros((n_cells, n_neighbors, n_cells), dtype=float)
    for i_c in range(n_cells):
        # note, n_neighbors <= n_faces_per_cell. I need to keep the vectorized version using n_faces_per_cell for consistency and add zero lines
        dX = np.zeros((n_neighbors, dim), dtype=float)
        R_loc = np.zeros((n_neighbors, n_cells), dtype=float)
        A_loc = np.zeros((dim, n_neighbors), dtype=float)
        R_loc[:, i_c] = -1.0
        for i_neighbor, neighbor in enumerate(neighbors[i_c]):
            R_loc[i_neighbor, neighbor] = 1.0
            assert not np.allclose(cell_centers[neighbor], cell_centers[i_c])
            for d in range(dim):
                dX[i_neighbor, d] = cell_centers[neighbor][d] - cell_centers[i_c][d]
        A_loc = np.linalg.inv(dX.T @ dX) @ dX.T
        A_glob[i_c, :, :] = np.einsum("ij, jk->ik", A_loc, R_loc)
        R_glob[i_c, :, :] = R_loc
    return A_glob, R_glob


def least_squares_reconstruction_local(
    n_cells, dim, n_neighbors, neighbors, cell_centers, polynomial_degree=1
):
    A_glob = np.zeros((n_cells, dim, n_neighbors), dtype=float)
    # R_glob = np.zeros((n_cells, n_neighbors, n_cells), dtype=float)
    for i_c in range(n_cells):
        # note, n_neighbors <= n_faces_per_cell. I need to keep the vectorized version using n_faces_per_cell for consistency and add zero lines
        dX = np.zeros((n_neighbors, dim), dtype=float)
        R_loc = np.zeros((n_neighbors, n_cells), dtype=float)
        A_loc = np.zeros((dim, n_neighbors), dtype=float)
        R_loc[:, i_c] = -1.0
        for i_neighbor, neighbor in enumerate(neighbors[i_c]):
            R_loc[i_neighbor, neighbor] = 1.0
            assert not np.allclose(cell_centers[neighbor], cell_centers[i_c])
            for d in range(dim):
                dX[i_neighbor, d] = cell_centers[neighbor][d] - cell_centers[i_c][d]
        # A_loc = np.linalg.inv(dX.T @ dX) @ dX.T
        A_loc = np.linalg.pinv(dX)
        A_glob[i_c, :, :] = A_loc
        # R_glob[i_c, :, :] = R_loc
    return A_glob, A_glob


def get_physical_boundary_labels(filepath):
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


# def construct_ghost_cells(dm, boundary_dict):
#     dm.constructGhostCells()
#     (eStart, eEnd) = dm.getDepthStratum(1)
#     ghost_cells = {k: [] for k in boundary_dict.values()}
#     for e in range(eStart, eEnd):
#         label = dm.getLabelValue("Face Sets", e)
#         # 2 cells support an edge. Ghost cell is the one with the higher number
#         ghost_cell = dm.getSupport(e).max()
#         if label > -1:
#             ghost_cells[label].append(ghost_cell)
#     return dm, ghost_cells


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


# def compute_edge_data(dm, boundary_dict):
#     gdm = dm.clone()
#     gdm.constructGhostCells()
#     (cStart, cEnd) = gdm.getHeightStratum(0)
#     (vStart, vEnd) = gdm.getDepthStratum(0)
#     (eStart, eEnd) = gdm.getDepthStratum(1)
#     boundary_cells = {k: [] for k in boundary_dict.values()}
#     boundary_face_vertices = {k: [] for k in boundary_dict.values()}
#     for e in range(eStart, eEnd):
#         label = gdm.getLabelValue("Face Sets", e)
#         # 2 cells support an edge. Ghost cell is the one with the higher number
#         boundary_cell = gdm.getSupport(e).min() - cStart
#         if label > -1:
#             boundary_cells[label].append(boundary_cell)
#             boundary_face_vertices[label].append(gdm.getCone(e)-vStart)
#     return dm, boundary_cells, boundary_face_vertices


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


# def load_gmsh_to_dm(filepath):
#     dm = PETSc.DMPlex().createFromFile(filepath, comm=PETSc.COMM_WORLD)
#     boundary_dict = get_physical_boundary_labels(filepath)
#     dm, ghost_cells_dict = construct_ghost_cells(dm, boundary_dict)
#     return dm, boundary_dict, ghost_cells_dict


def load_gmsh_to_fvm(filepath):
    dm = PETSc.DMPlex().createFromFile(filepath, comm=PETSc.COMM_WORLD)
    boundary_dict = get_physical_boundary_labels(filepath)
    dm, boundary_cells_dict, boundary_face_vertices_dict = compute_boundary_data(
        dm, boundary_dict
    )
    boundary_face_corresponding_element = boundary_dict_to_list(boundary_cells_dict)
    boundary_face_vertices = boundary_dict_to_list(boundary_face_vertices_dict)
    return dm, boundary_dict, ghost_cells_dict


def _get_mesh_type(dm):
    return "quad"


def convert_to_fvm_mesh(dm, boundary_dict, ghost_cells_dict):
    dimension = dm.getDimension()
    type = _get_mesh_type(dm)
    assert dimension == 2
    (cStart, cEnd) = dm.getHeightStratum(0)
    (vStart, vEnd) = dm.getDepthStratum(0)
    (eStart, eEnd) = dm.getDepthStratum(1)

    n_elements = cEnd - cStart
    n_vertices = vEnd - vStart
    n_boundary_elements = []
    boundary_face_corresponding_element = []
    boundary_face_vertices = []

    for be_list_key, be_list_values in ghost_cells_dict.items():
        n_boundary_elements += len(be_list_values)
        boundary_face_corresponding_element += be_list_values
        for i in len(be_list_values):
            boundary_face_tag.append(be_list_key)
    boundary_tag_names = boundary_dict.keys()


def _get_neighberhood(dm, cell, cStart=0):
    neighbors = (
        np.array(
            [dm.getSupport(f)[dm.getSupport(f) != cell][0] for f in dm.getCone(cell)],
            dtype=int,
        )
        - cStart
    )
    return neighbors


@register_static_pytree
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
    boundary_conditions_sorted_physical_tags: IArray
    boundary_conditions_sorted_names: CArray
    lsq_gradQ: FArray
    deltaQ: FArray
    z_ordering: IArray

    @classmethod
    def create_1d(cls, domain: tuple[float, float], n_inner_cells: int):
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

        cell_centers = np.zeros((n_cells, dimension), dtype=float)
        cell_centers[:n_inner_cells, 0] = np.arange(xL + dx / 2, xR, dx)
        cell_centers[n_inner_cells, 0] = xL - dx / 2
        cell_centers[n_inner_cells + 1, 0] = xR + dx / 2
        cell_neighbors = (n_cells + 1) * np.ones((n_cells, n_faces_per_cell), dtype=int)

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
        boundary_face_ghosts = np.array([n_inner_cells, n_inner_cells + 1], dtype=int)
        boundary_face_function_numbers = np.empty((n_boundary_faces), dtype=int)
        boundary_face_function_numbers[0] = 0
        boundary_face_function_numbers[1] = 1
        boundary_face_physical_tags = np.array([0, 1], dtype=int)
        boundary_face_face_indices = np.array([0, n_faces - 1], dtype=int)

        face_cells = np.empty((n_faces, 2), dtype=int)
        # face_cell_face_index = (n_faces + 1)*np.ones((n_faces, 2), dtype=int)
        face_cells[1 : n_faces - 1, 0] = list(range(0, n_inner_cells - 1))
        face_cells[1 : n_faces - 1, 1] = list(range(1, n_inner_cells))
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

        polynomial_degree = 1
        n_neighbors = n_faces_per_cell * polynomial_degree
        dim = 1
        lsq_gradQ, deltaQ = least_squares_reconstruction_local(
            n_cells, dim, n_neighbors, cell_neighbors, cell_centers
        )

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
            boundary_conditions_sorted_physical_tags,
            boundary_conditions_sorted_names,
            lsq_gradQ,
            deltaQ,
            z_ordering,
        )

    def _compute_ascending_order_structured_axis(self):
        cell_centers = self.cell_centers
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
    def from_gmsh(cls, filepath, allow_z_integration=False):
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
        cell_vertices = np.zeros((n_inner_cells, n_vertices_per_cell), dtype=int)
        cell_faces = np.zeros((n_inner_cells, n_faces_per_cell), dtype=int)
        cell_centers = np.zeros((n_cells, dim), dtype=float)
        # I create cell_volumes of size n_cells because then I can avoid an if clause in the numerical flux computation. The values will be delted after using apply_boundary_conditions anyways
        cell_volumes = np.ones((n_cells), dtype=float)
        cell_inradius = compute_cell_inradius(dm)
        for i_c, c in enumerate(range(cStart, cEnd)):
            cell_volume, cell_center, cell_normal = dm.computeCellGeometryFVM(c)
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
            cell_centers[i_c, :] = cell_center
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
            face_volume, face_center, face_normal = gdm.computeCellGeometryFVM(e)
            face_center = gdm.computeCellGeometryFVM(e)[1]

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
                cell_centers[_face_ghost] = (
                    cell_centers[_face_cell]
                    + 2
                    * ((face_center - cell_centers[_face_cell]) @ face_normal)
                    * face_normal
                )
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
        cell_neighbors = (n_cells + 1) * np.ones((n_cells, n_neighbors), dtype=int)

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

        lsq_gradQ, deltaQ = least_squares_reconstruction_local(
            n_cells, dim, n_neighbors, cell_neighbors, cell_centers, polynomial_degree=1
        )

        n_face_neighbors = (2 * n_faces_per_cell - 1) * polynomial_degree
        face_neighbors = (n_cells + 1) * np.ones((n_cells, n_face_neighbors), dtype=int)

        # VECTORIZED CASE
        # cell_neighbors = (n_cells+1)*np.ones((n_cells, n_faces_per_cell+1), dtype=int)
        # lsq_A = []
        # lsq_D = np.zeros((n_cells, n_faces_per_cell+1, n_cells), dtype=float)
        # for i_c, c in enumerate(range(cgStart, cgEnd )):
        #     neighbors = _get_neighberhood(gdm, c, cStart=cgStart)
        #     assert not (i_c == neighbors).any()
        #     n_neighbors = neighbors.shape[0]
        #     if n_neighbors == 1:
        #         neighbors_of_neighbor = _get_neighberhood(gdm, neighbors[0]+cgStart, cStart = cgStart)
        #         assert len(neighbors_of_neighbor) == n_faces_per_cell
        #         neighbors = np.union1d(neighbors_of_neighbor, neighbors)
        #         n_neighbors = neighbors.shape[0]
        #     cell_neighbors[i_c, :n_neighbors]= neighbors
        #     # note, n_neighbors <= n_faces_per_cell. I need to keep the vectorized version using n_faces_per_cell for consistency and add zero lines
        #     dX = np.zeros((n_neighbors, dim), dtype=float)
        #     mat = np.zeros((dim , n_faces_per_cell+1), dtype=float)
        #     for i_neighbor, neighbor in enumerate(neighbors):
        #         lsq_D[i_c, i_neighbor, i_c] = -1.
        #         lsq_D[i_c, i_neighbor, neighbor] = 1.
        #         # lsq_D[neighbor, i_neighbor, i_c] = 1.
        #         for d in range(dim):
        #             dX[i_neighbor, d ] = cell_centers[neighbor][d] - cell_centers[i_c][d]
        #     mat[:, :n_neighbors] = np.linalg.inv(dX.T @ dX) @ dX.T
        #     lsq_A.append(mat)
        # lsq_A = np.array(lsq_A, dtype=float)
        # # TODO: sparse matrix.
        # lsq_gradQ = np.einsum('...ij, ...jk -> ...ik', lsq_A, lsq_D)

        face_volumes = np.array(face_volumes, dtype=float)
        face_centers = np.array(face_centers, dtype=float)
        face_normals = np.array(face_normals, dtype=float)
        face_subvolumes = np.array(face_subvolumes, dtype=float)

        face_cells = np.array(face_cells, dtype=int)
        # face_cell_face_index = np.array(face_cell_face_index, dtype=int)
        boundary_face_function_numbers = _boundary_dict_indices(boundary_face_cells)

        # get rid of empty keys in the boundary_dict (e.g. no surface values in 2d)
        boundary_dict_inverted = {v: k for k, v in boundary_dict.items()}
        boundary_dict_reduced = {k: boundary_dict_inverted[k] for k in allowed_keys}

        # sort the dict by the values
        sorted_keys = np.array(list(boundary_dict_reduced.keys()), dtype=int)
        sorted_keys.sort()
        boundary_dict = {k: boundary_dict_reduced[k] for k in sorted_keys}
        boundary_face_cells = {k: boundary_face_cells[k] for k in sorted_keys}
        boundary_face_ghosts = {k: boundary_face_ghosts[k] for k in sorted_keys}
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
            boundary_conditions_sorted_physical_tags,
            boundary_conditions_sorted_names,
            lsq_gradQ,
            deltaQ,
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
        vertex_coordinates = extrude.extrude_points(msh.vertex_coordinates.T, Z).T
        cell_vertices = extrude.extrude_element_vertices(
            msh.cell_vertices.T, msh.n_vertices, n_layers
        ).T

        cell_centers = np.empty((n_cells, 3), dtype=float)
        cell_volumes = np.empty((n_cells), dtype=float)
        cell_inradius = np.empty((n_cells), dtype=float)
        cell_face_areas = np.empty((n_cells, n_faces_per_cell), dtype=float)
        cell_face_normals = np.empty((n_cells, n_faces_per_cell, 3), dtype=float)
        cell_n_neighbors = np.empty((n_cells), dtype=int)
        cell_neighbors = np.empty((n_cells, n_faces_per_cell), dtype=int)
        cell_neighbors_face_index = np.empty((n_cells, n_faces_per_cell), dtype=int)
        for i_elem, elem in enumerate(cell_vertices.T):
            # cell_inradius[i_elem] = mesh_util.inradius(
            #    vertex_coordinates, elem, mesh_type
            # )
            # cell_volumes[i_elem] = mesh_util.volume(vertex_coordinates, elem, mesh_type)
            cell_centers[i_elem] = mesh_util.center(vertex_coordinates.T, elem)
            # cell_face_areas[i_elem, :] = mesh_util.face_areas(
            #    vertex_coordinates, elem, mesh_type
            # )
            # cell_face_normals[i_elem, :] = mesh_util.face_normals(
            #    vertex_coordinates, elem, mesh_type
            # )
            # (
            #    cell_n_neighbors[i_elem],
            #    cell_neighbors[i_elem, :],
            #    cell_neighbors_face_index[i_elem, :],
            # ) = mesh_util.get_element_neighbors(cell_vertices, elem, mesh_type)

        # boundaries
        # convenction: 1. bottom 2. sides 3. top
        # boundary_face_vertices = extrude.extrude_boundary_face_vertices(msh, n_layers)
        # boundary_face_corresponding_cell = (
        #    extrude.extrude_boundary_face_corresponding_element(msh, n_layers)
        # )

        # boundary_face_cell_face_index = np.empty((n_boundary_faces), dtype=int)
        # boundary_face_tag = np.empty((n_boundary_faces), dtype=int)
        # get a unique list of tags
        # boundary_tag_names = np.array(
        #    list(msh.boundary_tag_names) + [b"bottom", b"top"]
        # )
        # boundary_tags = extrude.extrude_boundary_face_tags(msh, n_layers)
        # for i_boundary_face, boundary_tag in enumerate(boundary_tags):
        #    for i_tag, tag in enumerate(boundary_tag_names):
        #        if tag == boundary_tag:
        #            boundary_face_tag[i_boundary_face] = i_tag
        # for i_face, face in enumerate(boundary_face_vertices):
        #    boundary_face_cell_face_index[i_face] = mesh_util.find_edge_index(
        #        cell_vertices[boundary_face_corresponding_cell[i_face]],
        #        face,
        #        mesh_type,
        #    )

        # truncate normals and positions from 3d to dimendion-d
        vertex_coordinates = vertex_coordinates.T[:, :dimension].T
        cell_centers = cell_centers[:, :dimension]
        # cell_face_normals = cell_face_normals[:, :, :dimension]

        # TODO
        # empty fields
        cell_faces = np.empty((n_inner_cells, n_faces_per_cell), dtype=int)
        boundary_face_cells = np.array([0, n_inner_cells - 1], dtype=int)
        boundary_face_ghosts = np.array([n_inner_cells, n_inner_cells + 1], dtype=int)
        boundary_face_function_numbers = np.empty((n_boundary_faces), dtype=int)
        boundary_face_physical_tags = np.array([0, 1], dtype=int)
        boundary_face_face_indices = np.array([0, n_faces - 1], dtype=int)
        face_cells = np.empty((n_faces, 2), dtype=int)
        face_subvolumes = np.empty((n_faces, 2), dtype=float)
        boundary_conditions_sorted_physical_tags = np.array([0, 1], dtype=int)
        boundary_conditions_sorted_names = np.array(["left", "right"])
        lsq_gradQ = np.zeros((n_cells, dimension, n_cells), dtype=float)
        deltaQ = np.zeros((n_cells, n_faces_per_cell, n_cells), dtype=float)
        face_normals = np.zeros((n_faces, dimension + 1), dtype=float)
        face_volumes = np.zeros((n_faces), dtype=float)
        face_centers = np.zeros((n_faces, dimension + 1), dtype=float)

        return cls(
            msh.dimension + 1,
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
            boundary_conditions_sorted_physical_tags,
            boundary_conditions_sorted_names,
            lsq_gradQ,
            deltaQ,
        )

    def write_to_hdf5(self, filepath: str):
        main_dir = os.getenv("SMS")
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
            mesh.create_dataset("vertex_coordinates", data=self.vertex_coordinates)
            mesh.create_dataset("cell_vertices", data=self.cell_vertices)
            mesh.create_dataset("cell_faces", data=self.cell_faces)
            mesh.create_dataset("cell_volumes", data=self.cell_volumes)
            mesh.create_dataset("cell_centers", data=self.cell_centers)
            mesh.create_dataset("cell_inradius", data=self.cell_inradius)
            mesh.create_dataset("cell_neighbors", data=self.cell_neighbors)
            mesh.create_dataset("boundary_face_cells", data=self.boundary_face_cells)
            mesh.create_dataset("boundary_face_ghosts", data=self.boundary_face_ghosts)
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
            mesh.create_dataset(
                "boundary_conditions_sorted_physical_tags",
                data=np.array(self.boundary_conditions_sorted_physical_tags),
            )
            mesh.create_dataset(
                "boundary_conditions_sorted_names",
                data=np.array(self.boundary_conditions_sorted_names, dtype="S"),
            )
            mesh.create_dataset("lsq_gradQ", data=np.array(self.lsq_gradQ))
            mesh.create_dataset("deltaQ", data=np.array(self.deltaQ))
            mesh.create_dataset("z_ordering", data=np.array(self.z_ordering))

    @classmethod
    def from_hdf5(cls, filepath: str):
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
                file["mesh"]["boundary_conditions_sorted_physical_tags"][()],
                np.array(
                    file["mesh"]["boundary_conditions_sorted_names"][()], dtype="str"
                ),
                file["mesh"]["lsq_gradQ"][()],
                file["mesh"]["deltaQ"][()],
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
            vertex_coords_3d,
            [(self.type, self.cell_vertices.T)],
            cell_data=d_fields,
            point_data=point_data,
        )
        path, _ = os.path.split(filepath)
        filepath, file_ext = os.path.splitext(filepath)
        if not os.path.exists(path) and path != "":
            os.mkdir(path)
        meshout.write(filepath + ".vtk")


@define(frozen=True, slots=True)
class MeshJAX(Mesh):
    dimension: int = attr.ib()
    type: str = attr.ib()
    n_cells: int = attr.ib()
    n_inner_cells: int = attr.ib()
    n_faces: int = attr.ib()
    n_vertices: int = attr.ib()
    n_boundary_faces: int = attr.ib()
    n_faces_per_cell: int = attr.ib()
    vertex_coordinates: jnp.ndarray = attr.ib()
    cell_vertices: jnp.ndarray = attr.ib()
    cell_faces: jnp.ndarray = attr.ib()
    cell_volumes: jnp.ndarray = attr.ib()
    cell_centers: jnp.ndarray = attr.ib()
    cell_inradius: jnp.ndarray = attr.ib()
    cell_neighbors: jnp.ndarray = attr.ib()
    boundary_face_cells: jnp.ndarray = attr.ib()
    boundary_face_ghosts: jnp.ndarray = attr.ib()
    boundary_face_function_numbers: jnp.ndarray = attr.ib()
    boundary_face_physical_tags: jnp.ndarray = attr.ib()
    boundary_face_face_indices: jnp.ndarray = attr.ib()
    face_cells: jnp.ndarray = attr.ib()
    # face_cell_face_index: jnp.ndarray = attr.ib()  # If needed
    face_normals: jnp.ndarray = attr.ib()
    face_volumes: jnp.ndarray = attr.ib()
    face_centers: jnp.ndarray = attr.ib()
    face_subvolumes: jnp.ndarray = attr.ib()
    boundary_conditions_sorted_physical_tags: jnp.ndarray = attr.ib()
    boundary_conditions_sorted_names: Any = attr.ib()  # Keeping as NumPy char array
    lsq_gradQ: jnp.ndarray = attr.ib()
    deltaQ: jnp.ndarray = attr.ib()
    z_ordering: jnp.ndarray = attr.ib()


def convert_mesh_to_jax(mesh: Mesh) -> MeshJAX:
    return MeshJAX(
        dimension=mesh.dimension,
        type=mesh.type,
        n_cells=mesh.n_cells,
        n_inner_cells=mesh.n_inner_cells,
        n_faces=mesh.n_faces,
        n_vertices=mesh.n_vertices,
        n_boundary_faces=mesh.n_boundary_faces,
        n_faces_per_cell=mesh.n_faces_per_cell,
        vertex_coordinates=jnp.array(mesh.vertex_coordinates),
        cell_vertices=jnp.array(mesh.cell_vertices),
        cell_faces=jnp.array(mesh.cell_faces),
        cell_volumes=jnp.array(mesh.cell_volumes),
        cell_centers=jnp.array(mesh.cell_centers),
        cell_inradius=jnp.array(mesh.cell_inradius),
        cell_neighbors=jnp.array(mesh.cell_neighbors),
        boundary_face_cells=jnp.array(mesh.boundary_face_cells),
        boundary_face_ghosts=jnp.array(mesh.boundary_face_ghosts),
        boundary_face_function_numbers=jnp.array(mesh.boundary_face_function_numbers),
        boundary_face_physical_tags=jnp.array(mesh.boundary_face_physical_tags),
        boundary_face_face_indices=jnp.array(mesh.boundary_face_face_indices),
        face_cells=jnp.array(mesh.face_cells),
        # face_cell_face_index=jnp.array(mesh.face_cell_face_index),  # If needed
        face_normals=jnp.array(mesh.face_normals),
        face_volumes=jnp.array(mesh.face_volumes),
        face_centers=jnp.array(mesh.face_centers),
        face_subvolumes=jnp.array(mesh.face_subvolumes),
        boundary_conditions_sorted_physical_tags=jnp.array(
            mesh.boundary_conditions_sorted_physical_tags
        ),
        boundary_conditions_sorted_names=list(
            mesh.boundary_conditions_sorted_names
        ),  # Kept as NumPy array
        lsq_gradQ=jnp.array(mesh.lsq_gradQ),
        deltaQ=jnp.array(mesh.deltaQ),
        z_ordering=jnp.array(mesh.z_ordering),
    )


if __name__ == "__main__":
    path = "/home/ingo/Git/sms/meshes/quad_2d/mesh_coarse.msh"
    path2 = "/home/ingo/Git/sms/meshes/quad_2d/mesh_fine.msh"
    path3 = "/home/ingo/Git/sms/meshes/quad_2d/mesh_finest.msh"
    path4 = "/home/ingo/Git/sms/meshes/triangle_2d/mesh_coarse.msh"
    labels = get_physical_boundary_labels(path)
    # print(labels)

    # dm, boundary_dict, ghost_cells_dict = load_gmsh(path)
    # print(ghost_cells_dict)

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
