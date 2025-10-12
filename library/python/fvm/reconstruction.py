import numpy as np
from attrs import define


from library.python.misc.custom_types import IArray, FArray, CArray
import library.python.mesh.fvm_mesh as fvm_mesh
import library.python.mesh.mesh as petsc_mesh

# TODO get rid of the boundary_conditions requirement
# HOW: rewrite mesh:segments. The mesh should already allocate indices for the ghost cells for each inner element (in particular element_neighbor ids)
# Then the mapping is always clear from a perspective of the inner elements.
# The segment does not need to hold information like normal and such, probably only about the indes mapping?
# ALTERNATIVE: Just update the element_neighbor ids with the proper index of the ghost cells (take care of the correct ordering according to the normals)
# then I do not need the extra loop over the boundary conditions. Do not update the n_neigbours count (treat it as an inner_neighbor count) to differentiate between
# ghost and inner cells.


def create_map_elements_to_edges(mesh):
    n_edges = mesh.n_edges
    n_all_elements = mesh.n_elements + mesh.n_boundary_edges
    elements_to_edges_i = np.empty((n_edges), dtype=int)
    elements_to_edges_j = np.empty((n_edges), dtype=int)

    edge = 0
    # inner elements
    for elem in range(mesh.n_elements):
        for i_neighbor in range(mesh.element_n_neighbors[elem]):
            neighbor = mesh.element_neighbors[elem, i_neighbor]
            # I can only count each edge once (otherwise I need double the memory)
            if elem < neighbor:
                elements_to_edges_j[edge] = neighbor
                elements_to_edges_i[edge] = elem
                edge += 1
    # boundary edges
    for elem, neighbor, be_normal, be_vertices in zip(
        mesh.boundary_edge_elements,
        mesh.boundary_edge_neighbors,
        mesh.boundary_edge_normal,
        mesh.boundary_edge_vertices,
    ):
        elements_to_edges_i[edge] = elem
        elements_to_edges_j[edge] = neighbor
        # if np.dot(be_normal, normal_ji) > 0:
        #     elements_to_edges_i[edge] = neighbor
        #     elements_to_edges_j[edge] = elem
        # else:
        #     elements_to_edges_i[edge] = elem
        #     elements_to_edges_j[edge] = neighbor
        edge += 1

    # plus, minus terms
    return elements_to_edges_j, elements_to_edges_i


def get_edge_geometry_data(mesh):
    n_edges = mesh.n_edges
    dim = mesh.dimension

    normals_ij = np.zeros((n_edges, dim), dtype=float)
    edge_length_ij = np.zeros((n_edges, dim), dtype=float)

    edge = 0
    # inner elements
    for elem in range(mesh.n_elements):
        for i_neighbor in range(mesh.element_n_neighbors[elem]):
            neighbor = mesh.element_neighbors[elem, i_neighbor]
            # I can only count each edge once (otherwise I need double the memory)
            if elem < neighbor:
                normals_ij[edge] = mesh.element_edge_normal[elem, i_neighbor]
                edge_length_ij[edge] = mesh.element_edge_length[elem, i_neighbor]
                edge += 1
    # boundary edges
    for elem, neighbor, normal, edge_length in zip(
        mesh.boundary_edge_elements,
        mesh.boundary_edge_neighbors,
        mesh.boundary_edge_normal,
        mesh.boundary_edge_length,
    ):
        normals_ij[edge] = normal
        edge_length_ij[edge] = edge_length
        edge += 1
    return normals_ij, edge_length_ij


def constant(mesh, fields, i_elem, i_th_neighbor):
    """
    fields: a list of fields (e.g. Q and Qaux)
    """
    # n_edges = mesh.n_edges
    dim = mesh.dimension
    n_variables = len(fields)
    n_dim_fields = [field.shape[1] for field in fields]
    field_i = [np.zeros((n_dim_fields[i]), dtype=float) for i in range(n_variables)]
    field_j = [np.zeros((n_dim_fields[i]), dtype=float) for i in range(n_variables)]

    # get neighborhood
    i_elem_neighbor = mesh.element_neighbors[i_elem, i_th_neighbor]

    # reconstruct
    for i_field in range(n_variables):
        field_i[i_field] = fields[i_field][i_elem]
        field_j[i_field] = fields[i_field][i_elem_neighbor]

    return field_i, field_j


def constant_edge(mesh, fields, field_ghost, i_elem):
    # n_edges = mesh.n_edges
    dim = mesh.dimension
    n_variables = len(fields)
    n_dim_fields = [field.shape[1] for field in fields]
    field_i = [np.zeros((n_dim_fields[i]), dtype=float) for i in range(n_variables)]
    field_j = [np.zeros((n_dim_fields[i]), dtype=float) for i in range(n_variables)]

    # get neighborhood

    # reconstruct
    for i_field in range(n_variables):
        field_i[i_field] = fields[i_field][i_elem]

    # assumption, the first field is Q. Use the ghost cell for that.
    # all further fields get a copy of the value of fields
    field_j[0] = field_ghost
    for i_field in range(1, n_variables):
        field_j[i_field] = fields[i_field][i_elem]

    return field_i, field_j


def constant_old(mesh, fields):
    n_edges = mesh.n_edges
    dim = mesh.dimension
    n_variables = len(fields)
    n_dim_fields = [field.shape[1] for field in fields]
    # n_eq = Q.shape[1]
    # n_aux_eq = Qaux.shape[1]

    field_i = [
        np.zeros((n_edges, n_dim_fields[i]), dtype=float) for i in range(n_variables)
    ]
    field_j = [
        np.zeros((n_edges, n_dim_fields[i]), dtype=float) for i in range(n_variables)
    ]

    # Qi = np.zeros((n_edges, n_eq), dtype=float)
    # Qj = np.zeros((n_edges, n_eq), dtype=float)
    # Qauxi = np.zeros((n_edges, n_aux_eq), dtype=float)
    # Qauxj = np.zeros((n_edges, n_aux_eq), dtype=float)

    edge = 0
    # inner elements
    for elem in range(mesh.n_elements):
        for i_neighbor in range(mesh.element_n_neighbors[elem]):
            neighbor = mesh.element_neighbors[elem, i_neighbor]
            # I can only count each edge once (otherwise I need double the memory)
            if elem < neighbor:
                for i_field in range(n_variables):
                    field_i[i_field][edge] = fields[i_field][elem]
                    field_j[i_field][edge] = fields[i_field][neighbor]
                # Qi[edge] = Q[elem]
                # Qj[edge] = Q[neighbor]
                # Qauxi[edge] = Qaux[elem]
                # Qauxj[edge] = Qaux[neighbor]
                edge += 1
    # boundary edges
    for elem, neighbor in zip(
        mesh.boundary_edge_elements, mesh.boundary_edge_neighbors
    ):
        for i_field in range(n_variables):
            field_i[i_field][edge] = fields[i_field][elem]
            field_j[i_field][edge] = fields[i_field][neighbor]
        # Qi[edge] = Q[elem]
        # Qj[edge] = Q[neighbor]
        # Qauxi[edge] = Qaux[elem]
        # Qauxj[edge] = Qaux[neighbor]
        edge += 1
        # return Qi, Qj, Qauxi, Qauxj
        return field_i, field_j


@define(slots=True, frozen=True)
class GradientMesh(fvm_mesh.Mesh):
    # TODO Problem: currently I use nx = [1, 0], ny = [0,1] - however, I think I need to do minmod(nx, -nx) or something similar in nd?
    element_face_coefficients: IArray

    @classmethod
    def fromMesh(cls, msh):
        dim = msh.dimension
        element_face_coefficients = np.zeros(
            (msh.n_elements, msh.n_faces_per_element, dim), dtype=float
        )
        normals = [np.eye(dim)[d, :] for d in range(dim)]

        for i_elem in range(msh.n_elements):
            x_self = msh.element_center[i_elem]
            for i_face in range(msh.element_n_neighbors[i_elem]):
                x_neighbor = msh.element_center[msh.element_neighbors[i_elem][i_face]]
                for d in range(dim):
                    element_face_coefficients[i_elem][i_face][d] = max(
                        np.dot(normals[d], msh.element_face_normals[i_elem, i_face]),
                        0.0,
                    )
                    element_face_coefficients[i_elem][i_face][d] /= np.linalg.norm(
                        x_neighbor - x_self
                    )
        return cls(
            msh.dimension,
            msh.type,
            msh.n_elements,
            msh.n_vertices,
            msh.n_boundary_elements,
            msh.n_faces_per_element,
            msh.vertex_coordinates,
            msh.element_vertices,
            msh.element_face_areas,
            msh.element_center,
            msh.element_volume,
            msh.element_inradius,
            msh.element_face_normals,
            msh.element_n_neighbors,
            msh.element_neighbors,
            msh.element_neighbors_face_index,
            msh.boundary_face_vertices,
            msh.boundary_face_corresponding_element,
            msh.boundary_face_element_face_index,
            msh.boundary_face_tag,
            msh.boundary_tag_names,
            element_face_coefficients,
        )

    def gradQ(self, Q):
        dim = self.dimension
        n_variables = Q.shape[1]
        gradQ = np.zeros((self.n_elements, n_variables, dim), dtype=float)
        for i_elem in range(self.n_elements):
            q_self = Q[i_elem]
            for i_face in range(self.element_n_neighbors[i_elem]):
                i_neighbor = self.element_neighbors[i_elem, i_face]
                q_neighbor = Q[i_neighbor]
                for d in range(dim):
                    gradQ[i_elem, :, d] += self.element_face_coefficients[
                        i_elem, i_face, d
                    ] * (q_neighbor - q_self)
        return gradQ


@define(slots=True, frozen=True)
class GradientPetscMesh(petsc_mesh.Mesh):
    # TODO Problem: currently I use nx = [1, 0], ny = [0,1] - however, I think I need to do minmod(nx, -nx) or something similar in nd?
    cell_face_coefficients: IArray

    @classmethod
    def fromMesh(cls, msh):
        dim = msh.dimension
        cell_face_coefficients = np.zeros(
            (msh.n_cells, msh.n_faces_per_cell, dim), dtype=float
        )
        normals = [np.eye(dim)[d, :] for d in range(dim)]

        for i_cell in range(msh.n_cells):
            x_self = msh.cell_centers[i_cell]
            for i_face in range(msh.cell_faces[i_cell]):
                x_neighbor = msh.cell_centers[msh.cell_neighbors[i_elem][i_face]]
                for d in range(dim):
                    cell_face_coefficients[i_elem][i_face][d] = max(
                        np.dot(normals[d], msh.cell_face_normals[i_elem, i_face]), 0.0
                    )
                    cell_face_coefficients[i_elem][i_face][d] /= np.linalg.norm(
                        x_neighbor - x_self
                    )
        return cls(
            msh.dimension,
            msh.type,
            msh.n_cells,
            msh.n_vertices,
            msh.n_boundary_cells,
            msh.n_faces_per_cell,
            msh.vertex_coordinates,
            msh.cell_vertices,
            msh.cell_face_areas,
            msh.cell_center,
            msh.cell_volume,
            msh.cell_inradius,
            msh.cell_face_normals,
            msh.cell_n_neighbors,
            msh.cell_neighbors,
            msh.cell_neighbors_face_index,
            msh.boundary_face_vertices,
            msh.boundary_face_corresponding_cell,
            msh.boundary_face_cell_face_index,
            msh.boundary_face_tag,
            msh.boundary_tag_names,
            cell_face_coefficients,
        )

    def gradQ(self, Q):
        dim = self.dimension
        n_variables = Q.shape[1]
        gradQ = np.zeros((self.n_cells, n_variables, dim), dtype=float)
        for i_elem in range(self.n_cells):
            q_self = Q[i_elem]
            for i_face in range(self.cell_n_neighbors[i_elem]):
                i_neighbor = self.cell_neighbors[i_elem, i_face]
                q_neighbor = Q[i_neighbor]
                for d in range(dim):
                    gradQ[i_elem, :, d] += self.cell_face_coefficients[
                        i_elem, i_face, d
                    ] * (q_neighbor - q_self)
        return gradQ
