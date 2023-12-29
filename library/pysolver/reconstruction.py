import numpy as np


from library.mesh.fvm_mesh import *

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
                edge+=1
    # boundary edges
    for elem, neighbor, be_normal, be_vertices in zip(mesh.boundary_edge_elements, mesh.boundary_edge_neighbors, mesh.boundary_edge_normal, mesh.boundary_edge_vertices ):
        elements_to_edges_i[edge] = elem
        elements_to_edges_j[edge] = neighbor
        # if np.dot(be_normal, normal_ji) > 0:
        #     elements_to_edges_i[edge] = neighbor
        #     elements_to_edges_j[edge] = elem
        # else:
        #     elements_to_edges_i[edge] = elem
        #     elements_to_edges_j[edge] = neighbor
        edge+=1

     
    #plus, minus terms
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
                edge+=1
    # boundary edges
    for elem, neighbor, normal, edge_length in zip(mesh.boundary_edge_elements, mesh.boundary_edge_neighbors, mesh.boundary_edge_normal, mesh.boundary_edge_length):
        normals_ij[edge] = normal
        edge_length_ij[edge] = edge_length
        edge+=1
    return normals_ij, edge_length_ij
    
def constant(mesh, fields, i_elem, i_th_neighbor):
    """
    fields: a list of fields (e.g. Q and Qaux)
    """
    # n_edges = mesh.n_edges
    dim = mesh.dimension
    n_fields = len(fields)
    n_dim_fields = [field.shape[1] for field in fields]
    field_i = [np.zeros((n_dim_fields[i]), dtype=float) for i in range(n_fields)]
    field_j = [np.zeros((n_dim_fields[i]), dtype=float) for i in range(n_fields)]

    # get neighborhood
    i_elem_neighbor = mesh.element_neighbors[ i_elem, i_th_neighbor ]

    # reconstruct
    for i_field in range(n_fields):
        field_i[i_field] = fields[i_field][i_elem]
        field_j[i_field] = fields[i_field][i_elem_neighbor]

    return field_i, field_j

def constant_edge(mesh, fields, field_ghost, i_elem):
    # n_edges = mesh.n_edges
    dim = mesh.dimension
    n_fields = len(fields)
    n_dim_fields = [field.shape[1] for field in fields]
    field_i = [np.zeros((n_dim_fields[i]), dtype=float) for i in range(n_fields)]
    field_j = [np.zeros((n_dim_fields[i]), dtype=float) for i in range(n_fields)]

    # get neighborhood

    # reconstruct
    for i_field in range(n_fields):
        field_i[i_field] = fields[i_field][i_elem]

    # assumption, the first field is Q. Use the ghost cell for that.
    # all further fields get a copy of the value of fields
    field_j[0] = field_ghost
    for i_field in range(1, n_fields):
        field_j[i_field] = fields[i_field][i_elem]

    return field_i, field_j
    

def constant_old(mesh, fields):
    n_edges = mesh.n_edges
    dim = mesh.dimension
    n_fields = len(fields)
    n_dim_fields = [field.shape[1] for field in fields]
    # n_eq = Q.shape[1]
    # n_aux_eq = Qaux.shape[1]

    field_i = [np.zeros((n_edges, n_dim_fields[i]), dtype=float) for i in range(n_fields)]
    field_j = [np.zeros((n_edges, n_dim_fields[i]), dtype=float) for i in range(n_fields)]

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
                for i_field in range(n_fields):
                    field_i[i_field][edge] = fields[i_field][elem]
                    field_j[i_field][edge] = fields[i_field][neighbor]
                # Qi[edge] = Q[elem] 
                # Qj[edge] = Q[neighbor] 
                # Qauxi[edge] = Qaux[elem]
                # Qauxj[edge] = Qaux[neighbor]
                edge+=1
    # boundary edges
    for elem, neighbor in zip(mesh.boundary_edge_elements, mesh.boundary_edge_neighbors):
        for i_field in range(n_fields):
            field_i[i_field][edge] = fields[i_field][elem]
            field_j[i_field][edge] = fields[i_field][neighbor]
        # Qi[edge] = Q[elem]
        # Qj[edge] = Q[neighbor]
        # Qauxi[edge] = Qaux[elem]
        # Qauxj[edge] = Qaux[neighbor]
        edge+=1
    # return Qi, Qj, Qauxi, Qauxj
        return field_i, field_j