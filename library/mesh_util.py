import numpy as np

def get_global_cell_index_from_vertices(cells, coordinates, return_first=True):
    hits = []
    for index, cell in enumerate(cells):
        if set(coordinates).issubset(set(cell)):
            hits.append(index)
            if return_first:
                return index
    if hits == []:
        assert False
    return hits

def get_element_neighbors(element_vertices, current_elem, mesh_type):
    num_vertices_per_face = _get_num_vertices_per_face(mesh_type)
    max_num_neighbors = _get_faces_per_element(mesh_type)
    element_neighbor_indices = np.zeros((max_num_neighbors), dtype=int)
    n_found_neighbors = 0
    for i_elem, elem in enumerate(element_vertices):
        n_found_overlapping_vertices = 0
        n_found_overlapping_vertices = len(set(elem).intersection(set(current_elem)))
        if n_found_overlapping_vertices == num_vertices_per_face:
            element_neighbor_indices[n_found_neighbors] = i_elem
            n_found_neighbors += 1
            if n_found_neighbors == max_num_neighbors:
                break
    return n_found_neighbors, element_neighbor_indices
    

def face_normals(coordinates, element, mesh_type) -> float:
    if mesh_type == "triangle":
        return face_normals_2d(coordinates, element, mesh_type)
    assert False

def face_normals_2d(coordinates, element, mesh_type) -> float:
    edges = _edge_order(element, mesh_type)
    ez = np.array([0, 0, 1], dtype=float)
    normals = np.zeros((len(edges),3), dtype=float)
    for i_edge, edge in enumerate(edges):
        edge_direction = coordinates[edge[1]] - coordinates[edge[0]]
        normals[i_edge, :]  = -np.cross(edge_direction, ez)
        normals[i_edge,:] /=  np.linalg.norm(normals[i_edge,:])
    return normals


    return edge_lengths

def face_areas(coordinates, element, mesh_type) -> float:
    if mesh_type == "triangle":
        return face_areas_triangle(coordinates, element)
    assert False

def face_areas_triangle(coordinates, element) -> float:
    edges = _edge_order_triangle(element)
    edge_lengths = np.zeros(3, dtype=float)
    for i_edge, edge in enumerate(edges):
        edge_lengths[i_edge] = edge_length(coordinates, edge)
    return edge_lengths
        
    

def center(coordinates, element) -> float:
    center = np.zeros(3, dtype=float)
    for vertex_coord in coordinates[element]:
        center += vertex_coord
    center /= element.shape[0]
    return center

def volume(coordinates, element, mesh_type) -> float:
    if mesh_type == "triangle":
        return area_triangle(coordinates, element)
    assert False

def area_triangle(coordinates, element) -> float:
    edges = _edge_order_triangle(element)
    perimeter = 0.0
    for edge in edges:
        perimeter += edge_length(coordinates, edge)
    s = perimeter / 2
    a = edge_length(coordinates, edges[0])
    b = edge_length(coordinates, edges[1])
    c = edge_length(coordinates, edges[2])
    return np.sqrt(s*(s-a)*(s-b)*(s-c))
    


def insphere(coordinates, element, mesh_type) -> float:
    if mesh_type == "triangle":
        return 2 * incircle_triangle(coordinates, element)
    elif mesh_type == "quad":
        return 2 * incircle_quad(coordinates, element)
    elif mesh_type == "tetra":
        # get shortest edge length
        # compute incircle via formula for regular tetra using the shortest edge length
        return 2 * insphere_tetra(coordinates, element)
    assert False

def incircle_triangle(coordinates, element) -> float:
    area = area_triangle(coordinates, element)
    edges = _edge_order_triangle(element)
    perimeter = 0.0
    for edge in edges:
        perimeter += edge_length(coordinates, edge)
    s = perimeter / 2
    result = 1.0
    for edge in edges:
        result *= s - edge_length(coordinates, edge)
    return float(np.sqrt(result))


# def insphere_tetra(mesh: MeshCompas, face: int) -> float:
#     # get shortest edge length
#     edge_length_min = np.inf
#     for edge in mesh.face_halfedges(face):
#         edge_length_min = min(edge_length_min, mesh.edge_length(*edge))
#     # compute incircle via formula for regular tetra using the shortest edge length
#     return float(edge_length_min / np.sqrt(24))

# def incircle_quad(mesh: MeshCompas, face: int) -> float:
#     area = mesh.face_area(face)
#     edges = mesh.face_halfedges(face)
#     perimeter = 0.0
#     for edge in edges:
#         perimeter += mesh.edge_length(*edge)
#     s = perimeter / 2
#     return float(area / s)


def edge_length(coordinates, edge) -> float:
    x0 = coordinates[edge[0]]
    x1 = coordinates[edge[1]]
    return np.linalg.norm(x1-x0, 2)

def _edge_order(element, mesh_type):
    if mesh_type == 'triangle':
        return _edge_order_triangle(element)
    else:
        assert False

def _edge_order_triangle(element):
    return [(element[0], element[1]), (element[1], element[2]), (element[2], element[0])]

def _get_num_vertices_per_face(mesh_type) -> float:
    if mesh_type == "triangle":
        return 2
    assert False

def _get_dimension(mesh_type):
    if mesh_type == 'triangle':
        return 2
    elif mesh_type == 'quad':
        return 2
    elif mesh_type == 'tetra':
        return 3
    elif mesh_type == 'hex':
        return 3
    else:
        assert False

def _get_faces_per_element(mesh_type):
    if mesh_type == 'triangle':
        return 3
    elif mesh_type == 'quad':
        return 4
    elif mesh_type == 'tetra':
        return 4
    elif mesh_type == 'hex':
        return 6
    else:
        assert False



    
    
