"""Reference-element geometry (face order, normals, areas, volumes, inradii) for the supported cell
types, used by ``BaseMesh``.
"""

import numpy as np

def convert_mesh_type_to_meshio_mesh_type(mesh_type: str) -> str:
    """Map a Zoomy cell type (``wface`` = wedge) to the meshio cell-type name."""
    if mesh_type == "triangle":
        return "triangle"
    elif mesh_type == "hexahedron":
        return "hexahedron"
    elif mesh_type == "line":
        return "line"
    elif mesh_type == "quad":
        return "quad"
    elif mesh_type == "wface":
        return "wedge"
    elif mesh_type == "tetra":
        return "tetra"
    else:
        assert False

def _get_n_nodes_per_element(mesh_type: str) -> int:
    """Number of vertices of a cell of ``mesh_type``."""
    if (mesh_type) == "quad":
        return 4
    elif (mesh_type) == "triangle":
        return 3
    elif (mesh_type) == "wface":
        return 6
    elif (mesh_type) == "hexahedron":
        return 8
    elif (mesh_type) == "tetra":
        return 4
    else:
        assert False

def get_extruded_mesh_type(mesh_type: str) -> str:
    """Cell type obtained by extruding ``mesh_type`` one layer (line -> quad, triangle -> wface,
    quad -> hexahedron).
    """
    if (mesh_type) == "quad":
        return "hexahedron"
    elif (mesh_type) == "triangle":
        return "wface"
    elif (mesh_type) == "line":
        return "quad"
    else:
        assert False


def get_global_cell_index_from_vertices(
    cells, coordinates, return_first=True, offset=0
):
    """Index (plus ``offset``) of the first cell, or of all cells if ``return_first`` is False,
    containing every vertex in ``coordinates``.
    """
    hits = []
    for index, cell in enumerate(cells):
        if set(coordinates).issubset(set(cell)):
            hits.append(offset + index)
            if return_first:
                return offset + index
    # if hits == []:
    #     assert False
    return hits


def get_element_neighbors(element_vertices, current_elem, mesh_type):
    """Elements sharing a face with ``current_elem``; returns ``(n_found, neighbour_indices,
    local_face_indices)``.
    """
    num_vertices_per_face = _get_num_vertices_per_face(mesh_type)
    max_num_neighbors = _get_faces_per_element(mesh_type)
    element_neighbor_indices = np.zeros((max_num_neighbors), dtype=int)
    element_neighbor_face_index = np.zeros((max_num_neighbors), dtype=int)
    n_found_neighbors = 0
    for i_elem, elem in enumerate(element_vertices):
        found_overlapping_vertices = set(elem).intersection(set(current_elem))
        n_found_overlapping_vertices = len(found_overlapping_vertices)

        if n_found_overlapping_vertices == np.min(num_vertices_per_face):
            for i_elem_face_index, edge in enumerate(
                _face_order(current_elem, mesh_type)
            ):
                if set(edge).issubset(found_overlapping_vertices):
                    element_neighbor_face_index[n_found_neighbors] = i_elem_face_index
                    break
            element_neighbor_indices[n_found_neighbors] = i_elem
            n_found_neighbors += 1
            if n_found_neighbors == max_num_neighbors:
                break
    return n_found_neighbors, element_neighbor_indices, element_neighbor_face_index


def face_normals(coordinates, element, mesh_type) -> float:
    """Unit normals of the faces of one element, ``(n_faces, 3)``; orientation follows the vertex
    order.
    """
    if mesh_type == "triangle":
        return _face_normals_2d(coordinates, element, mesh_type)
    elif mesh_type == "quad":
        return _face_normals_2d(coordinates, element, mesh_type)
    elif mesh_type == "tetra":
        return _face_normals_3d(coordinates, element, mesh_type)
    elif mesh_type == "hexahedron":
        return _face_normals_3d(coordinates, element, mesh_type)
    elif mesh_type == "wface":
        return _face_normals_3d(coordinates, element, mesh_type)
    assert False


def _face_normals_2d(coordinates, element, mesh_type) -> float:
    """Unit edge normals of a 2D element (edge direction crossed with ez)."""
    edges = _face_order(element, mesh_type)
    ez = np.array([0, 0, 1], dtype=float)
    normals = np.zeros((len(edges), 3), dtype=float)
    for i_edge, edge in enumerate(edges):
        edge_direction = coordinates[edge[1]] - coordinates[edge[0]]
        normals[i_edge, :] = -np.cross(edge_direction, ez)
        normals[i_edge, :] /= np.linalg.norm(normals[i_edge, :])

    # DEBUG: check that the normal goes into the right direction
    # center_point = center(coordinates, element)
    # ec = coordinates[edges]
    # edge_centers = [np.mean(ec[i], axis=0) for i in range(4)]
    # for i in range(4):
    #     assert np.allclose(center_point + 0.125 * normals[i], edge_centers[i])

    return normals


def _face_normals_3d(coordinates, element, mesh_type) -> float:
    """Unit face normals of a 3D element from the cross product of each face's first two edges."""
    faces = _face_order(element, mesh_type)
    boundary_mesh_type = _get_boundary_element_type(mesh_type)
    normals = np.zeros((len(faces), 3), dtype=float)
    for i_face, face in enumerate(faces):
        # I will only consider the first 2 edges, regardless of the element type.
        # flat elements, where 2 edges already span the plane and normal
        edges = _face_order(face, boundary_mesh_type[i_face])
        edge_1 = edges[0]
        edge_2 = edges[1]
        edge_direction_1 = coordinates[edge_1[1]] - coordinates[edge_1[0]]
        edge_direction_2 = coordinates[edge_2[1]] - coordinates[edge_2[0]]
        normals[i_face, :] = -np.cross(edge_direction_1, edge_direction_2)
        normals[i_face, :] /= np.linalg.norm(normals[i_face, :])
    return normals


def _face_normals_wface(coordinates, element, mesh_type) -> float:
    """Unit face normals of a wedge element (same construction as ``_face_normals_3d``)."""
    faces = _face_order(element, mesh_type)
    boundary_mesh_type = _get_boundary_element_type(mesh_type)
    normals = np.zeros((len(faces), 3), dtype=float)
    for i_face, face in enumerate(faces):
        # I will only consider the first 2 edges, regardless of the element type.
        # flat elements, where 2 edges already span the plane and normal
        edges = _face_order(face, boundary_mesh_type[i_face])
        edge_1 = edges[0]
        edge_2 = edges[1]
        edge_direction_1 = coordinates[edge_1[1]] - coordinates[edge_1[0]]
        edge_direction_2 = coordinates[edge_2[1]] - coordinates[edge_2[0]]
        normals[i_face, :] = -np.cross(edge_direction_1, edge_direction_2)
        normals[i_face, :] /= np.linalg.norm(normals[i_face, :])
    return normals


def face_areas(coordinates, element, mesh_type) -> float:
    """Area (length in 2D) of each face of one element."""
    num_faces = _get_faces_per_element(mesh_type)
    faces = _face_order(element, mesh_type)
    boundary_mesh_type = _get_boundary_element_type(mesh_type)
    face_areas = np.zeros(num_faces, dtype=float)

    for i_face, face in enumerate(faces):
        face_areas[i_face] = volume(coordinates, face, boundary_mesh_type[i_face])

    return face_areas


def center(coordinates, element) -> float:
    """Vertex-average centre of ``element``."""
    center = np.zeros(3, dtype=float)
    dim = coordinates.shape[1]
    for vertex_coord in coordinates[np.array(element)]:
        center[:dim] += vertex_coord
    center /= np.array(element).shape[0]
    return center[:dim]


def volume(coordinates, element, mesh_type) -> float:
    """Volume (area in 2D, length for lines) of one element of ``mesh_type``."""
    if mesh_type == "line":
        return edge_length(coordinates, element)
    elif mesh_type == "triangle":
        return _area_triangle(coordinates, element)
    elif mesh_type == "quad":
        return _area_quad(coordinates, element)
    elif mesh_type == "tetra":
        return _volume_tetra(coordinates, element)
    elif mesh_type == "hexahedron":
        return _volume_hex(coordinates, element)
    elif mesh_type == "wface":
        return _volume_wface(coordinates, element)
    assert False


def _area_triangle(coordinates, element) -> float:
    """Triangle area from its edge lengths."""
    edges = _edge_order_triangle(element)
    return _area_triangle_heron_formula(coordinates, edges)


def _area_triangle_heron_formula(coordinates, edges):
    """Triangle area from its three edges by Heron's formula."""
    perimeter = 0.0
    for edge in edges:
        perimeter += edge_length(coordinates, edge)
    s = perimeter / 2
    a = edge_length(coordinates, edges[0])
    b = edge_length(coordinates, edges[1])
    c = edge_length(coordinates, edges[2])
    return np.sqrt(s * (s - a) * (s - b) * (s - c))


def _area_quad(coordinates, element) -> float:
    # compute area by splitting in 2 triangles.
    """Quad area as the sum of two triangles split along the 0-2 diagonal."""
    edges_tri_1 = [
        (element[0], element[1]),
        (element[1], element[2]),
        (element[2], element[0]),
    ]
    edges_tri_2 = [
        (element[2], element[3]),
        (element[3], element[0]),
        (element[0], element[2]),
    ]
    return _area_triangle_heron_formula(
        coordinates, edges_tri_1
    ) + _area_triangle_heron_formula(coordinates, edges_tri_2)


# formula from https://en.wikipedia.org/wiki/Tetrahedron, section 'general properties'
def _volume_tetra(coordinates, element) -> float:
    """Tetrahedron volume from the scalar triple product."""
    a = coordinates[element[0]]
    b = coordinates[element[1]]
    c = coordinates[element[2]]
    d = coordinates[element[3]]
    volume = np.abs(np.dot((a - d), np.cross((b - d), (c - d)))) / 6
    return volume


def _volume_hex(coordinates, element) -> float:
    """Hexahedron volume as 12 tetrahedra between the cell centre and the split faces."""
    volume = 0.0
    d = center(coordinates, element)
    faces = _face_order_hex(element)
    for face in faces:
        # Split quad face (0,1,2,3) into triangles (0,1,2) and (0,2,3)
        a, b, c = coordinates[face[0]], coordinates[face[1]], coordinates[face[2]]
        volume += np.abs(np.dot(a - d, np.cross(b - d, c - d))) / 6
        a, b, c = coordinates[face[0]], coordinates[face[2]], coordinates[face[3]]
        volume += np.abs(np.dot(a - d, np.cross(b - d, c - d))) / 6
    return volume


def _volume_wface(coordinates, element) -> float:
    """Wedge volume as triangle area times height."""
    volume = 0
    faces = _face_order_wface(element)
    area_triangle = _area_triangle(coordinates, faces[0])
    height = np.linalg.norm(coordinates[element[0]] - coordinates[element[3]])
    return area_triangle * height


def inradius(coordinates, element, mesh_type) -> float:
    """Inradius of one element of ``mesh_type``."""
    if mesh_type == "triangle":
        return _inradius_triangle(coordinates, element)
    elif mesh_type == "quad":
        return _inradius_quad(coordinates, element)
    elif mesh_type == "tetra":
        return _inradius_tetra(coordinates, element)
    elif mesh_type == "hexahedron":
        return _inradius_generic(coordinates, element, "hexahedron")
    elif mesh_type == "wface":
        return _inradius_generic(coordinates, element, "wface")
    assert False


def _inradius_triangle(coordinates, element) -> float:
    """Triangle inradius ``sqrt((s-a)(s-b)(s-c)/s)``."""
    area = _area_triangle(coordinates, element)
    edges = _edge_order_triangle(element)
    perimeter = 0.0
    for edge in edges:
        perimeter += edge_length(coordinates, edge)
    s = perimeter / 2
    result = 1.0
    for edge in edges:
        result *= s - edge_length(coordinates, edge)
    result /= s
    return float(np.sqrt(result))


def _inradius_quad(coordinates, element) -> float:
    """Quad inradius ``area / semi-perimeter``."""
    area = _area_quad(coordinates, element)
    edges = _edge_order_quad(element)
    perimeter = 0.0
    for edge in edges:
        perimeter += edge_length(coordinates, edge)
    s = perimeter / 2
    return float(area / s)


# see https://en.wikipedia.org/wiki/Tetrahedron, "Inradius"
def _inradius_tetra(coordinates, element) -> float:
    """Tetrahedron inradius ``3 V / total face area``."""
    faces = _face_order_tetra(element)
    area = 0
    for face in faces:
        area += _area_triangle(coordinates, face)
    volume = _volume_tetra(coordinates, element)
    return 3 * volume / area


def _inradius_generic(coordinates, element, mesh_type) -> float:
    """Inradius as the shortest normal distance from the element centre to any face plane."""
    faces = _face_order(element, mesh_type)
    normals = face_normals(coordinates, element, mesh_type)
    element_center = center(coordinates, element)
    n_faces = len(faces)
    distances = np.empty(n_faces, dtype=float)
    for i_face, face in enumerate(faces):
        face_center = center(coordinates, face)
        face_normal = normals[i_face]
        distances[i_face] = np.abs(np.dot(face_center - element_center, face_normal))
    return np.min(distances)


def edge_length(coordinates, edge) -> float:
    """Euclidean length of ``edge``."""
    x0 = coordinates[edge[0]]
    x1 = coordinates[edge[1]]
    return np.linalg.norm(x1 - x0, 2)


def _face_order(element, mesh_type):
    """Vertex tuples of the faces of ``element``, dispatched on ``mesh_type``."""
    if mesh_type == "triangle":
        return _edge_order_triangle(element)
    elif mesh_type == "quad":
        return _edge_order_quad(element)
    elif mesh_type == "tetra":
        return _face_order_tetra(element)
    elif mesh_type == "hexahedron":
        return _face_order_hex(element)
    elif mesh_type == "wface":
        return _face_order_wface(element)
    else:
        assert False


def _edge_order_triangle(element):
    """Edges of a triangle in order."""
    return [
        (element[0], element[1]),
        (element[1], element[2]),
        (element[2], element[0]),
    ]


def _edge_order_quad(element):
    """Edges of a quad in order."""
    return [
        (element[0], element[1]),
        (element[1], element[2]),
        (element[2], element[3]),
        (element[3], element[0]),
    ]


def _face_order_tetra(element):
    """Triangular faces of a tetrahedron in order."""
    return [
        (element[0], element[1], element[2]),
        (element[0], element[1], element[3]),
        (element[1], element[2], element[3]),
        (element[2], element[0], element[3]),
    ]


def _face_order_hex(element):
    """Quad faces of a hexahedron in order (bottom, top, then sides)."""
    return [
        (element[0], element[1], element[2], element[3]),
        (element[4], element[5], element[6], element[7]),
        (element[0], element[1], element[5], element[4]),
        (element[1], element[2], element[6], element[5]),
        (element[2], element[3], element[7], element[6]),
        (element[3], element[0], element[4], element[7]),
    ]


def _face_order_wface(element):
    """Faces of a wedge in order (two triangles, then three quads)."""
    return [
        (element[0], element[1], element[2]),
        (element[3], element[4], element[5]),
        (element[0], element[3], element[4], element[1]),
        (element[1], element[4], element[5], element[2]),
        (element[2], element[5], element[3], element[0]),
    ]


def _get_num_vertices_per_face(mesh_type) -> float:
    """Vertices per face for each face of ``mesh_type``."""
    if mesh_type == "triangle":
        return [2, 2, 2]
    elif mesh_type == "quad":
        return [2, 2, 2, 2]
    elif mesh_type == "tetra":
        return [3, 3, 3, 3]
    elif mesh_type == "hexahedron":
        return [4, 4, 4, 4, 4, 4]
    elif mesh_type == "wface":
        return [3, 3, 4, 4, 4]
    assert False


def _get_dimension(mesh_type):
    """Spatial dimension of ``mesh_type``."""
    if mesh_type == "triangle":
        return 2
    elif mesh_type == "quad":
        return 2
    elif mesh_type == "tetra":
        return 3
    elif mesh_type == "hexahedron":
        return 3
    else:
        assert False


def _get_faces_per_element(mesh_type):
    """Number of faces of ``mesh_type``."""
    if mesh_type == "line":
        return 2
    elif mesh_type == "triangle":
        return 3
    elif mesh_type == "quad":
        return 4
    elif mesh_type == "tetra":
        return 4
    elif mesh_type == "hexahedron":
        return 6
    elif mesh_type == "wface":
        return 5
    else:
        assert False


def _get_boundary_element_type(mesh_type):
    """Cell type of each face of ``mesh_type``."""
    if mesh_type == "triangle":
        return ["line", "line", "line"]
    elif mesh_type == "quad":
        return ["line", "line", "line", "line"]
    elif mesh_type == "tetra":
        return ["triangle", "triangle", "triangle", "triangle"]
    elif mesh_type == "hexahedron":
        return ["quad", "quad", "quad", "quad", "quad", "quad"]
    elif mesh_type == "wface":
        return ["triangle", "triangle", "quad", "quad", "quad"]
    else:
        assert False


def find_edge_index(element, edge_vertices, element_type):
    """Local index of the edge/face of ``element`` whose vertices lie in ``edge_vertices``."""
    edges = _face_order(element, element_type)
    for i_edge, edge in enumerate(edges):
        if set(edge).issubset(edge_vertices):
            return i_edge
    assert False


def compute_subvolume(face_vertices, cell_center, dim):
    """Volume of the sub-simplex spanned by a face and the cell centre (length, triangle,
    tetrahedron or pyramid for dim 1, 2, 3).

    Args:
        face_vertices: coordinates of the face vertices.
        cell_center: coordinates of the cell centre.
        dim: spatial dimension (1, 2 or 3).
    """
    if dim == 1:
        # 1D: Length of the line segment
        return np.abs(face_vertices[1] - face_vertices[0])

    elif dim == 2:
        # 2D: Area of the triangle
        v0, v1 = face_vertices
        return 0.5 * np.abs(np.cross(v1 - v0, cell_center - v0))

    elif dim == 3:
        if face_vertices.shape[0] == 3:
            # 3D: Volume of the tetrahedron
            v0, v1, v2 = face_vertices
            return np.abs(np.dot(np.cross(v1 - v0, v2 - v0), cell_center - v0)) / 6.0
        elif face_vertices.shape[0] == 4:
            # 3D: Volume of pyramid with quad base, split into 2 tetrahedra
            v0, v1, v2, v3 = face_vertices
            vol1 = np.abs(np.dot(np.cross(v1 - v0, v2 - v0), cell_center - v0)) / 6.0
            vol2 = np.abs(np.dot(np.cross(v2 - v0, v3 - v0), cell_center - v0)) / 6.0
            return vol1 + vol2
        else:
            assert False

    else:
        raise ValueError("Unsupported dimensionality")
