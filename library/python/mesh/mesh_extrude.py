import numpy as np


def extrude_points(points, Z):
    n_points = points.shape[0]
    dim = points.shape[1]
    Nz = Z.shape[0]
    points_ext = np.empty((n_points * (Nz), dim + 1), dtype=float)
    for i in range(n_points):
        for iz, z in enumerate(Z):
            offset = iz * n_points
            points_ext[i + offset, :dim] = points[i]
            points_ext[i + offset, -1] = z
    return points_ext


def extrude_element_vertices(element_vertices, n_vertices, Nz):

    n_elements = element_vertices.shape[0]
    n_vertices_per_element = element_vertices.shape[1]
    if n_vertices_per_element == 2:
        return extrude_element_vertices_line(element_vertices, n_vertices, Nz)
    element_vertices_ext = np.empty(
        (n_elements * Nz, 2 * n_vertices_per_element), dtype=int
    )
    for i in range(n_elements):
        for iz in range(Nz):
            offset = iz * n_elements
            element_vertices_ext[i + offset, :n_vertices_per_element] = (
                element_vertices[i, :] + iz * n_vertices
            )
            element_vertices_ext[i + offset, n_vertices_per_element:] = (
                element_vertices[i, :] + (iz + 1) * n_vertices
            )
    return element_vertices_ext

def extrude_element_vertices_line(element_vertices, n_vertices, Nz):
    n_elements = element_vertices.shape[0]
    n_vertices_per_element = 2
    element_vertices_ext = np.empty(
        (n_elements * Nz, 2 * n_vertices_per_element), dtype=int
    )
    for i in range(n_elements):
        for iz in range(Nz):
            offset = iz * n_elements
            element_vertices_ext[i + offset, [0, 3]] = (
                element_vertices[i, :] + iz * n_vertices
            )
            element_vertices_ext[i + offset, [1,2]] = (
                element_vertices[i, :] + (iz + 1) * n_vertices
            )
    return element_vertices_ext


def extrude_boundary_face_vertices(msh, Nz):
    """
    convenction: 1. bottom 2. sides 3. top
    """
    n_boundary_elements = msh.n_boundary_elements * Nz + 2 * msh.n_elements
    n_vertices_per_face = msh.boundary_face_vertices.shape[1]
    n_vertices_per_face_ext = 2 * n_vertices_per_face
    boundary_face_vertices = np.empty(
        (n_boundary_elements, n_vertices_per_face_ext), dtype=int
    )

    # for wface, boundary elements may have 3 or 4 vertices. In case of 3, I want to overwrite the other entries to be dublicates of existing entries
    side_offset = 0
    # bottom
    for i in range(msh.n_elements):
        boundary_face_vertices[i, : msh.element_vertices[i].shape[0]] = (
            msh.element_vertices[i]
        )
        boundary_face_vertices[i, msh.element_vertices[i].shape[0] :] = (
            msh.element_vertices[i][-1]
        )
    side_offset += msh.n_elements

    n_vertices_per_layer = msh.n_vertices
    # sides
    for i in range(msh.n_boundary_elements):
        face_vertices = msh.boundary_face_vertices[i]
        for iz in range(Nz):
            offset = iz * msh.n_boundary_elements + side_offset
            boundary_face_vertices[i + offset, :n_vertices_per_face] = (
                face_vertices + iz * n_vertices_per_layer
            )
            boundary_face_vertices[i + offset, n_vertices_per_face:] = (
                face_vertices + (iz + 1) * n_vertices_per_layer
            )
    side_offset += Nz * msh.n_boundary_elements

    # top
    for i in range(msh.n_elements):
        # boundary_face_vertices[i + side_offset] = msh.element_vertices[i] + Nz * msh.n_vertices
        boundary_face_vertices[i + side_offset, : msh.element_vertices[i].shape[0]] = (
            msh.element_vertices[i] + Nz * msh.n_vertices
        )
        boundary_face_vertices[i + side_offset, msh.element_vertices[i].shape[0] :] = (
            msh.element_vertices[i][-1] + Nz * msh.n_vertices
        )

    return boundary_face_vertices


def extrude_boundary_face_corresponding_element(msh, Nz):
    """
    convenction: 1. bottom 2. sides 3. top
    """
    n_boundary_elements = msh.n_boundary_elements * Nz + 2 * msh.n_elements
    boundary_face_corresponding_element = np.empty((n_boundary_elements), dtype=int)

    side_offset = 0
    # bottom
    for i in range(msh.n_elements):
        boundary_face_corresponding_element[i] = i
    side_offset += msh.n_elements

    # sides
    for i in range(msh.n_boundary_elements):
        i_elem = msh.boundary_face_corresponding_element[i]
        for iz in range(Nz):
            offset_boundary = iz * msh.n_boundary_elements + side_offset
            offset_elements = iz * msh.n_elements
            boundary_face_corresponding_element[i + offset_boundary] = (
                i_elem + offset_elements
            )
    side_offset += Nz * msh.n_boundary_elements

    # top
    for i in range(msh.n_elements):
        boundary_face_corresponding_element[i + side_offset] = (
            i + (Nz - 1) * msh.n_elements
        )

    return boundary_face_corresponding_element


def extrude_boundary_face_tags(msh, Nz):
    """
    convenction: 1. bottom 2. sides 3. top
    """
    n_boundary_elements = msh.n_boundary_elements * Nz + 2 * msh.n_elements
    boundary_face_tags = np.empty((n_boundary_elements), int)

    n_existing_tags = len(msh.boundary_tag_names)
    tag_bottom = n_existing_tags
    tag_top = n_existing_tags + 1

    side_offset = 0
    # bottom
    for i in range(msh.n_elements):
        boundary_face_tags[i] = tag_bottom
    side_offset += msh.n_elements

    # sides
    for i in range(msh.n_boundary_elements):
        i_elem = msh.boundary_face_corresponding_element[i]
        for iz in range(Nz):
            offset_boundary = iz * msh.n_boundary_elements + side_offset
            offset_elements = iz * msh.n_elements
            boundary_face_tags[i + offset_boundary] = msh.boundary_face_tag[i]
    side_offset += Nz * msh.n_boundary_elements

    # top
    for i in range(msh.n_elements):
        boundary_face_tags[i + side_offset] = tag_top

    return boundary_face_tags
