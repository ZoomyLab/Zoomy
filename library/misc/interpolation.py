import numpy as np
import library.mesh.mesh_util as mesh_util

def _find_bounding_element(mesh, position):
    """
    Strategy: the faces of the elements are outward facing. If I compute compte the intersection of the normal with the point, 
    resulting in  alpha * normal = minimal_distance; then the alpha needs to be negative for all faces
    see https://en.wikipedia.org/wiki/Hesse_normal_form
    """
    mesh_type = mesh.type
    for i_elem, vertices in enumerate(mesh.element_vertices):

        faces = mesh_util._face_order(vertices, mesh_type)
        face_centers = [mesh_util.center(mesh.vertex_coordinates, np.array(face)) for face in faces]
        vector_origin_to_plane = [face_center - position for face_center in face_centers]
        face_normals = mesh.element_face_normals[i_elem]

        if _is_point_inside_bounding_faces(face_normals, vector_origin_to_plane):
            return i_elem

    # outside of domain
    assert False




def _is_point_inside_bounding_faces(outward_face_normals, vectors_OP):
    for (n, p) in zip(outward_face_normals, vectors_OP):
        if np.dot(n, p) < 0:
            return False
    return True


def to_new_mesh(fields, mesh_old, mesh_new, interp='const', map_fields = None):
    assert interp=='const'

    fields_new = np.zeros_like(fields)

    for i_elem in range(mesh_new.n_elements):
        element_center = mesh_new.element_center[i_elem]
        i_elem_old = _find_bounding_element(mesh_old, element_center)
        fields_new[i_elem] = fields[i_elem_old]
    return fields_new
        
    
