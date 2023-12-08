import os
import numpy as np
import meshio
from compas.datastructures import Mesh as MeshCompas
from compas_gmsh.models import MeshModel
import h5py

from attr import define

from typing import Union, Tuple

from library.custom_types import IArray, FArray, CArray
from library.gmsh_loader import gmsh_to_domain_boundary_mesh
import library.mesh_util as mesh_util


@define(slots=True, frozen=True)
class Mesh:
    dimension: int
    type: str
    n_elements: int
    n_vertices: int
    n_boundary_elements: int
    n_faces_per_element: int
    vertex_coordinates: FArray
    element_vertices: IArray
    element_face_areas: IArray
    element_center: FArray
    element_volume: FArray
    element_inradius: FArray
    element_face_normals: FArray
    element_n_neighbors: IArray
    element_neighbors: IArray
    element_neighbors_face_index: IArray
    boundary_face_vertices: IArray
    boundary_face_corresponding_element: IArray
    boundary_face_element_face_index: IArray
    boundary_face_tag: IArray
    boundary_tag_names: CArray

    @classmethod
    def create_1d(cls, domain: tuple[float, float], n_elements: int):
        xL = domain[0]
        xR = domain[1]

        dimension = 1
        n_faces_per_element = 2
        dx = (xR - xL) / n_elements
        vertex_coordinates = np.zeros((n_elements + 1, 3))
        vertex_coordinates[:, 0] = np.linspace(
            xL, xR, n_elements + 1, dtype=float
        )
        element_vertices = np.zeros((n_elements, n_faces_per_element), dtype=int)
        element_vertices[:, 0] = np.linspace(0, n_elements - 1, n_elements, dtype=int)
        element_vertices[:, 1] = np.linspace(1, n_elements, n_elements, dtype=int)
        element_volume = dx * np.ones(n_elements, dtype=float)
        element_inradius = dx/2 * np.ones(n_elements, dtype=float)
        element_centers = np.zeros((n_elements, dimension), dtype=float)

        element_face_normals = np.zeros(
            (n_elements, n_faces_per_element, dimension), dtype=float
        )
        element_face_areas = np.ones(
            (n_elements, n_faces_per_element, dimension), dtype=float
        )

        element_centers[:, 0] = np.arange(xL + dx / 2, xR, dx)
        element_n_neighbors = 2*np.ones((n_elements), dtype=int)
        element_neighbors = np.empty((n_elements, n_faces_per_element), dtype=int)
        element_neighbors_face_index = np.empty((n_elements, n_faces_per_element), dtype=int)


        #inner elements
        for i_elem in range(0, n_elements):
            element_neighbors[i_elem, :] = np.array([i_elem - 1, i_elem + 1], dtype=int)
            element_face_normals[i_elem, :] = [
                np.array([-1.0], dtype=float),
                np.array([1.0], dtype=float),
            ]
            element_neighbors_face_index[i_elem, :] = np.array([0, 1], dtype=int)

        #first element
        element_n_neighbors[0] = 1
        element_neighbors_face_index[0, :] = np.array([1], dtype=int)
        
        #last element
        element_n_neighbors[n_elements -1] = 1
        element_neighbors_face_index[n_elements - 1, :] = np.array([0], dtype=int)

        n_boundary_elements = 2
        boundary_face_vertices = np.array([(0), (n_elements)], dtype=int)
        boundary_face_corresponding_element = np.array([0, n_elements-1], dtype=int)
        boundary_face_element_face_index = np.array([[0], [1]], dtype=int)
        boundary_face_tag = np.array([0, 1])
        boundary_tag_names = np.array(['left', 'right'], dtype='S10')

        boundary_face_vertices = np.array([0, n_elements - 1], dtype=int)

        return cls(dimension, 'line', n_elements, n_elements + 1, 2, n_faces_per_element, vertex_coordinates, element_vertices, element_face_areas, element_centers, element_volume, element_inradius, element_face_normals, element_n_neighbors, element_neighbors, element_neighbors_face_index, boundary_face_vertices, boundary_face_corresponding_element, boundary_face_element_face_index, boundary_face_tag, boundary_tag_names)


    @classmethod 
    def from_domain_boundary_mesh(cls, domain, boundary):
        # only allow for one group of one domain type
        assert len(domain.cells) == 1
        assert len(boundary.cells) == 1

        mesh_type = domain.cells[0].type
        dimension = mesh_util._get_dimension(mesh_type)
        n_elements = domain.cells[0].data.shape[0]
        n_vertices = domain.points.shape[0]
        boundary_element_type = mesh_util._get_boundary_element_type(mesh_type)
        n_boundary_faces = boundary.cells[0].data.shape[0]
        n_vertices_per_element = domain.cells[0].data.shape[1]
        n_faces_per_element = mesh_util._get_faces_per_element(mesh_type)
        vertex_coordinates = domain.points
        element_vertices = domain.cells[0].data

        # inner domain
        element_center = np.empty((n_elements, 3), dtype=float)
        element_volume = np.empty((n_elements), dtype=float)
        element_inradius = np.empty((n_elements), dtype=float)
        element_face_areas = np.empty((n_elements, n_faces_per_element), dtype=float)
        element_face_normals = np.empty((n_elements, n_faces_per_element, 3), dtype=float)
        element_n_neighbors = np.empty((n_elements), dtype=int)
        element_neighbors = np.empty((n_elements, n_faces_per_element), dtype=int)
        element_neighbors_face_index = np.empty((n_elements, n_faces_per_element), dtype=int)
        for i_elem, elem in enumerate(element_vertices):
            element_inradius[i_elem] = mesh_util.inradius(vertex_coordinates, elem, mesh_type)
            element_volume[i_elem] = mesh_util.volume(vertex_coordinates, elem, mesh_type)
            element_center[i_elem] = mesh_util.center(vertex_coordinates, elem)
            element_face_areas[i_elem, :] = mesh_util.face_areas(vertex_coordinates, elem, mesh_type)
            element_face_normals[i_elem, :] = mesh_util.face_normals(vertex_coordinates, elem, mesh_type)
            element_n_neighbors[i_elem], element_neighbors[i_elem, :], element_neighbors_face_index[i_elem, :] = mesh_util.get_element_neighbors(element_vertices, elem, mesh_type)

        
        # boundaries
        boundary_face_vertices = boundary.cells[0].data
        boundary_face_corresponding_element = boundary.cell_data['corresponding_cell'][0]
        boundary_face_element_face_index = np.empty((n_boundary_faces), dtype=int)
        # get a unique list of tags
        boundary_tag_names = list(set(boundary.cell_data['boundary_tag'][0]))
        boundary_tags = list(boundary.cell_data['boundary_tag'][0])
        boundary_face_tag = np.array([boundary_tags.index(tag) for tag in boundary_tag_names])
        for i_face, face in enumerate(boundary_face_vertices):
            boundary_face_element_face_index[i_face]  = mesh_util.find_edge_index(element_vertices[boundary_face_corresponding_element[i_face]], face, mesh_type)


        return cls(dimension, mesh_type, n_elements, n_vertices, n_boundary_faces, n_faces_per_element, vertex_coordinates, element_vertices, element_face_areas, element_center, element_volume, element_inradius, element_face_normals, element_n_neighbors, element_neighbors, element_neighbors_face_index, boundary_face_vertices, boundary_face_corresponding_element, boundary_face_element_face_index, boundary_face_tag, np.array(boundary_tag_names, dtype='S'))

    @classmethod
    def load_gmsh(cls, filepath, mesh_type):
        # load with gmsh_loader
        directory, mesh_name = os.path.split(filepath)
        domain, boundary = gmsh_to_domain_boundary_mesh(mesh_name, mesh_type, directory)
        return cls.from_domain_boundary_mesh(domain, boundary)
        

    @classmethod
    def load_cell_point_mesh(
        cls,
        cells, 
        points,
        mesh_type: str,
        dimension: int,
        boundary_tags: list[str],
    ):
        assert (mesh_type) == "quad" or (mesh_type) == "tri" or (mesh_type) == "tetra"
        mesh_type = mesh_type

        # points = mesh_io.points
        # cells = mesh_io.cells_dict[convert_mesh_type_to_meshio_mesh_type(mesh_type)]
        # make cells '1' based for compas
        mesh = MeshCompas.from_vertices_and_faces(points, cells)

        number_of_vertices = mesh.number_of_vertices()
        if dimension == 2:
            mesh.update_default_edge_attributes({"tag": "inner"})
        elif dimension == 3:
            mesh.update_default_face_attributes({"tag": "inner"})
        else: 
            assert False
        for boundary_tag in boundary_tags:
            # cell sets is a list containing !all! names quantities, e.g. physical curves/surfaces and the list of all gmsh:bounding_entities
            # for each entry, it contains a dict of all 'tags'. Therefore it can also be empty
            for i_cs, cs in enumerate(mesh_io.cell_sets[boundary_tag]):
                # get all lines contained by this physical entry
                for i_local_node in cs:
                    # get edge node ids and cast into tuple
                    face = tuple(mesh_io.cells[i_cs].data[i_local_node])
                    # set tag
                    if dimension == 2:
                        mesh.edge_attributes(face)["tag"] = boundary_tag
                    elif dimension == 3:
                        ## ERROR:  face (3 indices for the 3 nodes) 
                        ## but function takes single index, in constrast to the 2d version, where it takes an edge
                        ## Question: How to get this index, if the mesh_io.cells[].data[] gives only local indices, but no global ones?
                        # manual search for all face_attributes (containing 4 indices (all 4 nodes)), check when I match 3 of them?
                        face_index = mesh_util._get_global_cell_index_from_vertices(cells, face)
                        mesh.face_attributes(face_index)["tag"] = boundary_tag
                    else:
                        assert False


        return cls.from_compas_mesh(mesh, mesh_type, dimension)

    @classmethod
    def load_mesh(
        cls,
        filepath: str,
        mesh_type: str,
        dimension: int,
        boundary_tags: list[str],
    ):
        assert (mesh_type) == "quad" or (mesh_type) == "tri" or (mesh_type) == "tetra"
        mesh_type = mesh_type

        mesh_io = meshio.read(filepath)
        points = mesh_io.points
        cells = mesh_io.cells_dict[convert_mesh_type_to_meshio_mesh_type(mesh_type)]
        # make cells '1' based for compas
        mesh = MeshCompas.from_vertices_and_faces(points, cells)

        number_of_vertices = mesh.number_of_vertices()
        if dimension == 2:
            mesh.update_default_edge_attributes({"tag": "inner"})
        elif dimension == 3:
            mesh.update_default_face_attributes({"tag": "inner"})
        else: 
            assert False
        for boundary_tag in boundary_tags:
            # cell sets is a list containing !all! names quantities, e.g. physical curves/surfaces and the list of all gmsh:bounding_entities
            # for each entry, it contains a dict of all 'tags'. Therefore it can also be empty
            for i_cs, cs in enumerate(mesh_io.cell_sets[boundary_tag]):
                # get all lines contained by this physical entry
                for i_local_node in cs:
                    # get edge node ids and cast into tuple
                    face = tuple(mesh_io.cells[i_cs].data[i_local_node])
                    # set tag
                    if dimension == 2:
                        mesh.edge_attributes(face)["tag"] = boundary_tag
                    elif dimension == 3:
                        ## ERROR:  face (3 indices for the 3 nodes) 
                        ## but function takes single index, in constrast to the 2d version, where it takes an edge
                        ## Question: How to get this index, if the mesh_io.cells[].data[] gives only local indices, but no global ones?
                        # manual search for all face_attributes (containing 4 indices (all 4 nodes)), check when I match 3 of them?
                        face_index = mesh_util._get_global_cell_index_from_vertices(cells, face)
                        mesh.face_attributes(face_index)["tag"] = boundary_tag
                    else:
                        assert False


        return cls.from_compas_mesh(mesh, mesh_type, dimension)

    def from_comas_mesh_volume(mesh: MeshCompas, mesh_type: str, mesh_dimension: int):
        type = mesh_type
        dimension = mesh_dimension
        if mesh_dimension == 2:
            ez = np.array([0.0, 0.0, 1.0])
        n_nodes_per_element = get_n_nodes_per_element(mesh_type)

        n_vertices = mesh.number_of_vertices()
        n_edges = mesh.number_of_edges()
        n_elements = mesh.number_of_faces()
        vertex_coordinates = np.zeros((n_vertices, dimension), dtype=float)
        for i_vertex in range(n_vertices):
            vertex_coordinates[i_vertex, :] = mesh.vertex_coordinates(i_vertex)[
                :dimension
            ]
        element_vertices = np.zeros((n_elements, n_nodes_per_element), dtype=int)
        element_edge_length = np.zeros((n_elements, n_nodes_per_element), dtype=float)
        element_centers = np.zeros((n_elements, dimension), dtype=float)
        element_volume = np.zeros((n_elements), dtype=float)
        element_incircle = np.zeros((n_elements), dtype=float)
        element_edge_normal = -np.ones(
            (n_elements, n_nodes_per_element, dimension), dtype=float
        )
        element_edge_id = -np.ones((n_elements, n_nodes_per_element), dtype=float)

        # element_neighbors = mesh.face_adjacency()
        # manual computation of neighbors
        element_n_neighbors = np.zeros(n_elements, dtype=int)
        element_neighbors = np.zeros((n_elements, n_nodes_per_element), dtype=int)
        element_edge_normal = np.zeros(
            (n_elements, n_nodes_per_element, dimension), dtype=float
        )
        element_edge_length = np.zeros((n_elements, n_nodes_per_element), dtype=float)

        for face in np.fromiter(mesh.faces(), int):
            element_vertices[face] = np.array(mesh.face_vertices(face))
            element_volume[face] = mesh.face_area(face)
            element_incircle[face] = incircle(mesh, face, type)
            element_centers[face] = mesh.face_center(face)[:dimension]
            for i_edge, edge in enumerate(mesh.face_halfedges(face)):
                edge_faces = mesh.edge_faces(*edge)
                if edge_faces[0] != face and edge_faces[0] is not None:
                    element_neighbors[face, element_n_neighbors[face]] = edge_faces[0]
                    element_edge_normal[face, element_n_neighbors[face]] = -np.cross(
                        mesh.edge_direction(*edge), ez
                    )[:dimension]
                    element_edge_length[
                        face, element_n_neighbors[face]
                    ] = mesh.edge_length(*edge)
                    element_n_neighbors[face] += 1
                elif edge_faces[1] != face and edge_faces[1] is not None:
                    element_neighbors[face, element_n_neighbors[face]] = edge_faces[1]
                    element_edge_normal[face, element_n_neighbors[face]] = -np.cross(
                        mesh.edge_direction(*edge), ez
                    )[:dimension]
                    element_edge_length[
                        face, element_n_neighbors[face]
                    ] = mesh.edge_length(*edge)
                    element_n_neighbors[face] += 1
        return ( 
                dimension,
                type,
                n_elements,
                n_vertices,
                n_edges,
                n_nodes_per_element,
                vertex_coordinates,
                element_vertices,
                element_edge_length,
                element_centers,
                element_volume,
                element_incircle,
                element_edge_normal,
                element_neighbors,
                element_n_neighbors,
        )

    def from_comas_mesh_boundaries(mesh: MeshCompas, mesh_type: str, mesh_dimension: int):
        dimension = mesh_dimension

        edges_on_boundaries_list = []
        if dimension == 2:
            for edge in mesh.edges():
                if mesh.is_edge_on_boundary(*edge):
                    edges_on_boundaries_list.append(edge)
        elif dimension == 3:
            for edge in mesh.faces():
                if mesh.is_face_on_boundary(edge):
                    edges_on_boundaries_list.append(edge)
        edges_on_boundaries = np.array(edges_on_boundaries_list)
        boundary_edge_vertices = edges_on_boundaries
        n_boundary_edges = boundary_edge_vertices.shape[0]
        boundary_edge_elements = np.zeros(n_boundary_edges, dtype=int)
        boundary_edge_neighbors = np.zeros(n_boundary_edges, dtype=int)
        boundary_edge_tag = np.zeros(n_boundary_edges, dtype="S8")
        boundary_edge_length = np.zeros(n_boundary_edges, dtype=float)
        boundary_edge_normal = np.zeros((n_boundary_edges, dimension), dtype=float)

        if dimension == 2:
            for i_edge, edge in enumerate(edges_on_boundaries):
                edge_faces = mesh.edge_faces(*edge)
                boundary_edge_length[i_edge] = mesh.edge_length(*edge)
                boundary_edge_tag[i_edge] = mesh.edge_attributes(edge)["tag"]

                if edge_faces[0] == None:
                    boundary_edge_elements[i_edge] = edge_faces[1]
                    boundary_edge_normal[i_edge] = np.cross(mesh.edge_direction(*edge), ez)[
                        :dimension
                    ]
                elif edge_faces[1] == None:
                    boundary_edge_elements[i_edge] = edge_faces[0]
                    boundary_edge_normal[i_edge] = -np.cross(
                        mesh.edge_direction(*edge), ez
                    )[:dimension]
                else:
                    assert False
        elif dimension == 3:
            for i_edge, edge in enumerate(edges_on_boundaries):
                # mesh.face[4] -> seems to be the cells
                # mesh.edge_faces(27,34) seems to return the cells conected to one face.
                # edge_faces = mesh.edge_faces(edge)
                boundary_edge_length[i_edge] = mesh.face_area(edge)
                boundary_edge_tag[i_edge] = mesh.face_attributes(edge)["tag"]

                if edge_faces[0] == None:
                    boundary_edge_elements[i_edge] = edge_faces[1]
                    boundary_edge_normal[i_edge] = np.cross(mesh.edge_direction(*edge), ez)[
                        :dimension
                    ]
                elif edge_faces[1] == None:
                    boundary_edge_elements[i_edge] = edge_faces[0]
                    boundary_edge_normal[i_edge] = -np.cross(
                        mesh.edge_direction(*edge), ez
                    )[:dimension]
                else:
                    assert False

        
        n_edges_check , n_inner_edges = _compute_number_of_edges(n_elements, element_n_neighbors, n_nodes_per_element)
        assert(n_edges == n_edges_check)
        inner_edge_list = compute_edge_list_for_inner_domain(n_elements, element_n_neighbors, element_neighbors)
        
        return (
            n_inner_edges,
            n_boundary_edges,
            boundary_edge_vertices,
            boundary_edge_elements,
            boundary_edge_neighbors,
            boundary_edge_length,
            boundary_edge_normal,
            boundary_edge_tag,
        )


    @classmethod
    def from_compas_mesh(cls, mesh: MeshCompas, mesh_type: str, mesh_dimension: int):
        type = mesh_type
        dimension = mesh_dimension
        ez = np.array([0.0, 0.0, 1.0])
        n_nodes_per_element = get_n_nodes_per_element(mesh_type)

        n_vertices = mesh.number_of_vertices()
        n_edges = mesh.number_of_edges()
        n_elements = mesh.number_of_faces()
        vertex_coordinates = np.zeros((n_vertices, dimension), dtype=float)
        for i_vertex in range(n_vertices):
            vertex_coordinates[i_vertex, :] = mesh.vertex_coordinates(i_vertex)[
                :dimension
            ]
        element_vertices = np.zeros((n_elements, n_nodes_per_element), dtype=int)
        element_edge_length = np.zeros((n_elements, n_nodes_per_element), dtype=float)
        element_centers = np.zeros((n_elements, dimension), dtype=float)
        element_volume = np.zeros((n_elements), dtype=float)
        element_incircle = np.zeros((n_elements), dtype=float)
        element_edge_normal = -np.ones(
            (n_elements, n_nodes_per_element, dimension), dtype=float
        )
        element_edge_id = -np.ones((n_elements, n_nodes_per_element), dtype=float)

        # element_neighbors = mesh.face_adjacency()
        # manual computation of neighbors
        element_n_neighbors = np.zeros(n_elements, dtype=int)
        element_neighbors = np.zeros((n_elements, n_nodes_per_element), dtype=int)
        element_edge_normal = np.zeros(
            (n_elements, n_nodes_per_element, dimension), dtype=float
        )
        element_edge_length = np.zeros((n_elements, n_nodes_per_element), dtype=float)

        for face in np.fromiter(mesh.faces(), int):
            element_vertices[face] = np.array(mesh.face_vertices(face))
            element_volume[face] = mesh.face_area(face)
            element_incircle[face] = incircle(mesh, face, type)
            element_centers[face] = mesh.face_center(face)[:dimension]
            for i_edge, edge in enumerate(mesh.face_halfedges(face)):
                edge_faces = mesh.edge_faces(*edge)
                if edge_faces[0] != face and edge_faces[0] is not None:
                    element_neighbors[face, element_n_neighbors[face]] = edge_faces[0]
                    element_edge_normal[face, element_n_neighbors[face]] = -np.cross(
                        mesh.edge_direction(*edge), ez
                    )[:dimension]
                    element_edge_length[
                        face, element_n_neighbors[face]
                    ] = mesh.edge_length(*edge)
                    element_n_neighbors[face] += 1
                elif edge_faces[1] != face and edge_faces[1] is not None:
                    element_neighbors[face, element_n_neighbors[face]] = edge_faces[1]
                    element_edge_normal[face, element_n_neighbors[face]] = -np.cross(
                        mesh.edge_direction(*edge), ez
                    )[:dimension]
                    element_edge_length[
                        face, element_n_neighbors[face]
                    ] = mesh.edge_length(*edge)
                    element_n_neighbors[face] += 1

        edges_on_boundaries_list = []
        if dimension == 2:
            for edge in mesh.edges():
                if mesh.is_edge_on_boundary(*edge):
                    edges_on_boundaries_list.append(edge)
        elif dimension == 3:
            for edge in mesh.faces():
                if mesh.is_face_on_boundary(edge):
                    edges_on_boundaries_list.append(edge)
        edges_on_boundaries = np.array(edges_on_boundaries_list)
        boundary_edge_vertices = edges_on_boundaries
        n_boundary_edges = boundary_edge_vertices.shape[0]
        boundary_edge_elements = np.zeros(n_boundary_edges, dtype=int)
        boundary_edge_neighbors = np.zeros(n_boundary_edges, dtype=int)
        boundary_edge_tag = np.zeros(n_boundary_edges, dtype="S8")
        boundary_edge_length = np.zeros(n_boundary_edges, dtype=float)
        boundary_edge_normal = np.zeros((n_boundary_edges, dimension), dtype=float)

        if dimension == 2:
            for i_edge, edge in enumerate(edges_on_boundaries):
                edge_faces = mesh.edge_faces(*edge)
                boundary_edge_length[i_edge] = mesh.edge_length(*edge)
                boundary_edge_tag[i_edge] = mesh.edge_attributes(edge)["tag"]

                if edge_faces[0] == None:
                    boundary_edge_elements[i_edge] = edge_faces[1]
                    boundary_edge_normal[i_edge] = np.cross(mesh.edge_direction(*edge), ez)[
                        :dimension
                    ]
                elif edge_faces[1] == None:
                    boundary_edge_elements[i_edge] = edge_faces[0]
                    boundary_edge_normal[i_edge] = -np.cross(
                        mesh.edge_direction(*edge), ez
                    )[:dimension]
                else:
                    assert False
        elif dimension == 3:
            for i_edge, edge in enumerate(edges_on_boundaries):
                # mesh.face[4] -> seems to be the cells
                # mesh.edge_faces(27,34) seems to return the cells conected to one face.
                # edge_faces = mesh.edge_faces(edge)
                boundary_edge_length[i_edge] = mesh.face_area(edge)
                boundary_edge_tag[i_edge] = mesh.face_attributes(edge)["tag"]

                if edge_faces[0] == None:
                    boundary_edge_elements[i_edge] = edge_faces[1]
                    boundary_edge_normal[i_edge] = np.cross(mesh.edge_direction(*edge), ez)[
                        :dimension
                    ]
                elif edge_faces[1] == None:
                    boundary_edge_elements[i_edge] = edge_faces[0]
                    boundary_edge_normal[i_edge] = -np.cross(
                        mesh.edge_direction(*edge), ez
                    )[:dimension]
                else:
                    assert False

        
        n_edges_check , n_inner_edges = _compute_number_of_edges(n_elements, element_n_neighbors, n_nodes_per_element)
        assert(n_edges == n_edges_check)
        inner_edge_list = compute_edge_list_for_inner_domain(n_elements, element_n_neighbors, element_neighbors)
        
        return cls(
            dimension,
            type,
            n_elements,
            n_vertices,
            n_edges,
            n_inner_edges,
            n_boundary_edges,
            n_nodes_per_element,
            vertex_coordinates,
            element_vertices,
            element_edge_length,
            element_centers,
            element_volume,
            element_incircle,
            element_edge_normal,
            element_neighbors,
            element_n_neighbors,
            boundary_edge_vertices,
            boundary_edge_elements,
            boundary_edge_neighbors,
            boundary_edge_length,
            boundary_edge_normal,
            boundary_edge_tag,
            inner_edge_list,
        )
        

    @classmethod
    def from_compas_mesh_old(cls, mesh: MeshCompas, mesh_type: str, mesh_dimension: int):
        type = mesh_type
        dimension = mesh_dimension
        ez = np.array([0.0, 0.0, 1.0])
        n_nodes_per_element = get_n_nodes_per_element(mesh_type)

        n_vertices = mesh.number_of_vertices()
        n_edges = mesh.number_of_edges()
        n_elements = mesh.number_of_faces()
        vertex_coordinates = np.zeros((n_vertices, dimension), dtype=float)
        for i_vertex in range(n_vertices):
            vertex_coordinates[i_vertex, :] = mesh.vertex_coordinates(i_vertex)[
                :dimension
            ]
        element_vertices = np.zeros((n_elements, n_nodes_per_element), dtype=int)
        element_edge_length = np.zeros((n_elements, n_nodes_per_element), dtype=float)
        element_centers = np.zeros((n_elements, dimension), dtype=float)
        element_volume = np.zeros((n_elements), dtype=float)
        element_incircle = np.zeros((n_elements), dtype=float)
        element_edge_normal = -np.ones(
            (n_elements, n_nodes_per_element, dimension), dtype=float
        )
        element_edge_id = -np.ones((n_elements, n_nodes_per_element), dtype=float)

        # element_neighbors = mesh.face_adjacency()
        # manual computation of neighbors
        element_n_neighbors = np.zeros(n_elements, dtype=int)
        element_neighbors = np.zeros((n_elements, n_nodes_per_element), dtype=int)
        element_edge_normal = np.zeros(
            (n_elements, n_nodes_per_element, dimension), dtype=float
        )
        element_edge_length = np.zeros((n_elements, n_nodes_per_element), dtype=float)

        for face in np.fromiter(mesh.faces(), int):
            element_vertices[face] = np.array(mesh.face_vertices(face))
            element_volume[face] = mesh.face_area(face)
            element_incircle[face] = incircle(mesh, face, type)
            element_centers[face] = mesh.face_center(face)[:dimension]
            for i_edge, edge in enumerate(mesh.face_halfedges(face)):
                edge_faces = mesh.edge_faces(*edge)
                if edge_faces[0] != face and edge_faces[0] is not None:
                    element_neighbors[face, element_n_neighbors[face]] = edge_faces[0]
                    element_edge_normal[face, element_n_neighbors[face]] = -np.cross(
                        mesh.edge_direction(*edge), ez
                    )[:dimension]
                    element_edge_length[
                        face, element_n_neighbors[face]
                    ] = mesh.edge_length(*edge)
                    element_n_neighbors[face] += 1
                elif edge_faces[1] != face and edge_faces[1] is not None:
                    element_neighbors[face, element_n_neighbors[face]] = edge_faces[1]
                    element_edge_normal[face, element_n_neighbors[face]] = -np.cross(
                        mesh.edge_direction(*edge), ez
                    )[:dimension]
                    element_edge_length[
                        face, element_n_neighbors[face]
                    ] = mesh.edge_length(*edge)
                    element_n_neighbors[face] += 1

        edges_on_boundaries_list = []
        if dimension == 2:
            for edge in mesh.edges():
                if mesh.is_edge_on_boundary(*edge):
                    edges_on_boundaries_list.append(edge)
        elif dimension == 3:
            for edge in mesh.faces():
                if mesh.is_face_on_boundary(edge):
                    edges_on_boundaries_list.append(edge)
        edges_on_boundaries = np.array(edges_on_boundaries_list)
        boundary_edge_vertices = edges_on_boundaries
        n_boundary_edges = boundary_edge_vertices.shape[0]
        boundary_edge_elements = np.zeros(n_boundary_edges, dtype=int)
        boundary_edge_neighbors = np.zeros(n_boundary_edges, dtype=int)
        boundary_edge_tag = np.zeros(n_boundary_edges, dtype="S8")
        boundary_edge_length = np.zeros(n_boundary_edges, dtype=float)
        boundary_edge_normal = np.zeros((n_boundary_edges, dimension), dtype=float)

        if dimension == 2:
            for i_edge, edge in enumerate(edges_on_boundaries):
                edge_faces = mesh.edge_faces(*edge)
                boundary_edge_length[i_edge] = mesh.edge_length(*edge)
                boundary_edge_tag[i_edge] = mesh.edge_attributes(edge)["tag"]

                if edge_faces[0] == None:
                    boundary_edge_elements[i_edge] = edge_faces[1]
                    boundary_edge_normal[i_edge] = np.cross(mesh.edge_direction(*edge), ez)[
                        :dimension
                    ]
                elif edge_faces[1] == None:
                    boundary_edge_elements[i_edge] = edge_faces[0]
                    boundary_edge_normal[i_edge] = -np.cross(
                        mesh.edge_direction(*edge), ez
                    )[:dimension]
                else:
                    assert False
        elif dimension == 3:
            for i_edge, edge in enumerate(edges_on_boundaries):
                # mesh.face[4] -> seems to be the cells
                # mesh.edge_faces(27,34) seems to return the cells conected to one face.
                # edge_faces = mesh.edge_faces(edge)
                boundary_edge_length[i_edge] = mesh.face_area(edge)
                boundary_edge_tag[i_edge] = mesh.face_attributes(edge)["tag"]

                if edge_faces[0] == None:
                    boundary_edge_elements[i_edge] = edge_faces[1]
                    boundary_edge_normal[i_edge] = np.cross(mesh.edge_direction(*edge), ez)[
                        :dimension
                    ]
                elif edge_faces[1] == None:
                    boundary_edge_elements[i_edge] = edge_faces[0]
                    boundary_edge_normal[i_edge] = -np.cross(
                        mesh.edge_direction(*edge), ez
                    )[:dimension]
                else:
                    assert False

        
        n_edges_check , n_inner_edges = _compute_number_of_edges(n_elements, element_n_neighbors, n_nodes_per_element)
        assert(n_edges == n_edges_check)
        inner_edge_list = compute_edge_list_for_inner_domain(n_elements, element_n_neighbors, element_neighbors)
        
        return cls(
            dimension,
            type,
            n_elements,
            n_vertices,
            n_edges,
            n_inner_edges,
            n_boundary_edges,
            n_nodes_per_element,
            vertex_coordinates,
            element_vertices,
            element_edge_length,
            element_centers,
            element_volume,
            element_incircle,
            element_edge_normal,
            element_neighbors,
            element_n_neighbors,
            boundary_edge_vertices,
            boundary_edge_elements,
            boundary_edge_neighbors,
            boundary_edge_length,
            boundary_edge_normal,
            boundary_edge_tag,
            inner_edge_list,
        )

    @classmethod
    def from_hdf5(cls, filepath: str):
        with h5py.File(filepath, "r") as file:
            file_mesh = file["mesh"]
            mesh = cls(
                file_mesh["dimension"][()],
                (file_mesh["type"][()]).decode("utf-8"),
                file_mesh["n_elements"][()],
                file_mesh["n_vertices"][()],
                file_mesh["n_boundary_elements"][()],
                file_mesh["n_faces_per_element"][()],
                file_mesh["vertex_coordinates"][()],
                file_mesh["element_vertices"][()],
                file_mesh["element_face_areas"][()],
                file_mesh["element_center"][()],
                file_mesh["element_volume"][()],
                file_mesh["element_inradius"][()],
                file_mesh["element_face_normals"][()],
                file_mesh["element_n_neighbors"][()],
                file_mesh["element_neighbors"][()],
                file_mesh["element_neighbors_face_index"][()],
                file_mesh["boundary_face_vertices"][()],
                file_mesh["boundary_face_corresponding_element"][()],
                file_mesh["boundary_face_element_face_index"][()],
                file_mesh["boundary_face_tag"][()],
                file_mesh["boundary_tag_names"][()],
            )
        return mesh

    def write_to_file_vtk(
        self,
        filepath: str,
        fields: Union[FArray, None] = None,
        field_names: Union[list[str], None] = None,
        point_data: dict = {},
    ):
        d_fields = {}
        if fields is not None:
            if field_names is None:
                field_names = [str(i) for i in range(fields.shape[0])]
            for i_fields, field in enumerate(fields.T):
                d_fields[field_names[i_fields]] = [fields[:, i_fields]]
        meshout = meshio.Mesh(
            self.vertex_coordinates,
            [(self.type, self.element_vertices)],
            cell_data=d_fields,
            point_data=point_data,
        )
        path, _ = os.path.split(filepath)
        filepath, file_ext = os.path.splitext(filepath)
        if not os.path.exists(path) and path != "":
            os.mkdir(path)
        meshout.write(filepath + ".vtk")

    def write_to_hdf5(self, filepath: str):
        with h5py.File(filepath, "w") as f:
            attrs = f.create_group("mesh")
            attrs.create_dataset("dimension", data=self.dimension)
            attrs.create_dataset("type", data=self.type)
            attrs.create_dataset("n_elements", data=self.n_elements)
            attrs.create_dataset("n_vertices", data=self.n_vertices)
            attrs.create_dataset("n_boundary_elements", data=self.n_boundary_elements)
            attrs.create_dataset("n_faces_per_element", data=self.n_faces_per_element)
            attrs.create_dataset("vertex_coordinates", data=self.vertex_coordinates)
            attrs.create_dataset("element_vertices", data=self.element_vertices, dtype=int)
            attrs.create_dataset("element_face_areas", data=self.element_face_areas)
            attrs.create_dataset("element_center", data=self.element_center)
            attrs.create_dataset("element_volume", data=self.element_volume)
            attrs.create_dataset("element_inradius", data=self.element_inradius)
            attrs.create_dataset("element_face_normals", data=self.element_face_normals)
            attrs.create_dataset("element_n_neighbors", data=self.element_n_neighbors)
            attrs.create_dataset("element_neighbors", data=self.element_neighbors)
            attrs.create_dataset("element_neighbors_face_index", data=self.element_neighbors_face_index)
            attrs.create_dataset(
                "boundary_face_vertices", data=self.boundary_face_vertices
            )
            attrs.create_dataset(
                "boundary_face_corresponding_element", data=self.boundary_face_corresponding_element
            )
            attrs.create_dataset(
                "boundary_face_element_face_index", data=self.boundary_face_element_face_index
            )
            attrs.create_dataset("boundary_face_tag", data=self.boundary_face_tag)
            attrs.create_dataset("boundary_tag_names", data=self.boundary_tag_names)

def _compute_number_of_edges(n_elements, element_n_neighbors, n_nodes_per_element):
    n_edges = 0
    n_inner_edges = 0
    for elem in range(n_elements):
        n_neighbors = element_n_neighbors[elem]
        n_nodes = n_nodes_per_element
        # edges without neigbors count as one (outer edges)
        # edges with a neighbor get counted twice, therefore count as 0.5
        n_inner_edges += n_neighbors
        n_edges += (n_nodes - n_neighbors) * 1.0 + n_neighbors * 0.5
    #devide by two since I count all edges twice
    return int(n_edges), int(n_inner_edges/2)




def read_vtk_cell_fields(
    filename: str, n_fields: int, map_field_indices: list[int]
) -> FArray:
    mesh = meshio.read(filename)
    number_of_elements = mesh.cell_data["0"][0].shape[0]
    output = np.zeros((n_fields, number_of_elements))
    for k, v in mesh.cell_data.items():
        output[map_field_indices[int(k)]] = v[0]
    return output




def get_extruded_mesh_type(mesh_type: str) -> str:
    if (mesh_type) == "quad":
        return "hex"
    elif (mesh_type) == "tri":
        return "wface"
    else:
        assert False


def get_n_nodes_per_element(mesh_type: str) -> int:
    if (mesh_type) == "quad":
        return 4
    elif (mesh_type) == "tri":
        return 3
    elif (mesh_type) == "wface":
        return 6
    elif (mesh_type) == "hex":
        return 8
    elif (mesh_type) == "tetra":
        return 4
    else:
        assert False


def convert_mesh_type_to_meshio_mesh_type(mesh_type: str) -> str:
    if mesh_type == "tri":
        return "triangle"
    elif mesh_type == "hex":
        return "hexahedron"
    else:
        return mesh_type


def extrude_2d_element_vertices_mesh(
    mesh_type: str,
    vertex_coordinates: FArray,
    element_vertices: IArray,
    height: FArray,
    n_layers: int,
) -> Tuple[FArray, IArray, str]:
    n_vertices = vertex_coordinates.shape[0]
    n_elements = element_vertices.shape[0]
    num_nodes_per_element_2d = get_n_nodes_per_element(mesh_type)
    mesh_type = get_extruded_mesh_type(mesh_type)
    num_nodes_per_element = get_n_nodes_per_element(mesh_type)
    Z = np.linspace(0, 1, n_layers)
    points_3d = np.zeros(
        (
            vertex_coordinates.shape[0] * n_layers,
            3,
        ),
        dtype=float,
    )
    element_vertices_3d = np.zeros(
        (n_elements * (n_layers - 1), num_nodes_per_element), dtype=int
    )
    for i in range(n_vertices):
        points_3d[i * n_layers : (i + 1) * n_layers, :2] = vertex_coordinates[i]
        points_3d[i * n_layers : (i + 1) * n_layers, 2] = height[i] * Z

    # compute connectivity for mesh (element_vertices)
    for i_layer in range(n_layers - 1):
        element_vertices_3d[
            i_layer * n_elements : (i_layer + 1) * n_elements,
            :num_nodes_per_element_2d,
        ] = (
            i_layer + element_vertices * n_layers
        )
        element_vertices_3d[
            i_layer * n_elements : (i_layer + 1) * n_elements,
            num_nodes_per_element_2d:,
        ] = (
            i_layer + 1 + element_vertices * n_layers
        )
    return (points_3d, element_vertices_3d, mesh_type)


def write_to_file_vtk_from_vertices_edges(
    filepath,
    mesh_type,
    vertex_coordinates,
    element_vertices,
    fields=None,
    field_names=None,
    point_fields=None,
    point_field_names=None,
):
    assert (
        mesh_type == "tri"
        or mesh_type == "quad"
        or mesh_type == "wface"
        or mesh_type == "hex"
    )
    d_fields = {}
    if fields is not None:
        if field_names is None:
            field_names = [str(i) for i in range(fields.shape[0])]
        for i_fields, _ in enumerate(fields.T):
            d_fields[field_names[i_fields]] = [fields[:, i_fields]]
    point_d_fields = {}
    if point_fields is not None:
        if point_field_names is None:
            point_field_names = [str(i) for i in range(point_fields.shape[0])]
        for i_fields, _ in enumerate(point_fields):
            point_d_fields[point_field_names[i_fields]] = point_fields[i_fields]
    meshout = meshio.Mesh(
        vertex_coordinates,
        [(convert_mesh_type_to_meshio_mesh_type(mesh_type), element_vertices)],
        cell_data=d_fields,
        point_data=point_d_fields,
    )
    path, filename = os.path.split(filepath)
    filename_base, filename_ext = os.path.splitext(filename)
    os.makedirs(path, exist_ok=True)
    meshout.write(filepath + ".vtk")

def compute_edge_list_for_inner_domain(n_elements, element_n_neighbors, element_neighbors):
    # get the number of total edges
    n_edges = 0
    for i_elem in range(n_elements):
        for i_edge in range(element_n_neighbors[i_elem]):
            i_neighbor = element_neighbors[i_elem, i_edge]
            if i_elem < i_neighbor:
                n_edges += 1
    
    # traverse elements and edges and compile list
    element_edge_list = np.empty((n_edges, 2), dtype=int)
    index = 0
    for i_elem in range(n_elements):
        for i_edge in range(element_n_neighbors[i_elem]):
            i_neighbor = element_neighbors[i_elem, i_edge]
            if i_elem < i_neighbor:
                element_edge_list[index][0] = i_elem
                element_edge_list[index][1] = i_edge
                index += 1
    return element_edge_list


