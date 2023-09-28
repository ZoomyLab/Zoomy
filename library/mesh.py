import os
import numpy as np
import meshio
from compas.datastructures import Mesh as MeshCompas
from compas_gmsh.models import MeshModel

from attr import define

# from typing import Union
from typing import TypeVar, Type, Any

# Create a generic variable that can be 'Mesh', or any subclass.
MeshType = TypeVar("MeshType", bound="Mesh")

from library.custom_types import IArray, FArray, CArray


@define(slots=True, frozen=True)
class Mesh:
    dim: int
    type: str
    n_elements: int
    n_vertices: int
    n_boundary_edges: int
    n_nodes_per_element: int
    vertex_coordinates: FArray
    element_vertices: IArray
    element_edge_length: IArray
    element_centers: FArray
    element_volume: FArray
    element_incircle: FArray
    element_edge_normal: FArray
    element_neighbors: IArray
    boundary_edge_vertices: IArray
    boundary_edge_elements: IArray
    boundary_edge_length: FArray
    boundary_edge_normal: FArray
    boundary_edge_tag: CArray

    @classmethod
    def create_1d(
        cls: Type[MeshType], domain: tuple[float, float], n_elements: int
    ) -> MeshType:
        xL = domain[0]
        xR = domain[1]

        dimension = 1
        n_nodes_per_element = 2
        dx = (xR - xL) / n_elements
        vertex_coordinates = np.zeros((n_nodes_per_element + 1, 3))
        vertex_coordinates[:, 0] = np.linspace(
            xL, xR, n_nodes_per_element + 1, dtype=float
        )
        element_vertices = np.zeros((n_elements, n_nodes_per_element), dtype=int)
        element_vertices[:, 0] = np.linspace(0, n_elements - 1, n_elements, dtype=int)
        element_vertices[:, 1] = np.linspace(1, n_elements, n_elements, dtype=int)
        element_area = dx * np.ones(n_elements, dtype=float)
        element_incircle = dx * np.ones(n_elements, dtype=float)
        element_vertices[:, 0] = np.linspace(0, n_elements - 1, n_elements, dtype=int)
        element_centers = np.zeros((n_elements, dimension), dtype=float)

        element_edge_normal = np.zeros(
            (n_elements, n_nodes_per_element, dimension), dtype=float
        )
        element_edge_length = np.ones(
            (n_elements, n_nodes_per_element, dimension), dtype=float
        )

        element_centers[:, 0] = np.arange(xL + dx / 2, xR, dx)
        element_n_neighbors = np.zeros((n_elements), dtype=int)
        element_neighbors = np.ones((n_elements, n_nodes_per_element), dtype=int)
        element_neighbors[0, 0] = 1
        element_n_neighbors[0] = 1
        element_edge_normal[0, 0] = np.array([1.0])
        for i_elem in range(1, n_elements - 1):
            element_neighbors[i_elem, :] = np.array([i_elem - 1, i_elem + 1], dtype=int)
            element_edge_normal[i_elem, :] = [
                np.array([-1.0], dtype=float),
                np.array([1.0], dtype=float),
            ]
        element_n_neighbors[n_elements - 1] = 1
        element_neighbors[n_elements - 1, 0] = n_elements - 2
        element_edge_normal[n_elements - 1, 0] = np.array([-1.0])

        n_of_boundary_edges = 2
        boundary_edge_elements = np.zeros(n_of_boundary_edges, dtype=int)
        boundary_edge_length = np.ones(n_of_boundary_edges, dtype=float)
        boundary_edge_normal = np.zeros((n_of_boundary_edges, 3), dtype=float)
        # Implicit ordering: 0: left, 1: right
        boundary_edge_tag = np.zeros(n_of_boundary_edges, dtype=str)
        boundary_edge_tag[0] = "left"
        boundary_edge_tag[1] = "right"

        boundary_edge_vertices = np.array([0, n_elements], dtype=int)
        boundary_edge_elements = np.array([0, n_elements - 1], dtype=int)
        boundary_edge_normal[0] = np.array([-1.0, 0.0, 0.0])
        boundary_edge_normal[1] = np.array([1.0, 0.0, 0.0])

        type = "line"
        n_vertices = n_elements + 1
        n_boundary_edges = n_of_boundary_edges
        num_nodes_per_element = n_nodes_per_element
        vertex_coordinates = vertex_coordinates
        element_vertices = element_vertices
        element_edge_length = element_edge_length
        element_centers = element_centers
        element_volume = element_area
        element_incircle = element_incircle
        element_edge_normal = element_edge_normal
        element_neighbors = element_neighbors
        boundary_edge_vertices = boundary_edge_vertices
        boundary_edge_element = boundary_edge_elements
        boundary_edge_length = boundary_edge_length
        boundary_edge_normal = boundary_edge_normal
        boundary_edge_tag = boundary_edge_tag
        return cls(
            dimension,
            type,
            n_elements,
            n_vertices,
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
            boundary_edge_vertices,
            boundary_edge_elements,
            boundary_edge_length,
            boundary_edge_normal,
            boundary_edge_tag,
        )

    @classmethod
    def load_mesh(cls, filepath, mesh_type, dimension, boundary_tags):
        assert mesh_type == "quad" or mesh_type == "triangle"

        mesh_io = meshio.read(filepath)
        points = mesh_io.points
        cells = mesh_io.cells_dict[mesh_type]
        # make cells '1' based for compas
        mesh = MeshCompas.from_vertices_and_faces(points, cells)

        number_of_vertices = mesh.number_of_vertices()
        mesh.update_default_edge_attributes({"tag": "inner"})
        for boundary_tag in boundary_tags:
            # cell sets is a list containing !all! names quantities, e.g. physical curves/surfaces and the list of all gmsh:bounding_entities
            # for each entry, it contains a dict of all 'tags'. Therefore it can also be empty
            for i_cs, cs in enumerate(mesh_io.cell_sets[boundary_tag]):
                # get all lines contained by this physical entry
                for i_local_node in cs:
                    # get edge node ids and cast into tuple
                    edge = tuple(mesh_io.cells[i_cs].data[i_local_node])
                    # set tag
                    mesh.edge_attributes(edge)["tag"] = boundary_tag

        return cls.from_compas_mesh(mesh, mesh_type, dimension)

    @classmethod
    def from_compas_mesh(cls, mesh, mesh_type, mesh_dimension):
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
        element_neighbors = np.empty((n_elements, n_nodes_per_element), dtype=int)
        element_edge_normal = np.empty(
            (n_elements, n_nodes_per_element, dimension), dtype=float
        )
        element_edge_length = np.empty((n_elements, n_nodes_per_element), dtype=float)

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

        edges_on_boundaries = []
        for edge in mesh.edges():
            if mesh.is_edge_on_boundary(*edge):
                edges_on_boundaries.append(edge)
        edges_on_boundaries = np.array(edges_on_boundaries)
        boundary_edge_vertices = edges_on_boundaries
        n_boundary_edges = boundary_edge_vertices.shape[0]
        boundary_edge_elements = np.zeros(n_boundary_edges, dtype=int)
        boundary_edge_tag = np.empty(n_boundary_edges, dtype=object)
        boundary_edge_length = np.zeros(n_boundary_edges, dtype=float)
        boundary_edge_normal = np.zeros((n_boundary_edges, dimension), dtype=float)

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
        return cls(
            dimension,
            type,
            n_elements,
            n_vertices,
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
            boundary_edge_vertices,
            boundary_edge_elements,
            boundary_edge_length,
            boundary_edge_normal,
            boundary_edge_tag,
        )

    def write_to_file_3d(
        self,
        filename,
        vertex_coordinates,
        element_vertices,
        fields=None,
        field_names=None,
        point_fields=None,
        point_field_names=None,
    ):
        if self.type == "triangle":
            new_type = "wedge"
        elif self.type == "quad":
            new_type = "hexahedron"
        else:
            assert False
        d_fields = {}
        if fields is not None:
            if field_names is None:
                field_names = [str(i) for i in range(fields.shape[0])]
            for i_fields, _ in enumerate(fields):
                d_fields[field_names[i_fields]] = [fields[i_fields]]
        point_d_fields = {}
        if point_fields is not None:
            if point_field_names is None:
                point_field_names = [str(i) for i in range(point_fields.shape[0])]
            for i_fields, _ in enumerate(point_fields):
                point_d_fields[point_field_names[i_fields]] = point_fields[i_fields]
        meshout = meshio.Mesh(
            vertex_coordinates,
            [(new_type, element_vertices)],
            cell_data=d_fields,
            point_data=point_d_fields,
        )
        path, _ = os.path.split(filename)
        filename, file_ext = os.path.splitext(filename)
        if not os.path.exists(path) and path != "":
            os.mkdir(path)
        meshout.write(filename + ".vtk")

    def write_to_file(self, filename, fields=None, field_names=None, point_data={}):
        d_fields = {}
        if fields is not None:
            if field_names is None:
                field_names = [str(i) for i in range(fields.shape[0])]
            for i_fields, field in enumerate(fields):
                d_fields[field_names[i_fields]] = [fields[i_fields]]
        meshout = meshio.Mesh(
            self.vertex_coordinates,
            [(self.type, self.element_vertices)],
            cell_data=d_fields,
            point_data=point_data,
        )
        path, _ = os.path.split(filename)
        filename, file_ext = os.path.splitext(filename)
        if not os.path.exists(path) and path != "":
            os.mkdir(path)
        meshout.write(filename + ".vtk")

    def load_file(filename, n_fields, map_fields):
        mesh = meshio.read(filename)
        number_of_elements = mesh.cell_data["0"][0].shape[0]
        output = np.zeros((n_fields, number_of_elements))
        for k, v in mesh.cell_data.items():
            output[map_fields[int(k)]] = v[0]
        return output

    def set_default_parameters(self):
        return

    def set_runtime_variables(self):
        return

    def write_to_file(self, filename, fields=None, field_names=None, point_data={}):
        assert False

    def load_file(filename, n_fields, map_fields):
        assert False


class Mesh1D(Mesh):
    yaml_tag = "!Mesh1D"
    dim = 1

    def set_default_parameters(self):
        self.number_of_elements = 20
        self.domain = [-1, 1]

    def set_runtime_variables(self):
        self.mesh = self.create_mesh()

    def create_mesh(self):
        xL = self.domain[0]
        xR = self.domain[1]
        num_nodes_per_element = 2
        dx = (xR - xL) / self.number_of_elements
        vertex_coordinates = np.zeros((num_nodes_per_element + 1, 3))
        vertex_coordinates[:, 0] = np.linspace(
            xL, xR, num_nodes_per_element + 1, dtype=float
        )
        element_vertices = np.zeros((self.number_of_elements, num_nodes_per_element))
        element_vertices[:, 0] = np.linspace(
            0, self.number_of_elements - 1, self.number_of_elements, dtype=int
        )
        element_vertices[:, 1] = np.linspace(
            1, self.number_of_elements, self.number_of_elements, dtype=int
        )
        # element_edge_length = np.ones(
        #     (self.number_of_elements, num_nodes_per_element), dtype=float
        # )
        element_area = dx * np.ones(self.number_of_elements, dtype=float)
        element_incircle = dx * np.ones(self.number_of_elements, dtype=float)
        element_vertices[:, 0] = np.linspace(
            0, self.number_of_elements - 1, self.number_of_elements, dtype=int
        )
        element_centers = np.zeros((self.number_of_elements, 3), dtype=float)
        # element_edge_normal = np.zeros((self.number_of_elements, 2, 3), dtype=float)
        # element_edge_normal[1:-1, 0, :] = -1.0
        # element_edge_normal[1:-1, 1, :] = 1.0
        # # boundaries: I willy traverse the first entry there, correpsonding to the neighboring element
        # element_edge_normal[0, 0, :] = 1.0
        # element_edge_normal[-1, 0, :] = -1.0

        element_edge_normal = {}
        element_edge_length = {}

        element_centers[:, 0] = np.arange(xL + dx / 2, xR, dx)
        element_neighbors = {}
        element_neighbors[0] = [1]
        element_edge_normal[0] = [np.array([1.0, 0.0, 0.0])]
        element_edge_length[0] = [1.0]
        for i_elem in range(1, self.number_of_elements - 1):
            element_neighbors[i_elem] = [i_elem - 1, i_elem + 1]
            element_edge_normal[i_elem] = [
                np.array([-1.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
            ]
            element_edge_length[i_elem] = element_edge_length[i_elem] = [1.0, 1.0]
        element_neighbors[self.number_of_elements - 1] = [self.number_of_elements - 2]
        element_edge_normal[self.number_of_elements - 1] = [np.array([-1.0, 0.0, 0.0])]
        element_edge_length[self.number_of_elements - 1] = [1.0]

        number_of_boundary_edges = 2
        boundary_edge_face = np.zeros(number_of_boundary_edges, dtype=int)
        boundary_edge_length = np.ones(number_of_boundary_edges, dtype=float)
        boundary_edge_normal = np.zeros((number_of_boundary_edges, 3), dtype=float)
        # Implicit ordering: 0: left, 1: right
        boundary_edge_tag = np.zeros(number_of_boundary_edges, dtype=object)
        boundary_edge_tag[0] = "left"
        boundary_edge_tag[1] = "right"

        boundary_edge_vertices = np.array([0, self.number_of_elements], dtype=int)
        boundary_edge_face = np.array([0, self.number_of_elements - 1], dtype=int)
        boundary_edge_normal[0] = np.array([-1.0, 0.0, 0.0])
        boundary_edge_normal[1] = np.array([1.0, 0.0, 0.0])

        self.type = "line"
        self.n_elements = self.number_of_elements
        self.n_vertices = self.number_of_elements + 1
        self.n_boundary_edges = number_of_boundary_edges
        self.num_nodes_per_element = num_nodes_per_element
        self.vertex_coordinates = vertex_coordinates
        self.element_vertices = element_vertices
        self.element_edge_length = element_edge_length
        self.element_centers = element_centers
        self.element_volume = element_area
        self.element_incircle = element_incircle
        self.element_edge_normal = element_edge_normal
        self.element_neighbors = element_neighbors
        self.boundary_edge_vertices = boundary_edge_vertices
        self.boundary_edge_element = boundary_edge_face
        self.boundary_edge_length = boundary_edge_length
        self.boundary_edge_normal = boundary_edge_normal
        self.boundary_edge_tag = boundary_edge_tag

        def write_to_file(
            self, filename, fields=None, field_names=None, point_data={}, data_3d={}
        ):
            X = self.element_centers[:, 0]
            if fields is None:
                fields = np.array([X])
                field_names = ["X"]
            else:
                if field_names is None:
                    field_names = []
                    for i in range(fields.shape[0]):
                        field_names.append(str(i))
                else:
                    field_names.insert(0, "X")
            fields_out = np.zeros((fields.shape[0] + 1, fields.shape[1]))
            fields_out[0] = X
            fields_out[1:] = fields
            path, _ = os.path.split(filename)
            if not os.path.exists(path) and path != "":
                os.mkdir(path)
            np.savetxt(filename, np.array(fields_out), delimiter=",")


class Mesh2D(Mesh):
    yaml_tag = "!Mesh2D"
    dim = 2

    def set_default_parameters(self):
        self.type = "quad"
        self.filename = "meshes/quad_2d/mesh_coarse.msh"
        self.boundary_tags = ["left", "right", "bottom", "top"]
        self.kwargs = {}

    def set_runtime_variables(self):
        self.mesh = self.load_mesh()


def incircle(mesh, face, type):
    if type == "triangle":
        return 2 * incircle_triangle(mesh, face)
    elif type == "quad":
        return 2 * incircle_quad(mesh, face)
    assert False


def incircle_triangle(mesh, face):
    area = mesh.face_area(face)
    edges = mesh.face_halfedges(face)
    perimeter = 0.0
    for edge in edges:
        perimeter += mesh.edge_length(*edge)
    # return 2 * area / perimeter
    s = perimeter / 2
    result = 1.0
    for edge in edges:
        result *= s - mesh.edge_length(*edge)
    return np.sqrt(result)


def incircle_quad(mesh, face):
    area = mesh.face_area(face)
    edges = mesh.face_halfedges(face)
    perimeter = 0.0
    for edge in edges:
        perimeter += mesh.edge_length(*edge)
    s = perimeter / 2
    return area / s


def edge_lengths(mesh, face):
    edges = mesh.face_halfedges(face)
    edge_length_ = np.zeros(len(edges))
    for i, edge in enumerate(edges):
        edge_length_[i] = mesh.edge_length(*edge)
    return edge_length_


def get_n_nodes_per_element(mesh_type: str) -> int:
    if mesh_type == "quad":
        return 4
    elif mesh_type == "tri":
        return 3
    else:
        assert False


def test_create_1d_mesh():
    mesh = Mesh.create_1d((-1, 1), 10)
    assert True


def test_load_2d_mesh():
    main_dir = os.getenv("SMPYTHON")
    mesh = Mesh.load_mesh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
        "quad",
        2,
        ["left", "right", "top", "bottom"],
    )
    print(mesh)
    assert True


test_create_1d_mesh()
test_load_2d_mesh()
