import os
import numpy as np
import meshio
from compas.datastructures import Mesh as MeshCompas
from compas_gmsh.models import MeshModel
import h5py

from attr import define

from typing import Union
from typing import TypeVar, Type, Any

# Create a generic variable that can be 'Mesh', or any subclass.
MeshType = TypeVar("MeshType", bound="Mesh")

from library.custom_types import IArray, FArray, CArray


@define(slots=True, frozen=True)
class Mesh:
    dimension: int
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
        boundary_edge_tag = np.zeros(n_of_boundary_edges, dtype="|S5")
        boundary_edge_tag[0] = "left"
        boundary_edge_tag[1] = "right"

        boundary_edge_vertices = np.array([0, n_elements], dtype=int)
        boundary_edge_elements = np.array([0, n_elements - 1], dtype=int)
        boundary_edge_normal[0] = np.array([-1.0, 0.0, 0.0])
        boundary_edge_normal[1] = np.array([1.0, 0.0, 0.0])

        type = "line"
        n_vertices = n_elements + 1
        n_boundary_edges = n_of_boundary_edges
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
    def load_mesh(
        cls: Type[MeshType],
        filepath: str,
        mesh_type: str,
        dimension: int,
        boundary_tags: list[str],
    ) -> MeshType:
        assert (mesh_type) == "quad" or (mesh_type) == "tri"
        mesh_type = mesh_type

        mesh_io = meshio.read(filepath)
        points = mesh_io.points
        cells = mesh_io.cells_dict[convert_mesh_type_to_meshio_mesh_type(mesh_type)]
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
    def from_compas_mesh(
        cls: Type[MeshType], mesh: MeshCompas, mesh_type: str, mesh_dimension: int
    ) -> MeshType:
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

        edges_on_boundaries = []
        for edge in mesh.edges():
            if mesh.is_edge_on_boundary(*edge):
                edges_on_boundaries.append(edge)
        edges_on_boundaries = np.array(edges_on_boundaries)
        boundary_edge_vertices = edges_on_boundaries
        n_boundary_edges = boundary_edge_vertices.shape[0]
        boundary_edge_elements = np.zeros(n_boundary_edges, dtype=int)
        boundary_edge_tag = np.zeros(n_boundary_edges, dtype="S5")
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

    @classmethod
    def from_hdf5(cls: Type[MeshType], filepath: str) -> MeshType:
        with h5py.File(filepath, "r") as file:
            file_mesh = file["mesh"]
            mesh = cls(
                file_mesh["dimension"][()],
                (file_mesh["type"][()]).decode("utf-8"),
                file_mesh["n_elements"][()],
                file_mesh["n_vertices"][()],
                file_mesh["n_boundary_edges"][()],
                file_mesh["n_nodes_per_element"][()],
                file_mesh["vertex_coordinates"][()],
                file_mesh["element_vertices"][()],
                file_mesh["element_edge_length"][()],
                file_mesh["element_centers"][()],
                file_mesh["element_volume"][()],
                file_mesh["element_incircle"][()],
                file_mesh["element_edge_normal"][()],
                file_mesh["element_neighbors"][()],
                file_mesh["boundary_edge_vertices"][()],
                file_mesh["boundary_edge_elements"][()],
                file_mesh["boundary_edge_length"][()],
                file_mesh["boundary_edge_normal"][()],
                file_mesh["boundary_edge_tag"][()],
            )
        return mesh

    @classmethod
    def extrude_2d_mesh(
        cls: Type[MeshType], mesh: MeshType, N_z=int, height: Union[FArray, None] = None
    ) -> MeshType:
        if height is None:
            height = np.ones(mesh.n_vertices)

    def write_to_file_vtk(
        self: Type[MeshType],
        filepath: str,
        fields: Union[FArray, None] = None,
        field_names: Union[CArray, None] = None,
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

    def write_to_hdf5(self: Type[MeshType], filepath: str):
        with h5py.File(filepath, "w") as f:
            attrs = f.create_group("mesh")
            attrs.create_dataset("dimension", data=self.dimension)
            attrs.create_dataset("type", data=self.type)
            attrs.create_dataset("n_elements", data=self.n_elements)
            attrs.create_dataset("n_vertices", data=self.n_vertices)
            attrs.create_dataset("n_boundary_edges", data=self.n_boundary_edges)
            attrs.create_dataset("n_nodes_per_element", data=self.n_nodes_per_element)
            attrs.create_dataset("vertex_coordinates", data=self.vertex_coordinates)
            attrs.create_dataset("element_vertices", data=self.element_vertices)
            attrs.create_dataset("element_edge_length", data=self.element_edge_length)
            attrs.create_dataset("element_centers", data=self.element_centers)
            attrs.create_dataset("element_volume", data=self.element_volume)
            attrs.create_dataset("element_incircle", data=self.element_incircle)
            attrs.create_dataset("element_edge_normal", data=self.element_edge_normal)
            attrs.create_dataset("element_neighbors", data=self.element_neighbors)
            attrs.create_dataset(
                "boundary_edge_vertices", data=self.boundary_edge_vertices
            )
            attrs.create_dataset(
                "boundary_edge_elements", data=self.boundary_edge_elements
            )
            attrs.create_dataset("boundary_edge_length", data=self.boundary_edge_length)
            attrs.create_dataset("boundary_edge_normal", data=self.boundary_edge_normal)
            attrs.create_dataset("boundary_edge_tag", data=self.boundary_edge_tag)


def read_vtk_cell_fields(
    filename: str, n_fields: int, map_field_indices: list[int]
) -> FArray:
    mesh = meshio.read(filename)
    number_of_elements = mesh.cell_data["0"][0].shape[0]
    output = np.zeros((n_fields, number_of_elements))
    for k, v in mesh.cell_data.items():
        output[map_field_indices[int(k)]] = v[0]
    return output


def incircle(mesh: Type[MeshType], face: int, mesh_type: str) -> FArray:
    if mesh_type == "tri":
        return 2 * incircle_triangle(mesh, face)
    elif mesh_type == "quad":
        return 2 * incircle_quad(mesh, face)
    assert False


def incircle_triangle(mesh: Type[MeshType], face: int) -> FArray:
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


def incircle_quad(mesh: Type[MeshType], face: int) -> FArray:
    area = mesh.face_area(face)
    edges = mesh.face_halfedges(face)
    perimeter = 0.0
    for edge in edges:
        perimeter += mesh.edge_length(*edge)
    s = perimeter / 2
    return area / s


def edge_lengths(mesh: Type[MeshType], face: int) -> FArray:
    edges = mesh.face_halfedges(face)
    edge_length_ = np.zeros(len(edges))
    for i, edge in enumerate(edges):
        edge_length_[i] = mesh.edge_length(*edge)
    return edge_length_


def get_extrudes_mesh_type(mesh_type: str) -> int:
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
) -> (FArray, IArray):
    n_vertices = vertex_coordinates.shape[0]
    n_elements = element_vertices.shape[0]
    num_nodes_per_element_2d = get_n_nodes_per_element(mesh_type)
    mesh_type = get_extrudes_mesh_type(mesh_type)
    num_nodes_per_element = get_n_nodes_per_element(mesh_type)
    Z = np.linspace(0, 1, n_layers)
    points_3d = np.zeros(
        (
            vertex_coordinates.shape[0] * n_layers,
            3,
        )
    )
    element_vertices_3d = np.zeros((n_elements * (n_layers - 1), num_nodes_per_element))
    for i in range(n_vertices):
        points_3d[i * n_layers : (i + 1) * n_layers, :2] = vertex_coordinates[i]
        points_3d[i * n_layers : (i + 1) * n_layers, 2] = height[i] * Z

    # compute connectivity for mesh (element_vertices)
    element_vertices_3d = np.zeros((n_elements * (n_layers - 1), num_nodes_per_element))
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
