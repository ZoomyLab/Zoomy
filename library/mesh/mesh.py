import sys
import os
import h5py
import petsc4py
from petsc4py import PETSc
import numpy as np
import meshio

from attr import define
from typing import Union

from library.misc.custom_types import IArray, FArray, CArray


# petsc4py.init(sys.argv)

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
    if inradius  <= 0:
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
        inradius.append(_compute_inradius_generic(cell_center, face_centers, face_normals))
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
    assert(False)

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
        indices +=  [index for v in values]
        index +=1
    return np.array(indices, dtype=int)


# def load_gmsh_to_dm(filepath):
#     dm = PETSc.DMPlex().createFromFile(filepath, comm=PETSc.COMM_WORLD)
#     boundary_dict = get_physical_boundary_labels(filepath)
#     dm, ghost_cells_dict = construct_ghost_cells(dm, boundary_dict)
#     return dm, boundary_dict, ghost_cells_dict

def load_gmsh_to_fvm(filepath):
    dm = PETSc.DMPlex().createFromFile(filepath, comm=PETSc.COMM_WORLD) 
    boundary_dict = get_physical_boundary_labels(filepath)
    dm, boundary_cells_dict, boundary_face_vertices_dict = compute_boundary_data(dm, boundary_dict)
    boundary_face_corresponding_element = boundary_dict_to_list(boundary_cells_dict)
    boundary_face_vertices = boundary_dict_to_list(boundary_face_vertices_dict)
    return dm, boundary_dict, ghost_cells_dict


def _get_mesh_type(dm):
    return 'quad'

def convert_to_fvm_mesh(dm, boundary_dict, ghost_cells_dict):
    dimension = dm.getDimension()
    type = _get_mesh_type(dm)
    assert dimension == 2
    (cStart, cEnd) = dm.getHeightStratum(0)
    (vStart, vEnd) = dm.getDepthStratum(0)
    (eStart, eEnd) = dm.getDepthStratum(1)

    n_elements = cEnd-cStart
    n_vertices = vEnd-vStart
    n_boundary_elements = []
    boundary_face_corresponding_element = []
    boundary_face_vertices = []

    for be_list_key, be_list_values in ghost_cells_dict.items():
        n_boundary_elements += len(be_list_values)
        boundary_face_corresponding_element += be_list_values
        for i in len(be_list_values):
            boundary_face_tag.append(be_list_key)
    boundary_tag_names = boundary_dict.keys()


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
    boundary_face_cells: IArray
    boundary_face_ghosts: IArray
    boundary_face_function_numbers: IArray
    boundary_face_physical_tags: IArray
    boundary_face_face_indices: IArray
    face_cells: IArray
    face_normals: FArray
    face_volumes: FArray
    face_centers: FArray
    boundary_conditions_sorted_physical_tags: IArray
    boundary_conditions_sorted_names: CArray
    lsq_matrix: FArray
    lsq_diff_matrix: FArray

    @classmethod
    def from_gmsh(cls, filepath):
        dm = PETSc.DMPlex().createFromFile(filepath, comm=PETSc.COMM_WORLD) 
        boundary_dict = get_physical_boundary_labels(filepath)
        (cStart, cEnd) = dm.getHeightStratum(0)
        (vStart, vEnd) = dm.getDepthStratum(0)
        (eStart, eEnd) = dm.getDepthStratum(1)
        gdm = dm.clone()
        gdm.constructGhostCells()
        (cgStart, cgEnd) = gdm.getHeightStratum(0)
        (vgStart, vgEnd) = gdm.getDepthStratum(0)
        (egStart, egEnd) = gdm.getDepthStratum(1)

        n_faces_per_cell = len(gdm.getCone(cgStart))
        dim = dm.getDimension()
        n_cells = cgEnd-cgStart
        n_inner_cells = cEnd-cStart
        n_vertices = vEnd-vStart
        cell_vertices = np.zeros((n_inner_cells, n_faces_per_cell), dtype=int)
        cell_faces = np.zeros((n_inner_cells, n_faces_per_cell), dtype=int)
        cell_centers = np.zeros((n_cells, dim), dtype=float)
        # I create cell_volumes of size n_cells because then I can avoid an if clause in the numerical flux computation. The values will be delted after using apply_boundary_conditions anyways
        cell_volumes = np.ones((n_cells), dtype=float)
        cell_inradius = compute_cell_inradius(dm)
        for i_c, c in enumerate(range(cStart, cEnd)):
            cell_volume, cell_center, cell_normal = dm.computeCellGeometryFVM(c)
            transitive_closure_points, transitive_closure_orientation = dm.getTransitiveClosure(c, useCone=True)
            _cell_vertices = transitive_closure_points[np.logical_and(transitive_closure_points >= vStart, transitive_closure_points <vEnd)]
            # _cell_vertices_orientation = transitive_closure_orientation[np.logical_and(transitive_closure_points >= vStart, transitive_closure_points <vEnd)]
            assert _cell_vertices.shape[0] == cell_vertices.shape[1]
            # assert (_cell_vertices_orientation == 0).all()
            cell_vertices[i_c,: ] = _cell_vertices - vStart
            cell_centers[i_c, :] = cell_center
            cell_volumes[i_c] = cell_volume

        vertex_coordinates = np.array(gdm.getCoordinates()).reshape((-1, dim))
        boundary_face_cells = {k: [] for k in boundary_dict.values()}
        boundary_face_ghosts = {k: [] for k in boundary_dict.values()}
        boundary_face_face_indices = {k: [] for k in boundary_dict.values()}
        boundary_face_physical_tags = {k: [] for k in boundary_dict.values()}
        face_cells = []
        face_normals = []
        face_volumes = []
        face_centers = []
        n_faces = egEnd-egStart
        allowed_keys = []
        vertex_coordinates = np.array(dm.getCoordinates()).reshape((-1, dim))
        for e in range(egStart, egEnd):
            label = gdm.getLabelValue("Face Sets", e)
            # 2 cells support an face. Ghost cell is the one with the higher number
            if label > -1:
                allowed_keys.append(label)
                boundary_cell = gdm.getSupport(e).min() - cStart
                boundary_ghost = gdm.getSupport(e).max() - cStart
                boundary_face_cells[label].append(boundary_cell)
                boundary_face_ghosts[label].append(boundary_ghost)
                boundary_face_face_indices[label].append(e-egStart)
                boundary_face_physical_tags[label].append(label)
                # boundary_face_vertices[label].append(gdm.getCone(e)-vStart)
                # for periodic boudnary conditions, I need the ghost cell to have a cell_center. I copy the one from the related inner cell.
                _face_cell = gdm.getSupport(e).min()
                _face_ghost = gdm.getSupport(e).max()
                cell_centers[_face_ghost] = cell_centers[_face_cell]

            face_center = gdm.computeCellGeometryFVM(e)[1]
            face_centers.append(face_center)
            _face_cells = gdm.getSupport(e)
            face_volume, face_center, face_normal = gdm.computeCellGeometryFVM(e)
            face_volumes.append(face_volume)
            face_normals.append(face_normal)
            face_cells.append(_face_cells)

        lsq_matrix = []
        lsq_diff_matrix = -np.eye(n_cells, dtype=float)
        # I only want to iterate over the inner cells, but in the ghosted mesh
        for i_c, c in enumerate(range(cgStart, cgStart + n_inner_cells)):
            faces = gdm.getCone(c) - egStart
            cell_faces[i_c,:] = faces

            neighbors = np.array(
                [
                    gdm.getSupport(f)[gdm.getSupport(f) != c][0]
                    for f in gdm.getCone(c)
                ]
            )
            # tc = gdm.getTransitiveClosure(c, useCone=True)[0]
            # neighbors = tc[np.logical_and(np.logical_and(tc >= cgStart, tc < cgEnd), tc != c)] - cStart
            n_neighbors = neighbors.shape[0]
            dX = np.zeros((n_neighbors * dim, dim), dtype=float)
            mat = np.zeros((dim , n_faces_per_cell * dim), dtype=float)
            for d in range(dim):
                for i_neighbor, neigbor in enumerate(neighbors):
                    dX[i_neighbor + d * n_neighbors, d ] = cell_centers[i_neighbor][d] - cell_centers[i_c][d]
            mat[:, :n_neighbors*dim] = np.linalg.inv(dX.T @ dX) @ dX.T
            lsq_matrix.append(mat)
            lsq_diff_matrix[i_c, neighbors] = 1.



        lsq_matrix = np.array(lsq_matrix, dtype=float)
        face_volumes = np.array(face_volumes, dtype=float)
        face_centers = np.array(face_centers, dtype=float)
        face_normals = np.array(face_normals, dtype=float)
        face_cells = np.array(face_cells, dtype=int)
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
        boundary_face_face_indices = {k: boundary_face_face_indices[k] for k in sorted_keys}

        boundary_conditions_sorted_physical_tags = np.array(list(boundary_dict.keys()), dtype='int')
        boundary_conditions_sorted_names = np.array(list(boundary_dict.values()), dtype='str')
        boundary_face_cells = np.array(_boundary_dict_to_list(boundary_face_cells), dtype=int)
        boundary_face_ghosts = np.array(_boundary_dict_to_list(boundary_face_ghosts), dtype=int)
        boundary_face_physical_tags = np.array(_boundary_dict_to_list(boundary_face_physical_tags), dtype=int)
        boundary_face_face_indices = np.array(_boundary_dict_to_list(boundary_face_face_indices), dtype=int)
        n_boundary_faces =  boundary_face_cells.shape[0]

        mesh_type = get_mesh_type_from_dm(n_faces_per_cell, dim)

        return cls(dim, mesh_type, n_cells, n_inner_cells, n_faces, n_vertices, n_boundary_faces, n_faces_per_cell, vertex_coordinates.T, cell_vertices.T, cell_faces.T, cell_volumes, cell_centers.T, cell_inradius, boundary_face_cells.T, boundary_face_ghosts.T, boundary_face_function_numbers, boundary_face_physical_tags, boundary_face_face_indices.T, face_cells.T, face_normals.T, face_volumes, face_centers, boundary_conditions_sorted_physical_tags, boundary_conditions_sorted_names, lsq_matrix, lsq_diff_matrix)

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
            mesh.create_dataset("boundary_face_cells", data=self.boundary_face_cells)
            mesh.create_dataset("boundary_face_ghosts", data=self.boundary_face_ghosts)
            mesh.create_dataset("boundary_face_function_numbers", data=self.boundary_face_function_numbers)
            mesh.create_dataset("boundary_face_physical_tags", data=self.boundary_face_physical_tags)
            mesh.create_dataset("boundary_face_face_indices", data=self.boundary_face_face_indices)
            mesh.create_dataset("face_cells", data=self.face_cells)
            mesh.create_dataset("face_normals", data=self.face_normals)
            mesh.create_dataset("face_volumes", data=self.face_volumes)
            mesh.create_dataset("face_centers", data=self.face_centers)
            mesh.create_dataset("boundary_conditions_sorted_physical_tags", data=np.array(self.boundary_conditions_sorted_physical_tags))
            mesh.create_dataset("boundary_conditions_sorted_names", data=np.array(self.boundary_conditions_sorted_names, dtype='S'))
            mesh.create_dataset("lsq_matrix", data=np.array(self.lsq_matrix))
            mesh.create_dataset("lsq_diff_matrix", data=np.array(self.lsq_diff_matrix))

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
                file["mesh"]["boundary_face_cells"][()],
                file["mesh"]["boundary_face_ghosts"][()],
                file["mesh"]["boundary_face_function_numbers"][()],
                file["mesh"]["boundary_face_physical_tags"][()],
                file["mesh"]["boundary_face_face_indices"][()],
                file["mesh"]["face_cells"][()],
                file["mesh"]["face_normals"][()],
                file["mesh"]["face_volumes"][()],
                file["mesh"]["face_centers"][()],
                file["mesh"]["boundary_conditions_sorted_physical_tags"][()],
                np.array(file["mesh"]["boundary_conditions_sorted_names"][()], dtype='str'),
                file["mesh"]["lsq_matrix"][()],
                file["mesh"]["lsq_diff_matrix"][()],
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
        vertex_coords_3d[:self.vertex_coordinates.shape[0], :] = self.vertex_coordinates
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


if __name__ == "__main__":
    path = '/home/ingo/Git/SMM/shallow-moments-simulation/meshes/quad_2d/mesh_coarse.msh'
    path2 = '/home/ingo/Git/SMM/shallow-moments-simulation/meshes/quad_2d/mesh_fine.msh'
    path3 = '/home/ingo/Git/SMM/shallow-moments-simulation/meshes/quad_2d/mesh_finest.msh'
    path4 = '/home/ingo/Git/SMM/shallow-moments-simulation/meshes/triangle_2d/mesh_coarse.msh'
    labels = get_physical_boundary_labels(path)
    # print(labels)

    # dm, boundary_dict, ghost_cells_dict = load_gmsh(path)
    # print(ghost_cells_dict)

    mesh = Mesh.from_gmsh(path)
    assert mesh.cell_faces.max() == mesh.n_faces -1
    assert mesh.cell_faces.min() == 0
    assert mesh.face_cells.max() == mesh.n_cells -1
    assert mesh.face_cells.min() == 0
    assert mesh.cell_vertices.max() == mesh.n_vertices-1
    assert mesh.cell_vertices.min() == 0

    mesh.write_to_hdf5('./test.h5')
    mesh = Mesh.from_hdf5('./test.h5')
    # mesh.write_to_vtk('./test.vtk')
    mesh.write_to_vtk('./test.vtk', fields=np.ones((2, mesh.n_inner_cells), dtype=float), field_names=['A', 'B'])
