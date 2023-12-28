## DISCLAIMER: This file is a modified version of the mesh2xdmf converter used in dolfin (https://github.com/floiseau/msh2xdmf)
## Modifications: 
## - should carry the boundary condition name as a tag, to be identifyable by name
## - allow for more element types

import argparse
import meshio
import os
import numpy as np
from configparser import ConfigParser
from compas.datastructures import Mesh as MeshCompas
import h5py 

from library.mesh_util import get_global_cell_index_from_vertices


def gmsh_to_domain_boundary_mesh(mesh_name, mesh_type='triangle', directory="."):
    """
    Function converting a MSH mesh into XDMF files.
    The XDMF files are:
        - "domain.xdmf": the domain;
        - "boundaries.xdmf": the boundaries physical groups from GMSH;
    """
    # Set cell type
    if mesh_type == 'triangle':
        cell_type = "triangle"
        dim = 2
    elif mesh_type == 'quad':
        cell_type = "quad"
        dim = 2
    elif mesh_type == 'tetra':
        cell_type = "tetra"
        dim = 3
    else:
        assert False

    # Get the mesh name has prefix
    prefix = mesh_name.split('.')[0]
    # Read the input mesh
    msh = meshio.read("{}/{}".format(directory, mesh_name))

    gmsh_association_table = _get_association_table(msh, prefix, directory)
    # Generate the domain as cells_points
    domain = export_domain(msh, mesh_type, directory, prefix)
    # msh = fvm_mesh.Mesh.load_cell_point_mesh(cells, points, cell_type, dim, [] )
    # compas_msh = MeshCompas.from_vertices_and_faces(points, cells)
    # ( 
    #     dimension,
    #     type,
    #     n_elements,
    #     n_vertices,
    #     n_edges,
    #     n_nodes_per_element,
    #     vertex_coordinates,
    #     element_vertices,
    #     element_edge_length,
    #     element_centers,
    #     element_volume,
    #     element_incircle,
    #     element_edge_normal,
    #     element_neighbors,
    #     element_n_neighbors,
    # ) = fvm_mesh.Mesh.from_comas_mesh_volume(compas_msh, mesh_type, dim)

    # Generate the boundaries as cells points
    boundaries = export_boundaries(msh, mesh_type, directory, prefix, gmsh_association_table)
    # compas_msh = MeshCompas.from_vertices_and_faces(points, cells)
    # runtime_boundaries_mesh = fvm_mesh.Mesh.from_comas_mesh_boundaries(compas_msh, mesh_type, dim)
    return domain, boundaries

def export_domain(msh, mesh_type, directory, prefix):
    """
    Export the domain as well as the subdomains values. Export types are
    - write to XDMF file
    - return (simple) cells and point data. Simple means only one (the first) element type is returned.
    """
    # Set cell type
    if mesh_type == 'triangle':
        cell_type = "triangle"
        dim = 2
    elif mesh_type == 'quad':
        cell_type = "quad"
        dim = 2
    elif mesh_type == 'tetra':
        cell_type = "tetra"
        dim = 3
    else:
        assert False
    # Generate the cell block for the domain cells
    data_array = []
    for obj in msh.cells:
        if obj.type == cell_type: 
            data_array.append(obj.data)
    # data_array = [arr for (t, arr) in msh.cells if t == cell_type]
    if len(data_array) == 0:
        print("WARNING: No domain physical group found.")
        return
    else:
        data = np.concatenate(data_array)
    cells = [
        meshio.CellBlock(
            cell_type=cell_type,
            data=data,
        )
    ]
    # Generate a meshio Mesh for the domain
    domain = meshio.Mesh(
        points=msh.points[:, :],
        cells=cells,
        # cell_data=cell_data,
    )

    return domain


def export_boundaries(msh, mesh_type, directory, prefix, gmsh_association_table):
    """
    Export the boundaries XDMF file.
    """
    # Set the cell type
    if mesh_type == 'triangle':
        cell_type = "line"
        dim = 2
    elif mesh_type == 'quad':
        cell_type = "line"
        dim = 2
    elif mesh_type == 'tetra':
        cell_type = "triangle"
        dim = 3
    else:
        assert False
    # Generate the cell block for the boundaries cells
    # data_array = [arr for (t, arr) in msh.cells if t == cell_type]
    offset = 0
    data_array = []
    sort_order_list = []
    for obj in msh.cells:
        if obj.type == cell_type: 
            data_array.append(obj.data)
            sort_order_list.append((offset)+_sort_order_for_periodic_boundary_conditions(dim, msh.points, obj.data))
            offset += obj.data.shape[0]

    if len(data_array) == 0:
        print("WARNING: No boundary physical group found.")
        return
    else:
        data = np.concatenate(data_array)
        sort_order = np.concatenate(sort_order_list)
    boundaries_cells = [
        meshio.CellBlock(
            cell_type=cell_type,
            data=data[sort_order],
        )
    ]
    # Generate the boundaries cells data
    cell_data = {
        "boundary_tag": [
            np.concatenate(
                [
                    [ gmsh_association_table[tag_id] for tag_id in msh.cell_data["gmsh:physical"][i] ]
                    for i, cellBlock in enumerate(msh.cells)
                    if cellBlock.type == cell_type
                ]
            )[sort_order]
        ],
        "corresponding_cell": [
            np.concatenate(
                [
                    _get_boundary_edges_cells(msh, cellBlock.data)
                    for i, cellBlock in enumerate(msh.cells)
                    if cellBlock.type == cell_type
                ]
            )[sort_order]
        ]
    }
    # Generate the meshio Mesh for the boundaries physical groups
    boundaries = meshio.Mesh(
        points=msh.points[:, :],
        cells=boundaries_cells,
        cell_data=cell_data,
    )

    return boundaries

def _sort_order_for_periodic_boundary_conditions(dimension, points, data):
    edge_coordinates = points[data]
    center_coordinates = np.array([np.mean(edge_coordinates[i], axis=0) for i in range(edge_coordinates.shape[0])])
    if dimension == 1:
        indices_sorted = np.lexsort((center_coordinates[:, 0],))
    elif dimension == 2:
        indices_sorted = np.lexsort(
            (
            center_coordinates[:, 0],
            center_coordinates[:, 1],
            )
            )
    elif dimension == 3:
        indices_sorted = np.lexsort(
            (
            center_coordinates[:, 0],
            center_coordinates[:, 1],
            center_coordinates[:, 2],
            )
            )
    else:
        assert False
    return indices_sorted


def _get_boundary_edges_cells(msh, list_of_edges):
    boundary_edges_cells = []
    for edge in list_of_edges:
        # find index of cell
        # TODO cells[-1] is not save
        hit = get_global_cell_index_from_vertices(msh.cells[-1].data, edge)
        boundary_edges_cells.append(hit)
    return boundary_edges_cells



def _get_association_table(msh, prefix='mesh', directory='.', verbose=True):
    """
    Display the association between the physical group label and the mesh
    value.
    """
    # Create association table
    association_table = {}

    for label, arrays in msh.cell_sets.items():
        # Get the index of the array in arrays
        for i, array in enumerate(arrays):
            if array.size != 0:
                index = i
        # Added check to make sure that the association table
        # doesn't try to import irrelevant information.
        if label != "gmsh:bounding_entities":
            value = msh.cell_data["gmsh:physical"][index][0]
            # Store the association table in a dictionnary
            association_table[value] = label
    return association_table


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "msh_file",
    #     help="input .msh file",
    #     type=str,
    # )
    # parser.add_argument(
    #     "-d",
    #     "--dimension",
    #     help="dimension of the domain",
    #     type=int,
    #     default=2,
    # )
    # args = parser.parse_args()
    # # Get current directory
    # current_directory = os.getcwd()
    # # Conert the mesh
    # msh2xdmf(args.msh_file, args.dimension, directory=current_directory)
    # msh2xdmf('./meshes/tetra_3d/mesh.msh', 'tetra', './')
    # msh2xdmf('./meshes/tetra_3d/test.msh', 'tetra', './')
    # msh2runtime_fvm_mesh_simple('./meshes/tetra_3d/test.msh', 'tetra', './')
    msh2runtime_fvm_mesh_simple('./meshes/quad_2d/mesh_coarse.msh', 'quad', './')

