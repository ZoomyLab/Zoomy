import os
import h5py
try:
    from petsc4py import PETSc
    _HAVE_PETSC = True
except ImportError:
    PETSc = None
    _HAVE_PETSC = False
    
import numpy as np
import meshio
import jax.numpy as jnp
import jax
from copy import deepcopy

import attr
from attr import define
from typing import Union, Any


from zoomy_core.misc.custom_types import IArray, FArray, CArray
from zoomy_core.mesh.mesh_util import compute_subvolume, get_extruded_mesh_type
import zoomy_core.mesh.mesh_extrude as extrude
import zoomy_core.mesh.mesh_util as mesh_util
from zoomy_core.model.boundary_conditions import Periodic

from zoomy_core.mesh.mesh import (
    Mesh,
    find_derivative_indices,
    get_physical_boundary_labels,
)

from itertools import product


# petsc4py.init(sys.argv)


def compute_derivatives(u, mesh, derivatives_multi_index=None):
    A_glob = mesh.lsq_gradQ  # shape (n_cells, n_monomials, n_neighbors)
    neighbors = mesh.lsq_neighbors  # list of neighbors per cell
    mon_indices = mesh.lsq_monomial_multi_index  # shape (n_monomials, dim)
    # scale_factors = scale_lsq_derivative(mon_indices)
    scale_factors = mesh.lsq_scale_factors

    if derivatives_multi_index is None:
        derivatives_multi_index = mon_indices
    indices = find_derivative_indices(mon_indices, derivatives_multi_index)

    def reconstruct_cell(A_loc, neighbor_idx, u_i):
        u_neighbors = u[neighbor_idx]
        delta_u = u_neighbors - u_i
        return (scale_factors * (A_loc.T @ delta_u)).T  # shape (n_monomials,)

    return jax.vmap(reconstruct_cell)(
        A_glob,
        neighbors,
        u,
    )[:, indices]  # shape (n_cells, n_monomials)



@define(frozen=True, slots=True)
class MeshJAX(Mesh):
    dimension: int = attr.ib()
    type: str = attr.ib()
    n_cells: int = attr.ib()
    n_inner_cells: int = attr.ib()
    n_faces: int = attr.ib()
    n_vertices: int = attr.ib()
    n_boundary_faces: int = attr.ib()
    n_faces_per_cell: int = attr.ib()
    vertex_coordinates: jnp.ndarray = attr.ib()
    cell_vertices: jnp.ndarray = attr.ib()
    cell_faces: jnp.ndarray = attr.ib()
    cell_volumes: jnp.ndarray = attr.ib()
    cell_centers: jnp.ndarray = attr.ib()
    cell_inradius: jnp.ndarray = attr.ib()
    cell_neighbors: jnp.ndarray = attr.ib()
    boundary_face_cells: jnp.ndarray = attr.ib()
    boundary_face_ghosts: jnp.ndarray = attr.ib()
    boundary_face_function_numbers: jnp.ndarray = attr.ib()
    boundary_face_physical_tags: jnp.ndarray = attr.ib()
    boundary_face_face_indices: jnp.ndarray = attr.ib()
    face_cells: jnp.ndarray = attr.ib()
    # face_cell_face_index: jnp.ndarray = attr.ib()  # If needed
    face_normals: jnp.ndarray = attr.ib()
    face_volumes: jnp.ndarray = attr.ib()
    face_centers: jnp.ndarray = attr.ib()
    face_subvolumes: jnp.ndarray = attr.ib()
    face_neighbors: jnp.ndarray = attr.ib()
    boundary_conditions_sorted_physical_tags: jnp.ndarray = attr.ib()
    boundary_conditions_sorted_names: Any = attr.ib()  # Keeping as NumPy char array
    lsq_gradQ: jnp.ndarray = attr.ib()
    lsq_neighbors: jnp.ndarray = attr.ib()
    lsq_monomial_multi_index: jnp.int64 = attr.ib()
    lsq_scale_factors: jnp.ndarray = attr.ib()
    z_ordering: jnp.ndarray = attr.ib()


def convert_mesh_to_jax(mesh: Mesh) -> MeshJAX:
    return MeshJAX(
        dimension=mesh.dimension,
        type=mesh.type,
        n_cells=mesh.n_cells,
        n_inner_cells=mesh.n_inner_cells,
        n_faces=mesh.n_faces,
        n_vertices=mesh.n_vertices,
        n_boundary_faces=mesh.n_boundary_faces,
        n_faces_per_cell=mesh.n_faces_per_cell,
        vertex_coordinates=jnp.array(mesh.vertex_coordinates),
        cell_vertices=jnp.array(mesh.cell_vertices),
        cell_faces=jnp.array(mesh.cell_faces),
        cell_volumes=jnp.array(mesh.cell_volumes),
        cell_centers=jnp.array(mesh.cell_centers),
        cell_inradius=jnp.array(mesh.cell_inradius),
        cell_neighbors=jnp.array(mesh.cell_neighbors),
        boundary_face_cells=jnp.array(mesh.boundary_face_cells),
        boundary_face_ghosts=jnp.array(mesh.boundary_face_ghosts),
        boundary_face_function_numbers=jnp.array(mesh.boundary_face_function_numbers),
        boundary_face_physical_tags=jnp.array(mesh.boundary_face_physical_tags),
        boundary_face_face_indices=jnp.array(mesh.boundary_face_face_indices),
        face_cells=jnp.array(mesh.face_cells),
        # face_cell_face_index=jnp.array(mesh.face_cell_face_index),  # If needed
        face_normals=jnp.array(mesh.face_normals),
        face_volumes=jnp.array(mesh.face_volumes),
        face_centers=jnp.array(mesh.face_centers),
        face_subvolumes=jnp.array(mesh.face_subvolumes),
        face_neighbors=jnp.array(mesh.face_neighbors),
        boundary_conditions_sorted_physical_tags=jnp.array(
            mesh.boundary_conditions_sorted_physical_tags
        ),
        boundary_conditions_sorted_names=list(
            mesh.boundary_conditions_sorted_names
        ),  # Kept as NumPy array
        lsq_gradQ=jnp.array(mesh.lsq_gradQ),
        lsq_neighbors=jnp.array(mesh.lsq_neighbors),
        lsq_monomial_multi_index=jnp.int64(mesh.lsq_monomial_multi_index),
        lsq_scale_factors=jnp.int64(mesh.lsq_scale_factors),
        z_ordering=jnp.array(mesh.z_ordering),
    )

def meshjax_flatten(mesh: MeshJAX):
    # Children: all jnp arrays only (no strings)
    children = (
        mesh.vertex_coordinates,
        mesh.cell_vertices,
        mesh.cell_faces,
        mesh.cell_volumes,
        mesh.cell_centers,
        mesh.cell_inradius,
        mesh.cell_neighbors,
        mesh.boundary_face_cells,
        mesh.boundary_face_ghosts,
        mesh.boundary_face_function_numbers,
        mesh.boundary_face_physical_tags,
        mesh.boundary_face_face_indices,
        mesh.face_cells,
        mesh.face_normals,
        mesh.face_volumes,
        mesh.face_centers,
        mesh.face_subvolumes,
        mesh.face_neighbors,
        mesh.boundary_conditions_sorted_physical_tags,
        mesh.lsq_gradQ,
        mesh.lsq_neighbors,
        mesh.z_ordering,
    )
    # Aux data: static metadata, excluding the string list
    aux_data = (
        mesh.dimension,
        mesh.type,
        mesh.n_cells,
        mesh.n_inner_cells,
        mesh.n_faces,
        mesh.n_vertices,
        mesh.n_boundary_faces,
        mesh.n_faces_per_cell,
        # boundary_conditions_sorted_names excluded here
        mesh.lsq_monomial_multi_index,
        mesh.lsq_scale_factors,
    )
    return children, aux_data


def meshjax_unflatten(aux_data, children):
    (
        dimension,
        type_,
        n_cells,
        n_inner_cells,
        n_faces,
        n_vertices,
        n_boundary_faces,
        n_faces_per_cell,
        # boundary_conditions_sorted_names is excluded here, so provide a default or None
        lsq_monomial_multi_index,
        lsq_scale_factors,
    ) = aux_data
    return MeshJAX(
        dimension=dimension,
        type=type_,
        n_cells=n_cells,
        n_inner_cells=n_inner_cells,
        n_faces=n_faces,
        n_vertices=n_vertices,
        n_boundary_faces=n_boundary_faces,
        n_faces_per_cell=n_faces_per_cell,
        vertex_coordinates=children[0],
        cell_vertices=children[1],
        cell_faces=children[2],
        cell_volumes=children[3],
        cell_centers=children[4],
        cell_inradius=children[5],
        cell_neighbors=children[6],
        boundary_face_cells=children[7],
        boundary_face_ghosts=children[8],
        boundary_face_function_numbers=children[9],
        boundary_face_physical_tags=children[10],
        boundary_face_face_indices=children[11],
        face_cells=children[12],
        face_normals=children[13],
        face_volumes=children[14],
        face_centers=children[15],
        face_subvolumes=children[16],
        face_neighbors=children[17],
        boundary_conditions_sorted_physical_tags=children[18],
        boundary_conditions_sorted_names=None,  # or [] or some default value
        lsq_gradQ=children[19],
        lsq_neighbors=children[20],
        lsq_monomial_multi_index=lsq_monomial_multi_index,
        lsq_scale_factors=lsq_scale_factors,
        z_ordering=children[21],
    )


jax.tree_util.register_pytree_node(MeshJAX, meshjax_flatten, meshjax_unflatten)


if __name__ == "__main__":
    path = "/home/ingo/Git/sms/meshes/quad_2d/mesh_coarse.msh"
    path2 = "/home/ingo/Git/sms/meshes/quad_2d/mesh_fine.msh"
    path3 = "/home/ingo/Git/sms/meshes/quad_2d/mesh_finest.msh"
    path4 = "/home/ingo/Git/sms/meshes/triangle_2d/mesh_coarse.msh"
    labels = get_physical_boundary_labels(path)

    # dm, boundary_dict, ghost_cells_dict = load_gmsh(path)

    mesh = Mesh.from_gmsh(path)
    assert mesh.cell_faces.max() == mesh.n_faces - 1
    assert mesh.cell_faces.min() == 0
    assert mesh.face_cells.max() == mesh.n_cells - 1
    assert mesh.face_cells.min() == 0
    assert mesh.cell_vertices.max() == mesh.n_vertices - 1
    assert mesh.cell_vertices.min() == 0

    mesh.write_to_hdf5("./test.h5")
    mesh = Mesh.from_hdf5("./test.h5")
    # mesh.write_to_vtk('./test.vtk')
    mesh.write_to_vtk(
        "./test.vtk",
        fields=np.ones((2, mesh.n_inner_cells), dtype=float),
        field_names=["A", "B"],
    )
