import numpy as np
import os
import h5py

import sympy as sym
from sympy import Matrix

from attr import define, field
from typing import Union, Optional, Callable, List, Dict

from library.misc.custom_types import IArray, FArray
import library.mesh.fvm_mesh as fvm_mesh
from library.mesh.mesh_util import center
from library.misc.misc import (
    project_in_x_y_and_recreate_Q,
    projection_in_normal_and_transverse_direction,
)

@define(slots=True, frozen=False, kw_only=True)
class BoundaryCondition:
    physical_tag: str

    """ 
    Default implementation. The required data for the 'ghost cell' is the data from the interior cell. Can be overwritten e.g. to implement periodic boundary conditions.
    """

    def get_boundary_condition_function(self, Q, Qaux,  parameters, normal):
        print("BoundaryCondition is a virtual class. Use one if its derived classes!")
        assert False


@define(slots=True, frozen=False, kw_only=True)
class Extrapolation(BoundaryCondition):
    def get_boundary_condition_function(self, Q, Qaux, parameters, normal):
        return Matrix(Q)


@define(slots=True, frozen=False, kw_only=True)
class InflowOutflow(BoundaryCondition):
    prescribe_fields: dict[int, float]

    def get_boundary_condition_function(self, Q, Qaux, parameters, normal):
        Qout = Matrix( Q )
        for k, v in self.prescribe_fields.items():
            Qout[k] = v
        return Qout
            

@define(slots=True, frozen=False, kw_only=True)
class Wall(BoundaryCondition):
    """
    momentum_field_indices: list(int): indicate which fields need to be mirrored at the wall
    permeability: float : 1.0 corresponds to a perfect reflection (impermeable wall)
    """
    momentum_field_indices: List[int] = [1,2]
    permeability: float = 1.0


    def get_boundary_condition_function(self, Q, Qaux, parameters, normal):
        out = Matrix( Q)
        dim = normal.shape[0]
        n_fields = Q.shape[0]
        momentum = Matrix([Q[k] for k in self.momentum_field_indices])
        h = Q[0]
        hu = momentum[0]
        u = hu / h
        p = parameters
        out = Matrix([0 for i in range(n_fields)])
        out[0] = h
        U = [u]
        normal_momentum_coef = momentum.dot(normal)
        transverse_momentum = momentum - normal_momentum_coef * normal
        momentum_wall = transverse_momentum - self.permeability * normal_momentum_coef * normal
        for i_k, k in enumerate(self.momentum_field_indices):
            out[k] = momentum_wall[i_k]
        return out


# @define(slots=True, frozen=False, kw_only=True)
# class Periodic(BoundaryCondition):
#     periodic_to_physical_tag: str

#     def get_boundary_condition_function(self, Q, Qaux, parameters, normal):
#         return Matrix(Q)

#     def fill_maps_for_boundary_conditions(self, mesh, map_required_elements, map_functions, map_boundary_function_name_to_index, map_boundary_index_to_function, Q, Qaux, parameters, normal):
#         hits = 0
#         for i_edge in range(mesh.n_boundary_elements):
#             boundary_tag_name = mesh.boundary_tag_names[mesh.boundary_face_tag[i_edge]].decode("utf-8")
#             if self.physical_tag == boundary_tag_name:
#                 j_hits = 0
#                 for j_edge in range(mesh.n_boundary_elements):
#                     j_boundary_tag_name = mesh.boundary_tag_names[mesh.boundary_face_tag[j_edge]].decode("utf-8")
#                     if self.periodic_to_physical_tag == j_boundary_tag_name:
#                         if hits == j_hits:
#                             # map_required_elements[i_edge] = mesh.boundary_face_corresponding_element[j_edge]
#                             # map_functions[i_edge] = self.get_boundary_condition_function()
#                             map_required_elements[i_edge] = mesh.boundary_face_corresponding_element[j_edge]
#                             if boundary_tag_name in map_boundary_function_name_to_index:
#                                 map_functions[i_edge] = map_boundary_function_name_to_index[boundary_tag_name]
#                             else:
#                                 index = len(map_boundary_function_name_to_index)
#                                 map_boundary_function_name_to_index[boundary_tag_name] = index
#                                 map_boundary_index_to_function[index] = self.get_boundary_condition_function(Matrix(Q.get_list()), Matrix(Qaux.get_list()), Matrix(parameters.get_list()), Matrix(normal.get_list()))
#                                 map_functions[i_edge] = map_boundary_function_name_to_index[boundary_tag_name]
#                         j_hits +=1
#                 hits +=1


@define(slots=True, frozen=False)
class BoundaryConditions:
    boundary_conditions: list[BoundaryCondition]
    # map_boundary_index_to_required_elements: IArray = np.empty(0, dtype=int)
    map_function_index_to_physical_index: List[int] = []
    boundary_functions: List[Callable] = []
    # boundary_functions_name: List[str] = []
    # runtime_bc: list[Callable] = []
    initialized: bool = False

    #TODO add variables, ghost_variables and so on. Pass it to full_maps...
    def initialize(self, map_boundary_name_to_index, Q, Qaux, parameters, normal):
        # self.map_boundary_index_to_required_elements = np.empty(mesh.n_boundary_elements, dtype=int)
        # self.map_boundary_index_to_boundary_function_index = [None]  * mesh.n_boundary_elements
        # _map_boundary_function_name_to_index: Dict[str, int] = {}

        # _map_boundary_index_to_function: Dict[int, Callable] = {}
        for i_bc, bc in enumerate(self.boundary_conditions):
            # bc.fill_maps_for_boundary_conditions(mesh, self.map_boundary_index_to_required_elements, self.map_boundary_index_to_boundary_function_index, _map_boundary_function_name_to_index, _map_boundary_index_to_function, Q, Qaux, parameters, normal)
            map_function_index_to_physical_index.append(map_boundary_name_to_index[bc.physical_tag])
            boundary_functions.append(bc.get_boundary_condition_function(Q, Qaux, parameters, normal))
        self.initialized=True

    def get_boundary_function_list(self):
        assert initialized
        return self.boundary_functions

    def get_map_function_index_to_physical_index(self):
        """
        for each boundary edge, I get the physical index. I want to know the function index. I need to search through the list for that, since I cannot pass a dict to C. Alternatively, I can change the physical index in petsc later in the C code to correspond to the bcs.
        """
        assert initialized
        return self.map_function_index_to_physical_index

    def find_function_index(self, physical_index):
        for i_func, idx in enumerate(self.map_function_index_to_physical_index):
            if idx == physical_index:
                return i_func 
        print("physical index not found.")
        assert False
        
        
    # def apply_all(self, Q_ghost, Q_neighbor, boundary_normals, momentum_eqns):
    #     """
    #     Q_ghost: output cells I want to write to
    #     Q_neighbor: cells used to derive the boundary conditions from, e.g. neighboring cells (can be the periodic cells)
    #     boundary_normals: outgoing normal of the Q_neighbor cell
    #     momentum_eqns: 
    #     """
    #     n_boundary_elements = self.map_boundary_index_to_required_elements.shape[0]
    #     assert Q_neighbor.shape[0] == n_boundary_elements
    #     for i_edge in range(n_boundary_elements):
    #         Q_ghost[i_edge] = self.runtime_bc[self.map_boundary_index_to_boundary_function_index[i_edge]](Q_neighbor[i_edge], boundary_normals[i_edge], momentum_eqns)

    # #TODO add Qaux_ghost to everything
    # #TODO add Q, Qghost, ... to func
    # #TODO delete i_corresponding_element
    # def apply(self, i_boundary_element, i_corresponding_element, Q, Qaux, parameters, boundary_normal):
    #     q = Q[self.map_boundary_index_to_required_elements[i_boundary_element]]
    #     qaux = Qaux[self.map_boundary_index_to_required_elements[i_boundary_element]]
    #     func = self.runtime_bc[self.map_boundary_index_to_boundary_function_index[i_boundary_element]]
    #     ## C
    #     # qout = np.zeros_like(q)
    #     # func(q, qaux, parameters, boundary_normal, qout)
    #     # return qout
    #     ## python
    #     qout = func(q, qaux, parameters, boundary_normal)
    #     return qout
    
    # def append_boundary_map_to_mesh_hdf5(self, filepath, filename='mesh.hdf5'):
    #     with h5py.File(os.path.join(filepath, filename), "a") as f:
    #         delete_datasets = ["boundary_function_index", "required_elements", "boundary_function_name"]
    #         for name in delete_datasets:
    #             if name in f:
    #                 del f[name]
    #         f.create_dataset("boundary_function_index", data=self.map_boundary_index_to_boundary_function_index)
    #         f.create_dataset("required_elements", data=self.map_boundary_index_to_required_elements)
    #         f.create_dataset("boundary_function_name", data=self.boundary_functions_name)
