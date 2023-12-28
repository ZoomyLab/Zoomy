import numpy as np
import os

from attr import define, field
from typing import Union, Optional, Callable, List

from library.custom_types import IArray, FArray
import library.fvm_mesh as fvm_mesh
from library.mesh_util import center
from library.misc import (
    project_in_x_y_and_recreate_Q,
    projection_in_normal_and_transverse_direction,
)

@define(slots=True, frozen=False, kw_only=True)
class BoundaryCondition:
    physical_tag: str

    """ 
    Default implementation. The required data for the 'ghost cell' is the data from the interior cell. Can be overwritten e.g. to implement periodic boundary conditions.
    """
    def fill_maps_for_boundary_conditions(self, mesh, map_required_elements, map_functions):
        for i_edge in range(mesh.n_boundary_elements):
            boundary_tag_name = mesh.boundary_tag_names[mesh.boundary_face_tag[i_edge]].decode("utf-8")
            if self.physical_tag == boundary_tag_name:
                map_required_elements[i_edge] = mesh.boundary_face_corresponding_element[i_edge]
                map_functions[i_edge] = self.get_boundary_condition_function()

    def get_boundary_condition_function(self):
        def f(Q: FArray, normal: FArray, momentum_eqns: List[int]):
            print("BoundaryCondition is a virtual class. Use one if its derived classes!")
            assert False
        return f


@define(slots=True, frozen=False, kw_only=True)
class Extrapolation(BoundaryCondition):
    def get_boundary_condition_function(self):
        def f(Q: FArray, normal: FArray, momentum_eqns: List[int]):
            return Q
        return f
            


@define(slots=True, frozen=False, kw_only=True)
class Wall(BoundaryCondition):
    # indices of which field indices of Q correspond to directed fields, e.g. x/y momentum
    # momentum_eqns: IArray = field(converter=np.ndarray)
    momentum_eqns: IArray = field(converter=lambda x: np.array(x, dtype=int))

    def get_boundary_condition_function(self):
        def f(Q: FArray, normal: FArray, momentum_eqns: List[int]):
            Qn, Qt = projection_in_normal_and_transverse_direction(
                Q, momentum_eqns, normal
            )
            # flip the normal direcion for impermeable wall
            return project_in_x_y_and_recreate_Q(
                -Qn, Qt, Q, momentum_eqns, normal
            )
        return f


@define(slots=True, frozen=False, kw_only=True)
class Periodic(BoundaryCondition):
    periodic_to_physical_tag: str

    def get_boundary_condition_function(self):
        def f(Q: FArray, normal: FArray, momentum_eqns: List[int]):
            return Q
        return f

    def fill_maps_for_boundary_conditions(self, mesh, map_required_elements, map_functions):
        hits = 0
        for i_edge in range(mesh.n_boundary_elements):
            boundary_tag_name = mesh.boundary_tag_names[mesh.boundary_face_tag[i_edge]].decode("utf-8")
            if self.physical_tag == boundary_tag_name:
                j_hits = 0
                for j_edge in range(mesh.n_boundary_elements):
                    j_boundary_tag_name = mesh.boundary_tag_names[mesh.boundary_face_tag[j_edge]].decode("utf-8")
                    if self.periodic_to_physical_tag == j_boundary_tag_name:
                        if hits == j_hits:
                            map_required_elements[i_edge] = mesh.boundary_face_corresponding_element[j_edge]
                            map_functions[i_edge] = self.get_boundary_condition_function()
                        j_hits +=1
                hits +=1


@define(slots=True, frozen=False)
class BoundaryConditions:
    boundary_conditions: list[BoundaryCondition]
    map_boundary_index_to_required_elements: IArray = np.empty(0, dtype=int)
    map_boundary_index_to_boundary_function_index: List[Callable] = []

    def initialize(self, mesh):
        self.map_boundary_index_to_required_elements = np.empty(mesh.n_boundary_elements, dtype=int)
        self.map_boundary_index_to_boundary_function_index = [None]  * mesh.n_boundary_elements
        for bc in self.boundary_conditions:
            bc.fill_maps_for_boundary_conditions(mesh, self.map_boundary_index_to_required_elements, self.map_boundary_index_to_boundary_function_index)
        
        for val in self.map_boundary_index_to_boundary_function_index:
            assert val is not None
    
    def collect_requried_element_data(self, Q_global):
        return Q_global[self.map_boundary_index_to_required_elements]
        
    def apply_all(self, Q_ghost, Q_neighbor, boundary_normals, momentum_eqns):
        """
        Q_ghost: output cells I want to write to
        Q_neighbor: cells used to derive the boundary conditions from, e.g. neighboring cells (can be the periodic cells)
        boundary_normals: outgoing normal of the Q_neighbor cell
        momentum_eqns: 
        """
        n_boundary_elements = self.map_boundary_index_to_required_elements.shape[0]
        assert Q_neighbor.shape[0] == n_boundary_elements
        for i_edge in range(n_boundary_elements):
            Q_ghost[i_edge] = self.map_boundary_index_to_boundary_function_index[i_edge](Q_neighbor[i_edge], boundary_normals[i_edge], momentum_eqns)

    def apply(self, i_boundary_element, i_corresponding_element, Q, boundary_normal, momentum_eqns):
        return self.map_boundary_index_to_boundary_function_index[i_boundary_element](Q[self.map_boundary_index_to_required_elements[i_boundary_element]], boundary_normal, momentum_eqns)
