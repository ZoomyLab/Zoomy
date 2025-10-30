import numpy as np
import os

from attr import define, field
from typing import Union, Optional, Callable, List, Dict

from library.python.misc.custom_types import IArray, FArray
import library.python.mesh.fvm_mesh as fvm_mesh
from library.python.mesh.mesh_util import center
from library.python.misc.misc import (
    project_in_x_y_and_recreate_Q,
    projection_in_normal_and_transverse_direction,
)


@define(slots=True, frozen=False, kw_only=True)
class BoundaryCondition:
    physical_tag: str

    """ 
    Default implementation. The required data for the 'ghost cell' is the data from the interior cell. Can be overwritten e.g. to implement periodic boundary conditions.
    """

    def fill_maps_for_boundary_conditions(
        self,
        mesh,
        map_required_elements,
        map_functions,
        map_boundary_function_name_to_index,
        map_boundary_index_to_function,
    ):
        for i_edge in range(mesh.n_boundary_elements):
            boundary_tag_name = mesh.boundary_tag_names[
                mesh.boundary_face_tag[i_edge]
            ].decode("utf-8")
            if self.physical_tag == boundary_tag_name:
                map_required_elements[i_edge] = (
                    mesh.boundary_face_corresponding_element[i_edge]
                )
                if boundary_tag_name in map_boundary_function_name_to_index:
                    map_functions[i_edge] = map_boundary_function_name_to_index[
                        boundary_tag_name
                    ]
                else:
                    index = len(map_boundary_function_name_to_index)
                    map_boundary_function_name_to_index[boundary_tag_name] = index
                    map_boundary_index_to_function[index] = (
                        self.compute_boundary_condition()
                    )

    def compute_boundary_condition(self):
        def f(Q: FArray, normal: FArray, momentum_eqns: List[int]):
            print(
                "BoundaryCondition is a virtual class. Use one if its derived classes!"
            )
            assert False

        return f


@define(slots=True, frozen=False, kw_only=True)
class Extrapolation(BoundaryCondition):
    def compute_boundary_condition(self):
        def f(Q: FArray, normal: FArray, momentum_eqns: List[int]):
            return Q

        return f


@define(slots=True, frozen=False, kw_only=True)
class InflowOutflow(BoundaryCondition):
    prescribe_fields: dict[int, float]

    def compute_boundary_condition(self):
        def f(Q: FArray, normal: FArray, momentum_eqns: List[int]):
            Qout = np.array(Q)
            for k, v in self.prescribe_fields.items():
                Qout[k] = v
            return Qout

        return f


@define(slots=True, frozen=False, kw_only=True)
class Wall(BoundaryCondition):
    def compute_boundary_condition(self):
        def f(Q: FArray, normal: FArray, momentum_eqns: List[int]):
            Qn, Qt = projection_in_normal_and_transverse_direction(
                Q, momentum_eqns, normal
            )
            # flip the normal direcion for impermeable wall
            return project_in_x_y_and_recreate_Q(-Qn, Qt, Q, momentum_eqns, normal)

        return f


@define(slots=True, frozen=False, kw_only=True)
class Periodic(BoundaryCondition):
    periodic_to_physical_tag: str

    def compute_boundary_condition(self):
        def f(Q: FArray, normal: FArray, momentum_eqns: List[int]):
            return Q

        return f

    def fill_maps_for_boundary_conditions(
        self,
        mesh,
        map_required_elements,
        map_functions,
        map_boundary_function_name_to_index,
        map_boundary_index_to_function,
    ):
        hits = 0
        for i_edge in range(mesh.n_boundary_elements):
            boundary_tag_name = mesh.boundary_tag_names[
                mesh.boundary_face_tag[i_edge]
            ].decode("utf-8")
            if self.physical_tag == boundary_tag_name:
                j_hits = 0
                for j_edge in range(mesh.n_boundary_elements):
                    j_boundary_tag_name = mesh.boundary_tag_names[
                        mesh.boundary_face_tag[j_edge]
                    ].decode("utf-8")
                    if self.periodic_to_physical_tag == j_boundary_tag_name:
                        if hits == j_hits:
                            # map_required_elements[i_edge] = mesh.boundary_face_corresponding_element[j_edge]
                            # map_functions[i_edge] = self.compute_boundary_condition()
                            map_required_elements[i_edge] = (
                                mesh.boundary_face_corresponding_element[j_edge]
                            )
                            if boundary_tag_name in map_boundary_function_name_to_index:
                                map_functions[i_edge] = (
                                    map_boundary_function_name_to_index[
                                        boundary_tag_name
                                    ]
                                )
                            else:
                                index = len(map_boundary_function_name_to_index)
                                map_boundary_function_name_to_index[
                                    boundary_tag_name
                                ] = index
                                map_boundary_index_to_function[index] = (
                                    self.compute_boundary_condition()
                                )
                                map_functions[i_edge] = (
                                    map_boundary_function_name_to_index[
                                        boundary_tag_name
                                    ]
                                )
                        j_hits += 1
                hits += 1


@define(slots=True, frozen=False)
class BoundaryConditions:
    boundary_conditions: list[BoundaryCondition]
    map_boundary_index_to_required_elements: IArray = np.empty(0, dtype=int)
    map_boundary_index_to_boundary_function_index: List[int] = []
    map_index_to_function: List[Callable] = []
    map_index_to_function_name: List[str] = []

    def initialize(self, mesh):
        self.map_boundary_index_to_required_elements = np.empty(
            mesh.n_boundary_elements, dtype=int
        )
        self.map_boundary_index_to_boundary_function_index = [
            None
        ] * mesh.n_boundary_elements
        _map_boundary_function_name_to_index: Dict[str, int] = {}
        _map_boundary_index_to_function: Dict[int, Callable] = {}
        for bc in self.boundary_conditions:
            bc.fill_maps_for_boundary_conditions(
                mesh,
                self.map_boundary_index_to_required_elements,
                self.map_boundary_index_to_boundary_function_index,
                _map_boundary_function_name_to_index,
                _map_boundary_index_to_function,
            )

        for val in self.map_boundary_index_to_boundary_function_index:
            assert val is not None

        self.map_index_to_function_name = list(
            _map_boundary_function_name_to_index.keys()
        )
        self.map_index_to_function = list(_map_boundary_index_to_function.values())

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
            Q_ghost[i_edge] = self.map_index_to_function[
                self.map_boundary_index_to_boundary_function_index[i_edge]
            ](Q_neighbor[i_edge], boundary_normals[i_edge], momentum_eqns)

    def apply(
        self,
        i_boundary_element,
        i_corresponding_element,
        Q,
        boundary_normal,
        momentum_eqns,
    ):
        return self.map_index_to_function[
            self.map_boundary_index_to_boundary_function_index[i_boundary_element]
        ](
            Q[self.map_boundary_index_to_required_elements[i_boundary_element]],
            boundary_normal,
            momentum_eqns,
        )
