import numpy as np
import os
import h5py
from time import time as get_time

from copy import deepcopy

import sympy
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

    def get_boundary_condition_function(self, time, X, dX,  Q, Qaux,  parameters, normal):
        print("BoundaryCondition is a virtual class. Use one if its derived classes!")
        assert False


@define(slots=True, frozen=False, kw_only=True)
class Extrapolation(BoundaryCondition):
    def get_boundary_condition_function(self, time, X, dX,  Q, Qaux, parameters, normal):
        return Matrix(Q)


@define(slots=True, frozen=False, kw_only=True)
class InflowOutflow(BoundaryCondition):
    prescribe_fields: dict[int, float]

    def get_boundary_condition_function(self, time, X, dX, Q, Qaux, parameters, normal):
        # Extrapolate all fields
        Qout = Matrix(Q)

        # Set the fields which are prescribed in boundary condition dict
        for k, v in self.prescribe_fields.items():
            Qout[k] = eval(v)
        return Qout


def _sympy_interpolate_data(time, timeline, data):
    assert timeline.shape[0] == data.shape[0]
    conditions = ((data[0], time <= timeline[0])),
    for i in range(timeline.shape[0]-1):
        t0 = timeline[i]
        t1 = timeline[i+1]
        y0 = data[i]
        y1 = data[i+1]
        conditions += (-(time-t1)/(t1-t0)*y0 +(time-t0)/(t1-t0)*y1 , time <= t1),
    conditions += ((data[-1], time > timeline[-1])),
    return sympy.Piecewise(*conditions)

@define(slots=True, frozen=False, kw_only=True)
class FromData(BoundaryCondition):
    prescribe_fields: dict[int, np.ndarray]
    timeline: np.ndarray

    def get_boundary_condition_function(self, time, X, dX, Q, Qaux, parameters, normal):
        # Extrapolate all fields
        Qout = Matrix(Q)

        # Set the fields which are prescribed in boundary condition dict
        time_start = get_time()
        for k, v in self.prescribe_fields.items():
            # interp_func = sympy.functions.special.bsplines.interpolating_spline(1, time, self.timeline, v)
            interp_func = _sympy_interpolate_data(time, self.timeline, v)
            Qout[k] = 2*interp_func-Q[k]
        return Qout

@define(slots=True, frozen=False, kw_only=True)
class FromDataGhost(BoundaryCondition):
    prescribe_fields: dict[int, np.ndarray]
    timeline: np.ndarray

    def get_boundary_condition_function(self, time, X, dX, Q, Qaux, parameters, normal):
        # Extrapolate all fields
        Qout = Matrix(Q)

        # Set the fields which are prescribed in boundary condition dict
        time_start = get_time()
        for k, v in self.prescribe_fields.items():
            # interp_func = sympy.functions.special.bsplines.interpolating_spline(1, time, self.timeline, v)
            interp_func = _sympy_interpolate_data(time, self.timeline, v)
            Qout[k] = interp_func
        return Qout
            

@define(slots=True, frozen=False, kw_only=True)
class Wall(BoundaryCondition):
    """
    momentum_field_indices: list(int): indicate which fields need to be mirrored at the wall
    permeability: float : 1.0 corresponds to a perfect reflection (impermeable wall)
    """
    momentum_field_indices: List[List[int]] = [[1,2]]
    permeability: float = 1.0
    wall_slip: float = 1.0


    def get_boundary_condition_function(self, time, X, dX, Q, Qaux, parameters, normal):
        q = Matrix(Q)
        n = Matrix(normal)
        dim = normal.length()
        n_fields = Q.length()
        momentum_list = [Matrix([q[k] for k in l]) for l in self.momentum_field_indices]
        zero = 10**(-20)*q[0]
        h = q[0]
        p = parameters
        out = Matrix([zero for i in range(n_fields)])
        out[0] = h
        momentum_list_wall = []
        for momentum in momentum_list:
            normal_momentum_coef = momentum.dot(n)
            transverse_momentum = momentum - normal_momentum_coef * n
            momentum_wall = self.wall_slip * transverse_momentum - self.permeability * normal_momentum_coef * n
            momentum_list_wall.append( momentum_wall )
        for l, momentum_wall in zip(self.momentum_field_indices, momentum_list_wall):
            for i_k, k in enumerate(l):
                out[k] = momentum_wall[i_k]
        return out


@define(slots=True, frozen=False, kw_only=True)
class Periodic(BoundaryCondition):
    periodic_to_physical_tag: str

    def get_boundary_condition_function(self,time, X, dX, Q, Qaux, parameters, normal):
        return Matrix(Q)

@define(slots=True, frozen=False)
class BoundaryConditions:
    boundary_conditions: list[BoundaryCondition]
    boundary_functions: List[Callable] = []
    initialized: bool = False


    def resolve_periodic_bcs(self, mesh):
        dict_physical_name_to_index = {v: i for i, v in enumerate(mesh.boundary_conditions_sorted_names)}
        dict_function_index_to_physical_tag = {i: v for i, v in enumerate(mesh.boundary_conditions_sorted_physical_tags)}

        mesh_copy = deepcopy(mesh)

        for i_bc, bc in enumerate(self.boundary_conditions):
            if type(bc) == Periodic:
                from_physical_tag = dict_function_index_to_physical_tag[dict_physical_name_to_index[bc.periodic_to_physical_tag]]
                to_physical_tag = dict_function_index_to_physical_tag[dict_physical_name_to_index[bc.physical_tag]]
                #this is not cells, this is yet the boundary  index -> extract the cell index from here.
                from_cells_boundary_face_index = mesh.boundary_face_physical_tags == from_physical_tag
                to_cells_boundary_face_index = mesh.boundary_face_physical_tags == to_physical_tag

                from_cells = mesh.boundary_face_cells[from_cells_boundary_face_index]
                to_cells = mesh.boundary_face_ghosts[to_cells_boundary_face_index]

                from_coords = mesh.cell_centers[:, from_cells]
                to_coords = mesh.cell_centers[:, to_cells]


                sort_order_from = np.lexsort([from_coords[d, :] for d in range(mesh.dimension)])
                sort_order_to = np.lexsort([to_coords[d, :] for d in range(mesh.dimension)])

                # advanded indexing creates copies. So I need to construct indexing sets to overwrite the content of 
                # mesh.boundary_face_ghosts[to_cells_boundary_face_index][sort_order_to]

                indices_ghosts = np.array(list(range(to_cells_boundary_face_index.shape[0])))
                indices_ghosts = indices_ghosts[to_cells_boundary_face_index] 
                indices_ghosts = indices_ghosts[sort_order_to] 

                indices_cells = np.array(list(range(to_cells_boundary_face_index.shape[0])))
                indices_cells = indices_cells[from_cells_boundary_face_index] 
                indices_cells = indices_cells[sort_order_from] 

                # mesh.boundary_face_ghosts[indices_ghosts] = mesh.boundary_face_cells[indices_cells]
                mesh.boundary_face_cells[indices_cells] = mesh_copy.boundary_face_cells[indices_ghosts]
        return mesh

    def initialize(self, mesh, time, X, dX, Q, Qaux, parameters, normal):

        dict_physical_name_to_index = {v: i for i, v in enumerate(mesh.boundary_conditions_sorted_names)}
        dict_index_to_function = {i: None for i, v in enumerate(mesh.boundary_conditions_sorted_names)}
        periodic_bcs_ghosts = []
        for i_bc, bc in enumerate(self.boundary_conditions):
            dict_index_to_function[dict_physical_name_to_index[bc.physical_tag]] = bc.get_boundary_condition_function(time, X, dX, Q,  Qaux, parameters, normal)
        #     if type(bc) == Periodic:
        #         function_index = dict_physical_name_to_index[bc.periodic_to_physical_tag]
        #         periodics_bcs_from = mesh.boundary_face_ghosts[mesh.boundary_face_function_numbers == function_index ]
        self.boundary_functions = list(dict_index_to_function.values())
        mesh = self.resolve_periodic_bcs(mesh)
        self.initialized=True
        return mesh

    def get_boundary_function_list(self):
        assert self.initialized
        return self.boundary_functions
    