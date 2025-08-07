import numpy as np
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

    def get_boundary_condition_function(self, time, X, dX, Q, Qaux, parameters, normal):
        print("BoundaryCondition is a virtual class. Use one if its derived classes!")
        assert False


@define(slots=True, frozen=False, kw_only=True)
class Extrapolation(BoundaryCondition):
    def get_boundary_condition_function(self, time, X, dX, Q, Qaux, parameters, normal):
        return Matrix(Q)


@define(slots=True, frozen=False, kw_only=True)
class InflowOutflow(BoundaryCondition):
    prescribe_fields: dict[int, float]

    def get_boundary_condition_function(self, time, X, dX, Q, Qaux, parameters, normal):
        Qout = Matrix(Q)
        for k, v in self.prescribe_fields.items():
            Qout[k] = eval(v)
        return Qout
    
@define(slots=True, frozen=False, kw_only=True)
class Lambda(BoundaryCondition):
    prescribe_fields: dict[int, float]

    def get_boundary_condition_function(self, time, X, dX, Q, Qaux, parameters, normal):
        Qout = Matrix(Q)
        for k, v in self.prescribe_fields.items():
            Qout[k] = v(time, X, dX, Q, Qaux, parameters, normal)
        return Qout


def _sympy_interpolate_data(time, timeline, data):
    assert timeline.shape[0] == data.shape[0]
    conditions = (((data[0], time <= timeline[0])),)
    for i in range(timeline.shape[0] - 1):
        t0 = timeline[i]
        t1 = timeline[i + 1]
        y0 = data[i]
        y1 = data[i + 1]
        conditions += (
            (-(time - t1) / (t1 - t0) * y0 + (time - t0) / (t1 - t0) * y1, time <= t1),
        )
    conditions += (((data[-1], time > timeline[-1])),)
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
            Qout[k] = 2 * interp_func - Q[k]
        return Qout

@define(slots=True, frozen=False, kw_only=True)
class PreciceCoupling(BoundaryCondition):
    dirichlet_fields: dict[str, int]
    dirichlet_vector_fields: dict[str, int]

    def get_boundary_condition_function(self, time, X, dX, Q, Qaux, parameters, normal):
        return Matrix(Q)


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

    momentum_field_indices: List[List[int]] = [[1, 2]]
    permeability: float = 1.0
    wall_slip: float = 1.0

    def get_boundary_condition_function(self, time, X, dX, Q, Qaux, parameters, normal):
        q = Matrix(Q)
        n = Matrix(normal)
        dim = normal.length()
        n_variables = Q.length()
        momentum_list = [Matrix([q[k] for k in l]) for l in self.momentum_field_indices]
        zero = 10 ** (-20) * q[0]
        h = q[0]
        p = parameters
        out = Matrix([zero for i in range(n_variables)])
        out[0] = h
        momentum_list_wall = []
        for momentum in momentum_list:
            normal_momentum_coef = momentum.dot(n)
            transverse_momentum = momentum - normal_momentum_coef * n
            momentum_wall = (
                self.wall_slip * transverse_momentum
                - self.permeability * normal_momentum_coef * n
            )
            momentum_list_wall.append(momentum_wall)
        for l, momentum_wall in zip(self.momentum_field_indices, momentum_list_wall):
            for i_k, k in enumerate(l):
                out[k] = momentum_wall[i_k]
        return out


@define(slots=True, frozen=False, kw_only=True)
class Periodic(BoundaryCondition):
    periodic_to_physical_tag: str

    def get_boundary_condition_function(self, time, X, dX, Q, Qaux, parameters, normal):
        return Matrix(Q)


@define(slots=True, frozen=False)
class BoundaryConditions:
    boundary_conditions: list[BoundaryCondition]
    boundary_functions: List[Callable] = []
    initialized: bool = False

    def resolve_periodic_bcs(self, mesh):
        """
        Goal: if 'apply_boundary_condition' is called, the ghost cell value is computed, given an input cell value funtion.
              In case of a periodic BC, this is NOT the adjacent cell. So I need to change the 'boundary_face_cell' for the periodic
              cells to point to the right data location!
              This is why we only overwrite the 'mesh.boundary_face_cell' in the end.
              Furthermore, I CANNOT alter any ordering! However, I need to sort the two boundaries such that the e.g.
              left and right border are mapped correctly, as the boundary cells are not ordered.
              As ghost/inner cells is confusing here, I rather like to use 'from' and 'to', as 'from' data from which bc is copied and 'to' stands for the boundary where it is copied to. As we only change the cells where the data is taken from (from the adjacent to the periodic cell), we only alter the 'boundary_face_cell' in the end.
        """
        dict_physical_name_to_index = {
            v: i for i, v in enumerate(mesh.boundary_conditions_sorted_names)
        }
        dict_function_index_to_physical_tag = {
            i: v for i, v in enumerate(mesh.boundary_conditions_sorted_physical_tags)
        }

        mesh_copy = deepcopy(mesh)

        for i_bc, bc in enumerate(self.boundary_conditions):
            if type(bc) == Periodic:
                from_physical_tag = dict_function_index_to_physical_tag[
                    dict_physical_name_to_index[bc.periodic_to_physical_tag]
                ]
                to_physical_tag = dict_function_index_to_physical_tag[
                    dict_physical_name_to_index[bc.physical_tag]
                ]

                mask_face_from = mesh.boundary_face_physical_tags == from_physical_tag
                mask_face_to = mesh.boundary_face_physical_tags == to_physical_tag

                # I want to copy from boundary_face_cells to boundary_face_ghosts!
                from_cells = mesh.boundary_face_cells[mask_face_from]
                to_cells = mesh.boundary_face_ghosts[mask_face_to]

                from_coords = mesh.cell_centers[:, from_cells]
                to_coords = mesh.cell_centers[:, to_cells]

                # sort not dimension by dimension, but most significant dimension to least significant dimension
                # determine significance by max difference
                significance_per_dimension = [
                    from_coords[d, :].max() - from_coords[d, :].min()
                    for d in range(mesh.dimension)
                ]
                _significance_per_dimension = [
                    to_coords[d, :].max() - to_coords[d, :].min()
                    for d in range(mesh.dimension)
                ]
                # reverse the order of lexsort such that the most important is first IS NOT NEEDED, since lexsort starts sorting by the last entry in the list
                sort_order_significance = np.lexsort([significance_per_dimension])

                # sort_order_from = np.lexsort([from_coords[d, :] for d in range(mesh.dimension)])
                # sort_order_to = np.lexsort([to_coords[d, :] for d in range(mesh.dimension)])
                from_cells_sort_order = np.lexsort(
                    [from_coords[d, :] for d in sort_order_significance]
                )
                to_cells_sort_order = np.lexsort(
                    [to_coords[d, :] for d in sort_order_significance]
                )

                # advanded indexing creates copies. So I need to construct indexing sets to overwrite the content of
                # mesh.boundary_face_ghosts[to_cells_boundary_face_index][sort_order_to]

                # generates indices from 0 to number of ghost_cells (total)
                indices = np.array(list(range(mask_face_to.shape[0])))
                # masks away all cells that do not belong to this tag
                indices_to = indices[mask_face_to]
                # sort the indices
                indices_to_sort = indices_to[to_cells_sort_order]

                indices_from = indices[mask_face_from]
                indices_from_sort = indices_from[from_cells_sort_order]

                mesh.boundary_face_cells[indices_to_sort] = (
                    mesh_copy.boundary_face_cells[indices_from_sort]
                )
                # mesh.boundary_face_cells = mesh.boundary_face_cells.at[indices_to_sort].set(
                #     mesh_copy.boundary_face_cells[indices_from_sort]
                # )

                # if not np.allclose(
                #     from_coords[sort_order_significance[-1], from_cells_sort_order],
                #     to_coords[sort_order_significance[-1], to_cells_sort_order],
                # ):
                #     print(
                #         "WARNING: Periodic boundary condition detected for incompatible mesh. The periodic sides of the mesh does not have the same face layout."
                #     )
        return mesh

    def initialize(self, mesh, time, X, dX, Q, Qaux, parameters, normal):
        dict_physical_name_to_index = {
            v: i for i, v in enumerate(mesh.boundary_conditions_sorted_names)
        }
        dict_index_to_function = {
            i: None for i, v in enumerate(mesh.boundary_conditions_sorted_names)
        }
        periodic_bcs_ghosts = []
        for i_bc, bc in enumerate(self.boundary_conditions):
            dict_index_to_function[dict_physical_name_to_index[bc.physical_tag]] = (
                bc.get_boundary_condition_function(
                    time, X, dX, Q, Qaux, parameters, normal
                )
            )
        #     if type(bc) == Periodic:
        #         function_index = dict_physical_name_to_index[bc.periodic_to_physical_tag]
        #         periodics_bcs_from = mesh.boundary_face_ghosts[mesh.boundary_face_function_numbers == function_index ]
        self.boundary_functions = list(dict_index_to_function.values())
        mesh = self.resolve_periodic_bcs(mesh)
        self.initialized = True
        return mesh

    def get_precice_boundary_indices_to_bc_name(self, mesh):
        dict_physical_name_to_index = {
            v: i for i, v in enumerate(mesh.boundary_conditions_sorted_names)
        }
        dict_index_to_physical_name = {
            v: i for i, v in enumerate(mesh.boundary_conditions_sorted_names)
        }
        out = {}
        for i_bc, bc in enumerate(self.boundary_conditions):
            if type(bc) == PreciceCoupling:
                out = {**out, dict_physical_name_to_index[bc]: bc}
        return out

    def get_boundary_function_list(self):
        assert self.initialized
        return self.boundary_functions
