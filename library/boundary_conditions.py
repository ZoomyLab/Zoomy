import numpy as np
import os

from attr import define, field
from typing import Union, Optional

from library.custom_types import IArray, FArray
import library.mesh as mesh
from library.misc import (
    project_in_x_y_and_recreate_Q,
    projection_in_normal_and_transverse_direction,
)

# TODO ideally, I do not only want to map elements2elements, but interpolate a bounday function
# (e.g. as a parametric function) in order to have triangular boudnary conditions as well


@define(slots=True, frozen=False)
class Segment:
    element_indices: IArray
    face_centers: FArray
    face_normals: FArray
    face_area: FArray
    tag: str
    ghost_element_indices: Optional[IArray] = None
    initialized: bool = False

    @classmethod
    def placeholder(cls):
        iarray = np.array([], dtype=int)
        farray = np.array([], dtype=float)
        tag = "empty"
        return cls(iarray, farray, farray, farray, tag)

    @classmethod
    def from_mesh(cls, mesh, be_tag, reverse_order=False):
        counter = 0
        be_elements = []
        be_face_centers = []
        be_face_area = []
        be_normals = []

        # compute boundary edge elements and normals
        for i_edge, element in enumerate(mesh.boundary_edge_elements):
            if mesh.boundary_edge_tag[i_edge].decode("utf-8") == be_tag:
                be_elements.append(int(element))
                be_normals.append(mesh.boundary_edge_normal[i_edge])
                be_face_centers.append(mesh.element_centers[element])
                be_face_area.append(mesh.boundary_edge_length[i_edge])
                counter += 1

        if len(be_elements) == 0:
            print(
                "Error: boundary_tag {} not found. Make sure the mesh contains this tag. Use the physical tag property of gmsh to define tags."
            )
            assert False

        if mesh.dimension == 1:
            indices_sorted = np.lexsort((np.array(be_face_centers)[:, 0],))
        elif mesh.dimension == 2:
            indices_sorted = np.lexsort(
                (
                    np.array(be_face_centers)[:, 0],
                    np.array(be_face_centers)[:, 1],
                )
            )
        elif mesh.dimension == 3:
            indices_sorted = np.lexsort(
                (
                    np.array(be_face_centers)[:, 0],
                    np.array(be_face_centers)[:, 1],
                    np.array(be_face_centers)[:, 2],
                )
            )
        else:
            assert False

        be_elements = np.array(be_elements)[indices_sorted]
        be_face_centers = np.array(be_face_centers)[indices_sorted]
        be_face_area = np.array(be_face_area)[indices_sorted]
        be_normals = np.array(be_normals)[indices_sorted]
        counter = 0

        if reverse_order:
            be_elements.reverse()
            be_normals.reverse()
            be_face_centers.reverse()
        return cls(
            be_elements, be_face_centers, be_normals, be_face_area, be_tag, None, None
        )

    # The solver needs to reserve space in the global array for the ghost elements
    # Once it figured out where to store the ghost cells, we can generate them and
    # store the corresponding global indices
    def set_ghost_cells(self, element_indices):
        self.ghost_element_indices = element_indices
        self.initialized = True


@define(slots=True, frozen=False, kw_only=True)
class BoundaryCondition:
    physical_tag: str
    initialized: bool = False
    segment: Segment = Segment.placeholder()

    def initialize(self, mesh: mesh.Mesh):
        self.segment = Segment.from_mesh(mesh, self.physical_tag)
        self.initialized = True

    def apply_boundary_condition(self, Q: FArray):
        assert self.initialized
        assert self.segment is not None
        assert self.segment.initialized == True
        print("BoundaryCondition is a virtual class. Use one if its derived classed!")
        assert False

    def get_length(self):
        return self.segment.element_indices.shape[0]

    def set_ghost_cells(self, element_indices):
        self.segment.set_ghost_cells(element_indices)


@define(slots=True, frozen=False, kw_only=True)
class Extrapolation(BoundaryCondition):
    def apply_boundary_condition(self, Q: FArray):
        assert self.initialized
        assert self.segment.initialized
        Q[self.segment.ghost_element_indices] = Q[self.segment.element_indices]


@define(slots=True, frozen=False, kw_only=True)
class Wall(BoundaryCondition):
    # indices of which field indices of Q correspond to directed fields, e.g. x/y momentum
    # momentum_eqns: IArray = field(converter=np.ndarray)
    momentum_eqns: IArray = field(converter=lambda x: np.array(x, dtype=int))

    def apply_boundary_condition(self, Q: FArray):
        assert self.initialized
        assert self.segment.initialized
        Qorig = Q[self.segment.element_indices]
        normals = self.segment.face_normals
        Qn, Qt = projection_in_normal_and_transverse_direction(
            Qorig, self.momentum_eqns, normals
        )
        # flip the normal direcion for impermeable wall
        Q[self.segment.ghost_element_indices] = project_in_x_y_and_recreate_Q(
            -Qn, Qt, Qorig, self.momentum_eqns, normals
        )


@define(slots=True, frozen=False, kw_only=True)
class Periodic(BoundaryCondition):
    periodic_to_physical_tag: str
    periodic_to_element_indices: Optional[IArray] = None

    def initialize(self, mesh: mesh.Mesh):
        self.segment = Segment.from_mesh(mesh, self.physical_tag)
        self.periodic_to_element_indices = Segment.from_mesh(
            mesh, self.periodic_to_physical_tag
        ).element_indices
        self.initialized = True

    def apply_boundary_condition(self, Q: FArray):
        assert self.initialized
        assert self.segment.initialized
        Q[self.segment.ghost_element_indices] = Q[self.periodic_to_element_indices]


def initialize(boundary_conditions, mesh):
    n_inner_elements = mesh.n_elements
    initialize_bc(boundary_conditions, mesh)
    n_ghosts = initialize_ghost_cells(boundary_conditions, n_inner_elements)
    return n_ghosts


def initialize_bc(boundary_conditions, mesh):
    for bc in boundary_conditions:
        bc.initialize(mesh)


def initialize_ghost_cells(boundary_conditions, n_inner_elements):
    n_ghost_cells = 0
    offset = n_inner_elements
    for bc in boundary_conditions:
        bc.set_ghost_cells(
            offset + np.linspace(0, bc.get_length() - 1, bc.get_length(), dtype=int)
        )
        offset += bc.get_length()
        n_ghost_cells += bc.get_length()
    return n_ghost_cells


def apply_boundary_conditions(boundary_conditions, Q):
    for bc in boundary_conditions:
        bc.apply_boundary_condition(Q)


@define(slots=True, frozen=False)
class BoundaryConditions:
    boundary_conditions: list[BoundaryCondition]

    def initialize(self, mesh):
        n_ghosts = initialize(self.boundary_conditions, mesh)
        return n_ghosts

    def apply(self, Q):
        apply_boundary_conditions(self.boundary_conditions, Q)
