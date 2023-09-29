import numpy as np
import os

from attr import define
from typing import Union, Optional

from library.custom_types import IArray, FArray
import library.mesh as mesh

# TODO ideally, I do not only want to map elements2elements, but interpolate a bounday function
# (e.g. as a parametric function) in order to have triangular boudnary conditions as well


@define(slots=True, frozen=False)
class Segment:
    be_index_to_local_index: IArray
    be_elements: IArray
    be_face_center: FArray
    be_normals: FArray
    tag: str
    be_ghost_elements: Optional[IArray]

    @classmethod
    def from_mesh(cls, mesh, be_tag, reverse_order=False):
        counter = 0
        be_local_index_to_edge_index = []
        be_index_to_local_index = -1 * np.ones(
            mesh.boundary_edge_elements.shape[0], dtype=int
        )
        be_elements = []
        be_face_centers = []
        be_normals = []

        # compute boundary edge elements and normals
        for i_edge, element in enumerate(mesh.boundary_edge_elements):
            if mesh.boundary_edge_tag[i_edge].decode("utf-8") == be_tag:
                be_local_index_to_edge_index.append(i_edge)
                be_elements.append(int(mesh.boundary_edge_elements[i_edge]))
                be_normals.append(mesh.boundary_edge_normal[i_edge])
                be_face_centers.append(mesh.element_centers[element])

                be_index_to_local_index[i_edge] = counter
                assert element == be_elements[be_index_to_local_index[i_edge]]
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
        be_normals = np.array(be_normals)[indices_sorted]
        be_face_centers = np.array(be_face_centers)[indices_sorted]
        counter = 0
        be_local_index_to_edge_index = np.array(be_local_index_to_edge_index)[
            indices_sorted
        ]
        be_index_to_local_index[be_local_index_to_edge_index] = np.linspace(
            0, len(indices_sorted) - 1, len(indices_sorted)
        )

        # sanity check
        counter = 0
        for i_edge, element in enumerate(mesh.boundary_edge_elements):
            if mesh.boundary_edge_tag[i_edge] == be_tag:
                local_index = be_index_to_local_index[i_edge]
                assert local_index != -1
                elem_local = be_elements[local_index]
                assert elem_local == element

        if reverse_order:
            be_elements.reverse()
            be_normals.reverse()
            be_face_centers.reverse()
        return cls(
            be_index_to_local_index,
            be_elements,
            be_face_centers,
            be_normals,
            be_tag,
            None,
        )

    def set_ghost_cells(self, element_indices):
        self.ghost_elements = element_indices


@define(slots=True, frozen=False, kw_only=True)
class BoundaryCondition:
    physical_tag: str
    segment: Optional[Segment] = None

    def initialize(self, mesh: mesh.Mesh):
        self.segment = Segment.from_mesh(mesh, self.physical_tag)

    def apply_boundary_condition(self, Q: FArray):
        print("BoundaryCondition is a virtual class. Use one if its derived classed!")
        assert False


# @define(slots=True, frozen=False)
class Extrapolation(BoundaryCondition):
    def apply_boundary_condition(self, Q: FArray):
        Q[self.segment.ghost_elements] = Q[self.segment.be_elements]


@define(slots=True, frozen=False)
class Wall(BoundaryCondition):
    velocity_components: IArray

    def apply_boundary_condition(self, Q: FArray):
        Q[self.segment.ghost_elements] = Q[self.segment.be_elements]
        Q[:, self.velocity_components] *= -1.0


@define(slots=True, frozen=False)
class Periodic(BoundaryCondition):
    periodic_to_physical_tag: str
    segment_periodic: Optional[Segment]

    def initialize(self, mesh: mesh.Mesh):
        self.segment = Segment.from_mesh(mesh, self.physical_tag)
        self.segment_periodic = Segment.from_mesh(mesh, self.peridic_to_physical_tag)

    def apply_boundary_condition(self, Q: FArray):
        Q[self.segment.ghost_elements] = Q[self.segment_periodic.be_elements]
