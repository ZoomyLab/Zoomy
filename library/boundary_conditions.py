import numpy as np
import os

from attr import define
from typing import Union

from library.custom_types import IArray
import library.mesh as mesh


@define(slots=True, frozen=True)
class Segment:
    be_index_to_local_index: IArray
    tag: Union[str, None] = None
    be_elements: list = []
    be_element_center: list = []
    be_normals: list = []

    @classmethod
    def from_mesh(cls, mesh, reverse_order=False):
        instance = cls()
        counter = 0
        be_local_index_to_edge_index = []

        # compute boundary edge elements and normals
        for i_edge, element in enumerate(mesh.boundary_edge_element):
            if mesh.boundary_edge_tag[i_edge] == instance.be_tag:
                be_local_index_to_edge_index.append(i_edge)
                instance.be_elements.append(int(mesh.boundary_edge_element[i_edge]))
                instance.be_normals.append(mesh.boundary_edge_normal[i_edge])
                # TODO using the element centers instread of the boundary notes is WRONG!!
                instance.be_element_center.append(mesh.element_centers[element])

                instance.be_index_to_local_index[i_edge] = counter
                assert (
                    element
                    == instance.be_elements[instance.be_index_to_local_index[i_edge]]
                )
                counter += 1

        assert len(instance.be_elements) > 0

        indices_sorted = np.lexsort(
            (
                np.array(instance.be_element_center)[:, 0],
                np.array(instance.be_element_center)[:, 1],
                np.array(instance.be_element_center)[:, 2],
            )
        )
        instance.be_elements = list(np.array(instance.be_elements)[indices_sorted])
        instance.be_normals = list(np.array(instance.be_normals)[indices_sorted])
        instance.be_element_center = list(
            np.array(instance.be_element_center)[indices_sorted]
        )
        counter = 0
        be_local_index_to_edge_index = np.array(be_local_index_to_edge_index)[
            indices_sorted
        ]
        # instance.be_index_to_local_index[be_local_index_to_edge_index] = instance.be_index_to_local_index[np.array(be_local_index_to_edge_index)[indices_sorted]]
        instance.be_index_to_local_index[be_local_index_to_edge_index] = np.linspace(
            0, len(indices_sorted) - 1, len(indices_sorted)
        )

        # sanity check
        counter = 0
        for i_edge, element in enumerate(mesh.boundary_edge_element):
            if mesh.boundary_edge_tag[i_edge] == instance.be_tag:
                local_index = instance.be_index_to_local_index[i_edge]
                assert local_index != -1
                elem_local = instance.be_elements[local_index]
                assert elem_local == element

        if reverse_order:
            instance.be_elements.reverse()
            instance.be_normals.reverse()
            instance.be_element_center.reverse()
        return instance

    def initialize(self, mesh, reverse_order=False):
        self.be_elements = []
        self.be_element_center = []
        self.be_normals = []
        self.be_index_to_local_index = -np.ones(
            mesh.boundary_edge_element.shape, dtype=int
        )


# Identifier for physical boundaries by name/index and segment path
# class Segment2:
#     # TODO this can maybe to over to the mesh function??
#     # TODO ideally, I do not only want to map elements2elements, but interpolate a bounday function (e.g. as a parametric function) in order to have triangular boudnary conditions as well
#     # TODO WARNING: depending on how the loop traverses the below elements, I get different and potentially WRONG (zig zag) curves!!
#     def __init__(self, boundary_edge_tag):
#         self.be_tag = boundary_edge_tag


class BoundaryCondition:
    yaml_tag = "!BoundaryCondition"

    def set_default_parameters(self):
        self.physical_tag = None

    def set_runtime_variables(self):
        assert self.periodic_to_physical_tag is not None
        self.segment = Segment(self.physical_tag)

    def initialize(self, mesh):
        self.segment.initialize(mesh)

    def apply_boundary_condition(self, boundary_edge_id, fields, **kwargs):
        local_index = self.segment.be_index_to_local_index[boundary_edge_id]
        # element is not part of this boundary
        if local_index == -1:
            return np.zeros_like(fields[:, 0])
        return local_index


class Periodic(BoundaryCondition):
    yaml_tag = "Periodic"

    def set_default_parameters(self):
        super().set_default_parameters()
        self.periodic_to_physical_tag = None

    def set_runtime_variables(self):
        super().set_runtime_variables()
        assert self.periodic_to_physical_tag is not None
        self.periodic_to_segment = Segment(self.periodic_to_physical_tag)

    def initialize(self, mesh):
        super().initialize(mesh)
        self.periodic_to_segment.initialize(mesh, reverse_order=False)

    def apply_boundary_condition(self, boundary_edge_id, fields, **kwargs):
        local_index = self.segment.be_index_to_local_index[boundary_edge_id]
        # element is not part of this boundary
        if local_index == -1:
            return np.zeros_like(fields[:, 0])
        # TODO ASSUMPTION: local_index corresponds to the CORRECT local of the period point on the other domain! DOES NOT HOLD IN GENERAL!
        return fields[:, self.periodic_to_segment.be_elements[local_index]]


class Extrapolation(BoundaryCondition):
    yaml_tag = "Extrapolation"

    def set_default_parameters(self):
        self.physical_tag = None

    def set_runtime_variables(self):
        assert self.physical_tag is not None

        self.segment = Segment(self.physical_tag)

    def initialize(self, mesh):
        self.segment.initialize(mesh)

    def apply_boundary_condition(self, boundary_edge_id, fields, **kwargs):
        local_index = self.segment.be_index_to_local_index[boundary_edge_id]
        # element is not part of this boundary
        if local_index == -1:
            return np.zeros_like(fields[:, 0])
        # TODO ASSUMPTION: local_index corresponds to the CORRECT local of the period point on the other domain! DOES NOT HOLD IN GENERAL!
        return fields[:, self.segment.be_elements[local_index]]


class Custom_extrapolation(Extrapolation):
    yaml_tag = "Custom_extrapolation"

    def set_default_parameters(self):
        self.physical_tag = None
        self.bc_function_dict = None

    def set_runtime_variables(self):
        assert self.physical_tag is not None
        assert self.bc_function_dict is not None

        self.segment = Segment(self.physical_tag)

    def initialize(self, mesh):
        self.segment.initialize(mesh)

    def apply_boundary_condition(self, boundary_edge_id, fields, **kwargs):
        time = kwargs["time"]
        local_index = self.segment.be_index_to_local_index[boundary_edge_id]
        # element is not part of this boundary
        if local_index == -1:
            return np.zeros_like(fields[:, 0])
        # TODO ASSUMPTION: local_index corresponds to the CORRECT local of the period point on the other domain! DOES NOT HOLD IN GENERAL!
        # Extapolation + overwrite fields for given dict
        result = super().apply_boundary_condition(boundary_edge_id, fields, **kwargs)
        for key, bc_func in self.bc_function_dict.items():
            result[key] = eval(bc_func)(
                time,
                fields[:, self.segment.be_elements[local_index]],
                self.segment.be_element_center[local_index],
            )
        return result


class Lambda(BoundaryCondition):
    yaml_tag = "Lambda"

    def set_default_parameters(self):
        self.physical_tag = None
        self.bc_function_dict = None

    def set_runtime_variables(self):
        assert self.physical_tag is not None
        assert self.bc_function_dict is not None

        self.segment = Segment(self.physical_tag)

    def initialize(self, mesh):
        self.segment.initialize(mesh)

    def apply_boundary_condition(self, boundary_edge_id, fields, **kwargs):
        local_index = self.segment.be_index_to_local_index[boundary_edge_id]
        # element is not part of this boundary
        if local_index == -1:
            return np.zeros_like(fields[:, 0])
        # TODO ASSUMPTION: local_index corresponds to the CORRECT local of the period point on the other domain! DOES NOT HOLD IN GENERAL!
        # Extapolation + overwrite fields for given dict
        result = np.array(fields[:, self.segment.be_elements[local_index]])
        for key, bc_func in self.bc_function_dict.items():
            result[key] = eval(bc_func)(
                fields, self.segment.be_element_center[local_index]
            )
        return result


class Wall(BoundaryCondition):
    yaml_tag = "Wall"

    def wall_function(self, Q, n, dim=1):
        result = Q
        result[1 : 1 + dim] -= 2 * (np.dot(result[1 : 1 + dim], n[:dim]) * n[:dim])
        return result

    def set_default_parameters(self):
        self.physical_tag = None
        self.func_wall_Q_normal = None

    def set_runtime_variables(self):
        assert self.physical_tag is not None

        self.segment = Segment(self.physical_tag)
        if self.func_wall_Q_normal is None:
            self.func_wall_Q_normal = self.wall_function

    def initialize(self, mesh):
        self.segment.initialize(mesh)

    def apply_boundary_condition(self, boundary_edge_id, fields, **kwargs):
        local_index = self.segment.be_index_to_local_index[boundary_edge_id]
        # element is not part of this boundary
        if local_index == -1:
            return np.zeros_like(fields[:, 0])
        # TODO ASSUMPTION: local_index corresponds to the CORRECT local of the period point on the other domain! DOES NOT HOLD IN GENERAL!
        # Extapolation + overwrite fields for given dict
        result = np.array(fields[:, self.segment.be_elements[local_index]])
        normal = self.segment.be_normals[local_index]
        result = self.func_wall_Q_normal(result, normal, dim=kwargs["dim"])
        return result


class Wall2D(Wall):
    yaml_tag = "Wall2D"

    def wall_function(self, Q, n, dim=1, offset=1):
        assert dim == 2
        result = np.array(Q)
        mom = np.zeros((offset, dim))
        for d in range(dim):
            mom[:, d] = Q[1 + d * offset : 1 + (d + 1) * offset]
        mom_normal = np.einsum("ik... ,k... -> i...", mom, n[:dim])
        n_trans = np.cross(n, np.array([0.0, 0.0, -1.0]))
        mom_trans = np.einsum("ik... ,k... -> i...", mom, n_trans[:dim])
        mom_normal *= -1.0
        nx = np.array([1.0, 0.0, 0.0])
        ny = np.array([0.0, 1.0, 0.0])
        result[1 : 1 + offset] = mom_normal * np.dot(n, nx)
        result[1 : 1 + offset] += mom_trans * np.dot(n_trans, nx)
        result[1 + offset : 1 + 2 * offset] = mom_normal * np.dot(n, ny)
        result[1 + offset : 1 + 2 * offset] += mom_trans * np.dot(n_trans, ny)
        return result

    def set_default_parameters(self):
        self.physical_tag = None
        self.func_wall_Q_normal_name = "wall_function"

    def set_runtime_variables(self):
        assert self.physical_tag is not None

        self.segment = Segment(self.physical_tag)
        self.func_wall_Q_normal = getattr(self, self.func_wall_Q_normal_name)

    def initialize(self, mesh):
        self.segment.initialize(mesh)

    def apply_boundary_condition(self, boundary_edge_id, fields, **kwargs):
        local_index = self.segment.be_index_to_local_index[boundary_edge_id]
        # element is not part of this boundary
        if local_index == -1:
            return np.zeros_like(fields[:, 0])
        # TODO ASSUMPTION: local_index corresponds to the CORRECT local of the period point on the other domain! DOES NOT HOLD IN GENERAL!
        # Extapolation + overwrite fields for given dict
        result = np.array(fields[:, self.segment.be_elements[local_index]])
        normal = self.segment.be_normals[local_index]
        try:
            offset = kwargs["model"].level + 1
        except:
            offset = 1
        dim = kwargs["model"].dimension
        result = self.func_wall_Q_normal(result, normal, dim=dim, offset=offset)
        return result


class Wall2DStrong(Wall2D):
    yaml_tag = "Wall2DStrong"

    def wall_function(self, Q, n, dim=1, offset=1):
        assert dim == 2
        result = np.array(Q)
        mom = np.zeros((offset, dim))
        for d in range(dim):
            mom[:, d] = Q[1 + d * offset : 1 + (d + 1) * offset]
        mom_normal = np.einsum("ik... ,k... -> i...", mom, n[:dim])
        n_trans = np.cross(n, np.array([0.0, 0.0, 1.0]))
        mom_trans = np.einsum("ik... ,k... -> i...", mom, n_trans[:dim])
        mom_normal *= 0.0
        nx = np.array([1.0, 0.0, 0.0])
        ny = np.array([0.0, 1.0, 0.0])
        result[1 : 1 + offset] = mom_normal * np.dot(n, nx)
        result[1 : 1 + offset] += mom_trans * np.dot(n_trans, nx)
        result[1 + offset : 1 + 2 * offset] = mom_normal * np.dot(n, ny)
        result[1 + offset : 1 + 2 * offset] += mom_trans * np.dot(n_trans, ny)
        return result


def initialize_boundary_conditions(bc_type, mesh):
    for bc in bc_type:
        bc.set_runtime_variables()
        bc.initialize(mesh)


def get_boundary_value(boundary_conditions, boundary_edge_index, fields, **kwargs):
    result = np.zeros_like(fields[:, 0])
    for bc in boundary_conditions:
        result += bc.apply_boundary_condition(boundary_edge_index, fields, **kwargs)
    # check wheater a boundry condition is found
    # assert not np.allclose(result, np.zeros_like(fields[:, 0]))
    return result


def get_boundary_flux_value(boundary_conditions, boundary_edge_index, fields, **kwargs):
    result = fields[:, 0]
    for bc in boundary_conditions:
        result = bc.apply_boundary_condition(boundary_edge_index, fields, **kwargs)
    return result

    initialize_boundary_conditions(bc_type, mesh_obj)
    bc_values = [
        get_boundary_value(bc_type, idx, fields)
        for idx in range(mesh_obj["n_boundary_edges"])
    ]
    for ib, b in enumerate([bc_left, bc_right, bc_bottom, bc_top]):
        # compute boundary edge elements and normals
        for i_edge, element in enumerate(mesh_obj["boundary_edge_element"]):
            if mesh_obj["boundary_edge_tag"][i_edge] == ib:
                b.append(bc_values[i_edge])

    assert np.allclose(np.flip(bc_left, axis=0).T, fields[:, right])
    assert np.allclose(np.flip(bc_right, axis=0).T, fields[:, left])
    assert np.allclose(np.flip(bc_bottom, axis=0).T, fields[:, top])
    assert np.allclose(np.flip(bc_top, axis=0).T, fields[:, bottom])


def test_segment():
    mesh_obj = mesh.create_rectangle(1, 10, [-1.0, 1.0])


test_segment()
