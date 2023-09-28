import numpy as np

import library.mesh
import library.boundary_conditions


def test_lambda_1D():
    mesh_obj = mesh.create_rectangle(1, 10, [-1.0, 1.0])
    fields = np.linspace(1, 30, 30).reshape((3, 10))
    bc_type = [
        Lambda(physical_tag=0, bc_function_dict={0: "lambda x,y,z: 10"}),
        Lambda(physical_tag=1, bc_function_dict={1: "lambda x,y,z:-10"}),
    ]
    initialize_boundary_conditions(bc_type, mesh_obj)
    left = 0
    right = 1
    bc_left = get_boundary_value(bc_type, left, fields)
    bc_right = get_boundary_value(bc_type, right, fields)
    res_left = fields[:, 0]
    res_left[0] = 10
    res_right = fields[:, -1]
    res_right[1] = -10
    assert np.allclose(bc_left.flatten(), res_left)
    assert np.allclose(bc_right.flatten(), res_right)


def test_wall_1D():
    mesh_obj = mesh.create_rectangle(1, 10, [-1.0, 1.0])
    fields = np.linspace(1, 30, 30).reshape((3, 10))
    bc_type = [Wall(physical_tag=0), Wall(physical_tag=1)]
    initialize_boundary_conditions(bc_type, mesh_obj)
    left = 0
    right = 1
    bc_left = get_boundary_value(bc_type, left, fields, dim=1)
    bc_right = get_boundary_value(bc_type, right, fields, dim=1)
    res_left = fields[:, 0]
    res_left[1] *= -1.0
    res_right = fields[:, -1]
    res_right[1] *= -1.0
    assert np.allclose(bc_left.flatten(), res_left)
    assert np.allclose(bc_right.flatten(), res_right)


def test_wall_2D():
    mesh_obj = mesh.create_rectangle(2, 100, [-1, 1, -1, 1])
    mesh_ids = np.linspace(0, 99, 100, dtype=int).reshape((10, 10))
    left = []
    right = []
    bottom = []
    top = []
    bc_left = []
    bc_right = []
    bc_bottom = []
    bc_top = []

    def wall_func(Q, n, **kwargs):
        dim = 2
        result = Q
        result[1 : 1 + dim] -= 2 * (np.dot(result[1 : 1 + dim], n[:dim]) * n[:dim])
        return result

    for ib, b in enumerate([left, right, bottom, top]):
        # compute boundary edge elements and normals
        for i_edge, element in enumerate(mesh_obj["boundary_edge_element"]):
            if mesh_obj["boundary_edge_tag"][i_edge] == ib:
                b.append(int(mesh_obj["boundary_edge_element"][i_edge]))

    fields = np.linspace(1, 300, 300).reshape((3, 100))
    bc_type = [
        Wall(physical_tag=0, func_wall_Q_normal=wall_func),
        Wall(physical_tag=1, func_wall_Q_normal=wall_func),
        Wall(physical_tag=2, func_wall_Q_normal=wall_func),
        Wall(physical_tag=3, func_wall_Q_normal=wall_func),
    ]

    initialize_boundary_conditions(bc_type, mesh_obj)
    bc_values = [
        get_boundary_value(bc_type, idx, fields, dim=2)
        for idx in range(mesh_obj["n_boundary_edges"])
    ]
    for ib, b in enumerate([bc_left, bc_right, bc_bottom, bc_top]):
        # compute boundary edge elements and normals
        for i_edge, element in enumerate(mesh_obj["boundary_edge_element"]):
            if mesh_obj["boundary_edge_tag"][i_edge] == ib:
                result = bc_values[i_edge]
                if ib == 0 or ib == 1:
                    result[1] *= -1
                else:
                    result[2] *= -1
                b.append(result)

    assert np.allclose(np.transpose(bc_left), fields[:, left])
    assert np.allclose(np.transpose(bc_right), fields[:, right])
    assert np.allclose(np.transpose(bc_bottom), fields[:, bottom])
    assert np.allclose(np.transpose(bc_top), fields[:, top])


def test_periodic_1D():
    mesh_obj = mesh.create_mesh()
    fields = np.linspace(1, 30, 30).reshape((3, 10))
    bc_type = [
        Periodic(physical_tag=0, periodic_to_physical_tag=1),
        Periodic(physical_tag=1, periodic_to_physical_tag=0),
    ]
    initialize_boundary_conditions(bc_type, mesh_obj)
    left = 0
    right = 1
    bc_left = get_boundary_value(bc_type, left, fields)
    bc_right = get_boundary_value(bc_type, right, fields)
    assert np.allclose(bc_left.flatten(), fields[:, -1])
    assert np.allclose(bc_right.flatten(), fields[:, 0])


def test_periodic_2D():
    mesh_obj = mesh.create_rectangle(2, 100, [-1, 1, -1, 1])
    mesh_ids = np.linspace(0, 99, 100, dtype=int).reshape((10, 10))
    left = []
    right = []
    bottom = []
    top = []
    bc_left = []
    bc_right = []
    bc_bottom = []
    bc_top = []

    for ib, b in enumerate([left, right, bottom, top]):
        # compute boundary edge elements and normals
        for i_edge, element in enumerate(mesh_obj["boundary_edge_element"]):
            if mesh_obj["boundary_edge_tag"][i_edge] == ib:
                b.append(int(mesh_obj["boundary_edge_element"][i_edge]))

    fields = np.linspace(1, 300, 300).reshape((3, 100))
    bc_type = [
        Periodic(physical_tag=0, periodic_to_physical_tag=1),
        Periodic(physical_tag=1, periodic_to_physical_tag=0),
        Periodic(physical_tag=2, periodic_to_physical_tag=3),
        Periodic(physical_tag=3, periodic_to_physical_tag=2),
    ]


def test_extrapolation_1D():
    mesh_obj = mesh.create_rectangle(1, 10, [-1.0, 1.0])
    fields = np.linspace(1, 30, 30).reshape((3, 10))
    bc_type = [Extrapolation(physical_tag=0), Extrapolation(physical_tag=1)]
    initialize_boundary_conditions(bc_type, mesh_obj)
    left = 0
    right = 1
    bc_left = get_boundary_value(bc_type, left, fields)
    bc_right = get_boundary_value(bc_type, right, fields)
    assert np.allclose(bc_left.flatten(), fields[:, 0])
    assert np.allclose(bc_right.flatten(), fields[:, -1])


def test_extrapolation_2D():
    mesh_obj = mesh.create_rectangle(2, 100, [-1, 1, -1, 1])
    mesh_ids = np.linspace(0, 99, 100, dtype=int).reshape((10, 10))
    left = []
    right = []
    bottom = []
    top = []
    bc_left = []
    bc_right = []
    bc_bottom = []
    bc_top = []

    for ib, b in enumerate([left, right, bottom, top]):
        # compute boundary edge elements and normals
        for i_edge, element in enumerate(mesh_obj["boundary_edge_element"]):
            if mesh_obj["boundary_edge_tag"][i_edge] == ib:
                b.append(int(mesh_obj["boundary_edge_element"][i_edge]))

    fields = np.linspace(1, 300, 300).reshape((3, 100))
    bc_type = [
        Extrapolation(physical_tag=0),
        Extrapolation(physical_tag=1),
        Extrapolation(physical_tag=2),
        Extrapolation(physical_tag=3),
    ]

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

    assert np.allclose(np.transpose(bc_left), fields[:, left])
    assert np.allclose(np.transpose(bc_right), fields[:, right])
    assert np.allclose(np.transpose(bc_bottom), fields[:, bottom])
    assert np.allclose(np.transpose(bc_top), fields[:, top])


def test_extrapolation_2D():
    mesh_obj = mesh.create_rectangle(2, 100, [-1, 1, -1, 1])
    mesh_ids = np.linspace(0, 99, 100, dtype=int).reshape((10, 10))
    left = []
    right = []
    bottom = []
    top = []
    bc_left = []
    bc_right = []
    bc_bottom = []
    bc_top = []

    for ib, b in enumerate([left, right, bottom, top]):
        # compute boundary edge elements and normals
        for i_edge, element in enumerate(mesh_obj["boundary_edge_element"]):
            if mesh_obj["boundary_edge_tag"][i_edge] == ib:
                b.append(int(mesh_obj["boundary_edge_element"][i_edge]))

    fields = np.linspace(1, 300, 300).reshape((3, 100))
    bc_type = [
        Extrapolation(physical_tag=0),
        Extrapolation(physical_tag=1),
        Extrapolation(physical_tag=2),
        Extrapolation(physical_tag=3),
    ]

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

    assert np.allclose(np.transpose(bc_left), fields[:, left])
    assert np.allclose(np.transpose(bc_right), fields[:, right])
    assert np.allclose(np.transpose(bc_bottom), fields[:, bottom])
    assert np.allclose(np.transpose(bc_top), fields[:, top])


if __name__ == "__main__":
    test_periodic_1D()
    # test_extrapolation_1D()
    # test_lambda_1D()
    # test_wall_1D()

    # test_periodic_2D()
    # test_extrapolation_2D()
    # test_wall_2D()
