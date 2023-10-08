import numpy as np

from library.boundary_conditions import *
from library.mesh import Mesh


def test_segment_1d():
    mesh = Mesh.create_1d((-1, 1), 10)
    segment_left = Segment.from_mesh(mesh, "left")
    segment_right = Segment.from_mesh(mesh, "right")
    assert (segment_left.element_indices == [0]).all()
    assert (segment_right.element_indices == [9]).all()


def test_segment_2d():
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_mesh(
        os.path.join(main_dir, "meshes/tri_2d/mesh_coarse.msh"),
        "tri",
        2,
        ["left", "right", "top", "bottom"],
    )
    segment = Segment.from_mesh(mesh, "bottom")
    assert segment.face_normals.shape[0] == 8


def test_boundary_condition_initialization():
    mesh = Mesh.create_1d((-1, 1), 10)
    bc = BoundaryCondition(physical_tag="left")
    bc.initialize(mesh)
    assert bc.initialized


def test_boundary_condition_extrapolation():
    mesh = Mesh.create_1d((-1, 1), 10)
    bcs = [Extrapolation(physical_tag="left"), Extrapolation(physical_tag="right")]
    initialize_bc(bcs, mesh)
    n_ghosts = initialize_ghost_cells(bcs, mesh.n_elements)
    n_all_elements = mesh.n_elements + n_ghosts
    Q = np.linspace(1, 2 * n_all_elements, 2 * n_all_elements).reshape(
        n_all_elements, 2
    )
    apply_boundary_conditions(bcs, Q)
    assert np.allclose(Q[-2], np.array([1.0, 2.0], dtype=float))
    assert np.allclose(Q[-1], np.array([19.0, 20.0], dtype=float))
    assert bcs[0].segment.element_indices == [0]
    assert bcs[1].segment.element_indices == [9]
    assert bcs[0].segment.ghost_element_indices == [10]
    assert bcs[1].segment.ghost_element_indices == [11]


def test_boundary_condition_periodic():
    mesh = Mesh.create_1d((-1, 1), 10)
    bcs = [
        Periodic(physical_tag="left", periodic_to_physical_tag="right"),
        Periodic(physical_tag="right", periodic_to_physical_tag="left"),
    ]
    n_ghosts = initialize(bcs, mesh)
    n_all_elements = mesh.n_elements + n_ghosts
    Q = np.linspace(1, 2 * n_all_elements, 2 * n_all_elements).reshape(
        n_all_elements, 2
    )
    apply_boundary_conditions(bcs, Q)
    assert np.allclose(Q[-2], np.array([19.0, 20.0], dtype=float))
    assert np.allclose(Q[-1], np.array([1.0, 2.0], dtype=float))
    assert bcs[0].segment.element_indices == [0]
    assert bcs[1].segment.element_indices == [9]
    assert bcs[0].segment.ghost_element_indices == [10]
    assert bcs[1].segment.ghost_element_indices == [11]


def test_boundary_condition_extrapolation_2d():
    main_dir = os.getenv("SMS")
    bc_tags = ["left", "right", "top", "bottom"]
    bcs = [Extrapolation(physical_tag=tag) for tag in bc_tags]
    mesh = Mesh.load_mesh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
        "quad",
        2,
        bc_tags,
    )
    n_ghosts = initialize(bcs, mesh)
    n_all_elements = mesh.n_elements + n_ghosts
    Q = np.linspace(1, 2 * n_all_elements, 2 * n_all_elements).reshape(
        n_all_elements, 2
    )
    apply_boundary_conditions(bcs, Q)
    assert True


def test_boundary_condition_wall():
    mesh = Mesh.create_1d((-1, 1), 10)
    bcs = [
        Wall(physical_tag="left", momentum_eqns=[1]),
        Wall(physical_tag="right", momentum_eqns=[1]),
    ]
    n_ghosts = initialize(bcs, mesh)
    n_all_elements = mesh.n_elements + n_ghosts
    Q = np.linspace(1, 2 * n_all_elements, 2 * n_all_elements).reshape(
        n_all_elements, 2
    )
    apply_boundary_conditions(bcs, Q)
    assert np.allclose(Q[-2], np.array([1.0, -2.0], dtype=float))
    assert np.allclose(Q[-1], np.array([19.0, -20.0], dtype=float))


def test_boundary_condition_wall_2d():
    main_dir = os.getenv("SMS")
    bc_tags = ["left", "right", "top", "bottom"]
    bcs = [Wall(physical_tag=tag, momentum_eqns=[1, 2]) for tag in bc_tags]
    mesh = Mesh.load_mesh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
        "quad",
        2,
        bc_tags,
    )
    n_ghosts = initialize(bcs, mesh)
    n_all_elements = mesh.n_elements + n_ghosts
    Q = np.linspace(1, 3 * n_all_elements, 3 * n_all_elements).reshape(
        n_all_elements, 3
    )
    apply_boundary_conditions(bcs, Q)
    for i, i_swapped_dim in enumerate([1, 1, 2, 2]):
        assert np.allclose(
            Q[bcs[i].segment.element_indices][:, i_swapped_dim],
            -Q[bcs[i].segment.ghost_element_indices][:, i_swapped_dim],
        )
    assert True


def test_boundary_conditions_collection_class():
    mesh = Mesh.create_1d((-1, 1), 10)

    bcs = BoundaryConditions(
        [Extrapolation(physical_tag="left"), Extrapolation(physical_tag="right")]
    )
    n_ghosts = bcs.initialize(mesh)

    bcs_list = [Extrapolation(physical_tag="left"), Extrapolation(physical_tag="right")]
    initialize_bc(bcs_list, mesh)
    n_ghosts_list = initialize_ghost_cells(bcs_list, mesh.n_elements)

    assert n_ghosts_list == n_ghosts

    n_all_elements = mesh.n_elements + n_ghosts
    Q = np.linspace(1, 2 * n_all_elements, 2 * n_all_elements).reshape(
        n_all_elements, 2
    )
    Q_list = np.linspace(1, 2 * n_all_elements, 2 * n_all_elements).reshape(
        n_all_elements, 2
    )

    apply_boundary_conditions(bcs_list, Q_list)
    bcs.apply(Q)
    assert np.allclose(Q, Q_list)


if __name__ == "__main__":
    test_segment_1d()
    test_segment_2d()
    test_boundary_condition_initialization()
    test_boundary_condition_extrapolation()
    test_boundary_condition_periodic()
    test_boundary_condition_extrapolation_2d()
    test_boundary_condition_wall()
    test_boundary_condition_wall_2d()
    test_boundary_conditions_collection_class()
