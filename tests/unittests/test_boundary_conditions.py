import numpy as np

from library.boundary_conditions import *
from library.mesh import Mesh


def test_segment_1d():
    mesh = Mesh.create_1d((-1, 1), 10)
    segment_left = Segment.from_mesh(mesh, "left")
    segment_right = Segment.from_mesh(mesh, "right")
    assert (segment_left.be_elements == [0]).all()
    assert (segment_right.be_elements == [9]).all()


def test_segment_2d():
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_mesh(
        os.path.join(main_dir, "meshes/tri_2d/mesh_coarse.msh"),
        "tri",
        2,
        ["left", "right", "top", "bottom"],
    )
    segment = Segment.from_mesh(mesh, "bottom")
    assert segment.be_normals.shape[0] == 8


def test_boundary_condition():
    mesh = Mesh.create_1d((-1, 1), 10)
    bc = BoundaryCondition(physical_tag="left")
    bc.initialize(mesh)
    assert True


def test_boundary_condition_extrapolation():
    mesh = Mesh.create_1d((-1, 1), 10)
    bc = Extrapolation(physical_tag="left")
    Q = np.linspace(1, 2 * mesh.n_elements, 2 * mesh.n_elements).reshape(
        mesh.n_elements, 2
    )
    bc.initialize(mesh)
    print(Q)
    bc.apply_boundary_condition(Q)
    print(Q)
    assert True


if __name__ == "__main__":
    test_segment_1d()
    test_segment_2d()
    test_boundary_condition()
    test_boundary_condition_extrapolation()
