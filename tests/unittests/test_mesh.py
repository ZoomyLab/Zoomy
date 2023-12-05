import pytest

from library.fvm_mesh import *
from library.misc import all_class_members_identical


@pytest.mark.critical
def test_create_1d_mesh():
    mesh = Mesh.create_1d((-1, 1), 10)
    assert True


@pytest.mark.critical
@pytest.mark.parametrize("mesh_type", ["quad", "tri"])
def test_load_2d_mesh(mesh_type: str):
    main_dir = os.getenv("SMS")
    # mesh = Mesh.load_mesh(
    #     os.path.join(main_dir, "meshes/{}_2d/mesh_coarse.msh".format(mesh_type)),
    #     mesh_type,
    #     2,
    #     ["left", "right", "top", "bottom"],
    # )
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/{}_2d/mesh_coarse.msh".format(mesh_type)),
        mesh_type,
    )
    assert True



@pytest.mark.critical
@pytest.mark.parametrize("mesh_type", ["tetra"])
def test_load_3d_mesh(mesh_type: str):
    main_dir = os.getenv("SMS")
    # mesh = Mesh.load_mesh(
    #     os.path.join(main_dir, "meshes/{}_3d/mesh.msh".format(mesh_type)),
    #     mesh_type,
    #     3,
    #     ["left", "right", "top", "bottom", "front", "back"],
    # )
    mesh = Mesh.load_gmsh(
        os.path.join(main_dir, "meshes/{}_3d/mesh.msh".format(mesh_type)),
        mesh_type,
    )
    assert True


@pytest.mark.critical
def test_write_to_hdf5():
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_mesh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
        "quad",
        2,
        ["left", "right", "top", "bottom"],
    )
    filepath = os.path.join(main_dir, "output/test.hdf5")
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    mesh.write_to_hdf5(filepath)
    assert True


@pytest.mark.critical
def test_write_to_file_vtk():
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_mesh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
        "quad",
        2,
        ["left", "right", "top", "bottom"],
    )
    filepath = os.path.join(main_dir, "output/test.vtk")
    cell_data = np.linspace(1, 2 * mesh.n_elements, 2 * mesh.n_elements).reshape(
        mesh.n_elements, 2
    )
    point_data = {
        "0": np.linspace(1, mesh.n_vertices, mesh.n_vertices),
        "1": np.linspace(1, mesh.n_vertices, mesh.n_vertices),
    }
    field_names = ["field_1", "field_2"]
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    mesh.write_to_file_vtk(
        filepath, fields=cell_data, field_names=field_names, point_data=point_data
    )
    assert True


@pytest.mark.critical
def test_from_hdf5():
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_mesh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
        "quad",
        2,
        ["left", "right", "top", "bottom"],
    )
    filepath = os.path.join(main_dir, "output/test.hdf5")
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    mesh.write_to_hdf5(filepath)
    mesh_loaded = Mesh.from_hdf5(filepath)
    members = [
        attr
        for attr in dir(mesh)
        if not callable(getattr(mesh, attr)) and not attr.startswith("__")
    ]
    for member in members:
        m_mesh = getattr(mesh, member)
        m_mesh_loaded = getattr(mesh_loaded, member)
        if type(m_mesh) == np.ndarray:
            if not ((getattr(mesh, member) == getattr(mesh_loaded, member)).all()):
                print(getattr(mesh, member))
                print(getattr(mesh_loaded, member))
                assert False
        else:
            if not ((getattr(mesh, member) == getattr(mesh_loaded, member))):
                print(getattr(mesh, member))
                print(getattr(mesh_loaded, member))
                assert False


@pytest.mark.critical
def test_read_vtk_cell_fields():
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_mesh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
        "quad",
        2,
        ["left", "right", "top", "bottom"],
    )
    cell_data = np.linspace(1, 2 * mesh.n_elements, 2 * mesh.n_elements).reshape(
        mesh.n_elements, 2
    )
    filepath = os.path.join(main_dir, "output/test.vtk")
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    mesh.write_to_file_vtk(filepath, fields=cell_data)
    fields = read_vtk_cell_fields(filepath, 4, [0, 2])
    assert True


@pytest.mark.critical
def test_extrude_and_write_3d_mesh():
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_mesh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
        "quad",
        2,
        ["left", "right", "top", "bottom"],
    )
    (vertices_3d, elements_3d, mesh_type) = extrude_2d_element_vertices_mesh(
        "quad",
        mesh.vertex_coordinates,
        mesh.element_vertices,
        np.ones(mesh.n_vertices),
        10,
    )
    filepath = os.path.join(main_dir, "output/test.vtk")
    write_to_file_vtk_from_vertices_edges(
        filepath,
        mesh_type,
        vertices_3d,
        elements_3d,
    )


if __name__ == "__main__":
    # test_create_1d_mesh()
    # test_load_2d_mesh("quad")
    test_load_2d_mesh("triangle")
    # test_load_3d_mesh("tetra")
    # test_write_to_hdf5()
    # test_from_hdf5()
    # test_write_to_file_vtk()
    # test_read_vtk_cell_fields()
    # test_extrude_and_write_3d_mesh()
