import pytest
import os

# from zoomy_core.mesh.fvm_mesh import *
from zoomy_core.mesh.mesh import Mesh
import zoomy_core.misc.io as io
from zoomy_core import misc as misc


@pytest.mark.critical
def test_create_1d_mesh():
    mesh = Mesh.create_1d((-1, 1), 10)
    assert True


@pytest.mark.critical
@pytest.mark.parametrize("mesh_type", ["quad", "tri"])
def test_load_2d_mesh(mesh_type: str):
    main_dir = misc.get_main_directory()

    mesh = Mesh.from_gmsh(
        os.path.join(main_dir, "meshes/{}_2d/mesh_coarse.msh".format(mesh_type)),
    )
    assert True


@pytest.mark.critical
@pytest.mark.parametrize("mesh_type", ["tetra"])
def test_load_3d_mesh(mesh_type: str):
    main_dir = misc.get_main_directory()

    mesh = Mesh.from_gmsh(
        os.path.join(main_dir, "meshes/{}_3d/mesh.msh".format(mesh_type)),
    )
    assert True


@pytest.mark.critical
def test_write_to_hdf5():
    main_dir = misc.get_main_directory()

    mesh = Mesh.from_gmsh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
        "quad",
    )
    filepath = os.path.join(main_dir, "output")
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    mesh.write_to_hdf5(filepath, filename="test.hdf5")
    assert True


@pytest.mark.critical
def test_write_to_file_vtk():
    main_dir = misc.get_main_directory()

    mesh = Mesh.from_gmsh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
        "quad",
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
    main_dir = misc.get_main_directory()

    mesh = Mesh.from_gmsh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
        "quad",
    )
    filepath = os.path.join(main_dir, "output")
    os.makedirs(filepath, exist_ok=True)
    mesh.write_to_hdf5(filepath, filename="test.hdf5")
    filepath = os.path.join(filepath, "test.hdf5")
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
            if not (getattr(mesh, member) == getattr(mesh_loaded, member)):
                print(getattr(mesh, member))
                print(getattr(mesh_loaded, member))
                assert False


@pytest.mark.critical
def test_read_vtk_cell_fields():
    main_dir = misc.get_main_directory()

    mesh = Mesh.from_gmsh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"), "quad"
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
    main_dir = misc.get_main_directory()

    mesh = Mesh.from_gmsh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"), "quad"
    )
    (vertices_3d, elements_3d, mesh_type) = extrude_2d_element_vertices_mesh(
        "quad",
        mesh.vertex_coordinates,
        mesh.element_vertices,
        np.ones(mesh.n_vertices),
        10,
    )
    filepath = os.path.join(main_dir, "output/test_extruded")
    io._write_to_vtk_from_vertices_edges(
        filepath,
        mesh_type,
        vertices_3d,
        elements_3d,
    )


@pytest.mark.critical
@pytest.mark.parametrize("mesh_type", ["quad", "triangle"])
def test_extrude_2d_mesh(mesh_type: str):
    main_dir = misc.get_main_directory()

    mesh = Mesh.from_gmsh(
        os.path.join(main_dir, f"meshes/{mesh_type}_2d/mesh_coarse.msh")
    )
    # mesh = Mesh.from_gmsh(
    #     os.path.join(main_dir, f"meshes/line/mesh.msh")
    # )

    mesh_ext = Mesh.extrude_mesh(mesh, 3)

    filepath = os.path.join(main_dir, f"output/test_{mesh_type}.vtk")
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    # mesh.write_to_vtk(filepath)
    mesh_ext.write_to_vtk(filepath)


@pytest.mark.critical
def test_extract_z_axis_on_extruded_mesh():
    main_dir = misc.get_main_directory()

    print("start")
    path = os.path.join(main_dir, "meshes/channel_straight_long/mesh_3d_5572.msh")
    mesh = Mesh.from_gmsh(path, allow_z_integration=True)
    print("done")
    assert True


if __name__ == "__main__":
    #test_create_1d_mesh()
    #test_load_2d_mesh("quad")
    #test_load_2d_mesh("triangle")
    # test_load_3d_mesh("tetra")
    # test_write_to_hdf5()
    # test_from_hdf5()
    # test_write_to_file_vtk()
    # test_read_vtk_cell_fields()
    # test_extrude_and_write_3d_mesh()
    # test_extrude_2d_mesh('quad')
    test_extrude_2d_mesh('triangle')
    # test_extract_z_axis_on_extruded_mesh()
