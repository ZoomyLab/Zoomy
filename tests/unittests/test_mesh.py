from library.mesh import *
from library.misc import all_class_members_identical


def test_create_1d_mesh():
    mesh = Mesh.create_1d((-1, 1), 10)
    assert True


def test_load_2d_mesh():
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_mesh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
        "quad",
        2,
        ["left", "right", "top", "bottom"],
    )
    assert True

def test_write_to_hdf5():
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_mesh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
        "quad",
        2,
        ["left", "right", "top", "bottom"],
    )
    filepath = os.path.join(main_dir, 'output/test.hdf5')
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    mesh.write_to_hdf5(filepath)
    assert True

def test_from_hdf5():
    main_dir = os.getenv("SMS")
    mesh = Mesh.load_mesh(
        os.path.join(main_dir, "meshes/quad_2d/mesh_coarse.msh"),
        "quad",
        2,
        ["left", "right", "top", "bottom"],
    )
    filepath = os.path.join(main_dir, 'output/test.hdf5')
    os.makedirs(os.path.split(filepath)[0], exist_ok=True)
    mesh.write_to_hdf5(filepath)
    mesh_loaded = Mesh.from_hdf5(filepath)
    members = [attr for attr in dir(mesh) if not callable(getattr(mesh, attr)) and not attr.startswith("__")]
    for member in members:
        m_mesh = getattr(mesh, member)
        m_mesh_loaded = getattr(mesh_loaded, member)
        if type(m_mesh) == np.ndarray:
            if not ( (getattr(mesh, member) == getattr(mesh_loaded, member)).all()):
                print(getattr(mesh, member))
                print(getattr(mesh_loaded, member))
                assert False
        else:
            if not ( (getattr(mesh, member) == getattr(mesh_loaded, member))):
                print(getattr(mesh, member))
                print(getattr(mesh_loaded, member))
                assert False



test_create_1d_mesh()
test_load_2d_mesh()
test_write_to_hdf5()
test_from_hdf5()