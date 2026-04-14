"""Example: Create a structured 2D quad mesh and write to HDF5."""

from zoomy_core.mesh.base_mesh import BaseMesh

mesh = BaseMesh.create_2d((0, 10, 0, 5), nx=50, ny=25)
mesh.write_to_hdf5("mesh.h5")
