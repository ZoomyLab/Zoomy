"""Example: Create a uniform 1D mesh and write to HDF5."""

from zoomy_core.mesh.base_mesh import BaseMesh

mesh = BaseMesh.create_1d(domain=(0, 1), n_inner_cells=200)
mesh.write_to_hdf5("mesh.h5")
