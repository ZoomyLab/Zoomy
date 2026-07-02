"""zoomy_prepost — shared pre/postprocessing conversions.

The general place for format conversion so backends don't each ship custom
writers. Today: VTK / ParaView ``.pvd`` -> zoomy HDF5 (readable by
``zoomy_plotting`` / the GUI). Mesh conversion helpers can grow here too.
"""
from .vtk import vtk_to_hdf5

__all__ = ["vtk_to_hdf5"]
