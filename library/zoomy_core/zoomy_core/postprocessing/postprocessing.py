# The former `vtk_interpolate_to_3d` (bit-rotted: it called a non-existent
# `zoomy_core.mesh.lsq_reconstruction.Mesh`) has been removed. Use
# `zoomy_prepost.steps.lift3d` instead — it lifts a depth-averaged result to a
# (d+1)-D VTK series through the model's symbolic `interpolate_to_3d`.
"""Module `zoomy_core.postprocessing.postprocessing`."""
