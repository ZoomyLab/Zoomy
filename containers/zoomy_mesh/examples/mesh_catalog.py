"""Example: Download a mesh from the Zoomy mesh catalog."""

import shutil

from zoomy_core.mesh.mesh_catalog import MeshCatalog

catalog = MeshCatalog()
path = catalog.download("square_mesh", size="medium", filetype="msh")
shutil.copy(str(path), "mesh.msh")
