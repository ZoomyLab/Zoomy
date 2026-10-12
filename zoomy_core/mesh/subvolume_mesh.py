"""SubVolumeMesh: FVMMesh with cached face subvolumes for flux splitting."""

from __future__ import annotations

import numpy as np
import param

from zoomy_core.mesh.base_mesh import BaseMesh
from zoomy_core.mesh.fvm_mesh import FVMMesh


class SubVolumeMesh(FVMMesh):
    """FVMMesh plus precomputed ``face_subvolumes``."""

    _face_subvolumes = param.Array(default=None, allow_None=True)

    def face_subvolumes_computed(self) -> np.ndarray:
        """Return the cached face subvolumes."""
        return self._face_subvolumes

    @classmethod
    def from_fvm(cls, fvm: FVMMesh) -> "SubVolumeMesh":
        """Build a SubVolumeMesh from an FVMMesh by computing its face subvolumes."""
        svm = cls(
            dimension=fvm.dimension,
            type=fvm.type,
            n_cells=fvm.n_cells,
            n_inner_cells=fvm.n_inner_cells,
            n_faces=fvm.n_faces,
            n_vertices=fvm.n_vertices,
            n_boundary_faces=fvm.n_boundary_faces,
            n_faces_per_cell=fvm.n_faces_per_cell,
            vertex_coordinates=fvm.vertex_coordinates,
            cell_vertices=fvm.cell_vertices,
            cell_faces=fvm.cell_faces,
            face_cells=fvm.face_cells,
            cell_neighbors=fvm.cell_neighbors,
            boundary_face_cells=fvm.boundary_face_cells,
            boundary_face_ghosts=fvm.boundary_face_ghosts,
            boundary_face_function_numbers=fvm.boundary_face_function_numbers,
            boundary_face_physical_tags=fvm.boundary_face_physical_tags,
            boundary_face_face_indices=fvm.boundary_face_face_indices,
            boundary_conditions_sorted_physical_tags=fvm.boundary_conditions_sorted_physical_tags,
            boundary_conditions_sorted_names=fvm.boundary_conditions_sorted_names,
            z_ordering=fvm.z_ordering,
            _cell_centers=fvm._cell_centers,
            _cell_volumes=fvm._cell_volumes,
            _cell_inradius=fvm._cell_inradius,
            _face_normals=fvm._face_normals,
            _face_volumes=fvm._face_volumes,
            _face_centers=fvm._face_centers,
        )
        svm._face_subvolumes = BaseMesh.face_subvolumes_computed(svm)
        return svm

    @classmethod
    def from_msh(cls, filepath: str) -> "SubVolumeMesh":
        """Load a gmsh ``.msh`` file as a SubVolumeMesh."""
        fvm = FVMMesh.from_msh(filepath)
        return cls.from_fvm(fvm)

    @classmethod
    def from_hdf5(cls, filepath: str) -> "SubVolumeMesh":
        """Load an HDF5 mesh as a SubVolumeMesh."""
        fvm = FVMMesh.from_hdf5(filepath)
        return cls.from_fvm(fvm)

    @classmethod
    def create_1d(cls, domain: tuple, n_inner_cells: int) -> "SubVolumeMesh":
        """Structured 1D SubVolumeMesh on ``domain`` with ``n_inner_cells`` cells."""
        fvm = FVMMesh.create_1d(domain, n_inner_cells)
        return cls.from_fvm(fvm)

    @classmethod
    def create_2d(cls, domain: tuple, nx: int, ny: int) -> "SubVolumeMesh":
        """Structured 2D SubVolumeMesh on ``domain`` with ``nx`` x ``ny`` cells."""
        fvm = FVMMesh.create_2d(domain, nx, ny)
        return cls.from_fvm(fvm)

    @classmethod
    def create_3d(cls, domain: tuple, nx: int, ny: int, nz: int) -> "SubVolumeMesh":
        """Structured 3D SubVolumeMesh on ``domain`` with ``nx`` x ``ny`` x ``nz`` cells."""
        fvm = FVMMesh.create_3d(domain, nx, ny, nz)
        return cls.from_fvm(fvm)
