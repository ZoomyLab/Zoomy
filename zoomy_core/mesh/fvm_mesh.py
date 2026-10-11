"""FVMMesh — BaseMesh with precomputed geometry for FVM solvers.

Caches cell centers, volumes, inradii, face normals, face volumes,
and face centers so they are not recomputed every access.
"""

from __future__ import annotations

import numpy as np
import param

from zoomy_core.mesh.base_mesh import BaseMesh


class FVMMesh(BaseMesh):
    """BaseMesh + precomputed cell/face geometry."""

    # Precomputed geometry (overrides the on-the-fly methods in BaseMesh)
    _cell_centers = param.Array(default=None, allow_None=True)
    _cell_volumes = param.Array(default=None, allow_None=True)
    _cell_inradius = param.Array(default=None, allow_None=True)
    _face_normals = param.Array(default=None, allow_None=True)
    _face_volumes = param.Array(default=None, allow_None=True)
    _face_centers = param.Array(default=None, allow_None=True)

    def cell_centers_computed(self) -> np.ndarray:
        return self._cell_centers

    def cell_volumes_computed(self) -> np.ndarray:
        return self._cell_volumes

    def cell_inradius_computed(self) -> np.ndarray:
        return self._cell_inradius

    def face_normals_computed(self) -> np.ndarray:
        return self._face_normals

    def face_volumes_computed(self) -> np.ndarray:
        return self._face_volumes

    def face_centers_computed(self) -> np.ndarray:
        return self._face_centers

    # Public property accessors (compatible with old Mesh flat attributes)
    @property
    def cell_centers(self):
        return self._cell_centers

    @property
    def cell_volumes(self):
        return self._cell_volumes

    @property
    def cell_inradius(self):
        return self._cell_inradius

    @property
    def face_normals(self):
        return self._face_normals

    @property
    def face_volumes(self):
        return self._face_volumes

    @property
    def face_centers(self):
        return self._face_centers


    @classmethod
    def from_base(cls, base: BaseMesh) -> "FVMMesh":
        """Build an FVMMesh by computing geometry from a BaseMesh."""
        # Copy all base fields
        fvm = cls(
            dimension=base.dimension,
            type=base.type,
            n_cells=base.n_cells,
            n_inner_cells=base.n_inner_cells,
            n_faces=base.n_faces,
            n_vertices=base.n_vertices,
            n_boundary_faces=base.n_boundary_faces,
            n_faces_per_cell=base.n_faces_per_cell,
            vertex_coordinates=base.vertex_coordinates,
            cell_vertices=base.cell_vertices,
            cell_faces=base.cell_faces,
            face_cells=base.face_cells,
            cell_neighbors=base.cell_neighbors,
            boundary_face_cells=base.boundary_face_cells,
            boundary_face_ghosts=base.boundary_face_ghosts,
            boundary_face_function_numbers=base.boundary_face_function_numbers,
            boundary_face_physical_tags=base.boundary_face_physical_tags,
            boundary_face_face_indices=base.boundary_face_face_indices,
            boundary_conditions_sorted_physical_tags=base.boundary_conditions_sorted_physical_tags,
            boundary_conditions_sorted_names=base.boundary_conditions_sorted_names,
            z_ordering=base.z_ordering,
        )

        from zoomy_core.mesh import geometry_xp
        if str(base.type) in geometry_xp._SUPPORTED:   # vectorised
            g = geometry_xp.cell_face_geometry(fvm)
            fvm._face_centers = np.asarray(g["face_centers"])
            fvm._face_normals = np.asarray(g["face_normals"])
            fvm._cell_centers = np.asarray(g["cell_centers"])
            fvm._face_volumes = np.asarray(g["face_volumes"])
            fvm._cell_volumes = np.asarray(g["cell_volumes"])
            fvm._cell_inradius = np.asarray(g["cell_inradius"])
            return fvm

        # Compute geometry.  Order matters because of the
        # cell_centers ↔ face_normals dependency:
        #   * 2D+: cell_centers needs face_normals (for ghost-cell
        #     reflection across the boundary face).  face_normals (2D+
        #     branch in BaseMesh) computes inner-cell centers inline,
        #     so it has no cell_centers dependency.
        #   * 1D: face_normals needs cell_centers (for sign of normal).
        #     cell_centers (1D) does not need face_normals.
        # Compute face_normals first in 2D+, cell_centers first in 1D.
        fvm._face_centers = BaseMesh.face_centers_computed(fvm)
        if fvm.dimension > 1:
            fvm._face_normals = BaseMesh.face_normals_computed(fvm)
            fvm._cell_centers = BaseMesh.cell_centers_computed(fvm)
        else:
            fvm._cell_centers = BaseMesh.cell_centers_computed(fvm)
            fvm._face_normals = BaseMesh.face_normals_computed(fvm)
        fvm._face_volumes = BaseMesh.face_volumes_computed(fvm)
        fvm._cell_volumes = BaseMesh.cell_volumes_computed(fvm)
        fvm._cell_inradius = BaseMesh.cell_inradius_computed(fvm)

        return fvm

    @classmethod
    def from_msh(cls, filepath: str) -> "FVMMesh":
        """Load .msh and build FVMMesh with precomputed geometry."""
        base = BaseMesh.from_msh(filepath)
        return cls.from_base(base)

    @classmethod
    def from_hdf5(cls, filepath: str) -> "FVMMesh":
        """Load BaseMesh from H5 and compute geometry."""
        base = BaseMesh.from_hdf5(filepath)
        return cls.from_base(base)

    @classmethod
    def create_1d(cls, domain: tuple, n_inner_cells: int) -> "FVMMesh":
        base = BaseMesh.create_1d(domain, n_inner_cells)
        return cls.from_base(base)

    @classmethod
    def create_2d(cls, domain: tuple, nx: int, ny: int) -> "FVMMesh":
        base = BaseMesh.create_2d(domain, nx, ny)
        return cls.from_base(base)

    @classmethod
    def create_3d(cls, domain: tuple, nx: int, ny: int, nz: int) -> "FVMMesh":
        base = BaseMesh.create_3d(domain, nx, ny, nz)
        return cls.from_base(base)
