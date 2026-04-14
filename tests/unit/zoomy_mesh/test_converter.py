"""Round-trip tests for zoomy_mesh.converter (h5 <-> msh)."""

from __future__ import annotations

import tempfile
import os

import numpy as np
import pytest

from zoomy_core.mesh import BaseMesh
from zoomy_mesh.converter import h5_to_msh, msh_to_h5


# ── helpers ──────────────────────────────────────────────────────────────────

def _roundtrip_h5_msh_h5(mesh: BaseMesh) -> BaseMesh:
    """h5 -> msh -> h5 round-trip; return reloaded mesh."""
    with tempfile.TemporaryDirectory() as td:
        h5_a = os.path.join(td, "orig.h5")
        msh = os.path.join(td, "conv.msh")
        h5_b = os.path.join(td, "back.h5")

        mesh.write_to_hdf5(h5_a)
        h5_to_msh(h5_a, msh)
        msh_to_h5(msh, h5_b)

        return BaseMesh.from_hdf5(h5_b)


# ── tests ────────────────────────────────────────────────────────────────────

class TestH5ToMsh2DQuad:
    """2D quad mesh round-trip."""

    @pytest.fixture()
    def meshes(self):
        orig = BaseMesh.create_2d((0, 1, 0, 1), nx=5, ny=5)
        reloaded = _roundtrip_h5_msh_h5(orig)
        return orig, reloaded

    def test_inner_cell_count(self, meshes):
        orig, reloaded = meshes
        assert reloaded.n_inner_cells == orig.n_inner_cells

    def test_boundary_face_count(self, meshes):
        orig, reloaded = meshes
        assert reloaded.n_boundary_faces == orig.n_boundary_faces

    def test_boundary_names_preserved(self, meshes):
        orig, reloaded = meshes
        assert set(reloaded.boundary_conditions_sorted_names) == set(
            orig.boundary_conditions_sorted_names
        )

    def test_vertex_count(self, meshes):
        orig, reloaded = meshes
        assert reloaded.n_vertices == orig.n_vertices

    def test_dimension(self, meshes):
        orig, reloaded = meshes
        assert reloaded.dimension == orig.dimension

    def test_mesh_type(self, meshes):
        orig, reloaded = meshes
        assert reloaded.type == orig.type


class TestH5ToMsh2DQuadRectangular:
    """Non-square 2D quad mesh."""

    @pytest.fixture()
    def meshes(self):
        orig = BaseMesh.create_2d((0, 2, 0, 0.5), nx=8, ny=3)
        reloaded = _roundtrip_h5_msh_h5(orig)
        return orig, reloaded

    def test_inner_cell_count(self, meshes):
        orig, reloaded = meshes
        assert reloaded.n_inner_cells == orig.n_inner_cells

    def test_boundary_names_preserved(self, meshes):
        orig, reloaded = meshes
        assert set(reloaded.boundary_conditions_sorted_names) == set(
            orig.boundary_conditions_sorted_names
        )


class TestH5ToMsh1D:
    """1D mesh round-trip."""

    @pytest.fixture()
    def meshes(self):
        orig = BaseMesh.create_1d((0, 10), n_inner_cells=20)
        reloaded = _roundtrip_h5_msh_h5(orig)
        return orig, reloaded

    def test_inner_cell_count(self, meshes):
        orig, reloaded = meshes
        assert reloaded.n_inner_cells == orig.n_inner_cells

    def test_boundary_names_preserved(self, meshes):
        orig, reloaded = meshes
        assert set(reloaded.boundary_conditions_sorted_names) == set(
            orig.boundary_conditions_sorted_names
        )

    def test_vertex_count(self, meshes):
        orig, reloaded = meshes
        assert reloaded.n_vertices == orig.n_vertices


class TestMshToH5:
    """msh -> h5 direction (via BaseMesh.from_msh under the hood)."""

    def test_direct_msh_roundtrip(self):
        """Write a mesh to .msh, then msh_to_h5, then reload from h5."""
        mesh = BaseMesh.create_2d((0, 1, 0, 1), nx=4, ny=4)
        with tempfile.TemporaryDirectory() as td:
            h5_a = os.path.join(td, "a.h5")
            msh = os.path.join(td, "b.msh")
            h5_b = os.path.join(td, "c.h5")

            mesh.write_to_hdf5(h5_a)
            h5_to_msh(h5_a, msh)
            msh_to_h5(msh, h5_b)

            reloaded = BaseMesh.from_hdf5(h5_b)
            assert reloaded.n_inner_cells == mesh.n_inner_cells
            assert reloaded.n_boundary_faces == mesh.n_boundary_faces


class TestBoundaryTagConsistency:
    """Verify that boundary physical tags map correctly after round-trip."""

    def test_tag_face_count_per_boundary(self):
        """Each boundary name should have the same number of faces."""
        mesh = BaseMesh.create_2d((0, 1, 0, 1), nx=6, ny=4)
        reloaded = _roundtrip_h5_msh_h5(mesh)

        for name in mesh.boundary_conditions_sorted_names:
            idx_orig = mesh.boundary_conditions_sorted_names.index(name)
            tag_orig = mesh.boundary_conditions_sorted_physical_tags[idx_orig]
            count_orig = np.sum(mesh.boundary_face_physical_tags == tag_orig)

            idx_reload = reloaded.boundary_conditions_sorted_names.index(name)
            tag_reload = reloaded.boundary_conditions_sorted_physical_tags[idx_reload]
            count_reload = np.sum(
                reloaded.boundary_face_physical_tags == tag_reload
            )

            assert count_orig == count_reload, (
                f"Boundary '{name}': {count_orig} faces orig vs "
                f"{count_reload} reloaded"
            )
