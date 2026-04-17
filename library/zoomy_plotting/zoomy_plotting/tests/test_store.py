"""Unit tests for SimulationStore + Zstruct + field resolution."""

from __future__ import annotations

import numpy as np
import pytest
import sympy

from zoomy_plotting.store import SimulationStore, Zstruct


# --------------------------------------------------------------------------
# Zstruct
# --------------------------------------------------------------------------
def test_zstruct_attribute_and_string_access():
    z = Zstruct({"h": 0, "u": 1, "v": 2})
    assert z.h == 0
    assert z["u"] == 1
    assert "h" in z
    assert list(z.keys()) == ["h", "u", "v"]
    assert set(z.values()) == {0, 1, 2}


def test_zstruct_repr_is_stable():
    z = Zstruct({"b": 3, "a": 1})
    r = repr(z)
    assert "Zstruct(" in r
    # Sorted keys for deterministic repr.
    assert r.index("'a'") < r.index("'b'")


# --------------------------------------------------------------------------
# SimulationStore shape invariants
# --------------------------------------------------------------------------
def test_post_init_rejects_wrong_axis_order():
    # (dim=2, but vertices shape is (2, n) — needs transpose first).
    bad_verts = np.zeros((2, 5))
    cells = np.array([[0, 1, 2]])
    with pytest.raises(ValueError, match="vertices must have shape"):
        SimulationStore(
            dim=2, cell_type="triangle",
            vertices=bad_verts, cells=cells,
        )


def test_post_init_rejects_bad_dim():
    with pytest.raises(ValueError, match="dim must be 1, 2, or 3"):
        SimulationStore(
            dim=4, cell_type="triangle",
            vertices=np.zeros((3, 4)),
            cells=np.array([[0, 1, 2]]),
        )


# --------------------------------------------------------------------------
# Field resolution: int, str, sympy.Symbol
# --------------------------------------------------------------------------
def test_field_resolution_all_forms(store_2d_quad):
    s = store_2d_quad
    assert s._resolve(0, s.field) == 0
    assert s._resolve("h", s.field) == 0
    assert s._resolve("u", s.field) == 1
    assert s._resolve(sympy.Symbol("h"), s.field) == 0
    assert s._resolve(np.int64(1), s.field) == 1


def test_field_resolution_missing_name_lists_options(store_2d_quad):
    s = store_2d_quad
    with pytest.raises(KeyError, match=r"'\['h', 'u'\]'|h.*u|u.*h"):
        s._resolve("does_not_exist", s.field)


def test_field_resolution_rejects_unknown_type(store_2d_quad):
    s = store_2d_quad
    with pytest.raises(TypeError, match="unsupported field spec"):
        s._resolve(3.14, s.field)


# --------------------------------------------------------------------------
# Lazy cell access
# --------------------------------------------------------------------------
def test_get_cell_returns_expected_values(store_2d_quad):
    s = store_2d_quad
    values = s.get_cell(0, "h")
    assert values.shape == (s.n_cells,)
    # h at t=0 is cell-center x; cell count is 25.
    assert np.isfinite(values).all()


def test_get_cell_with_sympy_symbol(store_2d_quad):
    s = store_2d_quad
    v_sym = s.get_cell(0, sympy.Symbol("u"))
    v_idx = s.get_cell(0, 1)
    np.testing.assert_array_equal(v_sym, v_idx)


def test_get_cell_without_reader_raises(store_2d_quad):
    s = store_2d_quad
    s._cell_reader = None
    with pytest.raises(RuntimeError, match="cell reader"):
        s.get_cell(0, "h")


def test_cell_centers_cached(store_2d_quad):
    s = store_2d_quad
    c1 = s.cell_centers
    c2 = s.cell_centers
    assert c1 is c2
    assert c1.shape == (s.n_cells, 2)


# --------------------------------------------------------------------------
# n_snapshots / n_cells / n_vertices
# --------------------------------------------------------------------------
def test_counters(store_2d_quad):
    s = store_2d_quad
    assert s.n_snapshots == 3
    assert s.n_cells == 25
    assert s.n_vertices == 36
