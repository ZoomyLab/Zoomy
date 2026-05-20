"""``SystemModel.from_model`` auto-scans every non-state Function atom
and every Derivative atom, routing them to ``aux_state`` with a
structured :attr:`aux_registry`.  Solvers walk the registry to compute
per-cell aux values.

This test suite verifies the auto-scan + registry contract.
"""
from __future__ import annotations

import itertools

import numpy as np
import sympy as sp
import pytest

from zoomy_core.model.models.vam_galerkin import VAMModelGalerkin
from zoomy_core.model.models.system_model import SystemModel
from zoomy_core.transformation.to_numpy import NumpyRuntimeModel


def _all_derivative_atoms(sm):
    atoms = set()
    for M in (sm.flux, sm.hydrostatic_pressure,
              sm.nonconservative_matrix, sm.source, sm.mass_matrix):
        if isinstance(M, sp.Matrix):
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    atoms |= M[i, j].atoms(sp.Derivative)
        else:
            for idx in itertools.product(*[range(s) for s in M.shape]):
                atoms |= M[idx].atoms(sp.Derivative)
    return atoms


def _all_function_atoms(sm):
    atoms = set()
    for M in (sm.flux, sm.hydrostatic_pressure,
              sm.nonconservative_matrix, sm.source, sm.mass_matrix):
        if isinstance(M, sp.Matrix):
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    atoms |= M[i, j].atoms(sp.Function)
        else:
            for idx in itertools.product(*[range(s) for s in M.shape]):
                atoms |= M[idx].atoms(sp.Function)
    return atoms


@pytest.fixture
def sm_auto():
    """SystemModel.from_model auto-exposes every non-state atom."""
    m = VAMModelGalerkin(level=1)
    return SystemModel.from_model(m)


def test_auto_scan_clears_all_function_and_derivative_atoms(sm_auto):
    """After ``from_model`` runs, the matrices contain neither bare
    non-state ``Function`` atoms nor any ``Derivative`` atoms."""
    state_names = {str(s) for s in sm_auto.state}
    # Function atoms left: none — every non-state Function was routed
    # to aux.
    funcs = _all_function_atoms(sm_auto)
    leftover = {f for f in funcs if f.func.__name__ not in state_names}
    assert leftover == set(), (
        f"Function atoms left in matrices after auto-scan: {leftover}"
    )
    # Derivative atoms left: none.
    assert _all_derivative_atoms(sm_auto) == set()


def test_aux_registry_structure(sm_auto):
    """Every aux entry has ``kind``, ``row``, ``aux_symbol``, ``atom``;
    derivative entries additionally have ``target_name``,
    ``target_kind`` and ``multi_index``."""
    assert hasattr(sm_auto, "aux_registry")
    assert len(sm_auto.aux_registry) == len(sm_auto.aux_state)
    for entry in sm_auto.aux_registry:
        assert "kind" in entry
        assert entry["kind"] in {"function", "derivative"}
        assert entry["aux_symbol"] in sm_auto.aux_state
        assert isinstance(entry["row"], int)
        assert entry["atom"] is not None
        if entry["kind"] == "derivative":
            assert "target_name" in entry
            assert "target_kind" in entry
            assert entry["target_kind"] in {"state", "function", "unknown"}
            assert isinstance(entry["multi_index"], tuple)


def test_auto_scan_names_match_convention(sm_auto):
    """Aux Symbols are named ``{target}_{axes}`` (Derivative).  ``b``
    is now a STATE (with trivial ``∂_t b = 0`` evolution), so it
    isn't in ``aux_state``; only its spatial derivative ``b_x``
    appears."""
    aux_names = {str(s) for s in sm_auto.aux_state}
    state_names = {str(s) for s in sm_auto.state}
    # 1D chain: VAMModelGalerkin(level=1, dimension=2).
    assert "b" in state_names        # topography is state
    assert "b" not in aux_names      # and NOT aux
    assert "b_x" in aux_names        # ∂_x b — spatial deriv of state
    assert "h_x" in aux_names        # ∂_x h
    # The 2D chain (dimension=3) would additionally have b_y, h_y, …


def test_auto_scan_is_idempotent():
    """Calling ``expose_aux_atoms`` after ``from_model`` (which already
    invoked it) is a no-op."""
    m = VAMModelGalerkin(level=1)
    sm = SystemModel.from_model(m)
    aux_before = list(sm.aux_state)
    sm.expose_aux_atoms()
    aux_after = list(sm.aux_state)
    assert aux_before == aux_after


def test_runtime_lambdifies_after_auto_scan(sm_auto):
    """``NumpyRuntimeModel.from_system_model`` consumes the
    auto-exposed SystemModel cleanly — no Function atoms left to
    confuse lambdify."""
    rt = NumpyRuntimeModel.from_system_model(sm_auto)
    assert rt.n_aux_variables == len(sm_auto.aux_state)


def test_runtime_responds_to_aux_values(sm_auto):
    """Source row 4 (``zmom_j1``) responds when ``b``, ``b_x``,
    ``h_x`` values are supplied through Qaux at runtime."""
    rt = NumpyRuntimeModel.from_system_model(sm_auto)
    # ``sm.parameters`` is the Zstruct of name → Symbol; the numeric
    # defaults live on ``parameter_values``.  Pre-Zstruct-symbol-key
    # support, ``parameters[s]`` for a Symbol ``s`` raised TypeError,
    # so this line never actually ran — now it would return the Symbol
    # (not a float).  Use ``parameter_values`` for the numeric path.
    p = np.array([float(sm_auto.parameter_values[s])
                  for s in sm_auto.parameters])
    aux_names = [str(s) for s in sm_auto.aux_state]
    i_bx = aux_names.index("b_x")

    # 8-state baseline: [h, U_0, U_1, W_0, W_1, b, P_0, P_1] now that
    # VAMModelGalerkin promotes b to a state with ∂_t b = 0.
    Q = np.array([1.0, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    Qaux_zero = np.zeros(len(aux_names))
    src_zero = rt.source(Q, Qaux_zero, p).flatten()

    Qaux_b = Qaux_zero.copy(); Qaux_b[i_bx] = 0.1
    src_b = rt.source(Q, Qaux_b, p).flatten()
    assert abs(src_b[4] - src_zero[4]) > 1e-6, (
        "source[4] (zmom_j1) didn't respond to b_x"
    )


def test_aux_reverse_map_round_trip(sm_auto):
    """The ``_aux_reverse_map`` lets ``reconstruct_residuals`` display
    residuals in their original ``Derivative(…)`` form."""
    reverse = sm_auto._aux_reverse_map()
    assert len(reverse) > 0
    for aux_sym, atom in reverse.items():
        assert aux_sym in sm_auto.aux_state
        assert isinstance(atom, (sp.Function, sp.Derivative))
