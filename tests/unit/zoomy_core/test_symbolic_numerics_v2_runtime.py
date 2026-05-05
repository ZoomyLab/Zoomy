import numpy as np
import pytest

from tests.common.baseline_store import assert_or_create_array_baseline
from zoomy_core.fvm.riemann_solvers import PositiveNonconservativeRusanov
from zoomy_core.model.models.legacy.shallow_water_topo import ShallowWaterEquationsWithTopo


def _as_np(x):
    return np.asarray(x, dtype=float)


@pytest.mark.small
@pytest.mark.unittest
@pytest.mark.numpy
@pytest.mark.core
def test_symbolic_numerics_v2_numpy_runtime_regression():
    np.random.seed(1)
    model = ShallowWaterEquationsWithTopo(dimension=1, aux_variables=1)
    numerics = PositiveNonconservativeRusanov(
        model,
        field_map={
            "h": {"container": "q", "index": 1},
            "b": {"container": "q", "index": 0},
            "hinv": {"container": "qaux", "index": 0},
        },
    )
    runtime = numerics.to_runtime_numpy()

    n_faces = 8
    q_minus = np.zeros((model.n_variables, n_faces), dtype=float)
    q_plus = np.zeros((model.n_variables, n_faces), dtype=float)
    aux_minus = np.zeros((model.n_aux_variables, n_faces), dtype=float)
    aux_plus = np.zeros((model.n_aux_variables, n_faces), dtype=float)

    b_minus = 0.2 + 0.1 * np.sin(np.linspace(0.0, 1.0, n_faces))
    b_plus = 0.2 + 0.1 * np.cos(np.linspace(0.0, 1.0, n_faces))
    h_minus = np.maximum(1e-4, 0.05 + 0.3 * np.random.rand(n_faces))
    h_plus = np.maximum(1e-4, 0.05 + 0.3 * np.random.rand(n_faces))
    hu_minus = 0.02 * (2.0 * np.random.rand(n_faces) - 1.0)
    hu_plus = 0.02 * (2.0 * np.random.rand(n_faces) - 1.0)

    q_minus[0, :] = b_minus
    q_minus[1, :] = h_minus
    q_minus[2, :] = hu_minus
    q_plus[0, :] = b_plus
    q_plus[1, :] = h_plus
    q_plus[2, :] = hu_plus
    aux_minus[0, :] = 1.0 / np.maximum(h_minus, 1e-12)
    aux_plus[0, :] = 1.0 / np.maximum(h_plus, 1e-12)

    p = np.asarray(list(model.parameters.values()), dtype=float)
    normal = np.ones((1, n_faces), dtype=float)

    fluxes = []
    fluctuations = []
    for i in range(n_faces):
        flux_i = _as_np(
            runtime.numerical_flux(
                q_minus[:, i], q_plus[:, i], aux_minus[:, i], aux_plus[:, i], p, normal[:, i]
            )
        )
        fluct_i = _as_np(
            runtime.numerical_fluctuations(
                q_minus[:, i], q_plus[:, i], aux_minus[:, i], aux_plus[:, i], p, normal[:, i]
            )
        )
        fluxes.append(flux_i.reshape(-1))
        fluctuations.append(fluct_i.reshape(-1))

    flux = np.stack(fluxes, axis=1)
    fluct = np.stack(fluctuations, axis=1)
    assert np.isfinite(flux).all()
    assert np.isfinite(fluct).all()

    assert_or_create_array_baseline(
        "symbolic_numerics_v2_numpy_face_samples",
        {
            "flux": flux,
            "fluct": fluct,
            "max_abs_flux": np.asarray([np.max(np.abs(flux))]),
            "max_abs_fluct": np.asarray([np.max(np.abs(fluct))]),
        },
        rtol=1e-9,
        atol=1e-11,
    )

