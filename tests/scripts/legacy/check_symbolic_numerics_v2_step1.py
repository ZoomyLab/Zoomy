import numpy as np

from zoomy_core.fvm.symbolic_numerics_v2 import (
    PositiveNonconservativeRusanov as PositiveNonconservativeRusanovV2,
)
from zoomy_jax.transformation.to_jax import JaxRuntimeSymbolic
from zoomy_core.model.models.shallow_water_topo import ShallowWaterEquationsWithTopo


def _as_np(x):
    return np.asarray(x, dtype=float)


def _run_once(runtime, q_minus, q_plus, aux_minus, aux_plus, p, normal):
    # Evaluate per face to keep the runtime path robust while we still use
    # scalar symbolic kernels.
    n_faces = q_minus.shape[1]
    flux_list = []
    fluct_list = []
    for i in range(n_faces):
        qi_m = q_minus[:, i]
        qi_p = q_plus[:, i]
        ai_m = aux_minus[:, i]
        ai_p = aux_plus[:, i]
        ni = normal[:, i]
        flux_i = _as_np(runtime.numerical_flux(qi_m, qi_p, ai_m, ai_p, p, ni))
        fluct_i = _as_np(runtime.numerical_fluctuations(qi_m, qi_p, ai_m, ai_p, p, ni))
        flux_list.append(flux_i.reshape(-1))
        fluct_list.append(fluct_i.reshape(-1))
    flux = np.stack(flux_list, axis=1)
    fluct = np.stack(fluct_list, axis=1)
    return flux, fluct


def main():
    np.random.seed(1)
    # Use topo SWE with one aux slot so legacy/v2 PositiveRusanov can store hinv.
    model = ShallowWaterEquationsWithTopo(dimension=1, aux_variables=1)

    n_faces = 8
    n_q = model.n_variables
    n_aux = model.n_aux_variables

    q_minus = np.zeros((n_q, n_faces), dtype=float)
    q_plus = np.zeros((n_q, n_faces), dtype=float)
    aux_minus = np.zeros((n_aux, n_faces), dtype=float)
    aux_plus = np.zeros((n_aux, n_faces), dtype=float)

    # Layout for NumericalShallowWaterEquationsWithTopo in 1D: Q = [b, h, hu]
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

    # Aux[0] is hinv in this model variant.
    aux_minus[0, :] = 1.0 / np.maximum(h_minus, 1e-12)
    aux_plus[0, :] = 1.0 / np.maximum(h_plus, 1e-12)

    p = np.asarray(model.parameter_values, dtype=float)
    normal = np.ones((1, n_faces), dtype=float)

    numerics_v2 = PositiveNonconservativeRusanovV2(
        model,
        field_map={
            "h": {"container": "q", "index": 1},
            "b": {"container": "q", "index": 0},
            "hinv": {"container": "qaux", "index": 0},
        },
    )

    runtime_v2_np = numerics_v2.to_runtime_numpy()

    flux_v2, fluct_v2 = _run_once(
        runtime_v2_np, q_minus, q_plus, aux_minus, aux_plus, p, normal
    )

    print("step1: v2 numpy runtime")
    print("  max|flux|:", float(np.max(np.abs(flux_v2))))
    print("  max|fluct|:", float(np.max(np.abs(fluct_v2))))
    print("  all finite:", bool(np.isfinite(flux_v2).all() and np.isfinite(fluct_v2).all()))
    print("  flux shape:", flux_v2.shape, "fluct shape:", fluct_v2.shape)

    # Light check for alternative mapping support with b in Qaux.
    numerics_v2_b_aux = PositiveNonconservativeRusanovV2(
        model,
        field_map={
            "h": {"container": "q", "index": 1},
            "b": {"container": "qaux", "index": 0},
        },
        scaled_q_indices=[2],
    )
    runtime_v2_b_aux = numerics_v2_b_aux.to_runtime_numpy()
    _ = _run_once(runtime_v2_b_aux, q_minus, q_plus, aux_minus, aux_plus, p, normal)
    print("  alternate mapping instantiated/executed: b in Qaux (syntactic check)")

    try:
        runtime_v2_jax = JaxRuntimeSymbolic(numerics_v2)
        flux_v2_jax, fluct_v2_jax = _run_once(
            runtime_v2_jax, q_minus, q_plus, aux_minus, aux_plus, p, normal
        )
        print("step1: v2 numpy vs jax")
        print("  max|flux_numpy - flux_jax|:", float(np.max(np.abs(flux_v2 - flux_v2_jax))))
        print(
            "  max|fluct_numpy - fluct_jax|:",
            float(np.max(np.abs(fluct_v2 - fluct_v2_jax))),
        )
    except Exception as exc:
        print("step1: jax runtime check skipped:", repr(exc))


if __name__ == "__main__":
    main()
