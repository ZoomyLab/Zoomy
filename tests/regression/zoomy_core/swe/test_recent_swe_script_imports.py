import importlib

import pytest


MODULES = [
    "tests.scripts.zoomy_core.swe.check_symbolic_numerics_v2_step1",
    "tests.scripts.zoomy_core.swe.compare_jvp_fd_analytic_v2",
    "tests.scripts.zoomy_core.swe.compare_three_models_v2",
    "tests.scripts.zoomy_core.swe.benchmark_imex_source_modes_v2",
    "tests.scripts.zoomy_core.swe.benchmark_imex_numpy_vs_jax_v2",
    "tests.scripts.zoomy_core.swe.benchmark_precision_backend_matrix",
    "tests.scripts.zoomy_core.swe.benchmark_flux_vs_quasilinear_swe_v2",
    "tests.scripts.zoomy_core.swe.benchmark_swe_validation_ladder_v2",
    "tests.scripts.zoomy_core.swe.benchmark_suite_symbolic_numerics_v2",
    "tests.scripts.zoomy_core.swe.benchmark_swashes_wetdry_compare_v2",
    "tests.scripts.zoomy_core.swe.plot_swashes_convergence_v2",
    "tests.scripts.zoomy_core.swe.investigate_bathy_fullwet_hr_vs_quasilinear_v2",
    "tests.scripts.zoomy_core.swe.export_swashes_reference_v2",
]


@pytest.mark.small
@pytest.mark.regression
@pytest.mark.smoke
@pytest.mark.numpy
@pytest.mark.parametrize("module_name", MODULES)
def test_recent_swe_script_imports(module_name):
    mod = importlib.import_module(module_name)
    assert mod is not None
    assert hasattr(mod, "run") or hasattr(mod, "main")

