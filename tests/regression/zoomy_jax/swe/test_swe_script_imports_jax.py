"""Smoke-import SWE scripts that require JAX / zoomy_jax (kept out of core CI)."""

import importlib

import pytest

MODULES = [
    "tests.scripts.zoomy_core.swe.check_symbolic_numerics_v2_step1",
    "tests.scripts.zoomy_core.swe.benchmark_precision_backend_matrix",
    "tests.scripts.zoomy_core.swe.benchmark_imex_numpy_vs_jax_v2",
]


@pytest.mark.small
@pytest.mark.regression
@pytest.mark.smoke
@pytest.mark.jax
@pytest.mark.parametrize("module_name", MODULES)
def test_recent_swe_script_imports_jax(module_name):
    mod = importlib.import_module(module_name)
    assert mod is not None
    assert hasattr(mod, "run") or hasattr(mod, "main")
