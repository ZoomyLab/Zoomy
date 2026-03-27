import importlib

import pytest


MODULES = [
    "tutorials.swe.simple_swe_v2",
    "tutorials.swe.simple_gn_v2",
    "tutorials.swe.gn_classical_linear_analysis_v2",
    "tutorials.swe.gn_linear_analysis_v2",
    "tutorials.swe.simple_gn_classical_sim_v2",
    "tutorials.swe.beach_runup_swe_vs_gn_classical_v2",
]


@pytest.mark.small
@pytest.mark.regression
@pytest.mark.smoke
@pytest.mark.tutorial
@pytest.mark.numpy
@pytest.mark.core
@pytest.mark.parametrize("module_name", MODULES)
def test_tutorial_module_imports(module_name):
    mod = importlib.import_module(module_name)
    assert mod is not None
    # Trivial feature-runs check: each script must expose run() or main().
    assert hasattr(mod, "run") or hasattr(mod, "main")

