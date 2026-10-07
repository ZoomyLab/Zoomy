"""Solver-template gate tests (``tests/template_gates``).

* registers the ``template_gate`` marker;
* x64 for any jax a gate may touch (zoomy_core itself never imports jax);
* generated template code goes to a per-session cache directory.
"""
import os

import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "template_gate: solver-template sub-stage gate test")


@pytest.fixture(scope="session", autouse=True)
def _template_cache(tmp_path_factory):
    old = os.environ.get("ZOOMY_TEMPLATE_CACHE")
    os.environ["ZOOMY_TEMPLATE_CACHE"] = str(tmp_path_factory.mktemp("generated"))
    yield os.environ["ZOOMY_TEMPLATE_CACHE"]
    if old is None:
        os.environ.pop("ZOOMY_TEMPLATE_CACHE", None)
    else:
        os.environ["ZOOMY_TEMPLATE_CACHE"] = old
