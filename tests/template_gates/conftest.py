"""Solver-template gate tests (``tests/template_gates``).

* registers the ``template_gate`` marker;
* x64 on (the parent conftest asserts it; set it here as well so the gates
  also run standalone);
* generated template code goes to a per-session cache directory.
"""
import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)


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
