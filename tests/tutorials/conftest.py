"""Tutorial notebooks are opt-in: they run only when the `-m` expression names `tutorial`."""
import pytest


def pytest_collection_modifyitems(config, items):
    if "tutorial" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason="tutorial notebooks are opt-in: pytest tests/tutorials -m tutorial")
    for item in items:
        if item.get_closest_marker("tutorial"):
            item.add_marker(skip)
