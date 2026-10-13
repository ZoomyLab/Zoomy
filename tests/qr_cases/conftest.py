"""QR-case tests are opt-in: they run only when the `-m` expression names a `qr` marker."""
import pytest


def pytest_collection_modifyitems(config, items):
    if "qr" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason="QR cases are opt-in: pytest tests/qr_cases -m qr (or qr_live)")
    for item in items:
        if "qr" in item.keywords or "qr_live" in item.keywords:
            item.add_marker(skip)
