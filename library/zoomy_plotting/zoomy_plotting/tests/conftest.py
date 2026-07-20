"""Shared pytest fixtures for the postproc plotting suite.

All plotting tests run with matplotlib's ``Agg`` backend so they're headless,
and ``CONFIG`` is reset between tests so mutations don't leak.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # must happen before any pyplot import

import pytest

from zoomy_plotting.plot.style import reset_config


@pytest.fixture(autouse=True)
def _reset_config():
    reset_config()
    yield
    reset_config()
