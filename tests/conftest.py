import os

import pytest


def pytest_collection_modifyitems(config, items):
    markexpr = (config.option.markexpr or "").strip().lower()
    explicit_large = "large" in markexpr
    explicit_bench = "benchmark" in markexpr

    run_large = os.environ.get("ZOOMY_RUN_LARGE_TESTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    run_bench = os.environ.get("ZOOMY_RUN_BENCHMARK_TESTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    # If user explicitly selected an expensive marker expression with -m, trust that intent.
    # Many benchmark tests are also marked large, so selecting one should not be blocked by the other gate.
    explicit_expensive = explicit_large or explicit_bench
    run_large = run_large or explicit_expensive
    run_bench = run_bench or explicit_expensive

    skip_large = pytest.mark.skip(
        reason="set ZOOMY_RUN_LARGE_TESTS=1 or run with -m 'large' to execute large tests"
    )
    skip_bench = pytest.mark.skip(
        reason="set ZOOMY_RUN_BENCHMARK_TESTS=1 or run with -m 'benchmark' to execute benchmark tests"
    )

    for item in items:
        if ("large" in item.keywords) and (not run_large):
            item.add_marker(skip_large)
        if ("benchmark" in item.keywords) and (not run_bench):
            item.add_marker(skip_bench)

        try:
            path = str(item.path)
        except AttributeError:
            path = str(item.fspath)
        norm = path.replace("\\", "/")
        if "/zoomy_dmplex/" in norm:
            item.add_marker(pytest.mark.dmplex)
        if "/zoomy_fenicsx/" in norm:
            item.add_marker(pytest.mark.fenicsx)

