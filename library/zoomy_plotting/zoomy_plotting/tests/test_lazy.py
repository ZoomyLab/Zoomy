"""Assert that ``import zoomy_plotting`` is lightweight.

Matplotlib's import is ~1–2 s in Pyodide; paying it at package-import
time hurts the GUI's slider/selector code path which only needs to read
metadata. The contract:

* Importing ``zoomy_plotting`` must not load matplotlib.
* Instantiating ``MatplotlibPlotter(store)`` also must not load
  matplotlib (only method calls do).
* Calling ``plot_*`` triggers the import.

These tests run in a subprocess so a previous test that imported
matplotlib can't contaminate ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_snippet(code: str) -> str:
    """Run ``code`` in a fresh Python interpreter and return stdout."""
    out = subprocess.check_output(
        [sys.executable, "-c", textwrap.dedent(code)],
        text=True,
    )
    return out.strip()


def test_importing_zoomy_plotting_does_not_load_matplotlib():
    out = _run_snippet(
        """
        import sys
        import zoomy_plotting  # noqa: F401
        loaded = any(
            m == "matplotlib" or m.startswith("matplotlib.")
            for m in sys.modules
        )
        print("LOADED" if loaded else "LAZY")
        """
    )
    assert out == "LAZY", (
        "matplotlib leaked into sys.modules at zoomy_plotting import. "
        "Move the offending import(s) into a function body."
    )


def test_instantiating_plotter_does_not_load_matplotlib():
    out = _run_snippet(
        """
        import sys
        import numpy as np
        from zoomy_plotting import MatplotlibPlotter, SimulationStore, Zstruct

        store = SimulationStore(
            dim=1,
            cell_type="line",
            vertices=np.linspace(0, 1, 5).reshape(5, 1),
            cells=np.array([[0, 1], [1, 2], [2, 3], [3, 4]]),
            field=Zstruct({"h": 0}),
            _cell_reader=lambda t, i: np.arange(4, dtype=float),
        )
        p = MatplotlibPlotter(store)

        loaded = any(
            m == "matplotlib" or m.startswith("matplotlib.")
            for m in sys.modules
        )
        print("LOADED" if loaded else "LAZY")
        """
    )
    assert out == "LAZY", (
        "matplotlib was loaded at MatplotlibPlotter construction. "
        "Construction must only wire up state — not draw."
    )


def test_plot_call_does_load_matplotlib():
    """Inverse check: once plot_1d runs, matplotlib IS loaded.

    Guards against accidentally deleting the inner imports and still
    passing the previous two tests trivially.
    """
    out = _run_snippet(
        """
        import sys
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Now clear sys.modules of matplotlib so the test is meaningful
        for m in list(sys.modules):
            if m == "matplotlib" or m.startswith("matplotlib."):
                del sys.modules[m]

        from zoomy_plotting import MatplotlibPlotter, SimulationStore, Zstruct

        store = SimulationStore(
            dim=1,
            cell_type="line",
            vertices=np.linspace(0, 1, 5).reshape(5, 1),
            cells=np.array([[0, 1], [1, 2], [2, 3], [3, 4]]),
            field=Zstruct({"h": 0}),
            _cell_reader=lambda t, i: np.arange(4, dtype=float),
        )
        p = MatplotlibPlotter(store)
        assert "matplotlib" not in sys.modules   # still lazy before plot call

        import matplotlib.pyplot as plt2
        fig, ax = plt2.subplots()
        p.plot_1d(ax, 0, "h")
        print("LOADED" if "matplotlib" in sys.modules else "LAZY")
        """
    )
    assert out == "LOADED"
