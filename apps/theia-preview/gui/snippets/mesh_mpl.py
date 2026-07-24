"""Field on the mesh — matplotlib via ``zoomy_plotting.MatplotlibPlotter``.

The GUI injects ``store`` (a ``zoomy_plotting.SimulationStore``), ``time_step``
(timeline slider) and ``field_name`` (field selector). Unified 1D / 2D / 3D.
"""
import matplotlib
matplotlib.use("agg")            # headless worker — no GUI backend
import matplotlib.pyplot as plt
import zoomy_plotting as zp

if store is None:
    raise RuntimeError("No data yet — run a simulation first.")

field = field_name if ("field_name" in dir() and field_name) else next(iter(store.field.keys()))
step = int(time_step) if "time_step" in dir() else 0
kw = {} if store.dim == 1 else {"cmap": "viridis", "colorbar": True}

with zp.apply_style():
    if store.dim == 3:
        fig = plt.figure(); ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots()
    zp.MatplotlibPlotter(store).plot(ax, time_step=step, field=field, **kw)
    if store.times is not None and len(store.times):
        ax.set_title(f"{field} — t = {float(store.times[step]):.3f}")

display(fig)
