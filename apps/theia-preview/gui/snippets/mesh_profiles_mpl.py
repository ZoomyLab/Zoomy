"""Field + vertical velocity profiles at three cross-sections.

Top: the field on the mesh (same view as the Field Viewer). Bottom: the
vertical velocity profile u(zeta) at three stations, lifted the SAME way for
every model -- through the model's own symbolic ``interpolate_to_3d`` (the
coupling contract). zeta = 0 is the bed, 1 the free surface. SME expands a
polynomial profile, MLSME a piecewise per-layer one, VAM its own; the lift
reproduces each exactly, with no ansatz hard-coded here.

The GUI injects ``store`` (a ``zoomy_plotting.SimulationStore``), ``time_step``
(timeline slider) and ``field_name`` (field selector). ``model`` (the run's
SystemModel) persists in the shared exec scope, so the profiles come straight
from the run that produced the store.
"""
import matplotlib
matplotlib.use("agg")            # headless worker — no GUI backend
import matplotlib.pyplot as plt
import numpy as np
import zoomy_plotting as zp

if store is None:
    raise RuntimeError("No data yet — run a simulation first.")

step = int(time_step) if "time_step" in dir() else 0
names = list(store.field.keys())
field = field_name if ("field_name" in dir() and field_name) else names[0]

# --- cell centres, for stations and the x axis ------------------------------
verts, cells = np.asarray(store.vertices), np.asarray(store.cells)
centers = verts[cells].mean(axis=1)
n = store.n_inner_cells or len(centers)
centers, x = centers[:n], centers[:n, 0]

xs = np.quantile(x, [0.25, 0.50, 0.75])          # three stations across the span
probes = [int(np.argmin(np.abs(x - xq))) for xq in xs]


def _u_lift(M):
    """Lambdify the model's interpolate_to_3d ``u`` row (slot 2 of
    [b, h, u, v, w, p]) -> u(*Q, *Qaux, *params, x, y, zeta). Same lift as
    zoomy_prepost.steps.lift3d / column_plots.read_zoomyfoam."""
    import sympy as sp
    rows = getattr(M, "interpolate_to_3d", None)
    if rows is None:
        return None
    exprs = [sp.sympify(e) for e in np.asarray(rows, dtype=object).ravel()]
    state = list(M.state)
    aux = list(getattr(M, "aux_state", None) or [])
    params = list(M.parameters.values()) if M.parameters is not None else []
    pos = (list(M.position.values()) if getattr(M, "position", None) is not None
           else list(sp.symbols("x y z", real=True)))
    fn = sp.lambdify(state + aux + params + pos[:3], exprs[2], "numpy", dummify=True)
    pv = getattr(M, "parameter_values", None)
    pvals = [float(v) for v in pv.values()] if pv is not None else [0.0] * len(params)
    return fn, len(state), len(aux), pvals


M = globals().get("model")           # the run's SystemModel, from the shared scope
lift = _u_lift(M) if M is not None else None

with zp.apply_style():
    fig = plt.figure(figsize=(9.0, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.42, wspace=0.30)

    # ---- top: the field ----------------------------------------------------
    ax0 = fig.add_subplot(gs[0, :], projection="3d" if store.dim == 3 else None)
    kw = {} if store.dim == 1 else {"cmap": "viridis", "colorbar": True}
    zp.MatplotlibPlotter(store).plot(ax0, time_step=step, field=field, **kw)
    times = getattr(store, "times", None)
    t_now = float(times[step]) if times is not None and len(times) else None
    ax0.set_title(f"{field}" + (f" — t = {t_now:.3f}" if t_now is not None else ""))
    for k in probes:                                  # mark the stations
        if store.dim == 1:
            ax0.axvline(x[k], color="k", lw=1.0, ls="--", alpha=0.7)
        elif store.dim == 2:
            ax0.plot(centers[k, 0], centers[k, 1], "o", ms=7, mfc="none",
                     mew=2, color="k")

    # ---- bottom: u(zeta) at each station, via interpolate_to_3d ------------
    if lift is not None:
        fn, n_state, n_aux, pvals = lift
        zeta = np.linspace(0.0, 1.0, 60)
        aux0 = [0.0] * n_aux            # u row is a function of the moments, not aux
        for j, k in enumerate(probes):
            ax = fig.add_subplot(gs[1, j])
            Qk = [float(store.get_cell(step, i)[k]) for i in range(n_state)]
            u = np.asarray(fn(*Qk, *aux0, *pvals,
                              float(centers[k, 0]),
                              float(centers[k, 1] if centers.shape[1] > 1 else 0.0),
                              zeta), dtype=float)
            if u.ndim == 0:
                u = np.full_like(zeta, float(u))
            ax.plot(u, zeta, lw=2)
            ax.set_title(f"x = {x[k]:.3f}", fontsize=10)
            ax.set_xlabel(r"$u(\zeta)$")
            ax.set_ylim(0.0, 1.0)
            if j == 0:
                ax.set_ylabel(r"$\zeta$  (0 = bed, 1 = surface)")
    else:
        ax = fig.add_subplot(gs[1, :])
        ax.axis("off")
        why = ("run a model in this session first — the profiles lift through "
               "its interpolate_to_3d" if M is None else
               "this model defines no interpolate_to_3d")
        ax.text(0.5, 0.5, f"No vertical profile:\n{why}.",
                ha="center", va="center", fontsize=11)

display(fig)
