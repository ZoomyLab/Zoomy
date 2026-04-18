"""Matplotlib plotters — unified 1D/2D/3D interface.

Design rules:

1. Every visual parameter (color, linewidth, markersize, cmap, alpha)
   reads from :data:`~zoomy_plotting.plot.style.CONFIG`. Plot methods
   accept ``None``-defaulted kwargs that override the config for a
   single call. Never hardcode a literal here — ``tests/test_style.py``
   enforces this statically.
2. **Lazy matplotlib import**: importing this module must NOT load
   matplotlib. Only methods that actually draw touch matplotlib. This
   is checked by ``tests/test_lazy.py``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..mesh.faces import cell_to_vert_values
from .base import BasePlotter
from ._common import add_colorbar, resolve_viewpoint
from .style import CONFIG


def _mpl():
    """Lazy one-stop handle for the matplotlib submodules we use.

    First call imports matplotlib (~1 s in Pyodide); subsequent calls
    are cache hits. Returns a small namespace so callers don't have to
    juggle three import statements inside every method body.
    """
    global _MPL_CACHE
    if _MPL_CACHE is None:
        from matplotlib.collections import PolyCollection
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        import matplotlib.pyplot as plt

        class _Namespace:
            pass

        ns = _Namespace()
        ns.PolyCollection = PolyCollection
        ns.ScalarMappable = ScalarMappable
        ns.Normalize = Normalize
        ns.plt = plt
        _MPL_CACHE = ns
    return _MPL_CACHE


_MPL_CACHE = None


# Helper: pick between a kwarg and the CONFIG fallback.
def _pick(kw, cfg_attr):
    return getattr(CONFIG, cfg_attr) if kw is None else kw


class MatplotlibPlotter(BasePlotter):
    """Plot 1-D / 2-D / 3-D simulation data with matplotlib."""

    # ------------------------------------------------------------------
    # 1-D
    # ------------------------------------------------------------------
    def plot_1d(
        self,
        ax,
        time_step: int = 0,
        field=0,
        *,
        line_color: Optional[str] = None,
        line_linewidth: Optional[float] = None,
        line_linestyle: Optional[str] = None,
        show_nodes: Optional[bool] = None,
        node_color: Optional[str] = None,
        node_markersize: Optional[float] = None,
        node_marker: Optional[str] = None,
    ) -> dict:
        """Line plot of a 1-D mesh.

        Returns a dict of artist handles (``line``, optionally ``nodes``).
        """
        if self.store.dim != 1:
            raise ValueError(
                f"plot_1d requires dim==1, got dim={self.store.dim}; "
                "use plot_2d / plot_3d or slice first."
            )

        values = self._cell_values(time_step, field)
        x = self.store.cell_centers[:, 0]
        order = np.argsort(x)

        kw_line = {
            "linewidth": _pick(line_linewidth, "line_linewidth"),
            "linestyle": _pick(line_linestyle, "line_linestyle"),
        }
        lc = _pick(line_color, "line_color")
        if lc is not None:
            kw_line["color"] = lc

        (line,) = ax.plot(x[order], values[order], **kw_line)
        out = {"line": line}

        if _pick(show_nodes, "node_show"):
            node_kw = {
                "marker": _pick(node_marker, "node_marker"),
                "markersize": _pick(node_markersize, "node_markersize"),
                "color": _pick(node_color, "node_color"),
                "linestyle": "none",
            }
            (nodes,) = ax.plot(x[order], values[order], **node_kw)
            out["nodes"] = nodes

        if CONFIG.axes_grid_1d:
            ax.grid(True, alpha=CONFIG.grid_alpha)
        ax.set_xlabel("x")
        ax.set_ylabel(self._field_label(field))
        ax.set_title(f"{self._field_label(field)}  —  step {int(time_step)}")
        return out

    # ------------------------------------------------------------------
    # 2-D
    # ------------------------------------------------------------------
    def plot_2d(
        self,
        ax,
        time_step: int = 0,
        field=0,
        *,
        show_mesh: bool = False,
        interpolate: bool = False,
        cmap: Optional[str] = None,
        edgecolor: Optional[str] = None,
        linewidth: Optional[float] = None,
        alpha: Optional[float] = None,
        vlim=None,
        colorbar: Optional[bool] = None,
    ) -> dict:
        """Polygon mesh plot with cell-constant or gouraud-interpolated coloring."""
        if self.store.dim != 2:
            raise ValueError(
                f"plot_2d requires dim==2, got dim={self.store.dim}"
            )

        values = self._cell_values(time_step, field)
        vertices2d = self.store.vertices[:, :2]
        cells = self.store.cells

        vmin, vmax = (None, None) if vlim is None else (float(vlim[0]), float(vlim[1]))
        if vmin is None:
            vmin = float(np.min(values))
        if vmax is None:
            vmax = float(np.max(values))
        norm = _mpl().Normalize(vmin=vmin, vmax=vmax)

        cmap_ = _pick(cmap, "cmap")
        show_cb = _pick(colorbar, "colorbar_show")

        out = {}

        if interpolate:
            # Fan-triangulate to get matplotlib-friendly index arrays.
            ii, jj, kk, _ = self._triangulation
            vert_vals = cell_to_vert_values(self.store.n_vertices, cells, values)
            triangles = np.stack([ii, jj, kk], axis=1)
            tc = ax.tripcolor(
                vertices2d[:, 0],
                vertices2d[:, 1],
                vert_vals,                # per-vertex colors for gouraud
                triangles=triangles,
                shading="gouraud",
                cmap=cmap_,
                norm=norm,
            )
            out["mesh"] = tc
        else:
            polygons = [vertices2d[cell] for cell in cells]
            face_colors_rgba = _cmap_colors(cmap_, norm, values)
            # edgecolors="none" is structural ("draw no edges") — the
            # matplotlib convention, not a style choice, so it is permitted
            # by test_style.py. The linewidth fallback comes from CONFIG
            # even though it is invisible when edges are "none".
            coll = _mpl().PolyCollection(
                polygons,
                facecolors=face_colors_rgba,
                edgecolors="none",
                linewidths=_pick(linewidth, "mesh_linewidth"),
                alpha=_pick(alpha, "mesh_alpha"),
            )
            ax.add_collection(coll)
            out["mesh"] = coll

        if show_mesh:
            overlay = _mpl().PolyCollection(
                [vertices2d[cell] for cell in cells],
                facecolors="none",
                edgecolors=_pick(edgecolor, "mesh_edgecolor"),
                linewidths=_pick(linewidth, "mesh_linewidth"),
                alpha=_pick(alpha, "mesh_alpha"),
            )
            ax.add_collection(overlay)
            out["mesh_overlay"] = overlay

        ax.set_xlim(float(vertices2d[:, 0].min()), float(vertices2d[:, 0].max()))
        ax.set_ylim(float(vertices2d[:, 1].min()), float(vertices2d[:, 1].max()))
        ax.set_aspect("equal")
        if CONFIG.axes_grid_2d3d:
            ax.grid(True, alpha=CONFIG.grid_alpha)
        ax.set_title(f"{self._field_label(field)}  —  step {int(time_step)}")

        if show_cb:
            sm = _mpl().ScalarMappable(cmap=cmap_, norm=norm)
            sm.set_array([])
            out["colorbar"] = add_colorbar(
                ax.figure, ax, sm, label=self._field_label(field)
            )

        return out

    # ------------------------------------------------------------------
    # 3-D
    # ------------------------------------------------------------------
    def plot_3d(
        self,
        ax,
        time_step: int = 0,
        field=0,
        *,
        show_mesh: bool = False,
        interpolate: bool = False,
        viewpoint="auto",
        cmap: Optional[str] = None,
        edgecolor: Optional[str] = None,
        linewidth: Optional[float] = None,
        alpha: Optional[float] = None,
        vlim=None,
        colorbar: Optional[bool] = None,
    ) -> dict:
        """Boundary-surface plot of a 3-D mesh."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # lazy import

        if self.store.dim != 3:
            raise ValueError(
                f"plot_3d requires dim==3, got dim={self.store.dim}"
            )

        vertices = self.store.vertices
        cells = self.store.cells
        faces, parent_cells = self._boundary_3d
        if not faces:
            raise RuntimeError(
                f"no boundary faces extracted from {len(cells)} cells of "
                f"type {self.store.cell_type!r}"
            )

        values = self._cell_values(time_step, field)

        vmin, vmax = (None, None) if vlim is None else (float(vlim[0]), float(vlim[1]))
        if vmin is None:
            vmin = float(np.min(values))
        if vmax is None:
            vmax = float(np.max(values))
        norm = _mpl().Normalize(vmin=vmin, vmax=vmax)

        cmap_ = _pick(cmap, "cmap")
        show_cb = _pick(colorbar, "colorbar_show")
        alpha_ = _pick(alpha, "face_alpha")

        out: dict = {}

        if interpolate:
            vert_vals = cell_to_vert_values(self.store.n_vertices, cells, values)
            poly_coords = [vertices[f] for f in faces]
            # Average vertex values onto each face for flat fallback; Poly3D
            # doesn't support gouraud without a heavy lift — this is the
            # common compromise.
            face_vals = np.array(
                [float(vert_vals[f].mean()) for f in faces], dtype=float
            )
            face_colors = _cmap_colors(cmap_, norm, face_vals)
        else:
            poly_coords = [vertices[f] for f in faces]
            face_vals = np.array(
                [float(values[int(p)]) for p in parent_cells], dtype=float
            )
            face_colors = _cmap_colors(cmap_, norm, face_vals)

        # When show_mesh=False, edgecolors="none" hides edges regardless of
        # linewidth. We still source the width from CONFIG so behaviour stays
        # style-driven.
        coll = Poly3DCollection(
            poly_coords,
            facecolors=face_colors,
            edgecolors=_pick(edgecolor, "mesh_edgecolor") if show_mesh else "none",
            linewidths=_pick(linewidth, "mesh_linewidth"),
            alpha=alpha_,
        )
        ax.add_collection3d(coll)
        out["mesh"] = coll

        ax.set_xlim(float(vertices[:, 0].min()), float(vertices[:, 0].max()))
        ax.set_ylim(float(vertices[:, 1].min()), float(vertices[:, 1].max()))
        ax.set_zlim(float(vertices[:, 2].min()), float(vertices[:, 2].max()))

        elev, azim = resolve_viewpoint(viewpoint, vertices)
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        ax.set_title(f"{self._field_label(field)}  —  step {int(time_step)}")

        if show_cb:
            sm = _mpl().ScalarMappable(cmap=cmap_, norm=norm)
            sm.set_array([])
            out["colorbar"] = add_colorbar(
                ax.figure, ax, sm, label=self._field_label(field)
            )

        return out

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------
    def plot(self, ax, time_step: int = 0, field=0, **kwargs) -> dict:
        """Dispatch to :meth:`plot_1d` / :meth:`plot_2d` / :meth:`plot_3d` based on mesh dim."""
        d = self.store.dim
        if d == 1:
            return self.plot_1d(ax, time_step=time_step, field=field, **kwargs)
        if d == 2:
            return self.plot_2d(ax, time_step=time_step, field=field, **kwargs)
        if d == 3:
            return self.plot_3d(ax, time_step=time_step, field=field, **kwargs)
        raise ValueError(f"unsupported store.dim={d}")


# --------------------------------------------------------------------------
# Small helper used above. Lives here rather than in _common.py because it's
# only called from this module and keeps ``_common.py`` free of imports that
# depend on CONFIG-vs-kwarg semantics.
# --------------------------------------------------------------------------
def _cmap_colors(cmap_name: str, norm, values: np.ndarray) -> np.ndarray:
    """Map a 1-D array of scalars to RGBA via the given named colormap."""
    import matplotlib.pyplot as plt

    return _mpl().plt.get_cmap(cmap_name)(norm(values))
