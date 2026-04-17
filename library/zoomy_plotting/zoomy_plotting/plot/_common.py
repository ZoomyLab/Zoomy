"""Shared helpers used by the matplotlib plot functions."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .style import CONFIG


def compute_auto_viewpoint(vertices: np.ndarray) -> Tuple[float, float]:
    """Pick a decent ``(elev, azim)`` for a 3-D mesh.

    Heuristic: keep ``elev`` fixed (comes from ``CONFIG.viewpoint_elev``),
    and rotate azimuth so the longest horizontal bounding-box axis lies
    diagonally in the image plane. That way the widest part of the mesh
    gets the most screen real estate without being perfectly edge-on.
    """
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        return float(CONFIG.viewpoint_elev), 45.0

    bbox = vertices.max(axis=0) - vertices.min(axis=0)
    dx, dy = float(bbox[0]), float(bbox[1])
    # azim = 0 looks along +x; atan2 gives the direction of the longer axis.
    if dx == 0.0 and dy == 0.0:
        azim = 45.0
    else:
        azim = math.degrees(math.atan2(dy, dx)) + 45.0
    # Normalise to [-180, 180].
    azim = ((azim + 180.0) % 360.0) - 180.0
    elev = float(CONFIG.viewpoint_elev)
    # Clamp elev to matplotlib's legal range.
    elev = max(-90.0, min(90.0, elev))
    return elev, azim


def resolve_viewpoint(viewpoint, vertices: np.ndarray) -> Tuple[float, float]:
    """Coerce the ``viewpoint`` argument into ``(elev, azim)``.

    Accepted forms:
      * ``"auto"``: delegated to :func:`compute_auto_viewpoint`.
      * ``(elev, azim)``: used directly.
      * ``(x, y, z)``: interpreted as a camera eye position.
    """
    if viewpoint == "auto" or viewpoint is None:
        return compute_auto_viewpoint(vertices)
    vp = tuple(viewpoint)
    if len(vp) == 2:
        return float(vp[0]), float(vp[1])
    if len(vp) == 3:
        x, y, z = float(vp[0]), float(vp[1]), float(vp[2])
        r_xy = math.hypot(x, y)
        elev = math.degrees(math.atan2(z, r_xy)) if r_xy > 0 or z != 0 else float(CONFIG.viewpoint_elev)
        azim = math.degrees(math.atan2(y, x))
        return elev, azim
    raise ValueError(
        f"viewpoint must be 'auto' | (elev, azim) | (x, y, z); got {viewpoint!r}"
    )


def add_colorbar(fig, ax, mappable, *, label: str = "", shrink=None, pad=None):
    """Attach a colorbar sized from :data:`CONFIG`."""
    cb = fig.colorbar(
        mappable,
        ax=ax,
        shrink=CONFIG.colorbar_shrink if shrink is None else shrink,
        pad=CONFIG.colorbar_pad if pad is None else pad,
    )
    if label:
        cb.set_label(label)
    return cb
