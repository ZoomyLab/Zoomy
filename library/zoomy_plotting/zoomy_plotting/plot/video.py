"""Bundle matplotlib figures into a gif / mp4 — the generic animation engine.

Any per-frame figure composer works::

    from zoomy_plotting import animate

    animate(lambda fig, t: my_figure(fig, t), times, "out.gif", fps=8)

``draw(fig, item)`` receives a fresh figure and one element of ``frames``
(a time, a time-step index, a (path, t) tuple — whatever the composer
needs).  Rendering is off-screen (Agg) and headless-safe; ``.gif`` and
``.mp4`` are picked by the output extension (mp4 needs imageio-ffmpeg).

Style note: the composer draws under whatever rcParams are active —
call :func:`zoomy_plotting.use`\\ ("screen") first for animation-sized
fonts (gifs/slides), or "print" for figure-sized output.
"""

from __future__ import annotations

import numpy as np


def render_frame(draw, item, figsize=(10, 6), dpi=100) -> np.ndarray:
    """Render one composed figure to an RGB array (one gif frame)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, dpi=dpi)
    try:
        draw(fig, item)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    finally:
        plt.close(fig)


def animate(draw, frames, out, fps=8, figsize=(10, 6), dpi=100):
    """Render ``draw(fig, item)`` for every item in ``frames`` and bundle
    the figures into ``out`` (.gif or .mp4).  Returns ``out``."""
    import imageio.v2 as imageio

    imgs = [render_frame(draw, item, figsize=figsize, dpi=dpi)
            for item in frames]
    if not imgs:
        raise ValueError("frames is empty — nothing to animate")
    if str(out).endswith(".mp4"):
        imageio.mimsave(out, imgs, fps=fps)
    else:
        imageio.mimsave(out, imgs, fps=fps, loop=0)
    return out
