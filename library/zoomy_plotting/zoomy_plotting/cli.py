"""Small CLI for zoomy-plotting.

Installed as ``zoomy-plot`` via ``[project.scripts]``. Reads a zoomy_core
HDF5 file and produces a PNG (or saves the figure in any matplotlib format).

Examples::

    zoomy-plot sim.h5 --field h
    zoomy-plot sim.h5 --field h --time 5 --out out.png
    zoomy-plot sim.h5 --field q1 --time -1 --show-mesh
    zoomy-plot sim.h5 --list-fields
    zoomy-plot sim.h5 --field h --viewpoint 30,-60          # 3D only
    zoomy-plot sim.h5 --animate --field h --out movie.gif   # timeline to GIF
"""

from __future__ import annotations

import argparse
import sys

import matplotlib
import matplotlib.pyplot as plt

from . import MatplotlibPlotter, apply_style, read_hdf5


def _parse_viewpoint(s):
    if s is None or s == "auto":
        return "auto"
    parts = [p.strip() for p in s.split(",")]
    return tuple(float(p) for p in parts)


def _build_parser():
    p = argparse.ArgumentParser(
        prog="zoomy-plot",
        description="Plot a Zoomy simulation HDF5 snapshot (1D/2D/3D).",
    )
    p.add_argument("h5", help="Path to the HDF5 file.")
    p.add_argument("--field", "-f", default=None,
                   help="Field name or integer index (default: first available).")
    p.add_argument("--time", "-t", default=-1, type=int,
                   help="Snapshot index (default: -1, the last).")
    p.add_argument("--out", "-o", default=None,
                   help="Output file. Default: derived from --h5. Set to '-' to show on screen.")
    p.add_argument("--show-mesh", action="store_true",
                   help="Overlay mesh edges (2D/3D).")
    p.add_argument("--interpolate", action="store_true",
                   help="Per-vertex interpolation instead of per-cell flat shading.")
    p.add_argument("--cmap", default=None,
                   help="Override matplotlib colormap.")
    p.add_argument("--viewpoint", default="auto",
                   help="3D viewpoint: 'auto', 'elev,azim', or 'x,y,z'.")
    p.add_argument("--figsize", default=None,
                   help="Figure size as 'W,H' in inches.")
    p.add_argument("--dpi", default=130, type=int)
    p.add_argument("--list-fields", action="store_true",
                   help="List available fields and exit.")
    p.add_argument("--animate", action="store_true",
                   help="Write an animated GIF over the full timeline.")
    p.add_argument("--fps", default=10, type=int,
                   help="Frames per second for --animate.")
    return p


def _default_out(h5_path, animate=False):
    import os
    stem = os.path.splitext(os.path.basename(h5_path))[0]
    return f"{stem}.{'gif' if animate else 'png'}"


def _parse_field(spec, store):
    if spec is None:
        keys = list(store.field.keys())
        return keys[0] if keys else 0
    try:
        return int(spec)
    except ValueError:
        return spec


def _make_ax(fig, store):
    if store.dim == 3:
        return fig.add_subplot(111, projection="3d")
    return fig.add_subplot(111)


def _plot_once(plotter, ax, time_step, field, args):
    kwargs = {}
    if args.cmap is not None:
        kwargs["cmap"] = args.cmap
    if store_dim := plotter.store.dim:
        if store_dim >= 2:
            kwargs["show_mesh"] = args.show_mesh
            kwargs["interpolate"] = args.interpolate
        if store_dim == 3:
            kwargs["viewpoint"] = _parse_viewpoint(args.viewpoint)
    plotter.plot(ax, time_step=time_step, field=field, **kwargs)


def _run_static(args):
    matplotlib.use("Agg" if args.out != "-" else "TkAgg", force=True)
    store = read_hdf5(args.h5)
    try:
        if args.list_fields:
            print("cell fields:")
            for k, v in store.field.items():
                print(f"  [{v:>2}] {k}")
            print(f"snapshots: {store.n_snapshots}, dim: {store.dim}, "
                  f"cell_type: {store.cell_type}")
            return 0

        field = _parse_field(args.field, store)
        plotter = MatplotlibPlotter(store)
        figsize = tuple(float(x) for x in args.figsize.split(",")) if args.figsize \
                  else ((8, 6) if store.dim != 3 else (7, 6))
        with apply_style():
            fig = plt.figure(figsize=figsize, dpi=args.dpi)
            ax = _make_ax(fig, store)
            _plot_once(plotter, ax, args.time, field, args)
            fig.tight_layout()
            if args.out == "-":
                plt.show()
            else:
                out = args.out or _default_out(args.h5)
                fig.savefig(out)
                print(f"wrote {out}")
        plt.close(fig)
        return 0
    finally:
        store.close()


def _run_animation(args):
    import matplotlib.animation as anim
    matplotlib.use("Agg", force=True)
    store = read_hdf5(args.h5)
    try:
        field = _parse_field(args.field, store)
        plotter = MatplotlibPlotter(store)
        figsize = tuple(float(x) for x in args.figsize.split(",")) if args.figsize \
                  else ((8, 6) if store.dim != 3 else (7, 6))

        n = store.n_snapshots
        if n <= 1:
            print("file has only one snapshot; nothing to animate", file=sys.stderr)
            return 1

        with apply_style():
            fig = plt.figure(figsize=figsize, dpi=args.dpi)
            ax = _make_ax(fig, store)

            def draw(t):
                ax.clear()
                _plot_once(plotter, ax, t, field, args)
                return ax

            writer = anim.PillowWriter(fps=args.fps)
            out = args.out or _default_out(args.h5, animate=True)
            a = anim.FuncAnimation(fig, draw, frames=n, interval=1000 / args.fps)
            a.save(out, writer=writer)
            print(f"wrote {out}  ({n} frames)")
        plt.close(fig)
        return 0
    finally:
        store.close()


def main(argv=None):
    args = _build_parser().parse_args(argv)
    if args.animate:
        return _run_animation(args)
    return _run_static(args)


if __name__ == "__main__":
    raise SystemExit(main())
