"""vtk_to_gif — bundle a VTK snapshot series into an mp4 / gif.

Reads either a `.pvd` index, a `.vtk.series` file, or a directory
of `snap_*.vtu` snapshots; renders each frame with matplotlib via
`zoomy_core.postprocessing.plotting.plot_2d_mesh` (so the look stays
in sync with the rest of the codebase); then bundles the PNG frames
with `ffmpeg` into an mp4 (default) and optionally a gif.

Default field is `h`.  The color limits are computed once across
all frames so the colorbar is stable through the animation.

Pass `--right` to render two series **side-by-side** in each frame
(O1 vs O2 η-MUSCL, two solvers, …) — both must share the same
`cells`/`points` layout and ideally the same frame count.

Examples
--------
    # mp4 of h over the whole malpasset run
    python -m zoomy_core.postprocessing.vtk_to_gif \\
        --in outputs/malpasset_spmd_small_4dev_T20/vtk \\
        --field h

    # also write a gif at the same fps
    python -m zoomy_core.postprocessing.vtk_to_gif \\
        --in outputs/malpasset_spmd_small_4dev_T20/vtk \\
        --field h --gif

    # tighter color range + show the mesh
    python -m zoomy_core.postprocessing.vtk_to_gif \\
        --in outputs/malpasset_spmd_small_4dev_T20/vtk \\
        --field h --vmin 0 --vmax 60 --with-edges

    # side-by-side O1 vs O2 comparison
    python -m zoomy_core.postprocessing.vtk_to_gif \\
        --in  outputs/.../T20/vtk    --label "O1" \\
        --right outputs/.../T20_o2/vtk --right-label "O2 η-MUSCL" \\
        --field h --gif
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import meshio  # noqa: E402
import numpy as np  # noqa: E402

from zoomy_core.postprocessing import style  # noqa: E402
from zoomy_core.postprocessing.plotting import (  # noqa: E402
    list_series_frames,
    plot_2d_mesh,
    read_vtk_or_series,
)


def _resolve_inputs(in_path):
    """Resolve a --in / --right path to `(frame_meshio_paths, times, base_dir)`."""
    in_path = os.path.abspath(in_path)
    if os.path.isdir(in_path):
        # Sorted .vtu files; times unknown -> use frame index.
        vtus = sorted(glob.glob(os.path.join(in_path, "*.vtu")))
        if not vtus:
            raise SystemExit(f"No .vtu files in {in_path}")
        return vtus, list(range(len(vtus))), in_path
    if in_path.endswith((".pvd", ".vtk.series")):
        files, times = list_series_frames(in_path)
        base = os.path.dirname(in_path)
        paths = [os.path.join(base, f) for f in files]
        # Replace None timestamps with frame index for the title.
        times = [t if t is not None else i for i, t in enumerate(times)]
        return paths, times, base
    raise SystemExit(f"Unsupported path: {in_path}")


def _scan_field_range(frame_paths, field):
    """One-pass min/max of `field` across every frame."""
    fmin, fmax = np.inf, -np.inf
    for p in frame_paths:
        m = meshio.read(p)
        if field in m.cell_data_dict:
            for arr in m.cell_data_dict[field].values():
                fmin = min(fmin, float(np.min(arr)))
                fmax = max(fmax, float(np.max(arr)))
        elif field in m.point_data:
            arr = m.point_data[field]
            fmin = min(fmin, float(np.min(arr)))
            fmax = max(fmax, float(np.max(arr)))
        else:
            raise SystemExit(
                f"Field '{field}' not present in {p}. "
                f"Cell fields: {list(m.cell_data_dict.keys())}; "
                f"point fields: {list(m.point_data.keys())}"
            )
    if not np.isfinite(fmin) or not np.isfinite(fmax):
        raise SystemExit(f"No usable values for field '{field}'.")
    return fmin, fmax


def _render_frame(
    frame_path, field, t, vmin, vmax, cmap, edgecolors, linewidths,
    png_out, dpi, figsize, label=None,
    right_path=None, right_t=None, right_label=None,
):
    """Render one PNG frame — single panel, or side-by-side if
    `right_path` is given."""
    mL = read_vtk_or_series(frame_path, verbose=False)
    if right_path is None:
        fig, ax = plt.subplots(figsize=figsize)
        plot_2d_mesh(
            mL, ax,
            field_name=field, show_legend=True, cmap=cmap,
            vmin=vmin, vmax=vmax, edgecolors=edgecolors,
            linewidths=linewidths, colorbar_label=field,
        )
        title = f"{label}   " if label else ""
        ax.set_title(f"{title}{field}   t = {t:.3f}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    else:
        mR = read_vtk_or_series(right_path, verbose=False)
        fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)
        plot_2d_mesh(
            mL, axL, field_name=field, show_legend=False, cmap=cmap,
            vmin=vmin, vmax=vmax, edgecolors=edgecolors,
            linewidths=linewidths,
        )
        plot_2d_mesh(
            mR, axR, field_name=field, show_legend=True, cmap=cmap,
            vmin=vmin, vmax=vmax, edgecolors=edgecolors,
            linewidths=linewidths, colorbar_label=field,
        )
        axL.set_title(f"{label or 'left'}   t = {t:.2f}s")
        axR.set_title(f"{right_label or 'right'}   t = {right_t:.2f}s")
        for ax in (axL, axR):
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(png_out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _make_mp4(png_pattern, out_mp4, fps, crf):
    """ffmpeg → h264 mp4 (browser-playable)."""
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", png_pattern,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-movflags", "+faststart",
        out_mp4,
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)


def _make_gif(png_pattern, out_gif, fps, max_width):
    """ffmpeg → palette-based gif (small + nice colors)."""
    with tempfile.NamedTemporaryFile(
        suffix=".png", delete=False
    ) as palette:
        palette_path = palette.name
    try:
        vf_palette = (
            f"fps={fps},scale={max_width}:-1:flags=lanczos,palettegen"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", png_pattern, "-vf", vf_palette,
             palette_path],
            check=True, stderr=subprocess.DEVNULL,
        )
        lavfi = (
            f"fps={fps},scale={max_width}:-1:flags=lanczos [x];"
            "[x][1:v] paletteuse"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", png_pattern, "-i", palette_path,
             "-lavfi", lavfi, out_gif],
            check=True, stderr=subprocess.DEVNULL,
        )
    finally:
        os.unlink(palette_path)


def main(argv=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Bundle .vtu snapshots into mp4 / gif via matplotlib + ffmpeg.",
    )
    p.add_argument("--in", dest="in_path", required=True,
                   help="Directory of .vtu files, OR a .pvd / .vtk.series index.")
    p.add_argument("--right", default=None,
                   help="Optional second series → render side-by-side.")
    p.add_argument("--label", default=None,
                   help="Title for the (left) series.")
    p.add_argument("--right-label", default=None,
                   help="Title for the right series (with --right).")
    p.add_argument("--field", default="h",
                   help="Field to render (default: h).")
    p.add_argument("--out", default=None,
                   help="Output mp4 path (default: <input_dir>/<field>[_compare].mp4).")
    p.add_argument("--gif", action="store_true",
                   help="Also write a .gif next to the mp4.")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--cmap", default=None,
                   help="Colormap (default: the style scheme's continuous map).")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--with-edges", action="store_true",
                   help="Draw mesh cell edges (default off — too dense for 26k cells).")
    p.add_argument("--dpi", type=int, default=110)
    p.add_argument("--figsize", nargs=2, type=float, default=None,
                   metavar=("W", "H"),
                   help="Default 8x6 single / 14x5 side-by-side.")
    p.add_argument("--gif-width", type=int, default=900,
                   help="Max width of gif in pixels (mp4 uses raw PNG size).")
    p.add_argument("--crf", type=int, default=23,
                   help="x264 CRF (18..28; lower = better quality / bigger).")
    p.add_argument("--keep-frames", action="store_true",
                   help="Keep the per-frame PNGs (otherwise discarded).")
    args = p.parse_args(argv)

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")

    # Animations target the screen profile (scaled-up fonts on big canvases).
    style.use("screen")
    if args.cmap is None:
        args.cmap = style.CMAP_CONTINUOUS

    paired = args.right is not None
    figsize = tuple(args.figsize) if args.figsize is not None else (
        (14.0, 5.0) if paired else (8.0, 6.0))

    frame_paths, times, base_dir = _resolve_inputs(args.in_path)
    if paired:
        right_paths, right_times, _ = _resolve_inputs(args.right)
        n = min(len(frame_paths), len(right_paths))
        if len(frame_paths) != len(right_paths):
            print(f"WARN: frame counts differ "
                  f"({len(frame_paths)} vs {len(right_paths)}); using first {n}")
    else:
        right_paths, right_times = None, None
        n = len(frame_paths)
    print(f"Frames    : {n}  ({frame_paths[0]} … {frame_paths[n - 1]})")
    print(f"Field     : {args.field}" + ("  (side-by-side)" if paired else ""))

    # Global color range (over both series when paired).
    if args.vmin is not None and args.vmax is not None:
        vmin, vmax = args.vmin, args.vmax
        print(f"vmin/vmax : {vmin} / {vmax}  (user)")
    else:
        f0, f1 = _scan_field_range(frame_paths[:n], args.field)
        if paired:
            r0, r1 = _scan_field_range(right_paths[:n], args.field)
            f0, f1 = min(f0, r0), max(f1, r1)
        vmin = args.vmin if args.vmin is not None else f0
        vmax = args.vmax if args.vmax is not None else f1
        print(f"vmin/vmax : {vmin:.4g} / {vmax:.4g}  (auto)")

    default_name = f"{args.field}_compare.mp4" if paired else f"{args.field}.mp4"
    out_mp4 = args.out or os.path.join(base_dir, default_name)
    out_gif = os.path.splitext(out_mp4)[0] + ".gif"

    # Render PNG frames.
    edgecolors = "k" if args.with_edges else "none"
    linewidths = 0.15 if args.with_edges else 0.0
    if args.keep_frames:
        frames_dir = os.path.join(base_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        cleanup = False
    else:
        frames_dir = tempfile.mkdtemp(prefix="vtk_to_gif_")
        cleanup = True

    try:
        png_pattern = os.path.join(frames_dir, "f_%04d.png")
        for i in range(n):
            png_out = os.path.join(frames_dir, f"f_{i:04d}.png")
            _render_frame(
                frame_paths[i], args.field, times[i], vmin, vmax, args.cmap,
                edgecolors, linewidths, png_out, args.dpi, figsize,
                label=args.label,
                right_path=right_paths[i] if paired else None,
                right_t=right_times[i] if paired else None,
                right_label=args.right_label,
            )
            if (i + 1) % 10 == 0 or i == n - 1:
                print(f"  rendered {i + 1}/{n}")

        print(f"\nffmpeg → mp4: {out_mp4}")
        _make_mp4(png_pattern, out_mp4, args.fps, args.crf)
        mp4_kb = os.path.getsize(out_mp4) // 1024
        print(f"  size = {mp4_kb} KB")

        if args.gif:
            print(f"\nffmpeg → gif: {out_gif}")
            _make_gif(png_pattern, out_gif, args.fps, args.gif_width)
            gif_kb = os.path.getsize(out_gif) // 1024
            print(f"  size = {gif_kb} KB")
    finally:
        if cleanup:
            shutil.rmtree(frames_dir, ignore_errors=True)

    print(f"\nDone. mp4: {out_mp4}" + (f"  gif: {out_gif}" if args.gif else ""))


if __name__ == "__main__":
    main()
