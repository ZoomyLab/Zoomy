#!/usr/bin/env python3
"""Colour and assemble a title-slide capture into a video.

The simulation runner (TBD — JAX-based, coming when the JAX solver
returns) writes the raw SWE state into ``frames.npy``:

  * shape  ``(T, NY, NX, 3)``
  * dtype  ``float32``
  * channels  ``[h, hu, hv]``  (water height + the two discharges)
  * row order  ``j`` increasing south → north (postprocess vflips for the video)

…plus a ``manifest.json`` with at least ``{nx, ny, nFrames, endTime}``.
This script picks a field, computes the colour scale from the **whole
timeline** (so every frame uses the same saturation), applies a
colormap, and pipes RGB to ffmpeg.

Three user-facing knobs map directly to the requirements:

  * ``--duration``   target *video* length in seconds; independent of
                     how long the simulation ran.
  * ``--width``      output resolution (height follows the captured
                     aspect ratio unless ``--height`` is given).
  * ``--crf``        H.264 quality / size knob (lower = bigger / better;
                     visually lossless ≈ 18, heavy compression ≈ 28).

Usage::

    python apps/title-slide/postprocess.py FRAMES_DIR OUTPUT.mp4 \
        --field h --colormap water --duration 15 --width 1920

Requires ``ffmpeg``, ``numpy``, and ``matplotlib`` (used for its
``LinearSegmentedColormap`` builder).
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


# Colormap definitions — kept in lockstep with apps/title-slide/colormaps.js
# so the live preview and the postprocess output share the same palette.
_STOPS = {
    "water":      [(0.00, "#08233a"), (0.30, "#11507a"), (0.60, "#2b8fc7"),
                   (0.85, "#7ec8e3"), (1.00, "#e6f6fb")],
    "deepwater":  [(0.00, "#020912"), (0.40, "#093058"), (0.75, "#1f6aa0"),
                   (1.00, "#a8d8ee")],
    "ice":        [(0.00, "#f3f8fc"), (0.40, "#bfdbed"), (0.75, "#5e8fb6"),
                   (1.00, "#1a3b5e")],
    "viridis":    [(0.00, "#440154"), (0.25, "#3b528b"), (0.50, "#21918c"),
                   (0.75, "#5ec962"), (1.00, "#fde725")],
    "magma":      [(0.00, "#000004"), (0.25, "#3b0f70"), (0.50, "#8c2981"),
                   (0.75, "#de4968"), (1.00, "#fcfdbf")],
    "sunset":     [(0.00, "#0d1b2a"), (0.40, "#7a3c2a"), (0.75, "#f08a3e"),
                   (1.00, "#fde7c4")],
    "grayscale":  [(0.00, "#000000"), (1.00, "#ffffff")],
}


def build_lut(name: str) -> np.ndarray:
    """256×3 uint8 colormap matching the JS gradient builder."""
    from matplotlib.colors import LinearSegmentedColormap
    stops = _STOPS[name]
    cmap = LinearSegmentedColormap.from_list(name, stops, N=256)
    rgba = (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
    return rgba[:, :3]


WET = 1e-6


def derive_field(state: np.ndarray, field: str) -> np.ndarray:
    """state has shape (..., 3); channels are (h, hu, hv)."""
    h = state[..., 0]
    if field == "h":
        return h
    if field == "vmag":
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(h > WET, state[..., 1] / np.where(h > WET, h, 1.0), 0.0)
            v = np.where(h > WET, state[..., 2] / np.where(h > WET, h, 1.0), 0.0)
        return np.sqrt(u * u + v * v)
    raise ValueError(f"unknown field: {field!r}")


def parse_auto(s: str) -> float | None:
    if s == "auto" or s is None:
        return None
    return float(s)


def estimate_scale(frames: np.ndarray, field: str, vmin_user, vmax_user, pct: float):
    """Compute (vmin, vmax) from the whole timeline of the chosen field."""
    fld = derive_field(frames, field)
    vmin = vmin_user if vmin_user is not None else 0.0
    if vmax_user is not None:
        vmax = vmax_user
    else:
        # percentile across full T × NY × NX volume; robust to outlier
        # cells (a few super-supercritical jets won't crush the rest).
        vmax = float(np.percentile(fld, pct))
        if vmax <= vmin:
            vmax = float(fld.max())
        if vmax <= vmin:
            vmax = vmin + 1e-6
    return vmin, vmax


def colour_frame(fld: np.ndarray, vmin: float, vmax: float, lut: np.ndarray) -> np.ndarray:
    """fld: (NY, NX) float; returns (NY, NX, 3) uint8."""
    norm = np.clip((fld - vmin) / (vmax - vmin), 0.0, 1.0)
    idx = (norm * 255).astype(np.int32)
    return lut[idx]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("frames_dir", help="folder containing frames.npy + manifest.json")
    ap.add_argument("output", help="output video file (.mp4 / .mov / .webm)")

    ap.add_argument("--field", choices=["h", "vmag"], default="h",
                    help="which field to colour (default h)")
    ap.add_argument("--colormap", choices=list(_STOPS), default="water",
                    help="colormap name (default water)")

    ap.add_argument("--duration", type=float, default=15.0,
                    help="output video duration in seconds (default 15). Captured frames are stretched/compressed in time to fit.")
    ap.add_argument("--fps", type=int, default=30,
                    help="output frame rate (default 30)")
    ap.add_argument("--width", type=int, default=1920,
                    help="output width in pixels (default 1920)")
    ap.add_argument("--height", type=int, default=0,
                    help="output height (default 0 = derive from captured aspect)")
    ap.add_argument("--crf", type=int, default=20,
                    help="H.264 CRF: lower = bigger / better quality. Visually lossless ≈ 18, heavy ≈ 28 (default 20)")
    ap.add_argument("--preset", default="slow",
                    help="x264 preset: ultrafast … placebo (default slow)")
    ap.add_argument("--codec", default="libx264",
                    help="ffmpeg video codec (default libx264)")
    ap.add_argument("--pix-fmt", default="yuv420p",
                    help="ffmpeg pixel format (default yuv420p)")

    ap.add_argument("--scale-min", default="auto",
                    help="scale floor: 'auto' (=0) or float")
    ap.add_argument("--scale-max", default="auto",
                    help="scale ceiling: 'auto' (=percentile of timeline) or float")
    ap.add_argument("--scale-percentile", type=float, default=99.5,
                    help="percentile of timeline values for the auto scale ceiling (default 99.5)")

    ap.add_argument("--dry-run", action="store_true", help="print the ffmpeg command and exit")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None and not args.dry_run:
        sys.exit("ffmpeg not found on PATH — install ffmpeg first.")

    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        sys.exit(f"not a directory: {frames_dir}")
    npy_path = frames_dir / "frames.npy"
    if not npy_path.exists():
        sys.exit(f"no frames.npy in {frames_dir} — was the capture run?")

    manifest = {}
    mp = frames_dir / "manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text())

    # mmap_mode keeps the (T, NY, NX, 3) array on disk; we touch one
    # frame at a time so big captures (multi-GB) still fit on a laptop.
    frames = np.load(npy_path, mmap_mode="r")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        sys.exit(f"unexpected frames.npy shape {frames.shape}; expected (T, NY, NX, 3)")
    T, NY, NX, _ = frames.shape

    # If the capture stopped early, the header claims more frames than
    # were actually written. Trust the manifest's count.
    actual = manifest.get("framesActuallyWritten")
    if actual and actual < T:
        frames = frames[:actual]
        T = actual

    vmin, vmax = estimate_scale(
        frames, args.field,
        parse_auto(args.scale_min), parse_auto(args.scale_max),
        args.scale_percentile,
    )
    print(f"timeline: T={T}, NX={NX}, NY={NY}")
    print(f"field   : {args.field}")
    print(f"scale   : [{vmin:.4f}, {vmax:.4f}]  ({'auto p='+str(args.scale_percentile) if args.scale_max=='auto' else 'user'})")
    print(f"colormap: {args.colormap}")

    height = args.height
    if not height:
        height = int(round(args.width * NY / NX))
    if args.width % 2:  print(f"warning: width {args.width} odd; bumping +1.", file=sys.stderr)
    if height % 2:      print(f"warning: height {height} odd; bumping +1.", file=sys.stderr)
    width = args.width + (args.width % 2)
    height = height + (height % 2)

    input_fps = T / args.duration
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{NX}x{NY}",
        "-framerate", f"{input_fps:.6f}",
        "-i", "-",
        # The WebGL framebuffer has y=0 at the bottom, so the captured
        # rows go south → north; vflip puts north on top of the video.
        "-vf", f"vflip,scale={width}:{height}:flags=lanczos,fps={args.fps}",
        "-c:v", args.codec, "-crf", str(args.crf), "-preset", args.preset,
        "-pix_fmt", args.pix_fmt,
        str(args.output),
    ]
    print("ffmpeg  :", " ".join(cmd))
    if args.dry_run:
        return

    lut = build_lut(args.colormap)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for t in range(T):
            fld = derive_field(np.asarray(frames[t]), args.field)
            rgb = colour_frame(fld, vmin, vmax, lut)
            proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
            if t % 60 == 0 or t == T - 1:
                print(f"  frame {t+1}/{T}", end="\r", file=sys.stderr)
        print(file=sys.stderr)
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            sys.exit(f"ffmpeg failed with exit code {rc}")
    except (BrokenPipeError, KeyboardInterrupt):
        proc.kill()
        sys.exit("interrupted.")

    print("wrote", args.output)


if __name__ == "__main__":
    main()
