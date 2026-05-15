#!/usr/bin/env python3
"""Assemble title-slide PNG frame sequence into a video via ffmpeg.

The browser app writes ``frame_00001.png … frame_NNNNN.png`` plus a
``manifest.json`` into a chosen folder.  This script wraps ffmpeg with
three knobs that map directly to the user-facing requirements:

  * ``--duration``   target *video* length in seconds — independent of
                     how long the simulation ran.  Squeezing 60 s of sim
                     into a 15 s clip is just ``--duration 15``.
  * ``--width``      output resolution (height follows manifest aspect
                     unless ``--height`` is given).  Frames are scaled
                     up with Lanczos.
  * ``--crf``        H.264 CRF (lower = bigger / better quality; 18 is
                     visually lossless, 28 is heavily compressed).

Usage::

    python apps/title-slide/postprocess.py FRAMES_DIR OUTPUT.mp4 \
        --duration 15 --width 1920 --crf 20

Requires ``ffmpeg`` on PATH.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("frames_dir", help="directory containing frame_*.png")
    p.add_argument("output", help="output video file (.mp4 / .mov / .webm)")
    p.add_argument("--duration", type=float, default=15.0,
                   help="output video duration in seconds (default 15). All input frames are stretched/compressed to fit.")
    p.add_argument("--fps", type=int, default=30,
                   help="output frame rate in fps (default 30)")
    p.add_argument("--width", type=int, default=1920,
                   help="output width in pixels (default 1920). Frames are scaled up with Lanczos.")
    p.add_argument("--height", type=int, default=0,
                   help="output height (default 0 = derive from manifest aspect)")
    p.add_argument("--crf", type=int, default=20,
                   help="H.264 CRF; lower = better quality. Visually lossless ≈ 18, heavy compression ≈ 28 (default 20)")
    p.add_argument("--preset", default="slow",
                   help="x264 preset: ultrafast … placebo (default slow)")
    p.add_argument("--codec", default="libx264",
                   help="ffmpeg video codec (default libx264; try libx265 for .mkv, libvpx-vp9 for .webm)")
    p.add_argument("--pix-fmt", default="yuv420p",
                   help="ffmpeg pixel format (default yuv420p — broadly compatible)")
    p.add_argument("--pattern", default="frame_%05d.png",
                   help="frame filename pattern (default frame_%%05d.png)")
    p.add_argument("--dry-run", action="store_true", help="print the ffmpeg command and exit")
    args = p.parse_args()

    if shutil.which("ffmpeg") is None and not args.dry_run:
        sys.exit("ffmpeg not found on PATH — install ffmpeg first.")

    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        sys.exit(f"not a directory: {frames_dir}")

    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        sys.exit(f"no frame_*.png files in {frames_dir}")
    n_frames = len(frames)

    height = args.height
    if not height:
        manifest_path = frames_dir / "manifest.json"
        if manifest_path.exists():
            mf = json.loads(manifest_path.read_text())
            nx, ny = mf.get("nx", 0), mf.get("ny", 0)
            if nx > 0 and ny > 0:
                height = int(round(args.width * ny / nx))
        if not height:
            # fall back to 16:9 if no manifest
            height = int(round(args.width * 9 / 16))

    # Force even dimensions — yuv420p needs them.
    if args.width % 2: print(f"warning: width {args.width} is odd; bumping +1.", file=sys.stderr)
    if height % 2:    print(f"warning: height {height} is odd; bumping +1.", file=sys.stderr)
    width = args.width + (args.width % 2)
    height = height + (height % 2)

    input_fps = n_frames / args.duration

    cmd = [
        "ffmpeg", "-y",
        "-framerate", f"{input_fps:.6f}",
        "-i", str(frames_dir / args.pattern),
        "-vf", f"scale={width}:{height}:flags=lanczos,fps={args.fps}",
        "-c:v", args.codec,
        "-crf", str(args.crf),
        "-preset", args.preset,
        "-pix_fmt", args.pix_fmt,
        str(args.output),
    ]
    print("frames :", n_frames, "in", frames_dir)
    print("video  :", args.duration, "s @", args.fps, "fps →", width, "×", height)
    print("ffmpeg :", " ".join(cmd))

    if args.dry_run:
        return
    subprocess.run(cmd, check=True)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
