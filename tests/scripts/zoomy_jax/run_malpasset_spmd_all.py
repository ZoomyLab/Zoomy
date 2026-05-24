"""Run the full Malpasset SPMD scaling sweep with a single command:

    python tests/scripts/zoomy_jax/run_malpasset_spmd_all.py

Sets ``XLA_FLAGS`` + ``MALPASSET_MESH`` per case and dispatches
``malpasset_spmd_persistent.py`` as a subprocess for each cell-count
× device-count combination.  Each run is a fresh process to give JAX
a clean device count (XLA_FLAGS is set BEFORE jax imports).

Cases:
  (small mesh = 26k cells)  × {1, 4, 8} devices
  (large mesh = 104k cells) × {1, 4, 8} devices

For the 1-device case the script auto-falls-back to the single-
device baseline only (no SPMD overhead).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    "..", "..", ".."))
DATA = os.path.join(ROOT, "data", "malpasset")
SCRIPT = os.path.join(os.path.dirname(__file__),
                      "malpasset_spmd_persistent.py")

CASES = [
    # (label, mesh_filename, n_devices)
    ("small / 4 dev", "geo_malpasset-small.msh", 4),
    ("small / 8 dev", "geo_malpasset-small.msh", 8),
    ("large / 4 dev", "geo_malpasset-large.msh", 4),
    ("large / 8 dev", "geo_malpasset-large.msh", 8),
]


def run_case(label: str, mesh_filename: str, n_devices: int):
    mesh_path = os.path.join(DATA, mesh_filename)
    env = os.environ.copy()
    env["XLA_FLAGS"] = (
        f"--xla_force_host_platform_device_count={n_devices}"
    )
    env["MALPASSET_MESH"] = mesh_path
    env["JAX_PLATFORMS"] = "cpu"

    print(f"\n{'=' * 70}")
    print(f"CASE: {label}")
    print(f"  mesh   : {mesh_filename}")
    print(f"  devices: {n_devices}")
    print(f"{'=' * 70}")

    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, SCRIPT],
        env=env,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - t0

    # Print last 25 lines of stdout (the result table); drop the JAX
    # CUDA noise that prints to stderr on systems without a GPU.
    lines = proc.stdout.splitlines()
    tail = "\n".join(lines[-25:])
    print(tail)
    if proc.returncode != 0:
        print(f"  !! returncode = {proc.returncode}")
        # Surface only the actual error from stderr (skip CUDA boilerplate).
        err_lines = [ln for ln in proc.stderr.splitlines()
                     if "CUDA" not in ln and "cuda_versions" not in ln
                     and "xla_bridge" not in ln
                     and "jax_plugins" not in ln
                     and "Traceback" not in ln[:10]
                     and "RuntimeError: jaxlib" not in ln]
        if err_lines:
            print("  stderr (filtered):")
            print("\n".join(err_lines[-15:]))
    print(f"  (subprocess wall: {wall:.1f}s)")


def main():
    print("Malpasset SPMD scaling sweep — JAX persistent-buffer pattern")
    print(f"Repo: {ROOT}")
    print(f"Data: {DATA}")
    print(f"Script: {SCRIPT}")
    print(f"\n{len(CASES)} cases to run.  Each compiles the JAX flux op "
          "for the per-rank meshes (60-90s per case on CPU); total ~10-15 min.\n")

    t0 = time.perf_counter()
    for label, mesh_filename, n_devices in CASES:
        if not os.path.exists(os.path.join(DATA, mesh_filename)):
            print(f"\n[SKIP] {label}: mesh not found "
                  f"({mesh_filename})")
            continue
        run_case(label, mesh_filename, n_devices)
    print(f"\n{'=' * 70}")
    print(f"ALL CASES DONE — total wall = {time.perf_counter() - t0:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
