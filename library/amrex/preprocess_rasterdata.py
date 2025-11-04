#!/usr/bin/env python3
"""
update_inputs_dot.py  inputs.txt  dem.tif  release.tif  [--out new.txt]

* writes  <dem>.raw   <release>.raw  (float64 row-major)
* patches / creates the following keys in dot-syntax format

    geometry.n_cell_x
    geometry.n_cell_y
    geometry.phy_bb_x0  ...  geometry.phy_bb_y1
    geometry.dem_file
    geometry.release_file
"""

import argparse
import os
import re
from pathlib import Path
import rasterio
import numpy as np


# ---------------------------------------------------------------------- #
# helper: get raster size + bounds                                       #
# ---------------------------------------------------------------------- #
def dem_meta(fname: Path):
    with rasterio.open(fname) as ds:
        nx, ny = ds.width, ds.height
        T = ds.transform
        x0, y1 = T.c, T.f
        dx, dy = T.a, T.e  # dy negative in north-up images
        x1 = x0 + nx * dx
        y0 = y1 + ny * dy
    return nx, ny, x0, x1, y0, y1


# ---------------------------------------------------------------------- #
# main                                                                   #
# ---------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", help="existing dot-syntax inputs file")
    ap.add_argument("dem", help="DEM  GeoTIFF")
    ap.add_argument("release", help="Release-mass GeoTIFF")
    ns = ap.parse_args()
    preprocess(ns.inputs, ns.dem, ns.release)


def preprocess(inputs, dem, release):
    # ---------- 1. paths & raw files -----------------------------------
    dem_abs = Path(dem).resolve()
    rel_abs = Path(release).resolve()

    dem_raw = dem_abs.with_suffix(".raw")
    rel_raw = rel_abs.with_suffix(".raw")

    # write raw (float64) ------------------------------------------------
    for tif, raw in ((dem_abs, dem_raw), (rel_abs, rel_raw)):
        with rasterio.open(tif) as ds:
            ds.read(1).astype(np.float64).tofile(raw)

    # metadata from DEM --------------------------------------------------
    nx, ny, x0, x1, y0, y1 = dem_meta(dem_abs)

    # values we want to enforce -----------------------------------------
    updates = {
        "geometry.n_cell_x": str(nx),
        "geometry.n_cell_y": str(ny),
        "geometry.phy_bb_x0": str(x0),
        "geometry.phy_bb_x1": str(x1),
        "geometry.phy_bb_y0": str(y0),
        "geometry.phy_bb_y1": str(y1),
        "geometry.dem_file": f'"{dem_raw}"',
        "geometry.release_file": f'"{rel_raw}"',
    }

    # read original file -------------------------------------------------
    with open(inputs, "r") as f:
        lines = f.readlines()

    key_re = re.compile(r"^\s*([A-Za-z0-9_.]+)\s*=")

    # replace existing keys ---------------------------------------------
    remaining = updates.copy()
    for i, line in enumerate(lines):
        m = key_re.match(line)
        if not m:
            continue
        key = m.group(1)
        if key in remaining:
            # keep indentation / comment after value
            after_eq = line.split("=", 1)[1]
            comment = ""
            if "#" in after_eq:
                after_eq, comment = after_eq.split("#", 1)
                comment = "#" + comment
            lines[i] = f"{key} = {remaining[key]} {comment}".rstrip() + "\n"
            remaining.pop(key)

    # insert missing ones right after the "geometry" header -------------
    if remaining:
        geom_header = None
        for i, line in enumerate(lines):
            if line.lower().startswith("#") and "geometry" in line.lower():
                geom_header = i
                break
        insert_at = geom_header + 1 if geom_header is not None else len(lines)
        new_lines = [f"{k} = {v}\n" for k, v in remaining.items()]
        lines[insert_at:insert_at] = new_lines  # splice in

    # write result -------------------------------------------------------
    outfile = inputs
    with open(outfile, "w") as f:
        f.writelines(lines)

    print("✓ raw files:", dem_raw.name, rel_raw.name)
    print("✓ inputs updated →", os.path.abspath(outfile))


if __name__ == "__main__":
    main()
