"""CLI: a single call to convert VTK/.pvd output to zoomy HDF5.

    zoomy-convert simulation.pvd simulation.h5
"""
import argparse

from .vtk import vtk_to_hdf5


def main():
    ap = argparse.ArgumentParser(description="Convert VTK/.pvd -> zoomy HDF5.")
    ap.add_argument("source", help=".pvd collection, or a .vtu/.vtk file")
    ap.add_argument("output", help="destination .h5")
    args = ap.parse_args()
    out = vtk_to_hdf5(args.source, args.output)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
