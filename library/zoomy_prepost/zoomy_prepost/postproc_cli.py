"""zoomy-postproc — run one post-processing pipeline step.

    zoomy-postproc to-vtk  simulation.h5 out_dir/ [--basename simulation]
    zoomy-postproc to-h5   source(.pvd|.vtu|.vtk) out.h5
    zoomy-postproc lift3d  case_dir_or_model.py simulation.h5 out [--nz 10]

Completes the CLI story next to ``zoomy-convert`` (VTK -> HDF5 one-liner) and
``zoomy-case`` (case representations): ``to-vtk`` writes a ``.vtu`` series +
``.pvd`` from the zoomy HDF5 store, ``to-h5`` is ``zoomy-convert`` under the
step name, ``lift3d`` extrudes a depth-averaged run to a (d+1)-D VTK series
through the model's ``interpolate_to_3d``.
"""
import argparse

from .steps import hdf5_to_vtk, lift3d
from .vtk import vtk_to_hdf5


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("to-vtk", help="zoomy HDF5 store -> .vtu series + .pvd")
    p.add_argument("source", help="simulation .h5 (mesh + fields)")
    p.add_argument("out_dir", help="destination directory")
    p.add_argument("--basename", default="simulation")

    p = sub.add_parser("to-h5", help="VTK/.pvd series -> zoomy HDF5")
    p.add_argument("source", help=".pvd collection, or a .vtu/.vtk file")
    p.add_argument("output", help="destination .h5")

    p = sub.add_parser("lift3d", help="lift a run to (d+1)-D VTK via the model")
    p.add_argument("model", help="case folder (with model.py) or a model .py file")
    p.add_argument("source", help="simulation .h5 (mesh + fields)")
    p.add_argument("out", help="output base / directory / .pvd path")
    p.add_argument("--nz", type=int, default=10, help="vertical cell layers")

    a = ap.parse_args()
    if a.command == "to-vtk":
        out = hdf5_to_vtk(a.source, a.out_dir, basename=a.basename)
    elif a.command == "to-h5":
        out = vtk_to_hdf5(a.source, a.output)
    elif a.command == "lift3d":
        out = lift3d(a.model, a.source, a.out, Nz=a.nz)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
