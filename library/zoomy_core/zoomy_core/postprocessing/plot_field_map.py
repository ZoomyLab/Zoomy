"""Render a cell field from a triangular `.vtu` as a colored 2D map.

General-purpose VTU field plotter (a symmetric diverging colorbar makes
it well suited to signed fields such as sensitivities, but any cell
field works).  Two modes:

* **single** — one .vtu → one map:

```
      python -m zoomy_core.postprocessing.plot_field_map \\
          path/to/snapshot.vtu --field <name> --out map.png --clip 99
```

* **panels** — several .vtu side-by-side with a shared symmetric
  colorbar (e.g. a parameter / resolution sweep):

  ```
      python -m zoomy_core.postprocessing.plot_field_map \\
          --inputs run0/snapshot.vtu run1/snapshot.vtu run2/snapshot.vtu \\
          --field <name> --out compare.png --title "sweep"
  ```

`--clip P` symmetrically clips the colorbar to the P-th percentile
of |field|, so the dynamic range isn't blown out by outliers (e.g.
wet/dry artefacts).
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import meshio
from matplotlib.collections import PolyCollection

from . import style


def _read_field(vtu_path, field_name):
    """Return `(points2d, triangle_conn, cell_data, h_field)` for a .vtu."""
    m = meshio.read(vtu_path)
    pts = m.points[:, :2]
    tri = next(cb for cb in m.cells if cb.type == "triangle")
    data = m.cell_data_dict.get(field_name, {}).get("triangle")
    if data is None:
        avail = [k for k, v in m.cell_data_dict.items() if "triangle" in v]
        raise KeyError(
            f"Field {field_name!r} not in {vtu_path}; available: {avail}"
        )
    h = np.asarray(
        m.cell_data_dict.get("h", {}).get("triangle", np.zeros(len(tri.data)))
    )
    return pts, tri.data, np.asarray(data), h


def plot_map(vtu_path, field_name, out_path, clip=99.0,
             cmap=None, title=None):
    """Single .vtu → one colored 2D map PNG."""
    style.use()
    cmap = cmap or style.CMAP_DIVERGING
    pts, conn, data, _ = _read_field(vtu_path, field_name)

    # Robust symmetric colorbar via percentile.
    abs_data = np.abs(data[np.isfinite(data)])
    if abs_data.size:
        vmax = float(np.percentile(abs_data, clip)) or float(abs_data.max())
    else:
        vmax = 1.0
    if vmax == 0:
        vmax = 1.0
    vmin = -vmax

    polys = [pts[c] for c in conn]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    coll = PolyCollection(
        polys, array=data, cmap=cmap,
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
        edgecolors="none",
    )
    ax.add_collection(coll)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if title is None:
        title = field_name
    ax.set_title(
        f"{title}\n"
        f"min={data.min():.3g}  max={data.max():.3g}  "
        f"mean|·|={np.mean(np.abs(data)):.3g}  "
        f"(colorbar clipped at P{clip:.0f})"
    )
    cbar = plt.colorbar(coll, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(field_name)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_panels(inputs, field_name, out_path, clip=99.0,
                cmap=None, suptitle=None):
    """Several .vtu side-by-side with one shared symmetric colorbar.

    Each panel is titled by its parent dirname (e.g. `smt_lvl0_...`).
    The colorbar limit is the max of the per-panel P{clip} percentiles
    so all maps are directly comparable.
    """
    style.use()
    cmap = cmap or style.CMAP_DIVERGING
    panels = []
    for path in inputs:
        if not os.path.exists(path):
            print(f"skip {path} (not found)")
            continue
        pts, conn, data, h = _read_field(path, field_name)
        if not np.isfinite(data).any():
            print(f"skip {path}: all-NaN field {field_name!r}")
            continue
        panels.append((path, pts, conn, data, h))
    if not panels:
        raise SystemExit("no VTUs with finite data found.")

    vmax_candidates = []
    for _, _, _, d, _ in panels:
        finite = d[np.isfinite(d)]
        if finite.size:
            v = float(np.percentile(np.abs(finite), clip))
            if v > 0:
                vmax_candidates.append(v)
    vmax = max(vmax_candidates) if vmax_candidates else 1.0
    vmin = -vmax

    fig, axes = plt.subplots(
        1, len(panels), figsize=(5.0 * len(panels), 4.2),
        squeeze=False,
    )
    coll = None
    for ax, (path, pts, conn, data, h) in zip(axes[0], panels):
        polys = [pts[c] for c in conn]
        coll = PolyCollection(
            polys, array=data, cmap=cmap,
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
            edgecolors="none",
        )
        ax.add_collection(coll)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        tag = os.path.basename(os.path.dirname(path))
        ax.set_title(
            f"{tag}\n"
            f"min={data.min():.2g}, max={data.max():.2g}, "
            f"mean|·|={np.mean(np.abs(data)):.2g}"
        )
    axes[0, 0].set_ylabel("y [m]")
    cbar = fig.colorbar(
        coll, ax=axes[0, :].ravel().tolist(), fraction=0.012, pad=0.02,
    )
    cbar.set_label(f"{field_name}   (clipped P{clip:.0f})")
    if suptitle:
        fig.suptitle(suptitle)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("vtu", nargs="?",
                   help="single mode: one .vtu to render.")
    p.add_argument("--inputs", nargs="+", default=None,
                   help="panels mode: several .vtu rendered side-by-side "
                        "with a shared colorbar.")
    p.add_argument("--field", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--clip", type=float, default=99.0)
    p.add_argument("--title", default=None,
                   help="single: axis title; panels: figure suptitle.")
    args = p.parse_args(argv)

    if args.inputs:
        out = plot_panels(args.inputs, args.field, args.out,
                          clip=args.clip, suptitle=args.title)
    elif args.vtu:
        out = plot_map(args.vtu, args.field, args.out,
                       clip=args.clip, title=args.title)
    else:
        p.error("give either a positional VTU (single mode) "
                "or --inputs (panels mode)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
