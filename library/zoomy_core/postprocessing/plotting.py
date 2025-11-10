import os
import json
import meshio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


# ---------- I/O ----------


def read_vtk_or_series(path, index=0, verbose=True):
    """Read either .vtk or .vtk.series (relative paths handled)."""
    path = os.path.abspath(path)

    if path.endswith(".vtk"):
        if verbose:
            print(f"📁 Reading single VTK file: {path}")
        return meshio.read(path)

    if path.endswith(".vtk.series"):
        with open(path, "r") as f:
            series = json.load(f)

        entries = series.get("files", [])
        files, times = [], []
        for e in entries:
            fname = e.get("name") or e.get("filename") or e.get("file")
            if fname:
                files.append(fname)
                times.append(e.get("time", None))

        if not files:
            raise ValueError("No valid file entries found in .vtk.series.")

        n_files = len(files)
        if verbose:
            print(f"📂 Loaded VTK series: {path}")
            print(f"   • Available indices: {n_files}")
            if any(t is not None for t in times):
                print("   • Time values:", [t for t in times if t is not None])
            print(f"   • Reading index {index}")

        if not (0 <= index < n_files):
            raise IndexError(f"Index {index} out of range (0..{n_files - 1})")

        base_dir = os.path.dirname(path)
        file_path = os.path.join(base_dir, files[index])
        return meshio.read(file_path)

    raise ValueError("Expected a .vtk or .vtk.series file")


# ---------- Field listing ----------


def list_available_fields(mesh, print_out=True):
    """List all available point and cell fields."""
    fields = {
        "point_data": list(mesh.point_data.keys()),
        "cell_data": list(mesh.cell_data_dict.keys()),
    }

    if print_out:
        print("\n📊 Available fields:")
        if fields["point_data"]:
            print("  • Point data:")
            for name in fields["point_data"]:
                arr = mesh.point_data[name]
                print(f"     - {name} ({arr.shape})")
        else:
            print("  • Point data: none")

        if fields["cell_data"]:
            print("  • Cell data:")
            for name in fields["cell_data"]:
                entry = mesh.cell_data_dict[name]
                types = ", ".join(entry.keys())
                print(f"     - {name} (types: {types})")
        else:
            print("  • Cell data: none")
        print()

    return fields


# ---------- Cell helpers ----------


def get_cell_block(mesh, types):
    for block in mesh.cells:
        if block.type in types:
            return block
    return None


def get_cell_field_for_block(mesh, field_name, cell_block):
    """Get cell-centered data for the specified block."""
    if cell_block is None:
        return None
    ctype = cell_block.type
    if field_name is not None:
        field = mesh.cell_data_dict.get(field_name)
        if field is None:
            return None
        return field.get(ctype)
    for name, mapping in mesh.cell_data_dict.items():
        if ctype in mapping:
            return mapping[ctype]
    return None


# ---------- 1D Plotting ----------


def plot_1d_mesh(mesh, ax, field_name=None, **_):
    """
    Plot a 1D mesh as a line plot on the given Axes.
    Returns (vmin, vmax) for consistency.
    """
    points = mesh.points
    x = points[:, 0]

    line_block = get_cell_block(mesh, ("line", "line3"))
    if line_block is not None:
        connectivity = line_block.data
        unique_order = []
        for conn in connectivity:
            for i in conn:
                if i not in unique_order:
                    unique_order.append(i)
        x = x[unique_order]

    y = None
    field_data = get_cell_field_for_block(mesh, field_name, line_block)
    if field_data is not None:
        cell_centers = np.array(
            [np.mean(mesh.points[cell, 0]) for cell in line_block.data]
        )
        y = np.interp(x, cell_centers, field_data)
    elif field_name and field_name in mesh.point_data:
        y = mesh.point_data[field_name]
    elif mesh.point_data:
        field_name, y = next(iter(mesh.point_data.items()))

    if y is None:
        y = np.zeros_like(x)
        field_name = field_name or "default"

    ax.plot(x, y, "-o", markersize=3)
    return float(np.min(y)), float(np.max(y))


# ---------- 2D Plotting ----------


def plot_2d_mesh(
    mesh,
    ax,
    field_name=None,
    show_legend=True,
    legend_location="right",
    cmap="viridis",
):
    """
    Plot a 2D mesh (unstructured) using meshio connectivity.
    Adds polygons to the provided Axes.
    Returns (vmin, vmax).
    """
    points = mesh.points[:, :2]
    cell_block = get_cell_block(mesh, ("triangle", "quad", "polygon"))
    if cell_block is None:
        raise ValueError("No 2D cells found.")

    connectivity = cell_block.data
    field_data = get_cell_field_for_block(mesh, field_name, cell_block)
    if field_data is None:
        field_data = np.zeros(len(connectivity))
        field_name = field_name or "default"

    polygons = [points[cell] for cell in connectivity]
    vmin, vmax = float(field_data.min()), float(field_data.max())
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5

    coll = PolyCollection(
        polygons,
        array=field_data,
        cmap=cmap,
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
        edgecolors="k",
        linewidths=0.2,
    )
    ax.add_collection(coll)
    ax.autoscale()
    ax.set_aspect("equal")

    if show_legend:
        if legend_location == "bottom":
            cbar = plt.colorbar(
                coll, ax=ax, orientation="horizontal", fraction=0.05, pad=0.05
            )
        else:  # right
            cbar = plt.colorbar(coll, ax=ax)
        cbar.set_label(field_name)

    return vmin, vmax


# ---------- Auto Dispatch ----------


def plot_mesh(
    mesh,
    ax,
    field_name=None,
    show_legend=True,
    legend_location="right",
):
    """
    Auto-detect mesh type (1D or 2D) and plot on the provided Axes.
    Returns (vmin, vmax).
    """
    has_2d = get_cell_block(mesh, ("triangle", "quad", "polygon")) is not None
    has_1d = get_cell_block(mesh, ("line", "line3")) is not None

    if has_2d:
        return plot_2d_mesh(mesh, ax, field_name, show_legend, legend_location)
    elif has_1d:
        return plot_1d_mesh(mesh, ax, field_name)
    else:
        raise ValueError(f"Unsupported mesh types: {[c.type for c in mesh.cells]}")


# # ---------- Example ----------

# if __name__ == "__main__":
#     mesh = read_vtk_or_series("outputs/swe/out.vtk.series", index=0)
#     list_available_fields(mesh)

#     fig, ax = plt.subplots(figsize=(5, 4))
#     vmin, vmax = plot_mesh(mesh, ax, field_name=None, show_legend=True, legend_location="right")
#     print(f"Color range: {vmin:.3g} – {vmax:.3g}")
#     plt.show()
