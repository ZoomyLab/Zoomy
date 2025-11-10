import os
import numpy as np
import panel as pn
from glob import glob

try:
    import pyvista as pv

    _HAVE_PYVISTA = True
except ImportError:
    _HAVE_PYVISTA = False

import zoomy_core.misc.io as io
from zoomy_core.misc import misc as misc


if _HAVE_PYVISTA:
    pn.extension("vtk")


def pyvista_3d(folder, filename='out_3d', scale=1.):
    if not _HAVE_PYVISTA:
        raise ImportError("pyvista is required for pyvista_3d function.")
    # Set up
    main_dir = misc.get_main_directory()

    settings = io.load_settings(os.path.join(main_dir, folder))
    output_dir = os.path.join(main_dir, settings.output.directory)
    vtk_files = sorted(glob(os.path.join(output_dir, f"{filename}.*.vtk")))
    max_vtk_files = 10
    if len(vtk_files) > max_vtk_files:
        offset = int(len(vtk_files)  / 10)
        vtk_files = vtk_files[::offset]

    # Constants
    x_fixed, y_fixed = 0.3, 0.05
    n_points = 10
    scale_v = 0.3

    def scale_mesh_by_height(mesh, scale=1.):
        # Only modify points if necessary
        pmesh = mesh.cell_data_to_point_data()
        points = pmesh.points.copy()
        if "0" in pmesh.point_data:
            points[:, 2] = pmesh["0"] * points[:, 2] * scale
            mesh.points = points
        return mesh

    def add_velocity_field(mesh):
        # Add V field for vector (optional)
        try:
            f1 = mesh["1"]
            f2 = mesh["2"]
            V = np.column_stack((f1, f2, np.zeros_like(f1)))
            mesh["V"] = V
        except KeyError:
            pass
        return mesh

    # Utility: Load mesh & update field_selector
    def load_mesh(vtk_path):
        mesh = pv.read(vtk_path)
        mesh = add_velocity_field(mesh)
        mesh = scale_mesh_by_height(mesh, scale)

        return mesh

    meshes = [load_mesh(vtk_path) for vtk_path in vtk_files]


    # Widgets
    field_selector = pn.widgets.Select(name="Select Field", options=[], sizing_mode='stretch_width')
    time_slider = pn.widgets.IntSlider(name="Time Step", start=0, end=len(vtk_files) - 1, step=1, value=0, sizing_mode='stretch_width')
    show_mesh_checkbox = pn.widgets.Checkbox(name="Show Mesh", value=False)


    vtk_pane_container = pn.Column()
    plotter = pv.Plotter()
    plotter.set_background("lightgray")
    vtk_pane = pn.pane.VTK(plotter.ren_win, height=500, sizing_mode="stretch_width")
    vtk_pane_container.append(vtk_pane)




    def update_plot(event=None):
        plotter.clear()
        # plotter = pv.Plotter(off_screen=True)
        # plotter.set_background("lightgray")
        try:
            plotter.remove_scalar_bar()
        except:
            pass
        
        mesh = meshes[time_slider.value]




        # Setup available fields
        fields = list(mesh.cell_data.keys())
        
        if field_selector.options != fields:
            field_selector.options = fields
            field_selector.value = fields[0] if fields else None
        scalar_name = field_selector.value if field_selector.value in mesh.cell_data else "0"

        vmin, vmax = mesh.get_data_range(arr_var=scalar_name)
        plotter.add_mesh(mesh, scalars=scalar_name, opacity=0.5, clim=[vmin, vmax], 
                        scalar_bar_args=dict(       
                        title=scalar_name,
                        vertical=True,             
                        interactive=False,
                        outline=False,
                        title_font_size=35,
                        label_font_size=30,
                        fmt="%.5f",))

        # if show_mesh_checkbox.value == True:
        #     plotter.add_mesh(mesh, style='wireframe', color='black', opacity=0.3)
        plotter.reset_camera()
        vtk_pane.object = plotter.ren_win
        vtk_pane.param.trigger('object') 
        
    # Trigger update
    time_slider.param.watch(update_plot, "value")
    field_selector.param.watch(update_plot, "value")
    show_mesh_checkbox.param.watch(update_plot, "value")

    # Initial load
    update_plot()

    # Layout
    sidebar = pn.Column(
        "## Controls",
        time_slider,
        field_selector,
        show_mesh_checkbox,
        width=250,
    )

    layout = pn.Row(sidebar,pn.Spacer(width=5), vtk_pane_container)
    return layout