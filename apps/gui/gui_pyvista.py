import pyvista as pv
import panel as pn
import glob
import os

# Initialize the Panel extension for PyVista
pn.extension('vtk')

def create_pyvista_panel(vtk_folder, scalar_fields):
    """
    Create a Panel layout and controls for visualizing VTK series data with PyVista.
    
    Parameters:
    - vtk_folder: str
        Path to the folder containing VTK files and the out.vtk.series file.
    - scalar_fields: list of str
        Names of scalar fields available in the data.
    
    Returns:
    - layout: pn.Column
        The Panel layout containing the visualization.
    - controls: dict
        Dictionary containing the control widgets.
    """
    main_dir = os.getenv('SMS')
    vtk_path = os.path.join(main_dir, vtk_folder)
    
    # Check if the folder exists
    if not os.path.isdir(vtk_path):
        raise FileNotFoundError(f"The folder '{vtk_path}' does not exist.")

    # Read the VTK series files
    vtk_series_file = os.path.join(vtk_path, 'out.vtk.series')
    if os.path.isfile(vtk_series_file):
        # Parse the .vtk.series file to get the list of files and time steps
        import json
        with open(vtk_series_file, 'r') as f:
            series_data = json.load(f)
        
        # Construct the list of VTK file paths and time steps
        file_list = [os.path.join(vtk_path, item['name']) for item in series_data['files']]
        time_steps = [item.get('time', idx) for idx, item in enumerate(series_data['files'])]
    else:
        # If there's no .vtk.series file, get all .vtk files in the directory
        file_list = sorted(glob.glob(os.path.join(vtk_folder, '*.vtk')))
        time_steps = list(range(len(file_list)))

    if not file_list:
        raise FileNotFoundError(f"No VTK files found in folder '{vtk_path}'.")

    # Load the data into a list of meshes
    data_list = [pv.read(filename) for filename in file_list]

    # Check available scalar fields in the data
    available_scalar_fields = set()
    for mesh in data_list:
        available_scalar_fields.update(mesh.cell_data.keys())

    # Filter the scalar_fields to those that are actually available
    scalar_fields = [field for field in scalar_fields if field in available_scalar_fields]
    if not scalar_fields:
        raise ValueError("None of the specified scalar fields are available in the data.")

    # Create widgets for time step and scalar field selection
    time_slider = pn.widgets.IntSlider(
        name='Time Step',
        start=0,
        end=len(data_list)-1,
        step=1,
        value=0
    )
    scalar_selector = pn.widgets.Select(
        name='Scalar Field',
        options=scalar_fields,
        value=scalar_fields[0]
    )

    # Initialize the PyVista plotter and add the initial mesh
    plotter = pv.Plotter(window_size=[800, 600], notebook=False, off_screen=True)
    mesh = data_list[time_slider.value]
    actor = plotter.add_mesh(mesh, scalars=scalar_selector.value, cmap='viridis')
    plotter.view_xy()  # Set a consistent view direction

    # Create a Panel VTK pane to display the plotter's render window
    vtk_pane = pn.pane.VTK(plotter.ren_win, sizing_mode='stretch_both')

    # Define a function to update the visualization when the time step changes
    def update_time(event):
        time_index = time_slider.value
        scalar_field = scalar_selector.value
        mesh = data_list[time_index]
        
        # Clear the plotter and add the new mesh
        plotter.clear()
        plotter.add_mesh(mesh, scalars=scalar_field, cmap='viridis')
        plotter.view_xy()

        # Update the VTK pane with the new render window
        vtk_pane.object = plotter.ren_win

    # Define a function to update the scalar field when it changes
    def update_scalar(event):
        scalar_field = scalar_selector.value
        time_index = time_slider.value
        mesh = data_list[time_index]

        # Clear the plotter and add the mesh with the new scalar field
        plotter.clear()
        plotter.add_mesh(mesh, scalars=scalar_field, cmap='viridis')
        plotter.view_xy()

        # Update the VTK pane
        vtk_pane.object = plotter.ren_win

    # Attach the update functions to widget events
    time_slider.param.watch(update_time, 'value')
    scalar_selector.param.watch(update_scalar, 'value')

    # Create a dictionary of controls
    controls = {
        'time_slider': time_slider,
        'scalar_selector': scalar_selector
    }

    # Arrange the layout of the app
    layout = pn.Column(
        pn.Row(*controls.values()),
        vtk_pane
    )

    return layout, controls

# def create():
#     main, contr = create_pyvista_panel('outputs/test', ['0', '1', '2', '3', '4'])

#     pane = pn.Row(main, contr)
#     return pane


# pn.serve({'gui': create})
