import panel as pn

from panel.layout.gridstack import GridStack
from bokeh.plotting import figure

pn.extension('codeeditor', 'mathjax', 'katex')

import gui.test as test_code
from gui.docstring_crawler import get_class_code
from gui.gui_elements import MyBasicOrganizer
from library.model.model import ShallowWater

# code_file = open('./test.py', 'r')
# code_text = code_file.read()
code_text = ''
# code_text = get_class_code(ShallowWater, func_name='flux')
editor = pn.widgets.CodeEditor(value=code_text, sizing_mode='stretch_width', language='python', theme='monokai', height=300)

def update_terminal_from_editor(button, terminal, editor):
    terminal.value = editor.value


import numpy as np
import pyvista as pv

def make_points():
    """Helper to make XYZ points"""
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return np.column_stack((x, y, z))

def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points)-1, 3), 2, dtype=np.int32)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int32)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int32)
    poly.lines = cells
    return poly

points = make_points()
line = lines_from_points(points)
line["scalars"] = np.arange(line.n_points) # By default pyvista use viridis colormap
tube = line.tube(radius=0.1) #=> the object we will represent in the scene


pl = pv.Plotter()
pl.camera_position =  [(4.440892098500626e-16, -21.75168228149414, 4.258553981781006),
                       (4.440892098500626e-16, 0.8581731039809382, 0),
                       (0, 0.1850949078798294, 0.982720673084259)]

pl.add_mesh(tube, smooth_shading=True)
spline_pan = pn.panel(pl.ren_win, width=500, orientation_widget=True)
# spline_pan
pan_clone = spline_pan.clone() # we clone the panel to animate only this panel
pan_clone.unlink_camera() # we don't want to share the camera with the previous panel
player = pn.widgets.Player(name='Player', start=0, end=100, value=0, loop_policy='reflect', interval=100)
scalars = tube["scalars"]

def animate(value):
    tube["scalars"] = np.roll(scalars, 200*value)
    pan_clone.actors[0].SetOrientation(0, 0, -20*value)
    pan_clone.synchronize()

visu_window = pn.Column(pan_clone, player, pn.bind(animate, player))


main = GridStack(sizing_mode='stretch_both', min_height=60, min_width=190)
main[0:9, 0:6] = editor
main[0:9, 6:12] = visu_window
main[9:12, 0:6] = pn.Spacer(styles=dict(background='purple'))
main[9:12, 6:12] = pn.Spacer(styles=dict(background='yellow'))

visu_editor = MyBasicOrganizer(main)


checkbox_1 = pn.widgets.Checkbox(name='Show visu')
checkbox_2 = pn.widgets.Checkbox(name='Show Eigenvalues')
controls = pn.Column('# Editor controls', checkbox_1, checkbox_2)
visu_editor.attach_controls(controls)
