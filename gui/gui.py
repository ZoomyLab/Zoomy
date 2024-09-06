import panel as pn

from panel.layout.gridstack import GridStack
from bokeh.plotting import figure

import vtk
from vtk.util.colors import tomato

import numpy as np
from matplotlib.figure import Figure
from matplotlib import cm
import matplotlib.pyplot as plt

import param
from panel.reactive import ReactiveHTML
from panel.viewable import Viewer

from gui_mesh import meshes
from gui_model import models
from gui_model_editor import model_editor
from gui_solver import solvers
from gui_visualization import visu
from gui_elements import MyControls
# from gui_solver_selection import gui_solver_selection, gui_solver_selection_controls

pn.extension('gridstack', 'vtk', 'mathjax', 'katex', 'ipywidgets_bokeh', 'bokeh', 'codeeditor', 'terminal', console_output='disable')



###  General Controls ##########################################################
button_1 = pn.widgets.Button(name='Start', button_type='primary')
button_2 = pn.widgets.Button(name='Stop', button_type='primary')

general_controls = pn.Column('# Controls', button_1, button_2)
#############################################################


### Tabs ##########################################################
# tabs = pn.Tabs(('Mesh', gui_meshes), ('Model', gui_models), ('Solver', gui_solvers))
tabs_model = pn.Tabs(('Select', models.get_layout()), ('Editor', model_editor.get_layout()))
tabs = pn.Tabs(('Mesh', meshes.get_layout()), ('Model', tabs_model),('Simulation', solvers.get_layout()), ('Visualization', visu.get_layout()))
###################################################################


sidebar = pn.Column(general_controls, pn.layout.Divider(), pn.Column(*meshes.get_controls()), sizing_mode='stretch_width')



### Material Template ##########################################################
template = pn.template.BootstrapTemplate(
 # editable=True,
    title='Sim',
    logo='./data/logo_white.png',
)

template.sidebar.append(sidebar)
template.main.append(tabs)

@pn.depends(tabs_model.param.active, watch=True)
def insert_model_tab_controls(active_tab):
    if active_tab == 0:
        print('tab0')
        controls = pn.Column(*models.get_controls())
        sidebar[2] = controls
    elif active_tab == 1:
        print('tab1')
        controls = pn.Column(*model_editor.get_controls())
        sidebar[2] = controls


@pn.depends(tabs.param.active, watch=True)
def insert_tab_controls(active_tab):
    if active_tab == 0:
        controls = pn.Column(*meshes.get_controls())
        sidebar[2] = controls
    elif active_tab == 1:
        insert_model_tab_controls(tabs_model.active)
        # controls = pn.Column(*models.get_controls())
        # sidebar[2] = controls
    elif active_tab == 2:
        sidebar.append(*solvers.get_controls())
    elif active_tab == 3:
        sidebar.append(*visu.get_controls())


    
models.add_organizer(sidebar)
model_editor.add_organizer(sidebar)
meshes.add_organizer(sidebar)
solvers.add_organizer(sidebar)


template.servable();
