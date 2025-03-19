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
from panel.widgets import Tqdm

import os

from library.model.model import *
from apps.gui.gui_mesh import meshes
from apps.gui.gui_model import models
from apps.gui.gui_model_editor import model_editor
from apps.gui.gui_sim_editor import sim_editor, get_simulation_setup
from apps.gui.gui_visu_editor import visu_editor
from apps.gui.gui_solver import solvers
from apps.gui.gui_visualization import visu
from apps.gui.gui_elements import MyControls
from apps.gui.gui_simulate import run as run_simulation
from tests.simulations.test_minimal import test_smm_2d

pn.extension('gridstack', 'vtk', 'mathjax', 'katex', 'ipywidgets_bokeh', 'bokeh', 'codeeditor', 'terminal', notifications=True, console_output='disable')

main_dir = os.getenv("SMS")


###  General Controls ##########################################################
button_simulate = pn.widgets.Button(name='Start simulation', button_type='primary')
#progress_bar = Tqdm(total=100, desc="Progress", unit="%")
button_session_save = pn.widgets.Button(name='Save session (WIP)', button_type='danger')
button_session_load = pn.widgets.Button(name='Load session (WIP)', button_type='danger')

general_controls = pn.Column('# Controls', button_simulate, button_session_save, button_session_load)
#############################################################

def start_simulation(event):
    
    #if not event:
    #    return

    #model_cls = models.get_model()
    #level, bc, ic, settings = get_simulation_setup()
    #model = model_cls(
    #run_simulation()
    test_smm_2d()

    

pn.bind(start_simulation, button_simulate, watch=True)


### Tabs ##########################################################
# tabs = pn.Tabs(('Mesh', gui_meshes), ('Model', gui_models), ('Solver', gui_solvers))
tabs_model = pn.Tabs(('Select', models.get_layout()), ('Editor', model_editor.get_layout()))
tabs_sim = pn.Tabs(('Select', solvers.get_layout()), ('Editor', sim_editor.get_layout()))
tabs_visu = pn.Tabs(('Select', visu.get_layout()), ('Editor', visu_editor.get_layout()))
tabs = pn.Tabs(('Mesh', meshes.get_layout()), ('Model', tabs_model),('Simulation', tabs_sim), ('Visualization', tabs_visu))
###################################################################


sidebar = pn.Column(general_controls, pn.layout.Divider(), pn.Column(*meshes.get_controls()), sizing_mode='stretch_width')


def gui():

    ### Material Template ##########################################################
    template = pn.template.BootstrapTemplate(
     # editable=True,
        title='SMS - Shallow Moment Simulation',
        logo=os.path.join(main_dir, 'apps/gui/data/logo_white.png'),
    )
    
    template.sidebar.append(sidebar)
    template.main.append(tabs)
    
    @pn.depends(tabs_model.param.active, watch=True)
    def insert_model_tab_controls(active_tab):
        if active_tab == 0:
            controls = pn.Column(*models.get_controls())
            sidebar[2] = controls
        elif active_tab == 1:
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
            controls = pn.Column(*solvers.get_controls())
            sidebar[2] = controls
        elif active_tab == 3:
            controls = pn.Column(*visu.get_controls())
            sidebar[2] = controls
    
    
    models.add_organizer(sidebar)
    model_editor.add_organizer(sidebar)
    meshes.add_organizer(sidebar)
    solvers.add_organizer(sidebar)
    
    #info = pn.state.notifications.info('This GUI is WIP. Many functions are not implemented yet.', duration=10)
    
    return template
