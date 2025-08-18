import panel as pn
import vtk
from vtk.util.colors import tomato
import numpy as np
from matplotlib.figure import Figure
from matplotlib import cm
import matplotlib.pyplot as plt
from panel.viewable import Viewer
import param
import os

import library.mesh.mesh as petscMesh

from apps.gui.docstring_crawler import get_class_docstring
from apps.gui.mesh.load_gmsh import load_gmsh

main_dir = os.getenv("ZOOMY_DIR")

pn.extension('gridstack', 'vtk', 'mathjax', 'katex', 'ipywidgets_bokeh', 'codeeditor', 'terminal', console_output='disable')


class MyControls(param.Parameterized):
    def __init__(self, json):
        self.parameters = None
        if json is not None:
            parameters = []
            if 'parameters' in json:
                for k, v in json['parameters'].items():
                    if v['type'] == 'int':
                        parameters.append(pn.widgets.IntInput(name=k, value=v['value'], step=v['step']))
                    elif v['type'] == 'float':
                        parameters.append(pn.widgets.FloatInput(name=k, value=v['value'], step=v['step']))
                    elif v['type'] == 'string':
                        parameters.append(pn.widgets.TextInput(name=k, value=v['value']))
                    elif v['type'] == 'array':
                        parameters.append(pn.widgets.ArrayInput(name=k, value=v['value']))
                self.parameters = parameters


    def get_controls(self):
        if self.parameters is not None:
            return self.parameters
        else:
            return []

class MyBasicOrganizer(Viewer):

    def __init__(self, layout,  **params):
        self.organizer = None
        self.update_sidebar = None
        self.general_controls = []
        super().__init__(**params)
        self._layout = layout

    def add_organizer(self, organizer):
        self.organizer = organizer

    def add_card(self, card):
        self.cards.append(card)
        self._layout=pn.FlexBox(*self.cards)

    def get_layout(self):
        return self._layout

    def change_selected_card(self, card):
        if self.selected is not None and self.selected is not card:
            self.selected.deselect()
        self.selected = card
        card.select()

    def update_organizer_controls(self):
        if self.organizer is not None:
            self.organizer[2] = pn.Column(*self.get_controls())


    def attach_controls(self, controls):
        self.general_controls = controls
    
    def get_controls(self):
        return pn.Column(*self.general_controls)    



class MyOrganizer(Viewer):

    def __init__(self, **params):
        self.organizer = None
        self.cards = []
        self.update_sidebar = None
        self.selected = None
        self.general_controls = []
        self.card_controls = []
        super().__init__(**params)
        self._layout = pn.FlexBox()

    def add_organizer(self, organizer):
        self.organizer = organizer

    def add_card(self, card):
        self.cards.append(card)
        self._layout=pn.FlexBox(*self.cards)

    def get_layout(self):
        return self._layout

    def change_selected_card(self, card):
        if self.selected is not None and self.selected is not card:
            self.selected.deselect()
        self.selected = card
        card.select()

    def update_organizer_controls(self):
        if self.organizer is not None:
            self.organizer[2] = pn.Column(*self.get_controls())
        else:
            print('update_organizer_controls: No organizer found')


    def attach_controls(self, controls):
        self.general_controls = controls

    def attach_card_controls(self, controls):
        self.card_controls = controls
    
    def get_controls(self):
        return pn.Column(*self.general_controls, *self.card_controls)
        


class MyCard(Viewer):

    default_style = {'border': '1px solid white', 'border-radius': '10px'}
    selected_style = {'border': '1px solid black', 'border-radius': '10px'}

    def __init__(self, organizer, title='# Card', image=None, code=None, wip=False, **params):
        self.title = title
        self._code=code
        self._doc= get_class_docstring(self._code)
        self._control = MyControls(self._doc)
        if image is not None:
            self.fig = pn.pane.PNG(image, width=300)
        else:
            # image=os.path.join(main_dir, "apps/gui/data/sample.png")
            # self.fig = pn.pane.PNG(image, width=300)
            self.fig = None
        if not wip:
            self.button = pn.widgets.Button(name='Select', button_type='primary', width=300)
        else:
            self.button = pn.widgets.Button(name='WIP', button_type='danger', width=300)
        self.organizer = organizer
        super().__init__(**params)
        self._layout = pn.Column(self.title, self.fig, self.button, styles=self.default_style)

        def handle_select(event):
            self.organizer.change_selected_card(self)

        self.button.on_click(handle_select)

    def get_controls(self):
        return self._control.get_controls()

    def select(self):
        self._layout.styles = self.selected_style
        self.organizer.attach_card_controls(self.get_controls())
        self.organizer.update_organizer_controls()

    def deselect(self):
        self._layout.styles = self.default_style

    def __panel__(self):
        return self._layout

class MyModel(MyCard):

    def __init__(self, organizer, title='# Card', image=None, code=None, **params):
        super().__init__(organizer, title, image, code,**params)
        self._code=code
        self._doc= get_class_docstring(self._code)
        self._control = MyControls(self._doc)
        if image is not None:
            self.fig = pn.pane.PNG(image, width=300)

        else:
            latex = pn.pane.LaTeX( r'$\partial_t {Q} + \nabla \cdot {F}({Q}) + {NC}({Q}) : \nabla {Q} = {S}({Q})$', width=300)
            self.fig = latex

        self._layout = pn.Column(self.title, self.fig, self.button, styles=self.default_style)

    def get_model(self):
        return self._code

class MyMesh(MyCard):

    def __init__(self, organizer, title='# Card', image=None, code=None, path=None, **params):
        super().__init__(organizer, title, image, code,**params)
        if path is not None:
            plots, cells = load_gmsh(path)
            self.fig = plots[0]
            self.path = path
        elif image is not None:
            self.fig = pn.pane.PNG(image, width=300)
        else:
            image=os.path.join(main_dir, "apps/gui/data/sample_mesh.png")
            self.fig = pn.pane.PNG(image, width=300)
            
        self._layout = pn.Column(self.title, self.fig, self.button, styles=self.default_style)


    def get_mesh(self):
        mesh = petscMesh.Mesh.from_gmsh(
            os.path.join(main_dir, self.path),
        )
        return mesh


### Terminal ###############################################################

# terminal = pn.widgets.Terminal(
#     code_text,
#     options={"cursorBlink": False, "cursorInactiveStyle": None, "cursorWidth": 0},
#     sizing_mode='stretch_both'
# )
##################################################################

submit = pn.widgets.Button(name="Start the wind turbine")
def start_stop_wind_turbine(clicked):
    if submit.clicks % 2:
        submit.name = "Start the wind turbine"
    else:
        submit.name = "Stop the wind turbine"
    markdown.object = editor.value
pn.bind(start_stop_wind_turbine, submit, watch=True)
