import panel as pn

from panel.layout.gridstack import GridStack
from bokeh.plotting import figure

pn.extension('codeeditor', 'mathjax', 'katex')
from gui.gui_elements import MyCard, MyOrganizer, MyControls, MyModel
from library.model.model import *

models = MyOrganizer()

model_0 = MyModel(models, title='# ModelGUI \n 1d', code=ModelGUI)
model_1 = MyModel(models, title='# SWE \n 1d', code=ShallowWater)
model_2 = MyModel(models, title='# SWE \n 2d', code=ShallowWater2d)
model_3 = MyModel(models, title='# SME \n 1d')
model_4 = MyModel(models, title='# SME \n 2d')
model_5 = MyModel(models, title='# SFF \n 1d')
model_6 = MyModel(models, title='# SFF \n 2d')
model_7 = MyModel(models, title='# Steffer \n 1d')
model_8 = MyModel(models, title='# Steffler \n 2d')
model_9 = MyModel(models, title='# SME Ref. \n 2d')
model_10 = MyModel(models, title='# SME Ref. \n 3d')

models.add_card(model_0)
models.add_card(model_1)
models.add_card(model_2)
models.add_card(model_3)
models.add_card(model_4)
models.add_card(model_5)
models.add_card(model_6)
models.add_card(model_7)
models.add_card(model_8)
models.add_card(model_9)
models.add_card(model_10)
models.change_selected_card(models.cards[0])
gui_model_selection = models.get_layout()

###  Controls ##########################################################
button_1 = pn.widgets.Button(name='Load model', button_type='primary')
button_2 = pn.widgets.Button(name='Save model', button_type='primary')
controls = pn.Column('# Model controls', button_1, button_2)
models.attach_controls(controls)
#############################################################
