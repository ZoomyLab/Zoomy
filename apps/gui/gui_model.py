import panel as pn

from panel.layout.gridstack import GridStack
from bokeh.plotting import figure

pn.extension('codeeditor', 'mathjax', 'katex')
from apps.gui.gui_elements import MyCard, MyOrganizer, MyControls, MyModel
from library.model.model import ShallowWater, ShallowWater2d, ShallowMoments, ShallowMoments2d, ModelGUI

models = MyOrganizer()

models.add_card(MyModel(models, title='# New Model \n 1d', code=ModelGUI))
models.add_card(MyModel(models, title='# Shallow Water \n 1d', code=ShallowWater))
models.add_card(MyModel(models, title='# Shallow Water \n 2d', code=ShallowWater2d))
models.add_card(MyModel(models, title='# Shallow Moments \n 1d', code=ShallowMoments))
models.add_card(MyModel(models, title='# Shallow Moments \n 2d', code=ShallowMoments2d))
models.add_card(MyModel(models, title='# Shear Shallow Flow \n 1d', wip=True))
models.add_card(MyModel(models, title='# Shear Shallow Flow \n 2d', wip=True))
models.add_card(MyModel(models, title='# VAM \n 1d', wip=True))
models.add_card(MyModel(models, title='# VAM \n 2d', wip=True))
models.add_card(MyModel(models, title='# Shallow Moments Ref. \n 2d', wip=True))
models.add_card(MyModel(models, title='# Shallow Moments Ref. \n 3d', wip=True))

models.change_selected_card(models.cards[4])
gui_model_selection = models.get_layout()

###  Controls ##########################################################
button_upload = pn.widgets.Button(name='Upload model (WIP)', button_type='danger')
button_download = pn.widgets.Button(name='Download model (WIP)', button_type='danger')
controls = pn.Column('# Model controls', button_upload, button_download)
models.attach_controls(controls)
#############################################################
