import panel as pn

from panel.layout.gridstack import GridStack
from bokeh.plotting import figure

pn.extension()

from gui_elements import MyCard, MyOrganizer, MyControls

visu = MyOrganizer()

visu_1 = MyCard(visu, title='# generic')
visu_2 = MyCard(visu, title='# 1d')

visu.add_card(visu_1)
visu.add_card(visu_2)

visu.change_selected_card(visu.cards[0])

###  Controls ##########################################################
button_1 = pn.widgets.Button(name='Load model', button_type='primary')
button_2 = pn.widgets.Button(name='Save model', button_type='primary')
controls = pn.Column('# Model controls', button_1, button_2)
visu.attach_controls(controls)
#############################################################
