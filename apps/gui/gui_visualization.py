import panel as pn

from panel.layout.gridstack import GridStack
from bokeh.plotting import figure

pn.extension()

from apps.gui.gui_elements import MyCard, MyOrganizer, MyControls

visu = MyOrganizer()

visu.add_card(MyCard(visu, title='# VTK \n 1d, 2d, 3d', wip=True))
visu.add_card(MyCard(visu, title='# PyVista (VTK) \n 1d, 2d, 3d ', wip=False))
visu.add_card(MyCard(visu, title='# Matplotlib \n 1d, 2d', wip=True))
visu.add_card(MyCard(visu, title='# Plotly \n 1d, 2d', wip=True))

visu.change_selected_card(visu.cards[1])

###  Controls ##########################################################
button_1 = pn.widgets.Button(name='Download VTK', button_type='primary')
button_2 = pn.widgets.Button(name='Download current image', button_type='primary')
controls = pn.Column('# Visualization controls', button_1, button_2)
visu.attach_controls(controls)
#############################################################
