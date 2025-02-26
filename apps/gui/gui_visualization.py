import panel as pn

from panel.layout.gridstack import GridStack
from bokeh.plotting import figure

pn.extension()

from gui.gui_elements import MyCard, MyOrganizer, MyControls

visu = MyOrganizer()

visu_1 = MyCard(visu, title='# VTK \n 1d, 2d, 3d')
visu_2 = MyCard(visu, title='# PyVista (VTK) \n 1d, 2d, 3d ')
visu_3 = MyCard(visu, title='# Matplotlib \n 2d')
visu_4 = MyCard(visu, title='# Plotly \n 2d')

visu.add_card(visu_1)
visu.add_card(visu_2)
visu.add_card(visu_3)
visu.add_card(visu_4)

visu.change_selected_card(visu.cards[0])

###  Controls ##########################################################
button_1 = pn.widgets.Button(name='Load output', button_type='primary')
button_2 = pn.widgets.Button(name='Save current view', button_type='primary')
controls = pn.Column('# Visualization controls', button_1, button_2)
visu.attach_controls(controls)
#############################################################
