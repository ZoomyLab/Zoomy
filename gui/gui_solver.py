import panel as pn

from panel.layout.gridstack import GridStack
from bokeh.plotting import figure

pn.extension()

from gui_elements import MyCard, MyOrganizer, MyControls

solvers = MyOrganizer()

solver_1 = MyCard(solvers, title='# generic')
solver_2 = MyCard(solvers, title='# 1d')

solvers.add_card(solver_1)
solvers.add_card(solver_2)

solvers.change_selected_card(solvers.cards[0])

###  Controls ##########################################################
button_1 = pn.widgets.Button(name='Load model', button_type='primary')
button_2 = pn.widgets.Button(name='Save model', button_type='primary')
controls = pn.Column('# Model controls', button_1, button_2)
solvers.attach_controls(controls)
#############################################################
