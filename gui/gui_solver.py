import panel as pn

from panel.layout.gridstack import GridStack
from bokeh.plotting import figure

pn.extension()

from gui.gui_elements import MyCard, MyOrganizer, MyControls

solvers = MyOrganizer()

solver_1 = MyCard(solvers, title='# FVM generic')
solver_2 = MyCard(solvers, title='# FVM generic (Kokkos) \n GPU+CPU')
solver_3 = MyCard(solvers, title='# FVM generic (PETSc) \n MPI+AMR, CPU+GPU')
solver_4 = MyCard(solvers, title='# FVM generic (Jax) \n CPU + GPU')
solver_5 = MyCard(solvers, title='# FVM 1d')
solver_6 = MyCard(solvers, title='# FVM structured')
solver_7 = MyCard(solvers, title='# ADER-FVM structured')

solvers.add_card(solver_1)
solvers.add_card(solver_2)
solvers.add_card(solver_3)
solvers.add_card(solver_4)
solvers.add_card(solver_5)
solvers.add_card(solver_6)
solvers.add_card(solver_7)

solvers.change_selected_card(solvers.cards[0])

###  Controls ##########################################################
button_1 = pn.widgets.Button(name='Load solver', button_type='primary')
button_2 = pn.widgets.Button(name='Save solver', button_type='primary')
controls = pn.Column('# Solver controls', button_1, button_2)
solvers.attach_controls(controls)
#############################################################
