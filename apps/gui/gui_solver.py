import panel as pn

from panel.layout.gridstack import GridStack
from bokeh.plotting import figure

pn.extension()

from apps.gui.gui_elements import MyCard, MyOrganizer, MyControls

solvers = MyOrganizer()

solvers.add_card(MyCard(solvers, title='# FVM generic (Numpy) \n CPU'))
solvers.add_card(MyCard(solvers, title='# FVM generic (Jax) \n CPU + GPU + AD', wip=False))
solvers.add_card(MyCard(solvers, title='# FVM generic (Kokkos) \n GPU+CPU', wip=True))
solvers.add_card(MyCard(solvers, title='# FVM generic (PETSc) \n MPI+AMR, CPU+GPU', wip=True))
solvers.add_card(MyCard(solvers, title='# FVM structured (JAX) \n CPU + GPU + AD ', wip=True))
solvers.add_card(MyCard(solvers, title='# ADER-FVM structured (JAX) \n CPU + GPU + AD ', wip=True))

solvers.change_selected_card(solvers.cards[1])

###  Controls ##########################################################
button_1 = pn.widgets.Button(name='Load setup', button_type='primary')
button_2 = pn.widgets.Button(name='Save setup', button_type='primary')
controls = pn.Column('# Simulation controls', button_1, button_2)
solvers.attach_controls(controls)
#############################################################
