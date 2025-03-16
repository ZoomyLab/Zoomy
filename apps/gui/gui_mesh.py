import panel as pn

from panel.layout.gridstack import GridStack
from bokeh.plotting import figure

pn.extension()

from apps.gui.gui_elements import MyCard, MyOrganizer, MyMesh

meshes = MyOrganizer()


meshes.add_card(MyMesh(meshes, title='# Create \n 1d line', path = "/home/ingo/Git/sms/meshes/line/mesh.msh", wip=True))
meshes.add_card(MyMesh(meshes, title='# Create \n 2d square', wip=True))
meshes.add_card(MyMesh(meshes, title='# Square \n 2d, quad', path = "/home/ingo/Git/sms/meshes/quad_2d/mesh_coarse.msh"))
meshes.add_card(MyMesh(meshes, title='# Square \n 2d, triangle', path = "/home/ingo/Git/sms/meshes/triangle_2d/mesh_coarse.msh"))
meshes.add_card(MyMesh(meshes, title='# Channel \n 2d, with hole', path = "/home/ingo/Git/sms/meshes/channel_2d_hole_sym/mesh_coarse.msh"))
meshes.add_card(MyMesh(meshes, title='# Channel \n 2d, with junction', path = "/home/ingo/Git/sms/meshes/channel_junction/mesh_2d_coarse.msh"))
meshes.add_card(MyMesh(meshes, title='# Channel \n 2d, curved', path = "/home/ingo/Git/sms/meshes/curved_open_channel_extended/mesh.msh"))
meshes.add_card(MyMesh(meshes, title='# Nozzle \n 2d', path = "/home/ingo/Git/sms/meshes/quad_nozzle_2d/mesh_coarse.msh"))
meshes.add_card(MyMesh(meshes, title='# Box \n 3d, triangle', path = "/home/ingo/Git/sms/meshes/tetra_3d/mesh_coarse.msh"))

meshes.change_selected_card(meshes.cards[6])
gui_mesh_selection = meshes.get_layout()

###  Controls ##########################################################
button_1 = pn.widgets.Button(name='Upload GMSH (WIP)', button_type='danger')
# button_2 = pn.widgets.Button(name='Save', button_type='primary')
# meshes.attach_controls(['# Meshes controls', button_1, button_2])
controls = pn.Column('# Meshes controls', button_1)
meshes.attach_controls(controls)
#############################################################


