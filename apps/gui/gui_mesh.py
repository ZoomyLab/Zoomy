import panel as pn

from panel.layout.gridstack import GridStack
from bokeh.plotting import figure

pn.extension()

from gui.gui_elements import MyCard, MyOrganizer, MyMesh

meshes = MyOrganizer()


mesh_1 = MyMesh(meshes, title='# Create \n 1d line')
mesh_2 = MyMesh(meshes, title='# Create \n 2d square')
mesh_3 = MyMesh(meshes, title='# Create \n 3d box')
mesh_4 = MyMesh(meshes, title='# Square \n 2d, quad', path = "/home/ingo/Git/sms/meshes/quad_2d/mesh_coarse.msh")
mesh_4 = MyMesh(meshes, title='# Square \n 2d, triangle', path = "/home/ingo/Git/sms/meshes/triangle_2d/mesh_coarse.msh")
mesh_5 = MyMesh(meshes, title='# Channel \n 2d, with hole', path = "/home/ingo/Git/sms/meshes/channel_2d_hole_sym/mesh_coarse.msh")
mesh_6 = MyMesh(meshes, title='# Channel \n 2d, with junction', path = "/home/ingo/Git/sms/meshes/channel_junction/mesh_2d_coarse.msh")
mesh_7 = MyMesh(meshes, title='# Channel \n 2d, curved', path = "/home/ingo/Git/sms/meshes/curved_open_channel_extended/mesh.msh")
mesh_8 = MyMesh(meshes, title='# Nozzle \n 2d', path = "/home/ingo/Git/sms/meshes/quad_nozzle_2d/mesh_coarse.msh")
mesh_9 = MyMesh(meshes, title='# Box \n 3d, triangle', path = "/home/ingo/Git/sms/meshes/tetra_3d/mesh_coarse.msh")

meshes.add_card(mesh_1)
meshes.add_card(mesh_2)
meshes.add_card(mesh_3)
meshes.add_card(mesh_4)
meshes.add_card(mesh_5)
meshes.add_card(mesh_6)
meshes.add_card(mesh_7)
meshes.add_card(mesh_8)
meshes.add_card(mesh_9)
meshes.change_selected_card(meshes.cards[0])
gui_mesh_selection = meshes.get_layout()

###  Controls ##########################################################
button_1 = pn.widgets.Button(name='Load GMSH', button_type='primary')
# button_2 = pn.widgets.Button(name='Save', button_type='primary')
# meshes.attach_controls(['# Meshes controls', button_1, button_2])
controls = pn.Column('# Meshes controls', button_1)
meshes.attach_controls(controls)
#############################################################


