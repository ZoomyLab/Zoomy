import panel as pn

from apps.gui.gui import gui
#from apps.game.stream.swegame import start_game
#from apps.test_multiapp import app1

gui = gui()
gui.servable()
#pn.serve({'gui': gui})
