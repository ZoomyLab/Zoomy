import panel as pn

# from apps.gui.gui import gui
from apps.game.stream.swe_game import start_game
#from apps.test_multiapp import app1

#gui = gui()
#gui.servable()
game  = start_game()
#game, freehand, rect, stream, image  = start_game()
game.servable()
#pn.serve({'gui': gui})
