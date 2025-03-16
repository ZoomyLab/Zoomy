import panel as pn

from apps.gui.main import gui
from apps.test_multiapp import app1

pn.serve({'gui': gui, 'test': app1})
