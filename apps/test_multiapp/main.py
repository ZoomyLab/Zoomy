import panel as pn

from app1 import app1
from app2 import app2

pn.serve({'app1': app1, 'app2': app2})
