import panel as pn

from panel.layout.gridstack import GridStack
from bokeh.plotting import figure

pn.extension('codeeditor', 'mathjax', 'katex')

import apps.gui.test_eqn as test_code
from apps.gui.gui_elements import MyBasicOrganizer, MyOrganizer
from library.model.model import *


sim_setup_default = """
level = 0
settings = Settings(
    name="ShallowMoments2d",
    parameters={"g": 1.0, "C": 1.0, "nu": 0.1},
    reconstruction=recon.constant,
    num_flux=flux.LLF(),
    compute_dt=timestepping.adaptive(CFL=0.45),
    time_end=1.0,
    output_snapshots=100, output_dir='outputs/test')

inflow_dict = {i: 0.0 for i in range(1, 2 * (1 + level) + 1)}
inflow_dict[1] = 0.36
outflow_dict = {0: 1.0}

bcs = BC.BoundaryConditions(
    [
        BC.Wall(physical_tag="top"),
        BC.Wall(physical_tag="bottom"),
        BC.InflowOutflow(physical_tag="left", prescribe_fields=inflow_dict),
        BC.InflowOutflow(physical_tag="right", prescribe_fields= outflow_dict),
    ]
)
ic = IC.Constant(
    constants=lambda n_fields: np.array(
        [1.0, 0.1, 0.1] + [0.0 for i in range(n_fields - 3)]
    )
)
"""


editor_simulation = pn.widgets.CodeEditor(value=sim_setup_default, sizing_mode='stretch_width', language='python', theme='monokai', height=200)


controls = []

main = GridStack(sizing_mode='stretch_both', min_height=60, min_width=190)
main[0:8, 0:6] = editor_simulation
#main[0:8, 6:12] = editor_IC
#main[8:12, 0:6] = editor_BC
#main[8:12, 6:12] = editor_IC

sim_editor = MyBasicOrganizer(main)
sim_editor.attach_controls(controls)

def get_simulation_setup():
    exec(editor_simulation.value)
    return level, bcs, ics, settings


# Serve the panel
# pn.serve(main)
