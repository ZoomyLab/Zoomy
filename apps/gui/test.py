import panel as pn
import holoviews as hv
from holoviews import streams

pn.extension()
hv.extension('bokeh')

# Create an empty Path for drawing
path = hv.Path([], kdims=['x', 'y']).opts(
    height=400, width=600,
    tools=['freehand_draw'],          # Use only the FreehandDraw tool
    active_tools=['freehand_draw'],   # Activate FreehandDraw tool by default
    xlabel='', ylabel='',             # Hide axis labels
    xaxis=None, yaxis=None,           # Hide axes
    toolbar='disable',                # Disable the toolbar
    shared_axes=False                 # Do not share axes (prevents auto-scaling)
)

# Initialize the FreehandDraw stream
draw_stream = streams.FreehandDraw(source=path)

# Create the Panel layout
app = pn.Column(path)

# Serve the app
app.servable()
