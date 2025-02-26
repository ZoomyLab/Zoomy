import panel as pn
import holoviews as hv
from bokeh.models import FreehandDrawTool

# Initialize Panel and Holoviews extensions
pn.extension()
hv.extension('bokeh')

# Create an empty Path element (used for freehand drawing)
path = hv.Path([]).opts(line_width=2)

# Global variable to store renderer reference for easy access
renderer_ref = {}

# Add FreehandDrawTool to plot via hooks and store renderer reference
def add_freehand_tool(plot, element):
    tool = FreehandDrawTool(renderers=[plot.handles['glyph_renderer']])
    plot.state.add_tools(tool)
    plot.state.toolbar.active_drag = tool
    
    # Store renderer reference for later use
    renderer_ref['renderer'] = plot.handles['glyph_renderer']

dmap = hv.DynamicMap(lambda: path).opts(
    hooks=[add_freehand_tool],
    width=400,
    height=400,
)

# Function to print current path data directly from the renderer's ColumnDataSource
def print_current_data():
    if 'renderer' in renderer_ref:
        cds = renderer_ref['renderer'].data_source  # Get ColumnDataSource
        
        if 'xs' in cds.data and 'ys' in cds.data:
            for xs, ys in zip(cds.data['xs'], cds.data['ys']):
                print("New Segment:")
                for x, y in zip(xs, ys):
                    print(f"({x}, {y})")

# Layout with clear button functionality
def clear_canvas(event):
    if 'renderer' in renderer_ref:
        renderer_ref['renderer'].data_source.data.update(xs=[], ys=[])

reset_button = pn.widgets.Button(name='Clear Canvas')
reset_button.on_click(clear_canvas)

print_button = pn.widgets.Button(name='Print Data')
print_button.on_click(lambda event: print_current_data())

layout = pn.Column(reset_button, print_button, dmap)

layout.servable()
