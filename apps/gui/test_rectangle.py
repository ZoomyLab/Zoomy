import panel as pn
from bokeh.plotting import figure
from bokeh.models import BoxEditTool, ColumnDataSource

# Load Panel extension
pn.extension()

# Create a data source for rectangles
rect_source = ColumnDataSource(data=dict(x=[], y=[], width=[], height=[], angle=[]))

# Create a Bokeh figure
p = figure(
    toolbar_location='above',
    x_range=(0, 1),
    y_range=(0, 1),
    height=400,
    width=600
)

# Customize the axes for a cleaner look
p.xaxis.visible = False
p.yaxis.visible = False
p.grid.visible = False

# Add a rectangle glyph to the figure
rect_renderer = p.rect('x', 'y', 'width', 'height', source=rect_source,
                       fill_color='green', fill_alpha=0.4, line_color='black')

# Create the BoxEditTool and associate it with the rectangle renderer
box_edit_tool = BoxEditTool(renderers=[rect_renderer])

# Add the BoxEditTool to the figure's tools
p.add_tools(box_edit_tool)

# Set the BoxEditTool as the active drag tool
p.toolbar.active_drag = box_edit_tool

# Limit the toolbar to only the BoxEditTool
p.toolbar.logo = None
p.toolbar.tools = [box_edit_tool]

# Create the Panel layout
app = pn.Column(p)

# Serve the app
app.servable()
