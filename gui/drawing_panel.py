import panel as pn
from bokeh.plotting import figure
from bokeh.models import FreehandDrawTool, ColumnDataSource, BoxEditTool

# Load Panel extension
pn.extension()

# Create a data source for the freehand drawings
freehand_source = ColumnDataSource(data=dict(xs=[], ys=[]))

# Create a Bokeh figure with explicit ranges and no default tools
p = figure(
    tools=[],                  # Start with no default tools
    toolbar_location='above',  # Include the toolbar
    x_range=(0, 1),            # Set x-axis range from 0 to 1
    y_range=(0, 1),            # Set y-axis range from 0 to 1
    height=400,
    width=600
)

# Customize the axes for a cleaner look
p.xaxis.visible = False     # Hide x-axis
p.yaxis.visible = False     # Hide y-axis
p.grid.grid_line_color = None  # Hide gridlines

# Add a multi_line glyph to the figure for freehand drawing
renderer = p.multi_line('xs', 'ys', source=freehand_source, line_width=2)

# Create the FreehandDrawTool and associate it with the renderer
freehand_draw_tool = FreehandDrawTool(renderers=[renderer])

# Add the FreehandDrawTool to the figure's tools
p.add_tools(freehand_draw_tool)

# Set the FreehandDrawTool as the active drag tool
p.toolbar.active_drag = freehand_draw_tool

# Limit the toolbar to only the FreehandDrawTool
#p.toolbar.tools = [freehand_draw_tool]
#p.toolbar.logo = None  # Remove the Bokeh logo for a cleanerrect_source = ColumnDataSource(data=dict(x=[], y=[], width=[], height=[], angle=[])) look

rect_source = ColumnDataSource(data=dict(x=[], y=[], width=[], height=[], angle=[]))

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
#p.toolbar.logo = None
#p.toolbar.tools = [box_edit_tool]


# Add a button to print the drawing data
print_button = pn.widgets.Button(name='Print Drawing Data')

def print_data(event):
    print(freehand_source.data)

print_button.on_click(print_data)

# Update the app layout to include the button
app = pn.Column(p, print_button)

# Serve the app
app.servable()
