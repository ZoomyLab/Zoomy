import panel as pn
from bokeh.plotting import figure
from bokeh.models import FreehandDrawTool, ColumnDataSource, BoxEditTool

import apps.game.stream.tools as tools
import apps.game.stream.gui_elements  as ge
import apps.game.stream.flow as flow
import apps.game.stream.parameters as param

# Load Panel extension
pn.extension()

def start_game():
    # Create a Bokeh figure with explicit ranges and no default tools
    p = figure(
        tools=[],                  # Start with no default tools
        toolbar_location='above',  # Include the toolbar
        x_range=(0, 1),            # Set x-axis range from 0 to 1
        y_range=(0, 1),            # Set y-axis range from 0 to 1
        height=param.Nx,
        width=param.Ny,
        sizing_mode='stretch_both'
    )
    
    p.xaxis.visible = False
    p.yaxis.visible = False
    p.grid.grid_line_color = None  # Hide gridlines
    p.toolbar.logo = None  # Remove the Bokeh logo for a clean look
    
    
    freehand_renderer = ge.add_freehand(p)
    rect_renderer = ge.add_rect(p)
    draw_stream = flow.add_freehand_draw_stream()
    renderer_image = flow.add_raster_image(p)
    
    button_rasterize = pn.widgets.Button(name='Apply geometry', button_type="primary")
    def rasterize(event):
        flow.raster = tools.rasterize_boxes(flow.raster, ge.rect_source.data)
        flow.raster = tools.rasterize_segments(flow.raster, ge.freehand_source.data)
        ge.rect_source.data = dict(x=[], y=[], width=[], height=[], angle=[])
        ge.freehand_source.data = dict(xs=[], ys=[])
    button_rasterize.on_click(rasterize)
    
    button_clear = pn.widgets.Button(name='Clear Canvas', button_type="primary")
    def clear_canvas(event):
        flow.raster[:, :] = 0.
    button_clear.on_click(clear_canvas)
    
    button_reset = pn.widgets.Button(name='Reset simulation', button_type="primary")
    def reset_simulation(event):
        flow.setup()  
    button_reset.on_click(reset_simulation)
    
    button_start = pn.widgets.Button(name='Start/Stop', button_type="primary")
    def start_simulation(event):
        if flow.b_start == True:
             flow.b_start = False
        else:
            flow.b_start = True
    button_start.on_click(start_simulation)
    
    
    # Update the app layout to include the button
    app = pn.Column(p , pn.Row(button_start, button_rasterize, button_clear, button_reset))
    return app

# Serve the app
#app.servable()
#app.servable(static_dirs={'assets': './assets'})
