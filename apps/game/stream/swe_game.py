import panel as pn
import numpy as np
from bokeh.plotting import figure
from bokeh.models import FreehandDrawTool, ColumnDataSource, BoxEditTool

from panel.layout.gridstack import GridStack

import apps.game.stream.tools as tools
import apps.game.stream.gui_elements  as ge
import apps.game.stream.flow as flow
import apps.game.stream.parameters as param

# Load Panel extension
pn.extension("echarts", 'gridstack', 'mathjax')



# Create a Bokeh figure with explicit ranges and no default tools
p = figure(
    tools=[],                  # Start with no default tools
    toolbar_location=None,  # Include the toolbar
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
# rect_renderer = ge.add_rect(p)
draw_stream = flow.add_freehand_draw_stream()
renderer_image = flow.add_raster_image(p)

button_rasterize = pn.widgets.Button(name='Apply geometry', button_type="primary")
def rasterize(event):
    # flow.raster = tools.rasterize_boxes(flow.raster, ge.rect_source.data)
    flow.raster = tools.rasterize_segments(flow.raster, ge.freehand_source.data)
    # ge.rect_source.data = dict(x=[], y=[], width=[], height=[], angle=[])
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

gauges_top = [] 
for i in range(param.n_gauges_top):
    gauges_top.append(pn.indicators.Progress(name='', value=20, width=50))

gauges_out = [] 
for i in range(param.n_gauges_out):
    gauges_out.append(pn.Column(pn.indicators.Progress(name='', value=20, width=50), align='center', margin=0))

gauges_bot = [] 
for i in range(param.n_gauges_bot):
    gauges_bot.append(pn.Column(pn.indicators.Progress(name='', value=20, width=50), align='start', margin=0))



image_map = {
    'sad': 'apps/game/images/sad.png',
    'neutral': 'apps/game/images/neutral.png',
    'happy': 'apps/game/images/happy.png',
}


# Reactive image function
def image_from_value(value):
    if value < 50:
        choice = 'sad'
    elif value > 75:
        choice = 'happy'
    else:
        choice = 'neutral'
    return pn.Column(pn.pane.Image(image_map[choice], sizing_mode='stretch_both',
    margin=0))

image_gauges_top_0 = pn.bind(image_from_value, gauges_top[0].value)


app = GridStack(sizing_mode='stretch_both', min_height=600, allow_resize=False, allow_drag=False)

# row 0
app[0, 0:3] = pn.pane.Markdown(
    """
    # Gardeneer
    """
)

app[0, 3:10] = pn.pane.Markdown(
    """
    Overengineer what humanity is already doing since more than a millennium: *irrigation* 
    
    **Rules**: Draw an irregation system and make the farmers happy. Make sure you do not flood their fields.   
    """
    )
# row 1
# app[1 , 0:2] = pn.Spacer(styles=dict(background='orange'))
app[1 , 0:2] = pn.Spacer()

app[1 , 2:10] = pn.Row(pn.Spacer(width=80),image_gauges_top_0, pn.Column(gauges_top[0], margin=0), pn.Spacer(width=200), image_gauges_top_0, pn.Column(gauges_top[1], margin=0))

app[0:2, 10:12] = number = pn.indicators.Number(
    name='Your score', value=430,
    colors=[(200, 'red'), (400, 'gold'), (450, 'green')]
)
# row 2:10
app[2:10, 0:2] = pn.pane.Markdown(
    """
    # Highscore
    
    1. 10000
    2. 3000
    3. 1000
    4. 1000
    5. 0
    """)   
app[2:10, 2:10] = pn.Column(p, margin=5)
app[2:10, 10] = pn.Column(pn.Spacer(height=80), gauges_out[0], pn.Spacer(height=200), gauges_out[1])
app[2:10, 11] = pn.Spacer()  

# row 10
# app[10, 0:2] = pn.Spacer(styles=dict(background='blue'))    
app[10, 0:2] = pn.Spacer()    
app[10, 2:10] = pn.Row(pn.Spacer(width=200), gauges_bot[0], pn.Spacer(width=10))


# row 11
app[11, 0:2] = pn.Spacer()
app[11, 2:10] = pn.Row(button_start, button_rasterize, button_clear, button_reset)

app[10:12, 10:12] = pn.pane.PNG('./apps/game/images/logo.png', fixed_aspect=True)




# Update the app layout to include the button
# app = pn.Column(p , pn.Row(button_start, button_rasterize, button_clear, button_reset))
    #return app

# Serve the app
#app = start_game()
app.servable()
#app.servable(static_dirs={'assets': './assets'})
