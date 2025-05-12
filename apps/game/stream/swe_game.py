import panel as pn
import numpy as np
from bokeh.plotting import figure
from bokeh.models import FreehandDrawTool, ColumnDataSource, BoxEditTool

from panel.layout.gridstack import GridStack

import apps.game.stream.tools as tools
import apps.game.stream.gui_elements  as ge
import apps.game.stream.flow as flow
import apps.game.stream.parameters as param

from apps.game.stream.flow import outflow_register

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

button_clear = pn.widgets.Button(name='Clear drawing', button_type="primary")
def clear_canvas(event):
    flow.raster[:, :] = 0.
button_clear.on_click(clear_canvas)

button_reset = pn.widgets.Button(name='Reset simulation', button_type="primary")
def reset_simulation(event):
    flow.setup()  
    button_start.disabled=False

button_reset.on_click(reset_simulation)


button_start = pn.widgets.Button(name='Start irregation', button_type="primary")
def start_simulation(event):
    if flow.b_start == False:
         flow.b_start = True
         button_start.disabled=True
    # else:
    #     flow.b_start = True
    rasterize(event)
button_start.on_click(start_simulation)

gauges_top = [flow.progressbars[0], flow.progressbars[1]] 
# for i in range(param.n_gauges_top):
#     gauges_top.append(pn.indicators.Progress(name='', value=20, width=50))

gauges_out = [flow.progressbars[2], flow.progressbars[3]]
# for i in range(param.n_gauges_out):
#     gauges_out.append(pn.Column(pn.indicators.Progress(name='', value=20, width=50), align='center', margin=0))

gauges_bot = [flow.progressbars[4]]
# for i in range(param.n_gauges_bot):
#     gauges_bot.append(pn.Column(pn.indicators.Progress(name='', value=20, width=50), align='start', margin=0))



image_map = {
    'sad': 'apps/game/images/sad.png',
    'neutral': 'apps/game/images/neutral.png',
    'happy': 'apps/game/images/happy.png',
}



# Reactive image function
def image_from_value(value):
    score = value
    if score < 50:
        choice = 'sad'
    elif score > 75:
        choice = 'happy'
    else:
        choice = 'neutral'
    return pn.pane.Image(image_map[choice], sizing_mode='stretch_both')
    


def update_image(image_pane):
    def update(value):
        if value < 50:
            choice = 'sad'
        elif value > 75:
            choice = 'happy'
        else:
            choice = 'neutral'
        image_pane.object = image_map[choice]
        return image_pane
    return update



image_gauges_top_0 = image_from_value(0)
image_gauges_top_1 = image_from_value(0)
image_gauges_out_0 = image_from_value(0)
image_gauges_out_1 = image_from_value(0)
image_gauges_bot_0 = image_from_value(0)


image_gauges_top_0 = pn.bind(update_image(image_gauges_top_0), flow.progressbars[0].param.value)
image_gauges_top_1 = pn.bind(update_image(image_gauges_top_1), flow.progressbars[1].param.value)
image_gauges_out_0 = pn.bind(update_image(image_gauges_out_0), flow.progressbars[2].param.value)
image_gauges_out_1 = pn.bind(update_image(image_gauges_out_1), flow.progressbars[3].param.value) 
image_gauges_bot_0 = pn.bind(update_image(image_gauges_bot_0), flow.progressbars[4].param.value)
         
         

    
  


app = GridStack(sizing_mode='stretch_both', min_height=600, allow_resize=False, allow_drag=False)

# row 0
# app[0, 0:2] = pn.pane.Markdown(
#     """
#     # Gardeneer
#     """
# )

app[0:2, 0:2] = pn.Column(flow.sim_time)

app[0, 3:10] = pn.pane.Markdown(
    """
    # Supersonic irregation
    
    Draw an irregation system and make the farmers happy. Make sure you do not flood their fields.   
    """
    )
# row 1
# app[1 , 0:2] = pn.Spacer(styles=dict(background='orange'))
# app[1 , 0:2] = pn.Spacer()

app[1 , 2:4] = pn.Spacer()
app[1 , 4:6] = pn.Row(image_gauges_top_0, gauges_top[0])
app[1 , 6] = pn.Spacer()
app[1, 7:9]  = pn.Row(image_gauges_top_1, gauges_top[1])
# app[1 , 10] = pn.Spacer()

app[0:2, 10:12] = flow.local_score
# row 2:10
app[2:4, 0:2] = pn.Spacer()
app[4:6, 0:2] = pn.Row(pn.Spacer(height=50), pn.pane.PNG('./apps/game/images/inflow.png', fixed_aspect=True, sizing_mode='stretch_both'), sizing_mode='stretch_both')
app[6:10, 0:2] = flow.md_highscore

app[2:10, 2:10] = pn.Column(p, margin=5)

app[2, 10] = pn.Spacer()
app[3, 10] = pn.Column(image_gauges_out_0)
app[4, 10] = pn.Column(gauges_out[0])
app[5:6, 10] = pn.Row()
app[6, 10] = pn.Column(image_gauges_out_1)
app[7, 10] = pn.Column(gauges_out[1])

app[8:10, 10] = pn.Spacer()


app[2:10, 11] = pn.Spacer()  

# row 10
app[10, 0:6] = pn.Spacer()    
app[10, 6:8] = pn.Row(image_gauges_bot_0, gauges_bot[0])
app[10, 8:10] = pn.Spacer()    



# row 11
app[11, 0:2] = pn.Spacer()
# app[11, 2:10] = pn.Row(button_start, button_rasterize, button_clear, button_reset)
app[11, 2:10] = pn.Row(button_start, button_clear, button_reset)


app[10:12, 10:12] = pn.pane.PNG('./apps/game/images/logo.png', fixed_aspect=True)



# 3) For display, we can bind again, returning some text or an indicator
# score_display = pn.bind(lambda s: pn.pane.Markdown(f"**Total Score**: {s}"), total_score)

app.servable()
#app.servable(static_dirs={'assets': './assets'})


#TODO
# time as a clock that rotates?
# submit rating to score after try
# update score
# bind images to progress bars (currently disables)  