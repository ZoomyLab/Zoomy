import numpy as np
import jax.numpy as jnp
import holoviews as hv
from holoviews import streams
import panel as pn
from bokeh.models import ColumnDataSource, FreehandDrawTool, LinearColorMapper
from bokeh.palettes import Viridis256, Blues256

from time import time as get_time

import apps.game.stream.tools
import apps.game.stream.parameters as param

import apps.game.swe.swe._2d.solver as sim

# Initialize HoloViews and Panel extensions
hv.extension('bokeh')
pn.extension()

global path
global raster
global flow
global Q
global sim_step
global outflow_register
global b_finished
global time
global highscore
global b_submitted

highscore = []

b_submitted = False
time=0
b_finished = False

n = len(param.o_out) + len(param.o_top) + len(param.o_bot)
outflow_register = [0.] * n
progressbars = [pn.indicators.Progress(name='', value=0, max=100, width=50, bar_color = 'success') for i in range(n)]


sim_time = pn.indicators.Number(
    name='Zeit', value=param.end_time,font_size='36pt',  format='{value:.1f}')   

local_score = pn.indicators.Number(
    name='Punkte', value=0, font_size='36pt',
    colors=[(200, 'red'), (400, 'gold'), (450, 'green')]       )    

md_highscore = pn.pane.Markdown(
    """
    ## Highscore


    """)   

b_start = False

path = hv.Path([])
raster = np.zeros((param.Nx+2*param.n_ghosts, param.Ny+2*param.n_ghosts), dtype=np.uint8)

global color_mapper
# color_mapper = LinearColorMapper(palette=Viridis256, low=0, high=255, nan_color='black')
color_mapper = LinearColorMapper(palette=Blues256, low=0, high=255, nan_color='black')
global image_source



def setup():
    global Q
    global sim_step
    global outflow_register
    global b_finished
    global b_start
    global time
    global b_submitted
    b_submitted = False
    outflow_register = [0.] * n
    b_finished = False
    b_start = False
    time = 0.
    for bar in progressbars:
        bar.value = 0
    sim_time.value = param.end_time
    
    
    
    Q, sim_step = sim.setup()

setup()

def generate_image():
    global raster
    flow = Q[0, 1:-1, 1:-1]
    #flow = Q[0, :, :]
    flow = flow / flow.max() * 255
    ng = param.n_ghosts

    flow = np.where(raster[ng:-ng, ng:-ng] > 0, np.nan, flow)
    
    for o0, o1 in param.convert_to_wall(param.o_in):
        flow[o0:o1, 0:ng] = np.nan
    for o0, o1 in param.convert_to_wall(param.o_out):
        flow[o0:o1, -ng:] = np.nan  
    for o0, o1 in param.convert_to_wall(param.o_top):
        flow[-ng:, o0:o1] = np.nan
    for o0, o1 in param.convert_to_wall(param.o_bot):
        flow[0:ng, o0:o1] = np.nan
        
        
    #flow = flow / 0.5  * 255
    #flow = np.where(flow >= 250, np.nan, flow)
    #flow = np.roll(flow, 1, axis=0)

    out = np.zeros((param.Nx+2*ng, param.Ny+2*ng), dtype=np.uint8)
    out[ng:-ng, ng:-ng] = flow
    out = np.where(raster > 0, np.nan, out)
    for o0, o1 in param.o_in:
        for i in range(ng):
            out[ng+o0:ng+o1, i] = flow[o0:o1, 0]
    for o0, o1 in param.o_out:
        for i in range(ng):
            out[ng+o0:ng+o1, -i-1] = flow[o0:o1, -1]
    for o0, o1 in param.o_top:  
        for i in range(ng):
            out[-i-1, ng+o0:ng+o1] = flow[-1, o0:o1]
    for o0, o1 in param.o_bot:
        for i in range(ng):
            out[i, ng+o0:ng+o1] = flow[0, o0:o1]
            
    

    # return flow + np.where(raster > 0, np.nan, 0)
    return out

image_source = ColumnDataSource(data=dict(image=[generate_image()]))

def value_to_score(value, goal=3.):
    if value < goal:
        out =  value/goal * 100
    else:
        out =  max(0, 2*goal - value) / goal * 100
    return min(100, max(0, int(out+0.5)))

def value_to_color(value, goal=3.):
    if value < goal:
        return 'success'
    else:
        return 'danger'



def sum_values(event):
    local_score.value = sum(bar.value for bar in progressbars)


for bar in progressbars:
    bar.param.watch(sum_values, 'value')  
    
def submit_highscore(event):
    global b_finished
    global b_submitted
    if b_finished == True and b_submitted == False:
        b_submitted = True
        local_score.value = sum(bar.value for bar in progressbars)
        # print('submit highscore')
        global highscore
        highscore.append(local_score.value)
        
        highscore.sort(reverse=True)
        highscore = highscore[:10]
        highscore_text = "# Highscore\n"
        highscore_text += "\n".join([f"{i+1}. {s}" for i, s in enumerate(highscore)])
        # print(highscore_text)
        md_highscore.object = highscore_text

    
sim_time.param.watch(submit_highscore, 'value')

def update_progress():
    global outflow_register
    # print(outflow_register)
    # print([value_to_score(reg) for reg in outflow_register])
    for i, reg in enumerate(outflow_register):
        progressbars[i].value = value_to_score(reg)
        progressbars[i].bar_color = value_to_color(reg)
        
    sim_time.value=param.end_time - time




    

def update_image():
    global image_source
    new_image = generate_image()
    image_source.data = dict(image=[new_image])
    global Q
    global raster
    global sim_step
    global outflow_register
    global b_finished
    global b_start
    global time
    #I = np.where(raster > 0, 0., 1.)
    #Q = Q.at[0, 1:-1, 1:-1].multiply(I)
    #Q = Q.at[1, 1:-1, 1:-1].multiply(I)
    #Q = Q.at[2, 1:-1, 1:-1].multiply(I)
    tstart = get_time()
    ng  = param.n_ghosts
    if b_start:
        dt_acc = 0.
        for i in range(param.n_timesteps):
            Q, dt = sim_step(Q, outflow_register)
            time += dt
            dt_acc += dt
            if time > param.end_time:
                b_start = False
                b_finished = True
                time = param.end_time
        i_gauge = 0
        for i, [o0, o1] in enumerate(param.o_top):
            outflow_register[i_gauge] += float(np.sum(Q[2, -2, o0:o1]) * dt_acc)
            i_gauge += 1
        for i, [o0, o1] in enumerate(param.o_out):
            outflow_register[i_gauge] += float(np.sum(Q[1, o0:o1, -2]) * dt_acc)
            i_gauge += 1
        for i, [o0, o1] in enumerate(param.o_bot):
            outflow_register[i_gauge] += float(np.sum(-Q[2, 1, o0:o1]) * dt)
            i_gauge += 1
        # print(outflow_register)
        update_progress()


        Q = Q.at[3,1:-1,1:-1].set(raster[ng:-ng, ng:-ng])
        print(f'TIME FOR {param.n_timesteps} STEPS: {get_time()-tstart}')



def add_raster_image(p):
    global color_mapper
    renderer_image = p.image(
    'image',
    x=0,
    y=0,
    dw=1,
    dh=1,
    color_mapper=color_mapper,
    source=image_source,
    level="image"
    )
    return renderer_image

pn.state.add_periodic_callback(update_image, period=100)  


def on_draw(data):
    global raster
    raster = tools.rasterize_segments(raster, data)


def add_freehand_draw_stream():
    draw_stream = streams.FreehandDraw(source=path, num_objects=1)
    draw_stream.add_subscriber(on_draw)
    return draw_stream



#layout = pn.Column(reset_button, path.opts(line_color='blue', width=2, tools=['freehand_draw'],frame_width=800, frame_height=800) * dmap_flow)
#layout = pn.Column(reset_button, path.opts(line_color='blue', width=2, tools=['freehand_draw'],frame_width=800, frame_height=800) * dmap_flow)

