import numpy as np
import jax.numpy as jnp
import holoviews as hv
from holoviews import streams
import panel as pn
from bokeh.models import ColumnDataSource, FreehandDrawTool, LinearColorMapper
from bokeh.palettes import Viridis256, Blues256

from time import time as get_time

import apps.game.stream.tools as tools
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

b_start = False

path = hv.Path([])
raster = np.zeros((param.Nx, param.Ny), dtype=np.uint8)

def setup():
    global Q
    global sim_step
    Q, sim_step = sim.setup()

setup()

def generate_image():
    global raster
    flow = Q[0, 1:-1, 1:-1]
    flow = flow / flow.max() * 255
    flow = np.where(raster > 0, np.nan, flow)
    #flow = flow / 0.5  * 255
    #flow = np.where(flow >= 250, np.nan, flow)
    #flow = np.roll(flow, 1, axis=0)
    return flow + np.where(raster > 0, np.nan, 0)


def update_image():
    new_image = generate_image()
    image_source.data = dict(image=[new_image])
    global Q
    global raster
    global sim_step
    #I = np.where(raster > 0, 0., 1.)
    #Q = Q.at[0, 1:-1, 1:-1].multiply(I)
    #Q = Q.at[1, 1:-1, 1:-1].multiply(I)
    #Q = Q.at[2, 1:-1, 1:-1].multiply(I)
    tstart = get_time()
    if b_start:
        N = 10
        for i in range(N):
            Q = sim_step(Q)
        Q = Q.at[3,1:-1,1:-1].set(raster)
        print(f'TIME FOR {N} STEPS: {get_time()-tstart}')

image_source = ColumnDataSource(data=dict(image=[generate_image()]))
color_mapper = LinearColorMapper(palette=Viridis256, low=0, high=255, nan_color='black')

def add_raster_image(p):
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

