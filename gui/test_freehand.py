import numpy as np
import holoviews as hv
from holoviews import streams
import panel as pn

from stream_tools import rasterize

# Initialize HoloViews and Panel extensions
hv.extension('bokeh')
pn.extension()

path = hv.Path([])
raster = np.zeros((400, 400), dtype=np.uint8)
flow = np.zeros((400, 400), dtype=np.uint8)
flow[:10, :] = 255

############################################################################################3
#############################################FLOW###########################################3
############################################################################################3

def flow_image():
    global raster
    global flow
    flow = np.roll(flow, 1, axis=0)
    return flow + raster

def update_flow_image(next):
    return hv.Image(flow_image()).opts(alpha=0.3)

flow_stream = streams.Stream.define('Next', next=0)()

#dmap_flow = hv.DynamicMap(update_flow_image, streams=[flow_stream]).opts(tools=['reset'])

def periodic_callback():
    current_next = flow_stream.contents['next']
    flow_stream.event(next=current_next + 1)

dmap_flow = hv.DynamicMap(update_flow_image, streams=[flow_stream])

pn.state.add_periodic_callback(periodic_callback, period=200)  # Period in milliseconds

############################################################################################3
#############################################FREEHAND#######################################3
############################################################################################3

draw_stream = streams.FreehandDraw(source=path, num_objects=1)

def on_draw(data):
    global raster
    raster = rasterize(raster, data)

draw_stream.add_subscriber(on_draw)

############################################################################################3
#############################################GUI############################################3
############################################################################################3


def clear_canvas(event):
    global raster
    raster = np.zeros((400, 400), dtype=np.uint8)

reset_button = pn.widgets.Button(name='Clear Canvas')
reset_button.on_click(clear_canvas)

#layout = pn.Column(reset_button, path.opts(line_color='blue', width=2, tools=['freehand_draw'],frame_width=800, frame_height=800) * dmap_flow)
layout = pn.Column(reset_button, path.opts(line_color='blue', width=2, tools=['freehand_draw'],frame_width=800, frame_height=800) * dmap_flow)

if __name__.startswith("bokeh"):
    layout.servable()
