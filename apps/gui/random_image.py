import numpy as np
import holoviews as hv
from holoviews import streams
import panel as pn

# Initialize HoloViews and Panel extensions
hv.extension('bokeh')
pn.extension()

# Function to generate random image data
def random_image():
    return np.random.rand(400, 400)

# Function to update the image plot using a stream parameter
def update_image(next):
    return hv.Image(random_image())

# Create a stream that triggers updates every second with an initial value for 'next'
stream = streams.Stream.define('Next', next=0)()

# Create a DynamicMap with the update function and stream
dmap = hv.DynamicMap(update_image, streams=[stream])

# Panel to display the DynamicMap
panel = pn.panel(dmap)

# Use Panel's state management for periodic callbacks (Jupyter-friendly)
def periodic_callback():
    current_next = stream.contents['next']
    stream.event(next=current_next + 1)

pn.state.add_periodic_callback(periodic_callback, period=1000)  # Period in milliseconds

# If you're using this in a standalone script, make sure it's structured so it can be served.
if __name__.startswith("bokeh"):
    # This block ensures compatibility when running via 'panel serve'
    panel.servable()
